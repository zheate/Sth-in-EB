# title: 数据提取
"""壳体测试数据查询主页面 - 重构优化版"""

import os, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np, pandas as pd, streamlit as st, altair as alt

# 路径设置
_pages_dir = str(Path(__file__).parent)
if _pages_dir not in sys.path: sys.path.insert(0, _pages_dir)
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path: sys.path.insert(0, parent_dir)

# 导入模块
from data_fetch import (
    PLOT_ORDER, SANITIZED_PLOT_ORDER, SANITIZED_ORDER_LOOKUP, STATION_COLORS, DEFAULT_PALETTE,
    OUTPUT_COLUMNS, SHELL_COLUMN, TEST_TYPE_COLUMN, CURRENT_COLUMN, POWER_COLUMN, VOLTAGE_COLUMN,
    EFFICIENCY_COLUMN, LAMBDA_COLUMN, SHIFT_COLUMN, WAVELENGTH_COLD_COLUMN, CURRENT_TOLERANCE,
    MODULE_MODE, CHIP_MODE, CHIP_TEST_CATEGORY, MEASUREMENT_OPTIONS, TEST_CATEGORY_OPTIONS,
    interpret_folder_input, interpret_chip_folder_input, resolve_test_folder,
    find_measurement_file, find_chip_measurement_file, build_chip_measurement_index,
    build_module_measurement_index_cached, extract_lvi_data, extract_rth_data,
    extract_generic_excel, clear_extraction_caches, align_output_columns, merge_measurement_rows,
    ensure_prediction_libs_loaded, build_multi_shell_chart, build_single_shell_dual_metric_chart,
)
from data_fetch.constants import (
    EXTRACTION_STATE_KEY, EXTRACTION_MODE_OPTIONS, EXTRACTION_MODE_LOOKUP, CHIP_SUPPORTED_MEASUREMENTS,
)
from data_fetch.ui_components import (
    show_toast, trigger_scroll_if_needed, render_extraction_results_section,
    parse_folder_entries, parse_current_points, init_session_state,
)
from data_fetch.file_utils import build_chip_measurement_index_cached
from utils.data_cleaning import drop_zero_current


def _exclude_zero_current(df: pd.DataFrame) -> pd.DataFrame:
    """排除零电流数据"""
    return drop_zero_current(df, CURRENT_COLUMN, tol=CURRENT_TOLERANCE) if CURRENT_COLUMN in df.columns and not df.empty else df


def do_measurement(entry_id: str, test_category: str, measurement_label: str, file_path: Path,
                   file_mtime: float, multiple_found: bool, context_label: str,
                   current_points: Optional[List[float]], effective_output_columns: List[str]) -> Dict[str, Any]:
    """执行单个测量文件的数据提取"""
    try:
        info_parts = [f"找到文件: {context_label} -> {file_path.name}"]
        lvi_tuple, rth_tuple = None, None
        
        if measurement_label == "LVI":
            extracted, missing, lvi_full = extract_lvi_data(file_path=file_path, current_points=current_points, mtime=file_mtime)
            extracted, lvi_full = _exclude_zero_current(extracted), _exclude_zero_current(lvi_full)
            if missing: info_parts.append(f"{context_label}: 未找到电流点 {missing}")
            lvi_tuple = (entry_id, test_category, lvi_full, extracted if current_points else None)
        elif measurement_label == "Rth":
            extracted, missing, rth_full = extract_rth_data(file_path=file_path, current_points=current_points, mtime=file_mtime)
            extracted, rth_full = _exclude_zero_current(extracted), _exclude_zero_current(rth_full)
            if missing: info_parts.append(f"{context_label}: 未找到电流点 {missing}")
            baseline = extracted.attrs.get("lambda_baseline_current")
            if baseline and abs(baseline - 2.0) > CURRENT_TOLERANCE:
                info_parts.append(f"{context_label}: 波长shift基准使用 {baseline:.3f}A")
            rth_tuple = (entry_id, test_category, rth_full)
        else:
            extracted = extract_generic_excel(file_path, mtime=file_mtime)
        
        tagged = extracted.copy()
        tagged.insert(0, TEST_TYPE_COLUMN, test_category)
        tagged.insert(0, SHELL_COLUMN, entry_id)
        return {"tagged": align_output_columns(tagged, columns=effective_output_columns),
                "lvi": lvi_tuple, "rth": rth_tuple, "info": info_parts,
                "multiple": multiple_found, "context": context_label, "error": None}
    except Exception as exc:
        return {"tagged": None, "lvi": None, "rth": None, "info": [],
                "multiple": multiple_found, "context": context_label, "error": f"{context_label}: {exc}"}


def _set_analysis_mode(mode: str) -> None:
    """设置分析模式"""
    modes = {"single": "show_single_analysis", "multi_power": "show_multi_power",
             "multi_station": "show_multi_station", "boxplot": "show_boxplot"}
    for k, v in modes.items():
        st.session_state[v] = (k == mode)
    st.query_params.update({"section": mode})
    st.session_state.pending_scroll_target = mode


def _render_storage_section(result_df: Optional[pd.DataFrame], extraction_state: Optional[Dict]) -> None:
    """渲染数据存储区域（保存和加载）"""
    # Sidebar storage features removed per request
    return


def render_sidebar(result_df: Optional[pd.DataFrame], extraction_state: Optional[Dict]) -> None:
    """渲染侧边栏"""
    with st.sidebar:
        st.title("📑 功能导航")
        st.markdown("---")
        st.markdown("### 📊 数据分析")
        
        buttons = [("📈 单壳体分析", "single"), ("📉 多壳体分析", "multi_power"),
                   ("🔄 多站别分析", "multi_station"), ("📦 箱线图分析", "boxplot")]
        for label, mode in buttons:
            if st.button(label, use_container_width=True):
                _set_analysis_mode(mode)


def render_input_form(extraction_mode: str) -> Tuple[bool, bool, str, List[str], List[str], str]:
    """渲染输入表单"""
    is_module = extraction_mode == MODULE_MODE
    folder_label = "壳体号或Ldtd路径" if is_module else "芯片名称或路径"
    folder_help = ("可输入一个或多个壳体号，每行一个，例如 HHD550048。也支持直接粘贴完整路径。"
                   if is_module else "可输入一个或多个芯片名或完整路径，每行一个，例如 2019-12-120240。")
    meas_opts = list(MEASUREMENT_OPTIONS.keys()) if is_module else [k for k in MEASUREMENT_OPTIONS if k in CHIP_SUPPORTED_MEASUREMENTS]

    with st.form("input_form"):
        folder_input = st.text_area(folder_label, help=folder_help, key=f"folder_input_{extraction_mode}")
        if is_module:
            selected_tests = st.multiselect("选择测试类型", TEST_CATEGORY_OPTIONS, default=TEST_CATEGORY_OPTIONS, key="module_test_select")
        else:
            selected_tests = [CHIP_TEST_CATEGORY]
            st.info("芯片模式会自动递归查找最新的 LVI / Rth 测试文件。", icon="ℹ️")
        selected_measurements = st.multiselect("选择测试文件", meas_opts, default=meas_opts, key=f"measurement_select_{extraction_mode}")
        current_input = st.text_input("电流点", help="可选，默认最高电流点。输入 'a' 或 'A' 提取所有电流点。也可输入单值或范围（如 12~19）。", key=f"current_input_{extraction_mode}")
        c1, c2 = st.columns(2)
        submitted = c1.form_submit_button("🚀 开始抽取", use_container_width=True)
        force_refresh = c2.form_submit_button("♻️ 强制刷新缓存", use_container_width=True)
    return submitted, force_refresh, folder_input, selected_tests, selected_measurements, current_input


def process_extraction(folder_entries: List[str], selected_tests: List[str], selected_measurements: List[str],
                       current_points: Optional[List[float]], extraction_mode: str,
                       effective_output_columns: List[str]) -> Tuple[List[pd.DataFrame], List[str], List[str], Dict, Dict]:
    """执行数据提取处理"""
    combined_frames, error_messages, info_messages = [], [], []
    lvi_plot_sources: Dict[Tuple[str, str], Tuple[pd.DataFrame, Optional[pd.DataFrame]]] = {}
    rth_plot_sources: Dict[Tuple[str, str], pd.DataFrame] = {}
    
    total = len(folder_entries)
    entry_label = "壳体" if extraction_mode == MODULE_MODE else "芯片"
    if total >= 20: st.info(f"{entry_label}数量较多，正在使用多线程加速处理...")
    
    progress_text, progress_bar, status_text = st.empty(), st.progress(0.0), st.empty()
    progress_text.markdown(f"**正在处理 {total} 个{entry_label}...**")
    workers = max(8, min(32, (os.cpu_count() or 4) * 4))

    def process_module(entry: str):
        results, errors, infos = [], [], []
        try:
            base_path = interpret_folder_input(entry)
            infos.append(f"解析路径: {entry} -> {base_path}")
        except ValueError as e:
            return results, [f"{entry}: {e}"], infos
        for test in selected_tests:
            try:
                folder = resolve_test_folder(base_path, test)
                idx = build_module_measurement_index_cached(str(folder), folder.stat().st_mtime)
            except FileNotFoundError as e:
                errors.append(f"{entry}/{test}: {e}"); continue
            for meas in selected_measurements:
                try:
                    fp, multi, mt = find_measurement_file(folder, MEASUREMENT_OPTIONS[meas], index=idx)
                    results.append(do_measurement(entry, test, meas, fp, mt, multi, f"{entry}/{test}/{meas}", current_points, effective_output_columns))
                except (FileNotFoundError, KeyError, ValueError) as e:
                    errors.append(f"{entry}/{test}/{meas}: {e}")
        return results, errors, infos

    def process_chip(entry: str):
        results, errors, infos = [], [], []
        try:
            folder = interpret_chip_folder_input(entry)
            infos.append(f"解析芯片路径: {entry} -> {folder}")
        except (ValueError, FileNotFoundError, NotADirectoryError) as e:
            return results, [f"{entry}: {e}"], infos
        try:
            idx = build_chip_measurement_index(folder)
        except (FileNotFoundError, NotADirectoryError) as e:
            return results, [f"{entry}: {e}"], infos
        for meas in selected_measurements:
            try:
                fp, multi, mt = find_chip_measurement_file(folder, MEASUREMENT_OPTIONS[meas], index=idx)
                results.append(do_measurement(entry, CHIP_TEST_CATEGORY, meas, fp, mt, multi, f"{entry}/{meas}", current_points, effective_output_columns))
            except FileNotFoundError as e:
                errors.append(f"{entry}/{meas}: {e}")
        return results, errors, infos

    with ThreadPoolExecutor(max_workers=workers) as ex:
        proc = process_module if extraction_mode == MODULE_MODE else process_chip
        futures = [ex.submit(proc, e) for e in folder_entries]
    
    for i, fut in enumerate(as_completed(futures), 1):
        res_list, errs, infos = fut.result()
        error_messages.extend(errs); info_messages.extend(infos)
        for res in res_list:
            if res.get("error"): error_messages.append(res["error"])
            else:
                if res.get("tagged") is not None: combined_frames.append(res["tagged"])
                info_messages.extend(res.get("info", []))
                if res.get("multiple"): info_messages.append(f"{res.get('context')}: 使用最新文件")
                if res.get("lvi"): lvi_plot_sources[(res["lvi"][0], res["lvi"][1])] = (res["lvi"][2], res["lvi"][3])
                if res.get("rth"): rth_plot_sources[(res["rth"][0], res["rth"][1])] = res["rth"][2]
        progress_bar.progress(i / max(1, total)); status_text.text(f"已完成 {i}/{total} 个{entry_label}")
    
    progress_bar.empty(); progress_text.empty(); status_text.empty()
    return combined_frames, error_messages, info_messages, lvi_plot_sources, rth_plot_sources


def finalize_result_df(combined_frames: List[pd.DataFrame], effective_output_columns: List[str]) -> Optional[pd.DataFrame]:
    """整理最终结果 DataFrame"""
    if not combined_frames: return None
    
    valid = [f.dropna(how="all").loc[:, lambda x: ~x.isna().all()] for f in combined_frames if not f.empty]
    valid = [f for f in valid if not f.empty]
    if not valid: return None
    
    df = pd.concat(valid, ignore_index=True)
    if EFFICIENCY_COLUMN in df.columns:
        df[EFFICIENCY_COLUMN] = pd.to_numeric(df[EFFICIENCY_COLUMN], errors="coerce").multiply(100).round(3)
    
    df = merge_measurement_rows(df, columns=effective_output_columns)
    
    for col in [CURRENT_COLUMN, POWER_COLUMN, VOLTAGE_COLUMN, EFFICIENCY_COLUMN, LAMBDA_COLUMN, SHIFT_COLUMN]:
        if col in df.columns: df[col] = pd.to_numeric(df[col], errors="coerce").round(3)
    
    if TEST_TYPE_COLUMN in df.columns:
        df[TEST_TYPE_COLUMN] = pd.Categorical(df[TEST_TYPE_COLUMN], categories=PLOT_ORDER, ordered=True)
        sort_cols = [TEST_TYPE_COLUMN] + ([CURRENT_COLUMN] if CURRENT_COLUMN in df.columns else [])
        if CURRENT_COLUMN in df.columns: df[CURRENT_COLUMN] = pd.to_numeric(df[CURRENT_COLUMN], errors="coerce")
        df = df.sort_values(by=sort_cols, kind="stable")
        df[TEST_TYPE_COLUMN] = df[TEST_TYPE_COLUMN].astype("object").str.replace("测试", "", regex=False)
    return df


def _extract_metric_series(df: pd.DataFrame, cols: List[str]) -> Optional[pd.DataFrame]:
    """提取并清洗指标数据"""
    if df is None or df.empty: return None
    sub = df.dropna(subset=cols)
    if sub.empty: return None
    numeric = sub[cols].apply(pd.to_numeric, errors="coerce").dropna()
    return numeric if not numeric.empty else None


def render_multi_power_analysis(lvi_plot_sources: Dict, rth_plot_sources: Dict) -> None:
    """渲染多壳体功率分析"""
    st.markdown('<div id="multi_power"></div>', unsafe_allow_html=True)
    trigger_scroll_if_needed("multi_power")
    st.subheader("多壳体分析")
    
    shells = sorted({s for s, _ in lvi_plot_sources.keys()})
    if not shells: show_toast("请先抽取数据", icon="⚠️"); return
    if len(shells) > 10: show_toast(f"多壳体分析最多支持10个壳体，当前有{len(shells)}个", icon="⚠️"); return
    
    power_entries, eff_entries, lambda_entries = [], [], []
    for test in PLOT_ORDER:
        p_series, e_series, l_series = [], [], []
        for sid in shells:
            data = lvi_plot_sources.get((sid, test))
            if data:
                df_full = data[0]
                if (p := _extract_metric_series(df_full, [CURRENT_COLUMN, POWER_COLUMN])) is not None: p_series.append((sid, p))
                if (e := _extract_metric_series(df_full, [CURRENT_COLUMN, EFFICIENCY_COLUMN])) is not None: e_series.append((sid, e))
            rth = rth_plot_sources.get((sid, test))
            if (l := _extract_metric_series(rth, [CURRENT_COLUMN, LAMBDA_COLUMN])) is not None: l_series.append((sid, l))
        if p_series: power_entries.append((test, p_series))
        if e_series: eff_entries.append((test, e_series))
        if l_series: lambda_entries.append((test, l_series))
    
    if not any([power_entries, eff_entries, lambda_entries]):
        st.info("所选壳体在功率、效率和波长数据上缺少可对比的站别。"); return
    
    tabs = ["功率对比", "效率对比"] + (["波长对比"] if lambda_entries else [])
    main_tabs = st.tabs(tabs)
    with main_tabs[0]: _render_metric_comparison_tabs(power_entries, POWER_COLUMN, "功率(W)", "power")
    with main_tabs[1]:
        eff_pct = [(t, [(s, n.assign(**{EFFICIENCY_COLUMN: n[EFFICIENCY_COLUMN]*100})) for s, n in series]) for t, series in eff_entries]
        _render_metric_comparison_tabs(eff_pct, EFFICIENCY_COLUMN, "电光效率(%)", "eff")
    if lambda_entries:
        with main_tabs[2]: _render_metric_comparison_tabs(lambda_entries, LAMBDA_COLUMN, "波长(nm)", "lambda")


def _render_metric_comparison_tabs(tab_entries: List[Tuple[str, List]], metric_column: str, metric_label: str, key_prefix: str) -> None:
    """渲染指标对比标签页"""
    if not tab_entries: st.info(f"所选壳体在{metric_label}数据上缺少可对比的站别。"); return
    tabs = st.tabs([t.replace("测试", "") for t, _ in tab_entries])
    for tab, (test, series) in zip(tabs, tab_entries):
        with tab:
            chart = build_multi_shell_chart(series, metric_column, metric_label, test)
            if chart:
                st.altair_chart(chart, theme="streamlit", use_container_width=True)
            else:
                st.info("无法生成对比图表")


def render_single_analysis(extraction_state: Dict, lvi_plot_sources: Dict) -> None:
    """渲染单壳体分析"""
    st.markdown('<div id="single"></div>', unsafe_allow_html=True)
    trigger_scroll_if_needed("single")
    
    if not extraction_state: show_toast("请先抽取数据后再进行分析", icon="⚠️"); return
    entries = extraction_state["folder_entries"]
    if len(entries) != 1: show_toast("单壳体分析仅支持单个壳体号，请调整输入", icon="⚠️"); return
    
    shell_id = entries[0]
    st.subheader("电流-功率-电光效率曲线")
    
    available = []
    for test in PLOT_ORDER:
        data = lvi_plot_sources.get((shell_id, test))
        if not data or data[0] is None or data[0].empty: continue
        plot_df = data[0].dropna(subset=[CURRENT_COLUMN, POWER_COLUMN, EFFICIENCY_COLUMN])
        if not plot_df.empty: available.append((test, data[0], data[1], plot_df))
    
    if not available: show_toast("未找到可用于绘制的站别数据", icon="⚠️"); return
    
    tabs = st.tabs([e[0].replace("测试", "") for e in available])
    plotted = False
    for tab, (test, df_full, df_sel, plot_df) in zip(tabs, available):
        with tab:
            chart = build_single_shell_dual_metric_chart(plot_df, df_sel, shell_id, test)
            if chart:
                st.altair_chart(chart, theme="streamlit", use_container_width=True)
                plotted = True
            else:
                st.info("无法生成趋势图表")
    if not plotted: show_toast("未找到可绘制的 LVI 数据", icon="⚠️")


def _compute_station_changes(avg_df: pd.DataFrame, ordered_types: List[str]) -> List[Dict]:
    """计算站别间变化"""
    changes = []
    metrics = [(POWER_COLUMN, "功率变化(W)", 1), (EFFICIENCY_COLUMN, "效率变化(%)", 100),
               (VOLTAGE_COLUMN, "电压变化(V)", 1), (LAMBDA_COLUMN, "波长变化(nm)", 1), (SHIFT_COLUMN, "Shift变化(nm)", 1)]
    for i in range(len(ordered_types) - 1):
        f_type, t_type = ordered_types[i], ordered_types[i + 1]
        f_row, t_row = avg_df[avg_df[TEST_TYPE_COLUMN] == f_type], avg_df[avg_df[TEST_TYPE_COLUMN] == t_type]
        if f_row.empty or t_row.empty: continue
        row = {"变化": f"{f_type} -> {t_type}"}
        for col, name, mult in metrics:
            if col in avg_df.columns:
                fv, tv = f_row[col].iloc[0] * mult, t_row[col].iloc[0] * mult
                if pd.notna(fv) and pd.notna(tv): row[name] = tv - fv
        changes.append(row)
    return changes


def render_multi_station_analysis(lvi_plot_sources: Dict, rth_plot_sources: Dict, extraction_state: Dict) -> None:
    """渲染多站别分析"""
    if not lvi_plot_sources: st.info("请先抽取数据"); return
    
    st.markdown('---')
    st.markdown('<div id="multi_station"></div>', unsafe_allow_html=True)
    trigger_scroll_if_needed("multi_station")
    st.subheader("📊 多站别分析")

    shells = sorted({s for s, _ in lvi_plot_sources.keys()})
    
    if len(shells) > 1:
        st.markdown("**📊 所有壳体平均值变化分析**")
        all_data = []
        for sid in shells:
            for (s, test), (df, _) in lvi_plot_sources.items():
                if s == sid and df is not None and not df.empty:
                    tmp = df.assign(**{TEST_TYPE_COLUMN: test.replace("测试", ""), SHELL_COLUMN: sid})
                    all_data.append(tmp)
        
        if rth_plot_sources and isinstance(rth_plot_sources, dict):
            for i, df in enumerate(all_data):
                sid, test = df[SHELL_COLUMN].iloc[0], df[TEST_TYPE_COLUMN].iloc[0]
                rth = rth_plot_sources.get((sid, test + "测试"))
                if rth is None or (isinstance(rth, pd.DataFrame) and rth.empty):
                    rth = rth_plot_sources.get((sid, test))
                if rth is not None and isinstance(rth, pd.DataFrame) and not rth.empty:
                    rth_tmp = rth.assign(**{TEST_TYPE_COLUMN: test, SHELL_COLUMN: sid})
                    cols = [CURRENT_COLUMN, TEST_TYPE_COLUMN, SHELL_COLUMN] + [c for c in [LAMBDA_COLUMN, SHIFT_COLUMN] if c in rth_tmp.columns]
                    all_data[i] = pd.merge(df, rth_tmp[cols], on=[CURRENT_COLUMN, TEST_TYPE_COLUMN, SHELL_COLUMN], how="outer")
        
        if all_data:
            combined = pd.concat(all_data, ignore_index=True)
            agg = {c: 'mean' for c in [POWER_COLUMN, EFFICIENCY_COLUMN, VOLTAGE_COLUMN, LAMBDA_COLUMN, SHIFT_COLUMN] if c in combined.columns}
            avg = combined.groupby(TEST_TYPE_COLUMN).agg(agg).reset_index()
            ordered = [t for t in SANITIZED_PLOT_ORDER if t in avg[TEST_TYPE_COLUMN].unique()]
            changes = _compute_station_changes(avg, ordered)
            
            if changes:
                df_changes = pd.DataFrame(changes)
                num_cols = [c for c in df_changes.columns if c != "变化"]
                for c in num_cols:
                    df_changes[c] = df_changes[c].apply(lambda v: 0.0 if pd.notna(v) and abs(round(v, 3)) < 0.001 else round(v, 3) if pd.notna(v) else v)
                
                unit_map = {"(W)": "W", "(%)": "%", "(V)": "V", "(nm)": "nm"}
                for _, row in df_changes.iterrows():
                    st.markdown(f"**{row['变化']}**")
                    cols = st.columns(len(num_cols))
                    for i, col in enumerate(num_cols):
                        if col in row and pd.notna(row[col]):
                            unit = next((u for k, u in unit_map.items() if k in col), "")
                            label = col.replace(f"({unit})", "").strip() if unit else col
                            cols[i].metric(label=label, value=f"{abs(row[col]):.3f}{unit}", delta=f"{row[col]:+.3f}{unit}", delta_color="normal")
                    st.markdown("---")
        st.markdown("---")

    # 指标分析
    result_df = extraction_state.get("result_df") if extraction_state else None
    metrics = [POWER_COLUMN, VOLTAGE_COLUMN, EFFICIENCY_COLUMN, LAMBDA_COLUMN, SHIFT_COLUMN]
    avail_metrics = [c for c in metrics if result_df is not None and c in result_df.columns]
    
    if result_df is None or result_df.empty: st.info("无可用数据"); return
    
    per_type_records = []
    if avail_metrics and TEST_TYPE_COLUMN in result_df.columns:
        for test, grp in result_df.groupby(TEST_TYPE_COLUMN):
            for col in avail_metrics:
                s = pd.to_numeric(grp[col], errors="coerce").dropna()
                if not s.empty:
                    per_type_records.append({"站别": test, "指标": col, "数量": int(s.count()),
                        "均值": round(s.mean(), 3), "中位数": round(s.median(), 3),
                        "标准差": round(s.std(ddof=1), 3) if s.count() > 1 else 0.0,
                        "最小值": round(s.min(), 3), "最大值": round(s.max(), 3)})
    
    if avail_metrics:
        with st.expander("📊 指标分析", expanded=True):
            test_types = [t for t in SANITIZED_PLOT_ORDER if TEST_TYPE_COLUMN in result_df.columns and t in result_df[TEST_TYPE_COLUMN].unique()]
            opts = ["全部"] + test_types if test_types else ["全部"]
            sel = st.selectbox("选择站别进行统计", opts, index=len(opts)-1, key="stats_test_type_select")
            
            data = result_df if sel == "全部" else result_df[result_df[TEST_TYPE_COLUMN] == sel]
            st.markdown(f"### 📈 {'全部' if sel == '全部' else sel + ' 站'}数据统计")
            
            num_data = data[avail_metrics].apply(pd.to_numeric, errors="coerce")
            summary = pd.DataFrame({"数量": num_data.notna().sum().astype("Int64"), "均值": num_data.mean(),
                "中位数": num_data.median(), "标准差": num_data.std(ddof=1).fillna(0.0),
                "最小值": num_data.min(), "最大值": num_data.max()})
            for c in ["均值", "中位数", "标准差", "最小值", "最大值"]: summary[c] = summary[c].round(3)
            summary.index.name = "指标"
            st.dataframe(summary.style.format({c: "{:.3f}" for c in ["均值", "中位数", "标准差", "最小值", "最大值"]}), use_container_width=True)
    else:
        st.info("按站别统计缺少有效的数值列")
    
    if per_type_records:
        with st.expander("📋 按站别详细统计", expanded=False):
            df = pd.DataFrame(per_type_records)[["站别", "指标", "数量", "均值", "中位数", "标准差", "最小值", "最大值"]]
            for metric in df["指标"].unique():
                mdata = df[df["指标"] == metric].drop(columns=["指标"]).assign(__o=lambda x: x["站别"].map(SANITIZED_ORDER_LOOKUP)).sort_values("__o").drop(columns=["__o"]).set_index("站别")
                st.markdown(f"#### 🔹 {metric}")
                st.dataframe(mdata.style.format({"均值": "{:.3f}", "中位数": "{:.3f}", "标准差": "{:.3f}", "最小值": "{:.3f}", "最大值": "{:.3f}"}), use_container_width=True)
                if len(mdata) > 1:
                    c1, c2 = st.columns(2)
                    c1.caption("均值对比"); c1.bar_chart(mdata["均值"], use_container_width=True)
                    c2.caption("标准差对比"); c2.bar_chart(mdata["标准差"], use_container_width=True)


def _filter_by_current(df: pd.DataFrame, currents: List[float]) -> pd.DataFrame:
    """按电流点过滤数据"""
    if df is None or df.empty or CURRENT_COLUMN not in df.columns: return df
    if currents:
        mask = pd.Series(False, index=df.index)
        for c in currents: mask |= (df[CURRENT_COLUMN] - c).abs() <= CURRENT_TOLERANCE
        filtered = df.loc[mask]
        if not filtered.empty: return filtered
    max_c = df[CURRENT_COLUMN].max()
    return df.loc[(df[CURRENT_COLUMN] - max_c).abs() <= CURRENT_TOLERANCE] if pd.notna(max_c) else df


def render_boxplot_analysis(lvi_plot_sources: Dict, rth_plot_sources: Dict, extraction_state: Dict) -> None:
    """渲染箱线图分析"""
    if not lvi_plot_sources: st.info("请先抽取数据"); return
    
    st.markdown('---')
    st.markdown('<div id="boxplot"></div>', unsafe_allow_html=True)
    trigger_scroll_if_needed("boxplot")
    st.subheader("📊 箱线图分析")

    currents = (extraction_state.get("current_points", []) or []) if extraction_state else []
    
    # 收集 LVI 数据
    all_data = []
    for (sid, test), (df_full, df_sel) in lvi_plot_sources.items():
        if df_full is None or df_full.empty or CURRENT_COLUMN not in df_full.columns: continue
        base = df_sel if df_sel is not None and not df_sel.empty else _filter_by_current(df_full, currents)
        if not base.empty:
            all_data.append(base.assign(**{TEST_TYPE_COLUMN: test.replace("测试", ""), SHELL_COLUMN: sid}))
    
    if not all_data: st.info("无可用数据"); return
    combined = pd.concat(all_data, ignore_index=True)
    
    # 合并 Rth 数据
    if rth_plot_sources:
        rth_list = []
        for (sid, test), rth in rth_plot_sources.items():
            if rth is None or rth.empty or CURRENT_COLUMN not in rth.columns: continue
            filtered = _filter_by_current(rth, currents)
            if not filtered.empty:
                tmp = filtered.assign(**{TEST_TYPE_COLUMN: test.replace("测试", ""), SHELL_COLUMN: sid})
                cols = [SHELL_COLUMN, TEST_TYPE_COLUMN, CURRENT_COLUMN] + [c for c in [LAMBDA_COLUMN, SHIFT_COLUMN] if c in tmp.columns]
                rth_list.append(tmp[cols])
        if rth_list:
            rth_combined = pd.concat(rth_list, ignore_index=True)
            combined = combined.drop(columns=[c for c in [LAMBDA_COLUMN, SHIFT_COLUMN] if c in combined.columns], errors='ignore')
            combined = pd.merge(combined, rth_combined, on=[SHELL_COLUMN, TEST_TYPE_COLUMN, CURRENT_COLUMN], how="outer")

    # 渲染箱线图
    has_lambda = LAMBDA_COLUMN in combined.columns and combined[LAMBDA_COLUMN].notna().any()
    has_shift = SHIFT_COLUMN in combined.columns and combined[SHIFT_COLUMN].notna().any()
    tabs = ["功率", "效率", "电压"] + (["波长"] if has_lambda else []) + (["波长Shift"] if has_shift else [])
    box_tabs = st.tabs(tabs)
    
    configs = [(POWER_COLUMN, "功率(W)", None), (EFFICIENCY_COLUMN, "效率(%)", lambda s: pd.to_numeric(s, errors="coerce") * 100),
               (VOLTAGE_COLUMN, "电压(V)", None)]
    if has_lambda: configs.append((LAMBDA_COLUMN, "波长(nm)", None))
    if has_shift: configs.append((SHIFT_COLUMN, "波长Shift(nm)", None))
    
    for tab, (col, label, trans) in zip(box_tabs, configs):
        with tab: _render_boxplot(combined[[TEST_TYPE_COLUMN, col]].copy(), col, label, transform=trans)
    st.markdown('---')


def _render_boxplot(data: pd.DataFrame, value_col: str, value_label: str, transform=None) -> None:
    """渲染单个箱线图"""
    if transform: data = data.copy(); data[value_col] = transform(data[value_col])
    data = data.dropna()
    if data.empty: st.info(f"无{value_label}数据"); return

    counts = data.groupby(TEST_TYPE_COLUMN).size()
    enough = counts[counts >= 2].index.tolist()
    insufficient = counts[counts < 2].index.tolist()
    with_data = [s for s in enough if data[data[TEST_TYPE_COLUMN] == s][value_col].std() > 1e-10]
    no_var = [s for s in enough if s not in with_data]

    if not with_data:
        st.info(f"以下站别数据无变化：{', '.join(no_var)}" if no_var else "各站别数据点不足（至少需要 2 个壳体的数据）")
        return

    filtered = data[data[TEST_TYPE_COLUMN].isin(with_data)].assign(__o=lambda x: x[TEST_TYPE_COLUMN].map(SANITIZED_ORDER_LOOKUP)).sort_values("__o").drop(columns=["__o"])
    stations = [s for s in SANITIZED_PLOT_ORDER if s in with_data] + [s for s in with_data if s not in SANITIZED_PLOT_ORDER]
    colors = [STATION_COLORS.get(s, "#000084") for s in stations]

    chart = alt.Chart(filtered).mark_boxplot(extent="min-max", size=50).encode(
        x=alt.X(f"{TEST_TYPE_COLUMN}:N", title="Station", sort=stations, axis=alt.Axis(labelAngle=-45)),
        y=alt.Y(f"{value_col}:Q", title=value_label, scale=alt.Scale(zero=False)),
        color=alt.Color(f"{TEST_TYPE_COLUMN}:N", legend=None, scale=alt.Scale(domain=stations, range=colors)),
    ).properties(height=500, title=f"各站别{value_label}分布箱线图").configure_title(fontSize=16, anchor="middle")
    st.altair_chart(chart, use_container_width=True)

    if len(stations) > 1 and ensure_prediction_libs_loaded():
        _render_boxplot_statistics(filtered, value_col, stations)

    warns = []
    if insufficient: warns.append(f"数据点不足（至少需要 2 个壳体）：{', '.join(insufficient)}")
    if no_var: warns.append(f"数据无变化：{', '.join(no_var)}")
    if warns: st.caption("⚠️ " + "；".join(warns))


def _render_boxplot_statistics(filtered: pd.DataFrame, value_col: str, stations: List[str]) -> None:
    """渲染箱线图统计分析"""
    from data_fetch.models import get_stats_module
    stats_mod = get_stats_module()
    if stats_mod is None: return
    
    results = []
    for i in range(1, len(stations)):
        curr, prev = stations[i], stations[i-1]
        cs, ps = filtered[filtered[TEST_TYPE_COLUMN] == curr][value_col], filtered[filtered[TEST_TYPE_COLUMN] == prev][value_col]
        if cs.empty or ps.empty: continue
        cm, pm = cs.mean(), ps.mean()
        pct = (cm - pm) / abs(pm) * 100 if pm != 0 else np.nan
        try:
            _, pv = stats_mod.ttest_ind(cs, ps, equal_var=False, nan_policy='omit')
            sig = "***" if pv < 0.001 else "**" if pv < 0.01 else "*" if pv < 0.05 else "ns"
        except: pv, sig = np.nan, "N/A"
        results.append({"比较项": f"{curr} vs {prev}", "前序均值": pm, "当前均值": cm, "变化幅度(%)": pct, "P值": pv, "显著性": sig})
    
    if results:
        st.write("#### 📉 统计分析 (T-test)")
        st.caption("注：显著性标记 ***(p<0.001), **(p<0.01), *(p<0.05), ns(无显著差异)")
        df = pd.DataFrame(results)
        df["前序均值"] = df["前序均值"].apply(lambda x: f"{x:.4f}")
        df["当前均值"] = df["当前均值"].apply(lambda x: f"{x:.4f}")
        df["变化幅度(%)"] = df["变化幅度(%)"].apply(lambda x: f"{x:+.2f}%" if pd.notnull(x) else "N/A")
        df["P值"] = df["P值"].apply(lambda x: f"{x:.4e}" if pd.notnull(x) else "N/A")
        st.table(df)


def _auto_update_zh_database(result_df: pd.DataFrame, folder_entries: List[str], extraction_mode: str) -> None:
    """
    自动更新 Zh's DataBase 中已存在的壳体数据
    
    当用户在 Data_fetch 中查询壳体时，如果该壳体已存在于 Zh's DataBase 中，
    则自动更新其测试数据。
    """
    if result_df is None or result_df.empty:
        return
    
    if extraction_mode != MODULE_MODE:
        return  # 只处理模块模式
    
    try:
        # 动态导入以避免循环依赖
        from pages.Data_Manager import check_shell_in_database, update_shell_test_data
        
        updates = []
        for shell_id in folder_entries:
            shell_id = str(shell_id).strip()
            if not shell_id:
                continue
            
            # 检查壳体是否在数据库中
            if not check_shell_in_database(shell_id):
                continue
            
            # 获取该壳体的测试数据
            shell_data = result_df[result_df[SHELL_COLUMN] == shell_id] if SHELL_COLUMN in result_df.columns else pd.DataFrame()
            if shell_data.empty:
                continue
            
            # 收集测试数据
            test_data = {}
            for col in shell_data.columns:
                if col not in [SHELL_COLUMN, TEST_TYPE_COLUMN]:
                    # 取最新的非空值
                    values = shell_data[col].dropna()
                    if not values.empty:
                        test_data[col] = values.iloc[-1]
            
            # 获取最新站别
            current_station = None
            if TEST_TYPE_COLUMN in shell_data.columns:
                stations = shell_data[TEST_TYPE_COLUMN].dropna()
                if not stations.empty:
                    current_station = str(stations.iloc[-1])
            
            if test_data:
                updates.append({
                    "shell_id": shell_id,
                    "test_data": test_data,
                    "current_station": current_station
                })
        
        # 执行更新
        if updates:
            updated_count = 0
            for update in updates:
                if update_shell_test_data(
                    update["shell_id"],
                    update["test_data"],
                    update.get("current_station"),
                    source="data_fetch"
                ):
                    updated_count += 1
            
            if updated_count > 0:
                st.toast(f"✅ 已自动更新 Zh's DataBase 中 {updated_count} 个壳体的测试数据", icon="🗄️")
    
    except ImportError:
        pass  # Data_Manager 模块不可用时静默忽略
    except Exception as e:
        pass  # 更新失败时静默忽略，不影响主流程


def main() -> None:
    """主函数"""
    st.set_page_config(page_title="Excel 数据列提取", layout="wide")
    init_session_state()
    
    state = st.session_state.get(EXTRACTION_STATE_KEY)
    result_df = state["result_df"] if state else None
    render_sidebar(result_df, state)
    
    st.title("壳体测试数据查询")
    st.caption("支持输入多个壳体号，按测试类型与测试文件批量提取数据。")
    st.markdown('<div id="input"></div>', unsafe_allow_html=True)

    mode_label = st.radio("数据提取模式", [l for l, _ in EXTRACTION_MODE_OPTIONS], index=0, horizontal=True, key="data_fetch_mode")
    extraction_mode = EXTRACTION_MODE_LOOKUP.get(mode_label, MODULE_MODE)
    submitted, force_refresh, folder_input, selected_tests, selected_measurements, current_input = render_input_form(extraction_mode)
    
    action = submitted or force_refresh
    entry_label = "壳体" if extraction_mode == MODULE_MODE else "芯片"
    entry_prompt = "壳体号" if extraction_mode == MODULE_MODE else "芯片名或路径"
    extraction_state = state  # 使用之前获取的 state

    # 检查输入变化
    inputs_match = extraction_state and "form_folder_input" in extraction_state and all([
        folder_input == extraction_state.get("form_folder_input", ""),
        selected_tests == extraction_state.get("form_selected_tests", []),
        selected_measurements == extraction_state.get("form_selected_measurements", []),
        current_input == extraction_state.get("form_current_input", ""),
        extraction_mode == extraction_state.get("form_mode", MODULE_MODE)
    ])

    if extraction_state and not action and not inputs_match:
        st.session_state[EXTRACTION_STATE_KEY] = extraction_state = None

    if force_refresh:
        clear_extraction_caches()
        for k in [EXTRACTION_STATE_KEY, "lvi_plot_sources", "rth_plot_sources"]: st.session_state.pop(k, None)
        extraction_state, inputs_match = None, False

    recompute = force_refresh or extraction_state is None or (action and not inputs_match)
    if not action and extraction_state is None: st.info("填写参数后点击「开始提取」按钮"); return

    if action:
        for k in ["show_multi_station", "show_boxplot", "show_single_analysis", "show_multi_power"]: st.session_state[k] = False
        st.session_state.pending_scroll_target = None
        
        if not folder_input: st.toast(f"⚠️请填写{entry_prompt}", icon="⚠️"); return
        if extraction_mode == MODULE_MODE and not selected_tests: st.toast("⚠️请至少选择一个测试类型", icon="⚠️"); return
        if not selected_measurements: st.toast("⚠️请至少选择一个测试文件", icon="⚠️"); return

        folder_entries = parse_folder_entries(folder_input)
        if not folder_entries: st.toast(f"⚠️未识别到有效的{entry_label}输入，请检查格式", icon="⚠️"); return

        current_points: Optional[List[float]] = []
        if current_input.strip():
            try: current_points = parse_current_points(current_input)
            except ValueError as e: st.toast(f"⚠️{e}", icon="⚠️"); return

        if recompute:
            st.session_state.lvi_plot_sources = st.session_state.rth_plot_sources = {}
            out_cols = [c for c in OUTPUT_COLUMNS if not (extraction_mode == MODULE_MODE and c == WAVELENGTH_COLD_COLUMN)]
            
            frames, errors, infos, lvi_src, rth_src = process_extraction(folder_entries, selected_tests, selected_measurements, current_points, extraction_mode, out_cols)
            st.session_state.lvi_plot_sources, st.session_state.rth_plot_sources = lvi_src, rth_src
            
            result_df = finalize_result_df(frames, out_cols)
            if result_df is None:
                st.toast("❌ 未能汇总出任何数据", icon="❌")
                if errors:
                    with st.expander(f"失败详情（{len(errors)} 条）", expanded=False):
                        for m in errors: st.markdown(f"- {m}")
                st.session_state[EXTRACTION_STATE_KEY] = None; return

            st.session_state[EXTRACTION_STATE_KEY] = extraction_state = {
                "folder_entries": folder_entries, "combined_frames": frames, "error_messages": errors,
                "info_messages": infos, "result_df": result_df, "current_points": current_points,
                "form_folder_input": folder_input, "form_selected_tests": selected_tests,
                "form_selected_measurements": selected_measurements, "form_current_input": current_input, "form_mode": extraction_mode,
            }
            
            # 自动更新 Zh's DataBase 中已存在的壳体数据
            _auto_update_zh_database(result_df, folder_entries, extraction_mode)
    else:
        result_df, errors, infos = extraction_state["result_df"], extraction_state["error_messages"], extraction_state["info_messages"]

    render_extraction_results_section(st.container(), result_df, extraction_state.get("error_messages", []), extraction_state.get("info_messages", []), entity_label=entry_label)
    lvi_src, rth_src = st.session_state.get('lvi_plot_sources', {}), st.session_state.get('rth_plot_sources', {})

    if st.session_state.get('show_multi_power'): render_multi_power_analysis(lvi_src, rth_src)
    if st.session_state.get('show_multi_station'): render_multi_station_analysis(lvi_src, rth_src, extraction_state)
    if st.session_state.get('show_boxplot'): render_boxplot_analysis(lvi_src, rth_src, extraction_state)
    if st.session_state.get('show_single_analysis'): render_single_analysis(extraction_state, lvi_src)


if __name__ == "__main__":
    main()
