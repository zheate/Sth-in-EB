# title: 数据提取
"""
壳体测试数据查询主页面
重构后的模块化版本
"""

import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# 添加 pages 目录到路径以支持子模块导入
_pages_dir = str(Path(__file__).parent)
if _pages_dir not in sys.path:
    sys.path.insert(0, _pages_dir)

# 导入重构后的模块
from data_fetch import (
    # 常量
    PLOT_ORDER,
    SANITIZED_PLOT_ORDER,
    SANITIZED_ORDER_LOOKUP,
    STATION_COLORS,
    DEFAULT_PALETTE,
    OUTPUT_COLUMNS,
    SHELL_COLUMN,
    TEST_TYPE_COLUMN,
    CURRENT_COLUMN,
    POWER_COLUMN,
    VOLTAGE_COLUMN,
    EFFICIENCY_COLUMN,
    LAMBDA_COLUMN,
    SHIFT_COLUMN,
    WAVELENGTH_COLD_COLUMN,
    CURRENT_TOLERANCE,
    MODULE_MODE,
    CHIP_MODE,
    CHIP_TEST_CATEGORY,
    MEASUREMENT_OPTIONS,
    TEST_CATEGORY_OPTIONS,
    # 文件工具
    interpret_folder_input,
    interpret_chip_folder_input,
    resolve_test_folder,
    find_measurement_file,
    find_chip_measurement_file,
    build_chip_measurement_index,
    build_module_measurement_index_cached,
    # 数据提取
    extract_lvi_data,
    extract_rth_data,
    extract_generic_excel,
    clear_extraction_caches,
    align_output_columns,
    merge_measurement_rows,
    # 模型
    ensure_prediction_libs_loaded,
    # 图表
    build_multi_shell_chart,
    build_single_shell_dual_metric_chart,
)

from data_fetch.constants import (
    EXTRACTION_STATE_KEY,
    EXTRACTION_MODE_OPTIONS,
    EXTRACTION_MODE_LOOKUP,
    CHIP_SUPPORTED_MEASUREMENTS,
)

from data_fetch.ui_components import (
    show_toast,
    trigger_scroll_if_needed,
    render_extraction_results_section,
    parse_folder_entries,
    parse_current_points,
    init_session_state,
)

from data_fetch.file_utils import build_chip_measurement_index_cached

# 导入数据清洗工具
import sys
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from utils.data_cleaning import drop_zero_current


def _exclude_zero_current(df: pd.DataFrame) -> pd.DataFrame:
    """排除零电流数据"""
    if CURRENT_COLUMN not in df.columns or df.empty:
        return df
    return drop_zero_current(df, CURRENT_COLUMN, tol=CURRENT_TOLERANCE)


def do_measurement(
    entry_id: str,
    test_category: str,
    measurement_label: str,
    file_path: Path,
    file_mtime: float,
    multiple_found: bool,
    context_label: str,
    current_points: Optional[List[float]],
    effective_output_columns: List[str],
) -> Dict[str, Any]:
    """执行单个测量文件的数据提取"""
    try:
        if measurement_label == "LVI":
            extracted, missing_currents, lvi_full = extract_lvi_data(
                file_path=file_path,
                current_points=current_points,
                mtime=file_mtime,
            )
            extracted = _exclude_zero_current(extracted)
            lvi_full = _exclude_zero_current(lvi_full)
            selected_subset = extracted if current_points else None
            
            info_parts = []
            if missing_currents:
                info_parts.append(f"{context_label}: 未找到电流点 {missing_currents}")
            
            tagged = extracted.copy()
            tagged.insert(0, TEST_TYPE_COLUMN, test_category)
            tagged.insert(0, SHELL_COLUMN, entry_id)
            tagged = align_output_columns(tagged, columns=effective_output_columns)
            
            return {
                "tagged": tagged,
                "lvi": (entry_id, test_category, lvi_full, selected_subset),
                "rth": None,
                "info": [f"找到文件: {context_label} -> {file_path.name}"] + info_parts,
                "multiple": multiple_found,
                "context": context_label,
                "error": None,
            }
            
        elif measurement_label == "Rth":
            extracted, missing_currents, rth_full = extract_rth_data(
                file_path=file_path,
                current_points=current_points,
                mtime=file_mtime,
            )
            extracted = _exclude_zero_current(extracted)
            rth_full = _exclude_zero_current(rth_full)
            
            info_parts = []
            if missing_currents:
                info_parts.append(f"{context_label}: 未找到电流点 {missing_currents}")
            
            baseline_current = extracted.attrs.get("lambda_baseline_current")
            if baseline_current is not None and abs(baseline_current - 2.0) > CURRENT_TOLERANCE:
                info_parts.append(f"{context_label}: 波长shift基准使用 {baseline_current:.3f}A")
            
            tagged = extracted.copy()
            tagged.insert(0, TEST_TYPE_COLUMN, test_category)
            tagged.insert(0, SHELL_COLUMN, entry_id)
            tagged = align_output_columns(tagged, columns=effective_output_columns)
            
            return {
                "tagged": tagged,
                "lvi": None,
                "rth": (entry_id, test_category, rth_full),
                "info": [f"找到文件: {context_label} -> {file_path.name}"] + info_parts,
                "multiple": multiple_found,
                "context": context_label,
                "error": None,
            }
        else:
            extracted = extract_generic_excel(file_path, mtime=file_mtime)
            tagged = extracted.copy()
            tagged.insert(0, TEST_TYPE_COLUMN, test_category)
            tagged.insert(0, SHELL_COLUMN, entry_id)
            tagged = align_output_columns(tagged, columns=effective_output_columns)
            
            return {
                "tagged": tagged,
                "lvi": None,
                "rth": None,
                "info": [f"找到文件: {context_label} -> {file_path.name}"],
                "multiple": multiple_found,
                "context": context_label,
                "error": None,
            }
    except Exception as exc:
        return {
            "tagged": None,
            "lvi": None,
            "rth": None,
            "info": [],
            "multiple": multiple_found,
            "context": context_label,
            "error": f"{context_label}: {exc}",
        }


def render_sidebar(result_df: Optional[pd.DataFrame], extraction_state: Optional[Dict]) -> None:
    """渲染侧边栏"""
    with st.sidebar:
        st.title("📑 功能导航")
        st.markdown("---")
        
        st.markdown("### 📊 数据分析")
        
        if st.button("📈 单壳体分析", use_container_width=True):
            st.session_state.show_single_analysis = True
            st.session_state.show_multi_power = False
            st.session_state.show_multi_station = False
            st.session_state.show_boxplot = False
            st.query_params.update({"section": "single"})
            st.session_state.pending_scroll_target = "single"
        
        if st.button("📉 多壳体分析", use_container_width=True):
            st.session_state.show_multi_power = True
            st.session_state.show_single_analysis = False
            st.session_state.show_multi_station = False
            st.session_state.show_boxplot = False
            st.query_params.update({"section": "multi_power"})
            st.session_state.pending_scroll_target = "multi_power"
        
        if st.button("🔄 多站别分析", use_container_width=True):
            st.session_state.show_multi_station = True
            st.session_state.show_single_analysis = False
            st.session_state.show_multi_power = False
            st.session_state.show_boxplot = False
            st.query_params.update({"section": "multi_station"})
            st.session_state.pending_scroll_target = "multi_station"
        
        if st.button("📦 箱线图分析", use_container_width=True):
            st.session_state.show_boxplot = True
            st.session_state.show_single_analysis = False
            st.session_state.show_multi_power = False
            st.session_state.show_multi_station = False
            st.query_params.update({"section": "boxplot"})
            st.session_state.pending_scroll_target = "boxplot"
        
        st.markdown("---")
        
        # 快速统计信息
        if extraction_state and result_df is not None:
            st.markdown("### 📌 当前状态")
            col1, col2, col3 = st.columns(3)
            state_mode = extraction_state.get("form_mode", MODULE_MODE)
            sidebar_label = "壳体" if state_mode == MODULE_MODE else "芯片"
            
            with col1:
                st.metric(f"{sidebar_label}数", len(extraction_state.get("folder_entries", [])))
            with col2:
                st.metric("数据量", len(result_df))
            with col3:
                if TEST_TYPE_COLUMN in result_df.columns:
                    st.metric("站别数", result_df[TEST_TYPE_COLUMN].nunique())
            st.markdown("---")


def render_input_form(extraction_mode: str) -> Tuple[bool, bool, str, List[str], List[str], str]:
    """渲染输入表单"""
    folder_label = "壳体号或Ldtd路径" if extraction_mode == MODULE_MODE else "芯片名称或路径"
    folder_help = (
        "可输入一个或多个壳体号，每行一个，例如 HHD550048。也支持直接粘贴完整路径。"
        if extraction_mode == MODULE_MODE
        else "可输入一个或多个芯片名或完整路径，每行一个，例如 2019-12-120240。"
    )
    measurement_options = (
        [label for label in MEASUREMENT_OPTIONS.keys() if label in CHIP_SUPPORTED_MEASUREMENTS]
        if extraction_mode == CHIP_MODE
        else list(MEASUREMENT_OPTIONS.keys())
    )

    with st.form("input_form"):
        folder_input = st.text_area(
            folder_label,
            help=folder_help,
            key=f"folder_input_{extraction_mode}",
        )

        if extraction_mode == MODULE_MODE:
            selected_tests = st.multiselect(
                "选择测试类型",
                options=TEST_CATEGORY_OPTIONS,
                default=TEST_CATEGORY_OPTIONS,
                key="module_test_select",
            )
        else:
            selected_tests = [CHIP_TEST_CATEGORY]
            st.info("芯片模式会自动递归查找最新的 LVI / Rth 测试文件。", icon="ℹ️")

        selected_measurements = st.multiselect(
            "选择测试文件",
            options=measurement_options,
            default=measurement_options,
            key=f"measurement_select_{extraction_mode}",
        )

        current_input = st.text_input(
            "电流点",
            help="可选，默认最高电流点。输入 'a' 或 'A' 提取所有电流点。也可输入单值或范围（如 12~19）。",
            key=f"current_input_{extraction_mode}",
        )
        
        submit_col, refresh_col = st.columns(2)
        with submit_col:
            submitted = st.form_submit_button("🚀 开始抽取", use_container_width=True)
        with refresh_col:
            force_refresh = st.form_submit_button("♻️ 强制刷新缓存", use_container_width=True)

    return submitted, force_refresh, folder_input, selected_tests, selected_measurements, current_input


def process_extraction(
    folder_entries: List[str],
    selected_tests: List[str],
    selected_measurements: List[str],
    current_points: Optional[List[float]],
    extraction_mode: str,
    effective_output_columns: List[str],
) -> Tuple[List[pd.DataFrame], List[str], List[str], Dict, Dict]:
    """执行数据提取处理"""
    combined_frames: List[pd.DataFrame] = []
    error_messages: List[str] = []
    info_messages: List[str] = []
    lvi_plot_sources: Dict[Tuple[str, str], Tuple[pd.DataFrame, Optional[pd.DataFrame]]] = {}
    rth_plot_sources: Dict[Tuple[str, str], pd.DataFrame] = {}
    
    total_entries = len(folder_entries)
    entry_label = "壳体" if extraction_mode == MODULE_MODE else "芯片"
    
    if total_entries >= 20:
        st.info(f"{entry_label}数量较多，正在使用多线程加速处理...")
    
    # 创建进度显示
    progress_text = st.empty()
    progress_bar = st.progress(0.0)
    status_text = st.empty()
    progress_text.markdown(f"**正在处理 {total_entries} 个{entry_label}...**")
    
    # 增加线程数以提高并行度
    executor_workers = max(8, min(32, (os.cpu_count() or 4) * 4))
    futures = []
    total_tasks = 0
    
    def process_entry_module(entry: str, selected_tests: List[str], selected_measurements: List[str]):
        """并行处理单个模块条目的所有测试和测量"""
        results = []
        local_errors = []
        local_infos = []
        
        try:
            base_path = interpret_folder_input(entry)
            local_infos.append(f"解析路径: {entry} -> {base_path}")
        except ValueError as exc:
            local_errors.append(f"{entry}: {exc}")
            return results, local_errors, local_infos

        for test_category in selected_tests:
            try:
                test_folder = resolve_test_folder(base_path, test_category)
                measurement_index = build_module_measurement_index_cached(
                    str(test_folder), test_folder.stat().st_mtime
                )
            except FileNotFoundError as exc:
                local_errors.append(f"{entry}/{test_category}: {exc}")
                continue

            for measurement_label in selected_measurements:
                token = MEASUREMENT_OPTIONS[measurement_label]
                try:
                    file_path, multiple_found, file_mtime = find_measurement_file(
                        test_folder, token, index=measurement_index
                    )
                    # 直接执行测量提取
                    result = do_measurement(
                        entry, test_category, measurement_label,
                        file_path, file_mtime, multiple_found,
                        f"{entry}/{test_category}/{measurement_label}",
                        current_points, effective_output_columns,
                    )
                    results.append(result)
                except (FileNotFoundError, KeyError, ValueError) as exc:
                    local_errors.append(f"{entry}/{test_category}/{measurement_label}: {exc}")
                    continue
        
        return results, local_errors, local_infos

    def process_entry_chip(entry: str, selected_measurements: List[str]):
        """并行处理单个芯片条目的所有测量"""
        results = []
        local_errors = []
        local_infos = []
        
        try:
            chip_folder = interpret_chip_folder_input(entry)
            local_infos.append(f"解析芯片路径: {entry} -> {chip_folder}")
        except (ValueError, FileNotFoundError, NotADirectoryError) as exc:
            local_errors.append(f"{entry}: {exc}")
            return results, local_errors, local_infos

        try:
            measurement_index = build_chip_measurement_index(chip_folder)
        except (FileNotFoundError, NotADirectoryError) as exc:
            local_errors.append(f"{entry}: {exc}")
            return results, local_errors, local_infos

        for measurement_label in selected_measurements:
            token = MEASUREMENT_OPTIONS[measurement_label]
            try:
                file_path, multiple_found, file_mtime = find_chip_measurement_file(
                    chip_folder, token, index=measurement_index
                )
                result = do_measurement(
                    entry, CHIP_TEST_CATEGORY, measurement_label,
                    file_path, file_mtime, multiple_found,
                    f"{entry}/{measurement_label}",
                    current_points, effective_output_columns,
                )
                results.append(result)
            except FileNotFoundError as exc:
                local_errors.append(f"{entry}/{measurement_label}: {exc}")
                continue
        
        return results, local_errors, local_infos

    # 按条目级别并行处理（每个壳体/芯片一个任务）
    with ThreadPoolExecutor(max_workers=executor_workers) as executor:
        if extraction_mode == MODULE_MODE:
            futures = [
                executor.submit(process_entry_module, entry, selected_tests, selected_measurements)
                for entry in folder_entries
            ]
        else:
            futures = [
                executor.submit(process_entry_chip, entry, selected_measurements)
                for entry in folder_entries
            ]
        total_tasks = len(futures)
    
    # 收集结果（新格式：每个 future 返回一个条目的所有结果）
    completed = 0
    for fut in as_completed(futures):
        results_list, local_errors, local_infos = fut.result()
        
        # 收集错误和信息
        error_messages.extend(local_errors)
        info_messages.extend(local_infos)
        
        # 处理每个测量结果
        for res in results_list:
            if res.get("error"):
                error_messages.append(res["error"])
            else:
                tagged = res.get("tagged")
                if tagged is not None:
                    combined_frames.append(tagged)
                info_messages.extend(res.get("info", []))
                if res.get("multiple"):
                    info_messages.append(f"{res.get('context')}: 使用最新文件")
                
                lvi_tuple = res.get("lvi")
                if lvi_tuple:
                    e_id, t_cat, lvi_full, selected_subset = lvi_tuple
                    lvi_plot_sources[(e_id, t_cat)] = (lvi_full, selected_subset)
                
                rth_tuple = res.get("rth")
                if rth_tuple:
                    e_id, t_cat, rth_full = rth_tuple
                    rth_plot_sources[(e_id, t_cat)] = rth_full
        
        completed += 1
        progress_bar.progress(min(completed / max(1, total_tasks), 1.0))
        status_text.text(f"已完成 {completed}/{total_tasks} 个{entry_label}")

    progress_bar.empty()
    progress_text.empty()
    status_text.empty()
    
    return combined_frames, error_messages, info_messages, lvi_plot_sources, rth_plot_sources


def finalize_result_df(
    combined_frames: List[pd.DataFrame],
    effective_output_columns: List[str],
) -> Optional[pd.DataFrame]:
    """整理最终结果 DataFrame"""
    if not combined_frames:
        return None
    
    valid_frames: List[pd.DataFrame] = []
    for frame in combined_frames:
        if frame.empty:
            continue
        non_na_frame = frame.dropna(how="all")
        if non_na_frame.empty:
            continue
        non_na_frame = non_na_frame.loc[:, ~non_na_frame.isna().all()]
        if non_na_frame.empty:
            continue
        valid_frames.append(non_na_frame)
    
    if not valid_frames:
        return None
    
    result_df = pd.concat(valid_frames, ignore_index=True)
    
    # 效率转换为百分比
    if EFFICIENCY_COLUMN in result_df.columns:
        result_df[EFFICIENCY_COLUMN] = pd.to_numeric(result_df[EFFICIENCY_COLUMN], errors="coerce")
        result_df[EFFICIENCY_COLUMN] = result_df[EFFICIENCY_COLUMN].multiply(100).round(3)
    
    result_df = merge_measurement_rows(result_df, columns=effective_output_columns)
    
    # 数值列保留三位小数
    numeric_columns = [CURRENT_COLUMN, POWER_COLUMN, VOLTAGE_COLUMN, EFFICIENCY_COLUMN, LAMBDA_COLUMN, SHIFT_COLUMN]
    for col in numeric_columns:
        if col in result_df.columns:
            result_df[col] = pd.to_numeric(result_df[col], errors="coerce").round(3)
    
    # 按站别排序
    if TEST_TYPE_COLUMN in result_df.columns:
        result_df[TEST_TYPE_COLUMN] = pd.Categorical(
            result_df[TEST_TYPE_COLUMN], categories=PLOT_ORDER, ordered=True
        )
        if CURRENT_COLUMN in result_df.columns:
            result_df[CURRENT_COLUMN] = pd.to_numeric(result_df[CURRENT_COLUMN], errors="coerce")
            result_df = result_df.sort_values(by=[TEST_TYPE_COLUMN, CURRENT_COLUMN], kind="stable")
        else:
            result_df = result_df.sort_values(by=[TEST_TYPE_COLUMN], kind="stable")
        result_df[TEST_TYPE_COLUMN] = result_df[TEST_TYPE_COLUMN].astype("object").str.replace("测试", "", regex=False)
    
    return result_df


def render_multi_power_analysis(lvi_plot_sources: Dict, rth_plot_sources: Dict) -> None:
    """渲染多壳体功率分析"""
    st.markdown('<div id="multi_power"></div>', unsafe_allow_html=True)
    trigger_scroll_if_needed("multi_power")
    st.subheader("多壳体分析")
    
    shells = sorted({shell_id for shell_id, _ in lvi_plot_sources.keys()})
    
    if len(shells) == 0:
        show_toast("请先抽取数据", icon="⚠️")
        return
    
    if len(shells) > 10:
        show_toast(f"多壳体分析最多支持10个壳体，当前有{len(shells)}个壳体", icon="⚠️")
        return
    
    # 收集各指标数据
    power_tab_entries = []
    efficiency_tab_entries = []
    lambda_tab_entries = []
    
    for test_type in PLOT_ORDER:
        power_series = []
        efficiency_series = []
        lambda_series = []
        
        for shell_id in shells:
            data_tuple = lvi_plot_sources.get((shell_id, test_type))
            if not data_tuple:
                continue
            df_full, _ = data_tuple
            if df_full is None or df_full.empty:
                continue
            
            # 功率数据
            power_df = df_full.dropna(subset=[CURRENT_COLUMN, POWER_COLUMN])
            if not power_df.empty:
                power_numeric = power_df[[CURRENT_COLUMN, POWER_COLUMN]].copy()
                power_numeric[CURRENT_COLUMN] = pd.to_numeric(power_numeric[CURRENT_COLUMN], errors="coerce")
                power_numeric[POWER_COLUMN] = pd.to_numeric(power_numeric[POWER_COLUMN], errors="coerce")
                power_numeric = power_numeric.dropna()
                if not power_numeric.empty:
                    power_series.append((shell_id, power_numeric))
            
            # 效率数据
            efficiency_df = df_full.dropna(subset=[CURRENT_COLUMN, EFFICIENCY_COLUMN])
            if not efficiency_df.empty:
                efficiency_numeric = efficiency_df[[CURRENT_COLUMN, EFFICIENCY_COLUMN]].copy()
                efficiency_numeric[CURRENT_COLUMN] = pd.to_numeric(efficiency_numeric[CURRENT_COLUMN], errors="coerce")
                efficiency_numeric[EFFICIENCY_COLUMN] = pd.to_numeric(efficiency_numeric[EFFICIENCY_COLUMN], errors="coerce")
                efficiency_numeric = efficiency_numeric.dropna()
                if not efficiency_numeric.empty:
                    efficiency_series.append((shell_id, efficiency_numeric))
            
            # 波长数据
            rth_df = rth_plot_sources.get((shell_id, test_type))
            if rth_df is not None and not rth_df.empty:
                lambda_df = rth_df.dropna(subset=[CURRENT_COLUMN, LAMBDA_COLUMN])
                if not lambda_df.empty:
                    lambda_numeric = lambda_df[[CURRENT_COLUMN, LAMBDA_COLUMN]].copy()
                    lambda_numeric[CURRENT_COLUMN] = pd.to_numeric(lambda_numeric[CURRENT_COLUMN], errors="coerce")
                    lambda_numeric[LAMBDA_COLUMN] = pd.to_numeric(lambda_numeric[LAMBDA_COLUMN], errors="coerce")
                    lambda_numeric = lambda_numeric.dropna()
                    if not lambda_numeric.empty:
                        lambda_series.append((shell_id, lambda_numeric))
        
        if power_series:
            power_tab_entries.append((test_type, power_series))
        if efficiency_series:
            efficiency_tab_entries.append((test_type, efficiency_series))
        if lambda_series:
            lambda_tab_entries.append((test_type, lambda_series))
    
    if not power_tab_entries and not efficiency_tab_entries and not lambda_tab_entries:
        st.info("所选壳体在功率、效率和波长数据上缺少可对比的站别。")
        return
    
    # 创建主标签页
    tab_names = ["功率对比", "效率对比"]
    if lambda_tab_entries:
        tab_names.append("波长对比")
    main_tabs = st.tabs(tab_names)
    
    # 功率对比
    with main_tabs[0]:
        _render_metric_comparison_tabs(power_tab_entries, POWER_COLUMN, "功率(W)", "power")
    
    # 效率对比
    with main_tabs[1]:
        # 效率需要转换为百分比
        efficiency_percent_entries = []
        for test_type, series in efficiency_tab_entries:
            series_percent = []
            for shell_id, numeric in series:
                numeric_copy = numeric.copy()
                numeric_copy[EFFICIENCY_COLUMN] = numeric_copy[EFFICIENCY_COLUMN] * 100
                series_percent.append((shell_id, numeric_copy))
            efficiency_percent_entries.append((test_type, series_percent))
        _render_metric_comparison_tabs(efficiency_percent_entries, EFFICIENCY_COLUMN, "电光效率(%)", "eff")
    
    # 波长对比
    if lambda_tab_entries:
        with main_tabs[2]:
            _render_metric_comparison_tabs(lambda_tab_entries, LAMBDA_COLUMN, "波长(nm)", "lambda")


def _render_metric_comparison_tabs(
    tab_entries: List[Tuple[str, List]],
    metric_column: str,
    metric_label: str,
    key_prefix: str,
) -> None:
    """渲染指标对比标签页"""
    if not tab_entries:
        st.info(f"所选壳体在{metric_label}数据上缺少可对比的站别。")
        return
    
    tab_labels = [test_type.replace("测试", "") for test_type, _ in tab_entries]
    tabs = st.tabs(tab_labels)
    
    for tab, (test_type, series) in zip(tabs, tab_entries):
        with tab:
            chart = build_multi_shell_chart(series, metric_column, metric_label, test_type)
            if chart is not None:
                st.altair_chart(chart, theme="streamlit", use_container_width=True)
            else:
                st.info("无法生成对比图表")


def render_single_analysis(extraction_state: Dict, lvi_plot_sources: Dict) -> None:
    """渲染单壳体分析"""
    st.markdown('<div id="single"></div>', unsafe_allow_html=True)
    trigger_scroll_if_needed("single")
    
    if not extraction_state:
        show_toast("请先抽取数据后再进行分析", icon="⚠️")
        return
    
    folder_entries = extraction_state["folder_entries"]
    
    if len(folder_entries) != 1:
        show_toast("单壳体分析仅支持单个壳体号，请调整输入", icon="⚠️")
        return
    
    shell_id = folder_entries[0]
    st.subheader("电流-功率-电光效率曲线")

    available_entries = []
    for test_type in PLOT_ORDER:
        data_tuple = lvi_plot_sources.get((shell_id, test_type))
        if data_tuple is None:
            continue
        df_full, df_selected = data_tuple
        if df_full is None or df_full.empty:
            continue
        plot_df = df_full.dropna(subset=[CURRENT_COLUMN, POWER_COLUMN, EFFICIENCY_COLUMN])
        if plot_df.empty:
            continue
        available_entries.append((test_type, df_full, df_selected, plot_df))

    if not available_entries:
        show_toast("未找到可用于绘制的站别数据", icon="⚠️")
        return

    tab_labels = [entry[0].replace("测试", "") for entry in available_entries]
    tabs = st.tabs(tab_labels)
    
    plotted_any = False
    for tab, (test_type, df_full, df_selected, plot_df) in zip(tabs, available_entries):
        with tab:
            chart = build_single_shell_dual_metric_chart(plot_df, df_selected, shell_id, test_type)
            if chart is not None:
                st.altair_chart(chart, theme="streamlit", use_container_width=True)
                plotted_any = True
            else:
                st.info("无法生成趋势图表")

    if not plotted_any:
        show_toast("未找到可绘制的 LVI 数据", icon="⚠️")


def render_multi_station_analysis(lvi_plot_sources: Dict, rth_plot_sources: Dict, extraction_state: Dict) -> None:
    """渲染多站别分析"""
    if not lvi_plot_sources:
        st.info("请先抽取数据")
        return
    
    st.markdown('---')
    st.markdown('<div id="multi_station"></div>', unsafe_allow_html=True)
    trigger_scroll_if_needed("multi_station")
    st.subheader("📊 多站别分析")

    available_shells = sorted({shell_id for (shell_id, _) in lvi_plot_sources.keys()})

    # 多壳体平均值变化分析
    if len(available_shells) > 1:
        st.markdown("**📊 所有壳体平均值变化分析**")
        all_shells_data: List[pd.DataFrame] = []

        for shell_id in available_shells:
            for (sid, test_type), (df_full, _) in lvi_plot_sources.items():
                if sid == shell_id and df_full is not None and not df_full.empty:
                    temp_df = df_full.copy()
                    temp_df[TEST_TYPE_COLUMN] = test_type.replace("测试", "")
                    temp_df[SHELL_COLUMN] = shell_id
                    all_shells_data.append(temp_df)

            if rth_plot_sources:
                for (sid, test_type), rth_df in rth_plot_sources.items():
                    if sid == shell_id and rth_df is not None and not rth_df.empty:
                        for idx, lvi_df in enumerate(all_shells_data):
                            if (lvi_df[SHELL_COLUMN].iloc[0] == shell_id and 
                                lvi_df[TEST_TYPE_COLUMN].iloc[0] == test_type.replace("测试", "")):
                                rth_temp = rth_df.copy()
                                rth_temp[TEST_TYPE_COLUMN] = test_type.replace("测试", "")
                                rth_temp[SHELL_COLUMN] = shell_id
                                merged = pd.merge(
                                    lvi_df,
                                    rth_temp[[CURRENT_COLUMN, LAMBDA_COLUMN, SHIFT_COLUMN, TEST_TYPE_COLUMN, SHELL_COLUMN]],
                                    on=[CURRENT_COLUMN, TEST_TYPE_COLUMN, SHELL_COLUMN],
                                    how="outer",
                                )
                                all_shells_data[idx] = merged
                                break

        if all_shells_data:
            all_shells_df = pd.concat(all_shells_data, ignore_index=True)
            agg_dict: Dict[str, str] = {
                POWER_COLUMN: 'mean',
                EFFICIENCY_COLUMN: 'mean',
                VOLTAGE_COLUMN: 'mean',
            }
            if LAMBDA_COLUMN in all_shells_df.columns:
                agg_dict[LAMBDA_COLUMN] = 'mean'
            if SHIFT_COLUMN in all_shells_df.columns:
                agg_dict[SHIFT_COLUMN] = 'mean'

            avg_by_station = all_shells_df.groupby(TEST_TYPE_COLUMN).agg(agg_dict).reset_index()
            ordered_avg_types = [t for t in SANITIZED_PLOT_ORDER if t in avg_by_station[TEST_TYPE_COLUMN].unique()]

            avg_change_data: List[Dict[str, Union[str, float]]] = []
            for idx in range(len(ordered_avg_types) - 1):
                from_type = ordered_avg_types[idx]
                to_type = ordered_avg_types[idx + 1]
                from_row = avg_by_station[avg_by_station[TEST_TYPE_COLUMN] == from_type]
                to_row = avg_by_station[avg_by_station[TEST_TYPE_COLUMN] == to_type]
                if from_row.empty or to_row.empty:
                    continue

                avg_change_row: Dict[str, Union[str, float]] = {"变化": f"{from_type} -> {to_type}"}
                
                power_from = from_row[POWER_COLUMN].iloc[0]
                power_to = to_row[POWER_COLUMN].iloc[0]
                if pd.notna(power_from) and pd.notna(power_to):
                    avg_change_row["功率变化(W)"] = power_to - power_from

                eff_from = from_row[EFFICIENCY_COLUMN].iloc[0] * 100
                eff_to = to_row[EFFICIENCY_COLUMN].iloc[0] * 100
                if pd.notna(eff_from) and pd.notna(eff_to):
                    avg_change_row["效率变化(%)"] = eff_to - eff_from

                voltage_from = from_row[VOLTAGE_COLUMN].iloc[0]
                voltage_to = to_row[VOLTAGE_COLUMN].iloc[0]
                if pd.notna(voltage_from) and pd.notna(voltage_to):
                    avg_change_row["电压变化(V)"] = voltage_to - voltage_from

                if LAMBDA_COLUMN in avg_by_station.columns:
                    lambda_from = from_row[LAMBDA_COLUMN].iloc[0]
                    lambda_to = to_row[LAMBDA_COLUMN].iloc[0]
                    if pd.notna(lambda_from) and pd.notna(lambda_to):
                        avg_change_row["波长变化(nm)"] = lambda_to - lambda_from

                if SHIFT_COLUMN in avg_by_station.columns:
                    shift_from = from_row[SHIFT_COLUMN].iloc[0]
                    shift_to = to_row[SHIFT_COLUMN].iloc[0]
                    if pd.notna(shift_from) and pd.notna(shift_to):
                        avg_change_row["Shift变化(nm)"] = shift_to - shift_from

                avg_change_data.append(avg_change_row)

            if avg_change_data:
                avg_change_df = pd.DataFrame(avg_change_data)
                numeric_cols = [col for col in avg_change_df.columns if col != "变化"]
                for column in numeric_cols:
                    avg_change_df[column] = avg_change_df[column].apply(
                        lambda value: 0.0 if pd.notna(value) and abs(round(value, 3)) < 0.001
                        else round(value, 3) if pd.notna(value) else value
                    )

                for _, row in avg_change_df.iterrows():
                    st.markdown(f"**{row['变化']}**")
                    if not numeric_cols:
                        continue
                    cols = st.columns(len(numeric_cols))
                    for idx, column in enumerate(numeric_cols):
                        if column not in row or pd.isna(row[column]):
                            continue
                        value = row[column]
                        if "(W)" in column:
                            unit, label = "W", column.replace("(W)", "").strip()
                        elif "(%)" in column:
                            unit, label = "%", column.replace("(%)", "").strip()
                        elif "(V)" in column:
                            unit, label = "V", column.replace("(V)", "").strip()
                        elif "(nm)" in column:
                            unit, label = "nm", column.replace("(nm)", "").strip()
                        else:
                            unit, label = "", column
                        with cols[idx]:
                            st.metric(label=label, value=f"{abs(value):.3f}{unit}",
                                     delta=f"{value:+.3f}{unit}", delta_color="normal")
                    st.markdown("---")

        st.markdown("---")

    # 指标分析
    result_df_for_analysis = extraction_state.get("result_df") if extraction_state else None
    analysis_columns = [POWER_COLUMN, VOLTAGE_COLUMN, EFFICIENCY_COLUMN, LAMBDA_COLUMN, SHIFT_COLUMN]
    available_metrics = ([col for col in analysis_columns if col in result_df_for_analysis.columns]
                        if result_df_for_analysis is not None else [])
    per_type_records: List[Dict[str, Any]] = []

    if result_df_for_analysis is not None and not result_df_for_analysis.empty:
        if available_metrics and TEST_TYPE_COLUMN in result_df_for_analysis.columns:
            for test_type, group in result_df_for_analysis.groupby(TEST_TYPE_COLUMN):
                for column in available_metrics:
                    series = pd.to_numeric(group[column], errors="coerce").dropna()
                    if series.empty:
                        continue
                    per_type_records.append({
                        "站别": test_type, "指标": column,
                        "数量": int(series.count()),
                        "均值": round(series.mean(), 3),
                        "中位数": round(series.median(), 3),
                        "标准差": round(series.std(ddof=1), 3) if series.count() > 1 else 0.0,
                        "最小值": round(series.min(), 3),
                        "最大值": round(series.max(), 3),
                    })

        if available_metrics:
            with st.expander("📊 指标分析", expanded=True):
                if TEST_TYPE_COLUMN in result_df_for_analysis.columns:
                    available_test_types = [t for t in SANITIZED_PLOT_ORDER 
                                           if t in result_df_for_analysis[TEST_TYPE_COLUMN].unique()]
                    if available_test_types:
                        test_type_options = ["全部"] + available_test_types
                        selected_test_type = st.selectbox(
                            "选择站别进行统计", options=test_type_options,
                            index=len(test_type_options) - 1, key="stats_test_type_select"
                        )
                        if selected_test_type == "全部":
                            numeric_data = result_df_for_analysis[available_metrics].apply(pd.to_numeric, errors="coerce")
                            st.markdown("### 📈 全部数据统计")
                        else:
                            selected_test_df = result_df_for_analysis[result_df_for_analysis[TEST_TYPE_COLUMN] == selected_test_type]
                            numeric_data = selected_test_df[available_metrics].apply(pd.to_numeric, errors="coerce")
                            st.markdown(f"### 📈 {selected_test_type} 站数据统计")
                    else:
                        numeric_data = result_df_for_analysis[available_metrics].apply(pd.to_numeric, errors="coerce")
                        st.markdown("### 📈 全部数据统计")
                else:
                    numeric_data = result_df_for_analysis[available_metrics].apply(pd.to_numeric, errors="coerce")
                    st.markdown("### 📈 全部数据统计")

                counts = numeric_data.notna().sum()
                overall_summary = pd.DataFrame({
                    "数量": counts, "均值": numeric_data.mean(), "中位数": numeric_data.median(),
                    "标准差": numeric_data.std(ddof=1), "最小值": numeric_data.min(), "最大值": numeric_data.max(),
                })
                overall_summary["数量"] = overall_summary["数量"].astype("Int64")
                overall_summary["标准差"] = overall_summary["标准差"].fillna(0.0)
                summary_cols = ["均值", "中位数", "标准差", "最小值", "最大值"]
                overall_summary[summary_cols] = overall_summary[summary_cols].round(3)
                overall_summary.index.name = "指标"
                styled_summary = overall_summary.style.format({col: "{:.3f}" for col in summary_cols})
                st.dataframe(styled_summary, use_container_width=True)
        else:
            st.info("按站别统计缺少有效的数值列")

        if per_type_records:
            with st.expander("📋 按站别详细统计", expanded=False):
                ordered_cols = ["站别", "指标", "数量", "均值", "中位数", "标准差", "最小值", "最大值"]
                per_type_df = pd.DataFrame(per_type_records)[ordered_cols]
                unique_metrics = per_type_df["指标"].unique()

                for metric in unique_metrics:
                    metric_data = per_type_df[per_type_df["指标"] == metric].copy()
                    metric_data = metric_data.drop(columns=["指标"])
                    metric_data["__order"] = metric_data["站别"].map(SANITIZED_ORDER_LOOKUP)
                    metric_data = metric_data.sort_values("__order").drop(columns=["__order"])
                    metric_data = metric_data.set_index("站别")

                    st.markdown(f"#### 🔹 {metric}")
                    styled_metric = metric_data.style.format({
                        "均值": "{:.3f}", "中位数": "{:.3f}", "标准差": "{:.3f}",
                        "最小值": "{:.3f}", "最大值": "{:.3f}",
                    })
                    st.dataframe(styled_metric, use_container_width=True)

                    if len(metric_data) > 1:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.caption("均值对比")
                            st.bar_chart(metric_data["均值"], use_container_width=True)
                        with col2:
                            st.caption("标准差对比")
                            st.bar_chart(metric_data["标准差"], use_container_width=True)
    else:
        st.info("无可用数据")


def render_boxplot_analysis(lvi_plot_sources: Dict, rth_plot_sources: Dict, extraction_state: Dict) -> None:
    """渲染箱线图分析"""
    if not lvi_plot_sources:
        st.info("请先抽取数据")
        return
    
    st.markdown('---')
    st.markdown('<div id="boxplot"></div>', unsafe_allow_html=True)
    trigger_scroll_if_needed("boxplot")
    st.subheader("📊 箱线图分析")

    selected_currents: List[float] = []
    if extraction_state:
        selected_currents = extraction_state.get("current_points", []) or []

    # 收集箱线图数据
    all_data_for_boxplot: List[pd.DataFrame] = []
    
    for (shell_id, test_type), (df_full, df_selected) in lvi_plot_sources.items():
        if df_full is None or df_full.empty or CURRENT_COLUMN not in df_full.columns:
            continue

        if df_selected is not None and not df_selected.empty:
            base_df = df_selected.copy()
        else:
            base_df = df_full.copy()
            if selected_currents:
                filtered_mask = pd.Series(False, index=base_df.index)
                for current in selected_currents:
                    filtered_mask |= (base_df[CURRENT_COLUMN] - current).abs() <= CURRENT_TOLERANCE
                filtered_df = base_df.loc[filtered_mask]
                if not filtered_df.empty:
                    base_df = filtered_df.copy()
                else:
                    max_current = base_df[CURRENT_COLUMN].max()
                    if pd.notna(max_current):
                        base_df = base_df.loc[(base_df[CURRENT_COLUMN] - max_current).abs() <= CURRENT_TOLERANCE]
            else:
                max_current = base_df[CURRENT_COLUMN].max()
                if pd.notna(max_current):
                    base_df = base_df.loc[(base_df[CURRENT_COLUMN] - max_current).abs() <= CURRENT_TOLERANCE]

        if base_df.empty:
            continue

        tagged = base_df.copy()
        tagged[TEST_TYPE_COLUMN] = test_type.replace("测试", "")
        tagged[SHELL_COLUMN] = shell_id
        all_data_for_boxplot.append(tagged)

    if not all_data_for_boxplot:
        st.info("无可用数据")
        return

    combined_boxplot_df = pd.concat(all_data_for_boxplot, ignore_index=True)
    
    # 合并 Rth 数据
    if rth_plot_sources:
        rth_data_list: List[pd.DataFrame] = []
        for (shell_id, test_type), rth_df in rth_plot_sources.items():
            if rth_df is None or rth_df.empty or CURRENT_COLUMN not in rth_df.columns:
                continue
            rth_temp = rth_df.copy()
            if selected_currents:
                mask = pd.Series(False, index=rth_temp.index)
                for current in selected_currents:
                    mask |= (rth_temp[CURRENT_COLUMN] - current).abs() <= CURRENT_TOLERANCE
                filtered_rth = rth_temp.loc[mask]
                if filtered_rth.empty:
                    rth_max = rth_temp[CURRENT_COLUMN].max()
                    if pd.notna(rth_max):
                        filtered_rth = rth_temp.loc[(rth_temp[CURRENT_COLUMN] - rth_max).abs() <= CURRENT_TOLERANCE]
                rth_temp = filtered_rth
            else:
                rth_max = rth_temp[CURRENT_COLUMN].max()
                if pd.notna(rth_max):
                    rth_temp = rth_temp.loc[(rth_temp[CURRENT_COLUMN] - rth_max).abs() <= CURRENT_TOLERANCE]

            if rth_temp.empty:
                continue

            rth_temp[TEST_TYPE_COLUMN] = test_type.replace("测试", "")
            rth_temp[SHELL_COLUMN] = shell_id
            keep_cols = [SHELL_COLUMN, TEST_TYPE_COLUMN, CURRENT_COLUMN]
            if LAMBDA_COLUMN in rth_temp.columns:
                keep_cols.append(LAMBDA_COLUMN)
            if SHIFT_COLUMN in rth_temp.columns:
                keep_cols.append(SHIFT_COLUMN)
            rth_data_list.append(rth_temp[keep_cols])

        if rth_data_list:
            rth_combined = pd.concat(rth_data_list, ignore_index=True)
            cols_to_drop = [col for col in (LAMBDA_COLUMN, SHIFT_COLUMN) if col in combined_boxplot_df.columns]
            if cols_to_drop:
                combined_boxplot_df = combined_boxplot_df.drop(columns=cols_to_drop)
            combined_boxplot_df = pd.merge(
                combined_boxplot_df, rth_combined,
                on=[SHELL_COLUMN, TEST_TYPE_COLUMN, CURRENT_COLUMN],
                how="outer",
            )

    # 检查可用指标
    has_lambda = LAMBDA_COLUMN in combined_boxplot_df.columns and combined_boxplot_df[LAMBDA_COLUMN].notna().any()
    has_shift = SHIFT_COLUMN in combined_boxplot_df.columns and combined_boxplot_df[SHIFT_COLUMN].notna().any()

    tab_names = ["功率", "效率", "电压"]
    if has_lambda:
        tab_names.append("波长")
    if has_shift:
        tab_names.append("波长Shift")

    boxplot_tabs = st.tabs(tab_names)

    tab_idx = 0
    with boxplot_tabs[tab_idx]:
        tab_idx += 1
        _render_boxplot(combined_boxplot_df[[TEST_TYPE_COLUMN, POWER_COLUMN]].copy(), POWER_COLUMN, "功率(W)")

    with boxplot_tabs[tab_idx]:
        tab_idx += 1
        efficiency_data = combined_boxplot_df[[TEST_TYPE_COLUMN, EFFICIENCY_COLUMN]].copy()
        _render_boxplot(efficiency_data, EFFICIENCY_COLUMN, "效率(%)",
                       transform=lambda s: pd.to_numeric(s, errors="coerce") * 100)

    with boxplot_tabs[tab_idx]:
        tab_idx += 1
        _render_boxplot(combined_boxplot_df[[TEST_TYPE_COLUMN, VOLTAGE_COLUMN]].copy(), VOLTAGE_COLUMN, "电压(V)")

    if has_lambda:
        with boxplot_tabs[tab_idx]:
            tab_idx += 1
            _render_boxplot(combined_boxplot_df[[TEST_TYPE_COLUMN, LAMBDA_COLUMN]].copy(), LAMBDA_COLUMN, "波长(nm)")

    if has_shift:
        with boxplot_tabs[tab_idx]:
            _render_boxplot(combined_boxplot_df[[TEST_TYPE_COLUMN, SHIFT_COLUMN]].copy(), SHIFT_COLUMN, "波长Shift(nm)")

    st.markdown('---')


def _render_boxplot(data: pd.DataFrame, value_col: str, value_label: str, transform=None) -> None:
    """渲染单个箱线图"""
    if transform:
        data = data.copy()
        data[value_col] = transform(data[value_col])
    data = data.dropna()
    
    if data.empty:
        st.info(f"无{value_label}数据")
        return

    station_counts = data.groupby(TEST_TYPE_COLUMN).size()
    stations_with_enough = station_counts[station_counts >= 2].index.tolist()
    
    variance_threshold = 1e-10
    stations_with_data: List[str] = []
    stations_no_variance: List[str] = []
    
    for station in stations_with_enough:
        std_val = data[data[TEST_TYPE_COLUMN] == station][value_col].std()
        if std_val > variance_threshold:
            stations_with_data.append(station)
        else:
            stations_no_variance.append(station)

    if not stations_with_data:
        if stations_no_variance:
            st.info(f"以下站别数据无变化：{', '.join(stations_no_variance)}")
        else:
            st.info("各站别数据点不足（至少需要 2 个壳体的数据）")
        return

    filtered = data[data[TEST_TYPE_COLUMN].isin(stations_with_data)].copy()
    filtered["__order"] = filtered[TEST_TYPE_COLUMN].map(SANITIZED_ORDER_LOOKUP)
    filtered = filtered.sort_values("__order").drop(columns=["__order"])
    
    present_stations = [s for s in SANITIZED_PLOT_ORDER if s in stations_with_data]
    extras = [s for s in stations_with_data if s not in present_stations]
    present_stations.extend(extras)
    present_colors = [STATION_COLORS.get(s, "#000084") for s in present_stations]

    chart = (
        alt.Chart(filtered)
        .mark_boxplot(extent="min-max", size=50)
        .encode(
            x=alt.X(f"{TEST_TYPE_COLUMN}:N", title="Station", sort=present_stations,
                   axis=alt.Axis(labelAngle=-45)),
            y=alt.Y(f"{value_col}:Q", title=value_label, scale=alt.Scale(zero=False)),
            color=alt.Color(f"{TEST_TYPE_COLUMN}:N", legend=None,
                          scale=alt.Scale(domain=present_stations, range=present_colors)),
        )
        .properties(height=500, title=f"各站别{value_label}分布箱线图")
        .configure_title(fontSize=16, anchor="middle")
    )
    st.altair_chart(chart, use_container_width=True)

    # 统计分析
    if len(present_stations) > 1 and ensure_prediction_libs_loaded():
        _render_boxplot_statistics(filtered, value_col, present_stations)

    # 警告信息
    stations_insufficient = station_counts[station_counts < 2].index.tolist()
    warnings_list: List[str] = []
    if stations_insufficient:
        warnings_list.append(f"数据点不足（至少需要 2 个壳体）：{', '.join(stations_insufficient)}")
    if stations_no_variance:
        warnings_list.append(f"数据无变化：{', '.join(stations_no_variance)}")
    if warnings_list:
        st.caption("⚠️ " + "；".join(warnings_list))


def _render_boxplot_statistics(filtered: pd.DataFrame, value_col: str, present_stations: List[str]) -> None:
    """渲染箱线图统计分析"""
    from data_fetch.models import get_stats_module
    stats = get_stats_module()
    if stats is None:
        return
    
    stats_results = []
    
    for i in range(1, len(present_stations)):
        curr_name = present_stations[i]
        prev_name = present_stations[i-1]
        
        curr_series = filtered[filtered[TEST_TYPE_COLUMN] == curr_name][value_col]
        prev_series = filtered[filtered[TEST_TYPE_COLUMN] == prev_name][value_col]
        
        if curr_series.empty or prev_series.empty:
            continue
        
        curr_mean = curr_series.mean()
        prev_mean = prev_series.mean()
        
        # 变化百分比
        pct_change = (curr_mean - prev_mean) / abs(prev_mean) * 100 if prev_mean != 0 else np.nan
        
        # T-test
        p_value = np.nan
        sig_label = "N/A"
        try:
            _, p_val = stats.ttest_ind(curr_series, prev_series, equal_var=False, nan_policy='omit')
            p_value = p_val
            if p_val < 0.001:
                sig_label = "***"
            elif p_val < 0.01:
                sig_label = "**"
            elif p_val < 0.05:
                sig_label = "*"
            else:
                sig_label = "ns"
        except Exception:
            pass
        
        stats_results.append({
            "比较项": f"{curr_name} vs {prev_name}",
            "前序均值": prev_mean,
            "当前均值": curr_mean,
            "变化幅度(%)": pct_change,
            "P值": p_value,
            "显著性": sig_label
        })
    
    if stats_results:
        st.write("#### 📉 统计分析 (T-test)")
        st.caption("注：显著性标记 ***(p<0.001), **(p<0.01), *(p<0.05), ns(无显著差异)")
        
        df_stats = pd.DataFrame(stats_results)
        display_df = df_stats.copy()
        display_df["前序均值"] = display_df["前序均值"].apply(lambda x: f"{x:.4f}")
        display_df["当前均值"] = display_df["当前均值"].apply(lambda x: f"{x:.4f}")
        display_df["变化幅度(%)"] = display_df["变化幅度(%)"].apply(
            lambda x: f"{x:+.2f}%" if pd.notnull(x) else "N/A"
        )
        display_df["P值"] = display_df["P值"].apply(
            lambda x: f"{x:.4e}" if pd.notnull(x) else "N/A"
        )
        st.table(display_df)


def main() -> None:
    """主函数"""
    st.set_page_config(page_title="Excel 数据列提取", layout="wide")
    
    # 初始化 session state
    init_session_state()
    
    # 获取当前状态
    extraction_state = st.session_state.get(EXTRACTION_STATE_KEY)
    result_df = extraction_state["result_df"] if extraction_state else None
    
    # 渲染侧边栏
    render_sidebar(result_df, extraction_state)
    
    # 主标题
    st.title("壳体测试数据查询")
    st.caption("支持输入多个壳体号，按测试类型与测试文件批量提取数据。")
    st.markdown('<div id="input"></div>', unsafe_allow_html=True)

    # 模式选择
    mode_labels = [label for label, _ in EXTRACTION_MODE_OPTIONS]
    mode_label = st.radio(
        "数据提取模式",
        mode_labels,
        index=0,
        horizontal=True,
        key="data_fetch_mode",
    )
    extraction_mode = EXTRACTION_MODE_LOOKUP.get(mode_label, MODULE_MODE)

    # 渲染输入表单
    submitted, force_refresh, folder_input, selected_tests, selected_measurements, current_input = \
        render_input_form(extraction_mode)

    action_requested = submitted or force_refresh
    entry_label = "壳体" if extraction_mode == MODULE_MODE else "芯片"
    entry_prompt = "壳体号" if extraction_mode == MODULE_MODE else "芯片名或路径"

    # 检查输入是否变化
    previous_inputs_match = False
    if extraction_state and "form_folder_input" in extraction_state:
        previous_inputs_match = (
            folder_input == extraction_state.get("form_folder_input", "")
            and selected_tests == extraction_state.get("form_selected_tests", [])
            and selected_measurements == extraction_state.get("form_selected_measurements", [])
            and current_input == extraction_state.get("form_current_input", "")
            and extraction_mode == extraction_state.get("form_mode", MODULE_MODE)
        )

    # 输入变化时清除状态
    if extraction_state is not None and not action_requested and not previous_inputs_match:
        st.session_state[EXTRACTION_STATE_KEY] = None
        extraction_state = None

    # 强制刷新
    if force_refresh:
        clear_extraction_caches()
        st.session_state.pop(EXTRACTION_STATE_KEY, None)
        st.session_state.pop("lvi_plot_sources", None)
        st.session_state.pop("rth_plot_sources", None)
        extraction_state = None
        previous_inputs_match = False

    should_recompute = (
        force_refresh
        or extraction_state is None
        or (action_requested and not previous_inputs_match)
    )

    # 未请求操作且无状态时显示提示
    if not action_requested and extraction_state is None:
        st.info("填写参数后点击「开始提取」按钮")
        return

    # 重置分析状态
    if action_requested:
        st.session_state.show_multi_station = False
        st.session_state.show_boxplot = False
        st.session_state.show_single_analysis = False
        st.session_state.show_multi_power = False
        st.session_state.pending_scroll_target = None

        # 验证输入
        if not folder_input:
            st.toast(f"⚠️请填写{entry_prompt}", icon="⚠️")
            return
        if extraction_mode == MODULE_MODE and not selected_tests:
            st.toast("⚠️请至少选择一个测试类型", icon="⚠️")
            return
        if not selected_measurements:
            st.toast("⚠️请至少选择一个测试文件", icon="⚠️")
            return

        folder_entries = parse_folder_entries(folder_input)
        if not folder_entries:
            st.toast(f"⚠️未识别到有效的{entry_label}输入，请检查格式", icon="⚠️")
            return

        # 解析电流点
        current_points: Optional[List[float]] = []
        if current_input.strip():
            try:
                current_points = parse_current_points(current_input)
            except ValueError as exc:
                st.toast(f"⚠️{str(exc)}", icon="⚠️")
                return

        # 执行提取
        if should_recompute:
            st.session_state.lvi_plot_sources = {}
            st.session_state.rth_plot_sources = {}
            
            effective_output_columns = list(OUTPUT_COLUMNS)
            if extraction_mode == MODULE_MODE:
                if WAVELENGTH_COLD_COLUMN in effective_output_columns:
                    effective_output_columns.remove(WAVELENGTH_COLD_COLUMN)

            combined_frames, error_messages, info_messages, lvi_plot_sources, rth_plot_sources = \
                process_extraction(
                    folder_entries, selected_tests, selected_measurements,
                    current_points, extraction_mode, effective_output_columns
                )

            st.session_state.lvi_plot_sources = lvi_plot_sources
            st.session_state.rth_plot_sources = rth_plot_sources

            # 整理结果
            result_df = finalize_result_df(combined_frames, effective_output_columns)
            
            if result_df is None:
                st.toast("❌ 未能汇总出任何数据", icon="❌")
                if error_messages:
                    with st.expander(f"失败详情（{len(error_messages)} 条）", expanded=False):
                        for message in error_messages:
                            st.markdown(f"- {message}")
                st.session_state[EXTRACTION_STATE_KEY] = None
                return

            # 保存状态
            st.session_state[EXTRACTION_STATE_KEY] = {
                "folder_entries": folder_entries,
                "combined_frames": combined_frames,
                "error_messages": error_messages,
                "info_messages": info_messages,
                "result_df": result_df,
                "current_points": current_points,
                "form_folder_input": folder_input,
                "form_selected_tests": selected_tests,
                "form_selected_measurements": selected_measurements,
                "form_current_input": current_input,
                "form_mode": extraction_mode,
            }
            extraction_state = st.session_state[EXTRACTION_STATE_KEY]
    else:
        # 使用缓存的状态
        result_df = extraction_state["result_df"]
        error_messages = extraction_state["error_messages"]
        info_messages = extraction_state["info_messages"]

    # 渲染结果
    extraction_results_container = st.container()
    render_extraction_results_section(
        extraction_results_container,
        result_df,
        extraction_state.get("error_messages", []),
        extraction_state.get("info_messages", []),
        entity_label=entry_label,
    )

    # 获取绘图数据源
    lvi_plot_sources = st.session_state.get('lvi_plot_sources', {})
    rth_plot_sources = st.session_state.get('rth_plot_sources', {})

    # 渲染各分析模块
    if st.session_state.get('show_multi_power', False):
        render_multi_power_analysis(lvi_plot_sources, rth_plot_sources)

    if st.session_state.get('show_multi_station', False):
        render_multi_station_analysis(lvi_plot_sources, rth_plot_sources, extraction_state)

    if st.session_state.get('show_boxplot', False):
        render_boxplot_analysis(lvi_plot_sources, rth_plot_sources, extraction_state)

    if st.session_state.get('show_single_analysis', False):
        render_single_analysis(extraction_state, lvi_plot_sources)


if __name__ == "__main__":
    main()
