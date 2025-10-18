# title: 测试数据分析

from datetime import datetime
from pathlib import Path
from typing import List, Optional

import altair as alt
import pandas as pd
import streamlit as st
from utils.compat import inject_structured_clone_polyfill

REPORT_PREFIX = "常用测试数据报表"
ALLOWED_EXTENSIONS = (".xlsx", ".xls")
HOME_REPORTS_SESSION_KEY = "test_analysis_home_reports"
HOME_SELECTED_PATH_SESSION_KEY = "test_analysis_home_path"

STATION_ORDER: List[str] = ["耦合测试", "Pre测试", "低温储存后测试", "Post测试", "封盖测试"]

SUMMARY_COLUMNS: List[str] = ["最大效率", "功率", "电压", "最大电流", "热阻"]
NUMERIC_CANDIDATES: List[str] = SUMMARY_COLUMNS + [
    "峰值波长",
    "中心波长",
    "光谱全高宽",
    "NA",
]

TEST_TYPE_NORMALIZATION = {
    "耦合测试": "耦合测试",
    "耦合": "耦合测试",
    "pre测试": "Pre测试",
    "pretest": "Pre测试",
    "pre": "Pre测试",
    "post测试": "Post测试",
    "posttest": "Post测试",
    "post": "Post测试",
    "封盖测试": "封盖测试",
    "封盖": "封盖测试",
    "顶盖测试": "封盖测试",
    "顶盖": "封盖测试",
    "低温储存后测试": "低温储存后测试",
    "低温存储后测试": "低温储存后测试",
    "低温后测试": "低温储存后测试",
    "低温储存后試驗": "低温储存后测试",
    "低温储存后试验": "低温储存后测试",
    "complete": "已完成",
    "已完成": "已完成",
    "完成": "已完成",
}

# 字符标准化映射表（用于统一全角/半角字符等）
CHAR_NORMALIZATION = str.maketrans({
    "（": "(",
    "）": ")",
    "％": "%",
    "：": ":",
    "，": ",",
    "。": ".",
    "　": " ",  # 全角空格转半角
})


def is_supported_report(filename: str) -> bool:
    sanitized = filename.strip()
    lower_name = sanitized.lower()
    if not lower_name.endswith(ALLOWED_EXTENSIONS):
        st.error("仅支持扩展名为 .xlsx 或 .xls 的 Excel 报表")
        return False
    if not sanitized.startswith(REPORT_PREFIX):
        st.error(f"仅支持文件名以“{REPORT_PREFIX}”开头的 Excel 报表")
        return False
    return True


def load_report_from_path(file_path: str) -> Optional[pd.DataFrame]:
    path = Path(file_path)
    if not path.exists():
        st.error(f"选择的文件不存在：{file_path}")
        return None
    if not path.is_file():
        st.error(f"选择的路径不是文件：{file_path}")
        return None
    if not is_supported_report(path.name):
        return None
    try:
        df = pd.read_excel(path)
    except Exception as exc:  # pragma: no cover - surface to UI
        st.error(f"文件解析失败：{exc}")
        return None
    if df.empty:
        st.warning("选择的报表没有数据，请检查内容后重试。")
        return None
    return df


def normalize_text(value: object) -> str:
    text = "" if value is None else str(value)
    return text.translate(CHAR_NORMALIZATION).strip()


def normalize_test_type(value: object) -> Optional[str]:
    cleaned = normalize_text(value)
    if not cleaned:
        return None
    compact = cleaned.replace(" ", "").lower()
    return TEST_TYPE_NORMALIZATION.get(compact, cleaned)


def parse_uploaded_file(uploaded_file) -> Optional[pd.DataFrame]:
    if uploaded_file is None:
        return None

    try:
        if not is_supported_report(uploaded_file.name):
            return None
        df = pd.read_excel(uploaded_file)
    except Exception as exc:  # pragma: no cover - surface to UI
        st.error(f"文件解析失败：{exc}")
        return None

    if df.empty:
        st.warning("上传的文件没有数据，请检查内容后重试。")
        return None

    return df


def prepare_dataframe(raw: pd.DataFrame) -> Optional[pd.DataFrame]:
    df = raw.copy()
    df.rename(columns={col: normalize_text(col) for col in df.columns}, inplace=True)

    if "测试类型" not in df.columns or "壳体号" not in df.columns:
        st.error("文件缺少必需的列：需要包含“壳体号”和“测试类型”。")
        return None

    df["原始测试类型"] = df["测试类型"]
    df["标准测试站别"] = df["测试类型"].apply(normalize_test_type)
    df = df[df["标准测试站别"].isin(STATION_ORDER)].copy()

    if df.empty:
        st.warning("数据中未找到目标的 5 个测试站别，请确认文件内容。")
        return None

    df["壳体号"] = df["壳体号"].astype(str).str.strip()

    for optional in ["规格类型", "生产订单", "操作人"]:
        if optional in df.columns:
            df[optional] = df[optional].astype(str).str.strip()

    if "测试时间" in df.columns:
        df["测试时间"] = pd.to_datetime(df["测试时间"], errors="coerce")
        df["测试日期"] = df["测试时间"].dt.date
    else:
        df["测试日期"] = pd.NaT

    for column in NUMERIC_CANDIDATES:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    numeric_cols = [col for col in SUMMARY_COLUMNS if col in df.columns]
    sort_columns: List[str] = ["标准测试站别"]
    if "测试时间" in df.columns:
        sort_columns.append("测试时间")
    sort_columns.extend(numeric_cols)
    existing_sort_columns = [col for col in sort_columns if col in df.columns]
    if existing_sort_columns:
        df.sort_values(existing_sort_columns, inplace=True)

    return df.reset_index(drop=True)


def render_station_tab(station: str, station_df: pd.DataFrame) -> None:
    if station_df.empty:
        st.info(f"暂无 {station} 的数据。")
        return

    st.subheader(f"{station} 数据概览")

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("记录数", len(station_df))
    with col_b:
        if "最大效率" in station_df.columns and station_df["最大效率"].notna().any():
            st.metric("平均最大效率", f"{station_df['最大效率'].mean():.3f}")
        else:
            st.metric("平均最大效率", "—")
    with col_c:
        if "热阻" in station_df.columns and station_df["热阻"].notna().any():
            st.metric("平均热阻", f"{station_df['热阻'].mean():.3f}")
        else:
            st.metric("平均热阻", "—")

    stats_columns = [col for col in SUMMARY_COLUMNS if col in station_df.columns]
    if stats_columns:
        summary = station_df[stats_columns].agg(["count", "mean", "std", "min", "max"]).T
        summary.rename(
            columns={"count": "数量", "mean": "平均值", "std": "标准差", "min": "最小值", "max": "最大值"},
            inplace=True,
        )
        formatter_map = {col: "{:.3f}" for col in summary.columns if col != "数量"}
        st.dataframe(summary.style.format(formatter_map), width='stretch')

    base_columns = ["壳体号"]
    for optional in ["规格类型", "生产订单", "测试时间"]:
        if optional in station_df.columns:
            base_columns.append(optional)
    metric_columns = [col for col in NUMERIC_CANDIDATES if col in station_df.columns]
    display_cols = base_columns + metric_columns
    deduped = station_df[display_cols].copy()
    if "测试时间" in deduped.columns:
        deduped = deduped.sort_values("测试时间")
    else:
        deduped = deduped.sort_values("壳体号")
    st.dataframe(deduped.reset_index(drop=True), width='stretch', height=360)


def render_overview_table(filtered: pd.DataFrame) -> None:
    rows = []
    for station in STATION_ORDER:
        sub = filtered[filtered["标准测试站别"] == station]
        row = {"测试站别": station, "记录数": len(sub)}
        for metric in SUMMARY_COLUMNS:
            if metric in sub.columns and sub[metric].notna().any():
                row[f"{metric}均值"] = sub[metric].mean()
        # Add NA metric
        if "NA" in sub.columns and sub["NA"].notna().any():
            row["NA均值"] = sub["NA"].mean()
        rows.append(row)

    overview = pd.DataFrame(rows)
    # Transpose the table
    overview_transposed = overview.set_index("测试站别").T
    st.dataframe(
        overview_transposed.style.format(
            {
                col: "{:.3f}"
                for col in overview_transposed.columns
            },
            subset=pd.IndexSlice[overview_transposed.index != "记录数", :],
        ),
        width='stretch',
    )


alt.data_transformers.disable_max_rows()
st.set_page_config(page_title="常用测试数据分析", page_icon="📈", layout="wide")
inject_structured_clone_polyfill()

st.title("📈 常用测试数据分析")
st.markdown("上传常用测试数据报表，查看五个测试站别的指标表现。")

if "test_analysis_df" not in st.session_state:
    st.session_state.test_analysis_df = None
    st.session_state.test_analysis_filename = None
if HOME_SELECTED_PATH_SESSION_KEY not in st.session_state:
    st.session_state[HOME_SELECTED_PATH_SESSION_KEY] = None

uploaded = st.file_uploader(
    "上传测试数据（建议使用常用测试数据报表格式）",
    type=["xlsx", "xls"],
    help=f"仅支持文件名以“{REPORT_PREFIX}”开头的 Excel 报表。",
)

if uploaded is not None and uploaded.name != st.session_state.test_analysis_filename:
    with st.spinner("正在解析并加载数据..."):
        raw_df = parse_uploaded_file(uploaded)
        if raw_df is not None:
            prepared = prepare_dataframe(raw_df)
            if prepared is not None:
                st.session_state.test_analysis_df = prepared
                st.session_state.test_analysis_filename = uploaded.name
                st.session_state[HOME_SELECTED_PATH_SESSION_KEY] = None
                st.success(f"文件 {uploaded.name} 解析成功，共 {len(prepared)} 条记录。")

home_reports_raw = st.session_state.get(HOME_REPORTS_SESSION_KEY) or []
home_options_map: dict[str, str] = {}
selected_home_path: Optional[str] = None
reload_home_file = False

if home_reports_raw:
    st.markdown("#### 或从主页扫描的报表中选择")
    home_options = []
    for entry in home_reports_raw:
        if isinstance(entry, dict):
            candidate_path = entry.get("path")
            display_name = entry.get("name") or ""
        else:
            candidate_path = entry
            display_name = ""
        if not candidate_path:
            continue
        if not display_name:
            display_name = Path(candidate_path).name
        base_label = f"{display_name} | {candidate_path}"
        label = base_label
        suffix = 2
        while label in home_options_map:
            label = f"{base_label} ({suffix})"
            suffix += 1
        home_options_map[label] = candidate_path
        home_options.append(label)

    if home_options:
        select_col, refresh_col, reload_col = st.columns([5, 1, 1])
        with select_col:
            selected_label = st.selectbox(
                "主页识别的 Excel 报表",
                options=home_options,
                key="test_analysis_home_file_select",
            )
        with refresh_col:
            st.markdown("<div style='margin-top: 32px;'></div>", unsafe_allow_html=True)
            refresh_home_reports = st.button(
                "🔄 刷新",
                width='stretch',
                key="test_analysis_refresh_home_reports",
            )
        with reload_col:
            st.markdown("<div style='margin-top: 32px;'></div>", unsafe_allow_html=True)
            reload_home_file = st.button(
                "重新加载",
                width='stretch',
                key="test_analysis_reload_home_report",
            )
        if refresh_home_reports:
            st.rerun()
        selected_home_path = home_options_map[selected_label]
        st.caption("提示：列表来源于主页的数据文件浏览功能。")
    else:
        st.caption(f"未在主页找到以“{REPORT_PREFIX}”开头的 Excel 报表。")

if selected_home_path:
    last_loaded_path = st.session_state.get(HOME_SELECTED_PATH_SESSION_KEY)
    if reload_home_file:
        last_loaded_path = None
    if last_loaded_path != selected_home_path:
        display_name = Path(selected_home_path).name
        with st.spinner(f"正在加载 {display_name}..."):
            home_raw_df = load_report_from_path(selected_home_path)
            if home_raw_df is not None:
                prepared_home = prepare_dataframe(home_raw_df)
                if prepared_home is not None:
                    st.session_state.test_analysis_df = prepared_home
                    st.session_state.test_analysis_filename = display_name
                    st.session_state[HOME_SELECTED_PATH_SESSION_KEY] = selected_home_path
                    st.success(f"文件 {display_name} 加载成功，共 {len(prepared_home)} 条记录。")

dataframe = st.session_state.test_analysis_df
if dataframe is None:
    st.info("请先上传测试数据报表。")
    st.stop()

filtered_df = dataframe.copy()

filters_row = st.columns(4)

with filters_row[0]:
    part_options = sorted(filtered_df["规格类型"].dropna().unique()) if "规格类型" in filtered_df.columns else []
    selected_parts = st.multiselect("规格类型", part_options, default=part_options)
    if selected_parts:
        filtered_df = filtered_df[filtered_df["规格类型"].isin(selected_parts)]

with filters_row[1]:
    order_options = sorted(filtered_df["生产订单"].dropna().unique()) if "生产订单" in filtered_df.columns else []
    selected_orders = st.multiselect("生产订单", order_options, default=order_options)
    if selected_orders:
        filtered_df = filtered_df[filtered_df["生产订单"].isin(selected_orders)]

with filters_row[2]:
    station_options = STATION_ORDER
    selected_stations = st.multiselect("测试站别", station_options, default=station_options)
    if selected_stations:
        filtered_df = filtered_df[filtered_df["标准测试站别"].isin(selected_stations)]
    else:
        filtered_df = filtered_df.iloc[0:0]

with filters_row[3]:
    if "测试时间" in filtered_df.columns and filtered_df["测试时间"].notna().any():
        min_date = filtered_df["测试时间"].min().date()
        max_date = filtered_df["测试时间"].max().date()
        start, end = st.date_input(
            "测试日期区间",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )
        if start and end:
            mask = filtered_df["测试日期"].between(start, end)
            filtered_df = filtered_df[mask]
    else:
        st.write("测试时间缺失")

if filtered_df.empty:
    st.warning("筛选条件下没有数据，请调整过滤器。")
    st.stop()

col_left, col_mid, col_right = st.columns(3)
with col_left:
    st.metric("筛选后记录数", len(filtered_df))
with col_mid:
    unique_shells = filtered_df["壳体号"].nunique()
    st.metric("壳体数量", unique_shells)
with col_right:
    if "测试时间" in filtered_df.columns and filtered_df["测试时间"].notna().any():
        latest_time = filtered_df["测试时间"].max()
        st.metric("最新测试时间", latest_time.strftime("%Y-%m-%d %H:%M"))

st.markdown("### 站别概览")
render_overview_table(filtered_df)

csv_bytes = filtered_df.to_csv(index=False).encode("utf-8-sig")
st.download_button(
    "📥 下载筛选后的数据（CSV）",
    data=csv_bytes,
    file_name=f"测试数据筛选_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
    mime="text/csv",
)

station_tabs = st.tabs(STATION_ORDER)
for tab, station in zip(station_tabs, STATION_ORDER):
    with tab:
        station_data = filtered_df[filtered_df["标准测试站别"] == station]
        render_station_tab(station, station_data)
