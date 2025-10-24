# title: 测试数据分析

from datetime import datetime
from pathlib import Path
from typing import List, Optional
import hashlib
import io
import re
import sys

import altair as alt
import pandas as pd
import streamlit as st

APP_ROOT = Path(__file__).resolve().parent.parent
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from config import DEFAULT_DATA_FOLDER


def resolve_input_path(path_str: str) -> Path:
    normalized = path_str.strip()
    if not normalized:
        raise ValueError("路径不能为空")

    candidate = Path(normalized).expanduser()
    if not candidate.is_absolute():
        candidate = APP_ROOT / candidate
    return candidate.resolve()


REPORT_PREFIX = "常用测试数据报表"
ALLOWED_EXTENSIONS = (".xlsx", ".xls")
FOLDER_SOURCE_LABEL = "📁 从文件夹选择"
UPLOAD_SOURCE_LABEL = "📤 上传文件"
DATA_SOURCE_OPTIONS = [FOLDER_SOURCE_LABEL, UPLOAD_SOURCE_LABEL]

STATION_ORDER: List[str] = ["耦合测试", "Pre测试", "低温储存后测试", "Post测试", "封盖测试"]

SUMMARY_COLUMNS: List[str] = ["最大效率", "功率", "电压", "最大电流", "热阻", "NA"]
NUMERIC_CANDIDATES: List[str] = SUMMARY_COLUMNS + [
    "峰值波长",
    "中心波长",
    "光谱全高宽",
]

MAX_AUTOMATIC_SELECTION = 80
MAX_MULTISELECT_OPTIONS = 200

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


@st.cache_data(show_spinner=False)
def _read_report_from_disk(path_str: str, mtime: float) -> pd.DataFrame:
    return pd.read_excel(path_str)


@st.cache_data(show_spinner=False)
def _read_report_from_bytes(data: bytes, checksum: str) -> pd.DataFrame:
    return pd.read_excel(io.BytesIO(data))


def _invalidate_filter_cache() -> None:
    st.session_state.pop("test_analysis_filter_cache", None)
    st.session_state.pop("test_analysis_filter_cache_version", None)


def _ensure_filter_cache(df: pd.DataFrame) -> dict[str, object]:
    dataset_version = st.session_state.get("test_analysis_dataset_version", 0)
    cache_version = st.session_state.get("test_analysis_filter_cache_version")
    cache = st.session_state.get("test_analysis_filter_cache")
    if cache is None or cache_version != dataset_version:
        part_options: List[str] = []
        if "规格类型" in df.columns:
            part_options = sorted(
                df["规格类型"].dropna().astype(str).unique().tolist()
            )

        order_options: List[str] = []
        if "生产订单" in df.columns:
            order_options = sorted(
                df["生产订单"].dropna().astype(str).unique().tolist()
            )

        date_range = None
        if "测试时间" in df.columns:
            valid_times = df["测试时间"].dropna()
            if not valid_times.empty:
                date_range = (valid_times.min().date(), valid_times.max().date())

        cache = {
            "part_options": part_options,
            "order_options": order_options,
            "date_range": date_range,
        }
        st.session_state.test_analysis_filter_cache = cache
        st.session_state.test_analysis_filter_cache_version = dataset_version
    return cache


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
        resolved = path.resolve()
    except OSError:
        resolved = path
    try:
        mtime = path.stat().st_mtime
    except OSError as exc:
        st.error(f"无法获取文件信息：{exc}")
        return None
    try:
        df = _read_report_from_disk(str(resolved), mtime)
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
        file_bytes = uploaded_file.getvalue()
        checksum = hashlib.md5(file_bytes).hexdigest()
        df = _read_report_from_bytes(file_bytes, checksum)
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
    with st.expander("查看详细记录", expanded=True):
        preview = deduped.reset_index(drop=True)
        st.dataframe(
            preview,
            use_container_width=True,
            hide_index=True,
            height=min(600, max(200, len(preview) * 35)),
        )
        if len(deduped) > len(preview):
            st.caption(f"仅展示前 {len(preview)} 行完整记录，可通过下载功能获取全部数据。")


def render_overview_table(filtered: pd.DataFrame) -> None:
    rows = []
    for station in STATION_ORDER:
        sub = filtered[filtered["标准测试站别"] == station]
        row = {"测试站别": station, "记录数": len(sub)}
        for metric in SUMMARY_COLUMNS:
            if metric in sub.columns and sub[metric].notna().any():
                row[f"{metric}均值"] = sub[metric].mean()
        rows.append(row)

    overview = pd.DataFrame(rows)
    overview_display = overview.set_index("测试站别")
    numeric_cols = [col for col in overview_display.columns if col != "记录数"]
    if numeric_cols:
        overview_display[numeric_cols] = overview_display[numeric_cols].apply(lambda col: col.round(3))
    st.dataframe(
        overview_display.reset_index(),
        use_container_width=True,
        hide_index=True,
    )


alt.data_transformers.disable_max_rows()
st.set_page_config(page_title="常用测试数据分析", page_icon="📈", layout="wide")

st.title("📈 常用测试数据分析")
st.markdown("上传常用测试数据报表，查看五个测试站别的指标表现。")

if "test_analysis_df" not in st.session_state:
    st.session_state.test_analysis_df = None
if "test_analysis_filename" not in st.session_state:
    st.session_state.test_analysis_filename = None
if "test_analysis_dataset_version" not in st.session_state:
    st.session_state.test_analysis_dataset_version = 0
if "test_analysis_loaded_path" not in st.session_state:
    st.session_state.test_analysis_loaded_path = None
if "test_analysis_folder_path" not in st.session_state:
    st.session_state.test_analysis_folder_path = DEFAULT_DATA_FOLDER
if "test_analysis_data_source" not in st.session_state:
    st.session_state.test_analysis_data_source = FOLDER_SOURCE_LABEL

data_source = st.radio(
    "选择数据来源",
    options=DATA_SOURCE_OPTIONS,
    horizontal=True,
    key="test_analysis_data_source",
)

if data_source == UPLOAD_SOURCE_LABEL:
    uploaded = st.file_uploader(
        "上传测试数据（建议使用常用测试数据报表格式）",
        type=["xlsx", "xls"],
        help=f"仅支持文件名以“{REPORT_PREFIX}”开头的 Excel 报表",
    )

    if uploaded is not None and uploaded.name != st.session_state.test_analysis_filename:
        with st.spinner("正在解析并加载数据..."):
            raw_df = parse_uploaded_file(uploaded)
            if raw_df is not None:
                prepared = prepare_dataframe(raw_df)
                if prepared is not None:
                    st.session_state.test_analysis_df = prepared
                    st.session_state.test_analysis_filename = uploaded.name
                    st.session_state.test_analysis_loaded_path = None
                    st.session_state.test_analysis_dataset_version = (
                        st.session_state.get("test_analysis_dataset_version", 0) + 1
                    )
                    _invalidate_filter_cache()
                    st.success(f"文件 {uploaded.name} 解析成功，共 {len(prepared)} 条记录。")
else:
    col_path, col_refresh = st.columns([4, 1])
    with col_path:
        folder_path = st.text_input(
            "文件夹路径",
            value=st.session_state.test_analysis_folder_path,
            placeholder=f"默认: {DEFAULT_DATA_FOLDER}",
            key="test_analysis_folder_path",
        )
    with col_refresh:
        st.markdown("<div style='margin-top: 32px;'></div>", unsafe_allow_html=True)
        refresh_folder = st.button(
            "🔄 刷新",
            use_container_width=True,
            key="test_analysis_refresh_folder",
        )
    if refresh_folder:
        st.rerun()

    folder_path_value = (folder_path or "").strip()
    if folder_path_value:
        try:
            search_path = resolve_input_path(folder_path_value)
            if search_path.exists() and search_path.is_dir():
                excel_files = sorted(
                    list(search_path.glob("*.xlsx")) + list(search_path.glob("*.xls")),
                    key=lambda candidate: candidate.stat().st_mtime,
                    reverse=True,
                )
                preferred_files = [f for f in excel_files if f.name.startswith(REPORT_PREFIX)]
                display_files = preferred_files if preferred_files else excel_files

                if display_files:
                    options_map: dict[str, Path] = {}
                    option_labels: List[str] = []
                    for candidate in display_files:
                        size_kb = candidate.stat().st_size / 1024
                        base_label = f"{candidate.name} ({size_kb:.1f} KB)"
                        label = base_label
                        suffix = 2
                        while label in options_map:
                            label = f"{base_label} ({suffix})"
                            suffix += 1
                        options_map[label] = candidate
                        option_labels.append(label)

                    suffix_text = "（已优先显示常用测试报表）" if preferred_files else ""
                    selected_label = st.selectbox(
                        f"选择报表文件{suffix_text}",
                        options=option_labels,
                        key="test_analysis_folder_file_select",
                    )
                    selected_path = options_map.get(selected_label)

                    auto_load = (
                        st.session_state.test_analysis_df is None
                        and selected_path is not None
                        and display_files
                        and selected_path == display_files[0]
                        and selected_path.name.startswith(REPORT_PREFIX)
                    )

                    load_folder_file = st.button("📂 加载选中的数据", type="primary")

                    if selected_path and (load_folder_file or auto_load):
                        with st.spinner(f"正在加载 {selected_path.name}..."):
                            raw_df = load_report_from_path(str(selected_path))
                            if raw_df is not None:
                                prepared = prepare_dataframe(raw_df)
                                if prepared is not None:
                                    st.session_state.test_analysis_df = prepared
                                    st.session_state.test_analysis_filename = selected_path.name
                                    st.session_state.test_analysis_loaded_path = str(selected_path)
                                    st.session_state.test_analysis_dataset_version = (
                                        st.session_state.get("test_analysis_dataset_version", 0) + 1
                                    )
                                    _invalidate_filter_cache()
                                    st.success(
                                        f"文件 {selected_path.name} 加载成功，共 {len(prepared)} 条记录。"
                                    )
                                    if auto_load:
                                        st.rerun()
                else:
                    st.warning(f"`{search_path}` 中未找到 Excel 报表文件。")
            else:
                st.error(f"路径不存在或不是文件夹：{search_path}")
        except ValueError as value_error:
            st.error(str(value_error))
        except Exception as exc:
            st.error(f"读取文件夹时出错: {exc}")
    else:
        st.info("请输入有效的文件夹路径，或切换到上传文件。")

dataframe = st.session_state.test_analysis_df
if dataframe is None:
    st.info("请先上传测试数据报表。")
    st.stop()

filtered_df = dataframe
filter_cache = _ensure_filter_cache(dataframe)

filters_row = st.columns(4)

with filters_row[0]:
    part_options = filter_cache.get("part_options", [])
    if part_options:
        if len(part_options) <= MAX_MULTISELECT_OPTIONS:
            default_parts = part_options if len(part_options) <= MAX_AUTOMATIC_SELECTION else []
            selected_parts = st.multiselect(
                "规格类型",
                part_options,
                default=default_parts,
                placeholder="可输入关键字过滤",
                key="test_analysis_part_multiselect",
            )
            if len(part_options) > MAX_AUTOMATIC_SELECTION:
                st.caption(f"检测到 {len(part_options)} 种规格类型，默认不勾选，可手动挑选。")
            if selected_parts and "规格类型" in filtered_df.columns:
                filtered_df = filtered_df[filtered_df["规格类型"].isin(selected_parts)]
        else:
            st.caption(f"规格类型项较多（{len(part_options)} 项），请使用关键词筛选。")
            keywords = st.text_input(
                "规格类型关键词（逗号分隔）",
                key="test_analysis_part_keywords",
            )
            if keywords.strip() and "规格类型" in filtered_df.columns:
                terms = [token.strip() for token in keywords.split(',') if token.strip()]
                if terms:
                    series = filtered_df["规格类型"].astype(str)
                    mask = pd.Series(False, index=series.index)
                    for term in terms:
                        mask |= series.str.contains(re.escape(term), case=False, na=False)
                    filtered_df = filtered_df[mask]
    else:
        st.write("规格类型列缺失")

with filters_row[1]:
    order_options = filter_cache.get("order_options", [])
    if order_options:
        if len(order_options) <= MAX_MULTISELECT_OPTIONS:
            default_orders = order_options if len(order_options) <= MAX_AUTOMATIC_SELECTION else []
            selected_orders = st.multiselect(
                "生产订单",
                order_options,
                default=default_orders,
                placeholder="可输入订单编号",
                key="test_analysis_order_multiselect",
            )
            if len(order_options) > MAX_AUTOMATIC_SELECTION:
                st.caption(f"检测到 {len(order_options)} 个订单号，默认不勾选，可手动挑选。")
            if selected_orders and "生产订单" in filtered_df.columns:
                filtered_df = filtered_df[filtered_df["生产订单"].isin(selected_orders)]
        else:
            st.caption(f"订单号数量较多（{len(order_options)} 个），请输入关键词筛选。")
            order_keywords = st.text_input(
                "订单关键词（逗号分隔）",
                key="test_analysis_order_keywords",
            )
            if order_keywords.strip() and "生产订单" in filtered_df.columns:
                terms = [token.strip() for token in order_keywords.split(',') if token.strip()]
                if terms:
                    series = filtered_df["生产订单"].astype(str)
                    mask = pd.Series(False, index=series.index)
                    for term in terms:
                        mask |= series.str.contains(re.escape(term), case=False, na=False)
                    filtered_df = filtered_df[mask]
    else:
        st.write("生产订单列缺失")

with filters_row[2]:
    selected_stations = st.multiselect("测试站别", STATION_ORDER, default=STATION_ORDER)
    if selected_stations:
        if "标准测试站别" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["标准测试站别"].isin(selected_stations)]
    else:
        filtered_df = filtered_df.iloc[0:0]

with filters_row[3]:
    date_range = filter_cache.get("date_range")
    if date_range and "测试日期" in filtered_df.columns:
        start, end = st.date_input(
            "测试日期区间",
            value=date_range,
            min_value=date_range[0],
            max_value=date_range[1],
            key="test_analysis_date_range",
        )
        if start and end:
            mask = filtered_df["测试日期"].between(start, end)
            filtered_df = filtered_df[mask]
    else:
        st.write("测试时间缺失")

if filtered_df.empty:
    st.warning("筛选条件下没有数据，请调整过滤器。")
    st.stop()


st.caption('筛选结果已缓存，可在“数据分析”页统一保存。')

with st.expander("查看筛选结果预览", expanded=True):
    preview = filtered_df
    st.dataframe(
        preview,
        use_container_width=True,
        height=min(600, max(200, len(preview) * 35)),
    )
    if len(filtered_df) > len(preview):
        st.caption(f"仅展示前 {len(preview)} 行。完整数据可通过下方下载按钮获取。")

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


