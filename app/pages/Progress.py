# title: 进度追踪
from pathlib import Path
from typing import List, Optional, Union, Tuple
import sys
import time

import altair as alt
import pandas as pd
import streamlit as st

# Ensure we can import project modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DEFAULT_DATA_FOLDER, WIP_REPORT_KEYWORDS
from pages.data_manager.constants import (
    BASE_STATIONS,
    BASE_STATIONS_LOWER,
    STATION_MAPPING,
    STATION_MAPPING_LOWER,
    get_stations_for_part,
)
from pages.data_manager.product_type_service import ProductTypeService
from utils.local_storage import DataCategory, LocalDataStore
from utils.exceptions import StorageError
from utils.storage_widgets import render_load_selector

APP_ROOT = Path(__file__).resolve().parent.parent
ALLOWED_PATH_ROOTS = [APP_ROOT, Path(DEFAULT_DATA_FOLDER).resolve()]

PRODUCTION_ORDER_CANDIDATES: List[str] = [
    "生产订单",
    "ERP生产订单",
    "SAP生产订单",
    "生产订单号",
    "订单号",
    "工单号",
]


def _pick_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Pick a column from a list of candidates (case insensitive)."""
    lower_map = {str(c).lower(): c for c in df.columns}
    for name in candidates:
        if name in df.columns:
            return name
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    return None


def resolve_input_path(path_str: str) -> Path:
    """Resolve and validate a user supplied folder path."""
    normalized = path_str.strip()
    if not normalized:
        raise ValueError("路径不能为空")

    candidate = Path(normalized).expanduser()
    if not candidate.is_absolute():
        candidate = APP_ROOT / candidate
    resolved = candidate.resolve()

    if not resolved.exists():
        raise ValueError(f"路径不存在: {resolved}")
    if not resolved.is_dir():
        raise ValueError(f"路径不是文件夹: {resolved}")

    is_allowed = any(resolved == allowed or allowed in resolved.parents for allowed in ALLOWED_PATH_ROOTS)
    if not is_allowed:
        raise ValueError(f"路径不在允许的范围内: {resolved}")
    return resolved


def _compute_usecols(header_cols: List[str]) -> List[str]:
    cols_set = {str(c).strip() for c in header_cols}
    needed = {"壳体号", "料号", "当前站点", "上一站"}
    for name in PRODUCTION_ORDER_CANDIDATES:
        if name in cols_set:
            needed.add(name)
            break
    for excel_col in STATION_MAPPING.keys():
        time_col = f"{excel_col}时间"
        if time_col in cols_set:
            needed.add(time_col)
    return [c for c in header_cols if str(c).strip() in needed]


def read_data_file(file_path: Union[str, Path], usecols: Optional[List[str]] = None) -> pd.DataFrame:
    """Read CSV/Excel with basic encoding handling."""
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()

    if suffix == ".csv":
        for encoding in ("utf-8", "utf-8-sig", "gbk", "gb18030"):
            try:
                return pd.read_csv(file_path, encoding=encoding, usecols=usecols, low_memory=False)
            except UnicodeDecodeError:
                continue
        raise ValueError(f"无法解析 CSV 文件编码: {file_path.name}")
    if suffix in (".xls", ".xlsx"):
        engine = "openpyxl" if suffix == ".xlsx" else None
        return pd.read_excel(file_path, usecols=usecols, engine=engine)
    raise ValueError(f"不支持的文件格式: {suffix}")


def parse_uploaded_file(uploaded_file) -> Optional[pd.DataFrame]:
    """Parse uploaded CSV/Excel file."""
    try:
        name = uploaded_file.name.lower()
        if name.endswith(".csv"):
            for encoding in ("utf-8", "utf-8-sig", "gbk", "gb18030"):
                try:
                    uploaded_file.seek(0)
                    return pd.read_csv(uploaded_file, encoding=encoding)
                except UnicodeDecodeError:
                    continue
            st.error("无法解析 CSV 文件编码")
            return None
        if name.endswith((".xls", ".xlsx")):
            return pd.read_excel(uploaded_file)
        st.error("不支持的文件格式，请上传 CSV 或 Excel 文件")
        return None
    except Exception as e:
        st.error(f"文件解析失败: {str(e)}")
        return None


def normalize_station_name(station_name) -> str:
    """Normalize station names using the shared mapping."""
    if pd.isna(station_name) or station_name == "":
        return ""

    station_name = str(station_name).strip()
    station_name_lower = station_name.lower()

    if station_name_lower in STATION_MAPPING_LOWER:
        return STATION_MAPPING_LOWER[station_name_lower]

    if station_name.endswith("测试") or station_name_lower.endswith("测试"):
        base_name = station_name[:-2]
        base_name_lower = base_name.lower()
        if base_name_lower in STATION_MAPPING_LOWER:
            return STATION_MAPPING_LOWER[base_name_lower]

    test_name_lower = station_name_lower + "测试"
    if test_name_lower in STATION_MAPPING_LOWER:
        return STATION_MAPPING_LOWER[test_name_lower]

    if station_name_lower in BASE_STATIONS_LOWER:
        return BASE_STATIONS_LOWER[station_name_lower]

    # 返回原始名称
    return station_name


def extract_progress_data(df: pd.DataFrame, light: bool = False) -> pd.DataFrame:
    """Extract normalized progress information from raw WIP data."""
    if df is None or df.empty:
        return pd.DataFrame()

    shell_col = _pick_column(df, ["壳体号", "壳体编码", "壳体", "腔体号", "腔体编号", "Shell ID", "ShellID", "SN", "序列号"])
    current_col = _pick_column(df, ["当前站点", "当前站", "最新站点", "当前工序", "站别"])
    prev_col = _pick_column(df, ["上一站", "上一站点", "上一步"])
    part_col = _pick_column(df, ["料号", "产品料号", "物料号", "料号编码"])
    production_order_column = _pick_column(df, PRODUCTION_ORDER_CANDIDATES)

    if shell_col is None:
        st.error("未找到壳体号列，请检查文件内容")
        return pd.DataFrame()

    df = df.copy()
    df[shell_col] = df[shell_col].fillna("").astype(str).str.strip()
    df = df[df[shell_col] != ""].reset_index(drop=True)

    result = pd.DataFrame()
    result["壳体号"] = df[shell_col]
    result["料号"] = df[part_col].fillna("").astype(str).str.strip() if part_col else ""

    if production_order_column and production_order_column in df.columns:
        result["生产订单"] = df[production_order_column].fillna("").astype(str).str.strip()
    else:
        result["生产订单"] = ""

    if current_col and current_col in df.columns:
        result["当前站点原始"] = df[current_col].fillna("").astype(str)
    else:
        result["当前站点原始"] = ""
    result["当前站点"] = result["当前站点原始"].apply(normalize_station_name)
    result["上一站"] = df[prev_col].fillna("").astype(str) if prev_col and prev_col in df.columns else ""
    result["是否工程分析"] = result["当前站点"] == "工程分析"

    existing_station_time_cols = [
        (excel_col, STATION_MAPPING[excel_col], f"{excel_col}时间")
        for excel_col in STATION_MAPPING.keys()
        if f"{excel_col}时间" in df.columns
    ]

    def compute_completed_stations(row_idx: int):
        completed = []
        time_map = {}
        for _, standard_station, time_col in existing_station_time_cols:
            val = df.at[row_idx, time_col]
            if pd.notna(val) and str(val).strip():
                completed.append(standard_station)
                time_map[standard_station] = val
        return completed, time_map

    completed_data = [compute_completed_stations(i) for i in df.index]
    result["完成站别"] = [c[0] for c in completed_data]
    result["站别时间"] = [c[1] for c in completed_data]

    unrecognized = result[
        (result["当前站点原始"] != "")
        & (result["当前站点"] == result["当前站点原始"])
        & (~result["当前站点"].isin(BASE_STATIONS))
        & (~result["当前站点"].isin({"工程分析", "已完成", "未开始"}))
    ]["当前站点原始"].unique()

    if len(unrecognized) > 0:
        st.warning(f"⚠️ 发现未识别的站别名称: {', '.join(sorted(unrecognized))}")

    result.attrs["production_order_column"] = production_order_column
    result.attrs["time_cols"] = [tc for _, _, tc in existing_station_time_cols]
    result.attrs["shell_col"] = shell_col
    return result


def calculate_station_counts(progress_df: pd.DataFrame) -> pd.DataFrame:
    """统计各当前站别的壳体数量与占比"""
    if progress_df.empty:
        return pd.DataFrame(columns=["站别", "数量", "占比"])

    unknown_label = "未识别"
    station_series = progress_df["当前站点"].fillna("").astype(str).str.strip()
    station_series = station_series.replace({"": unknown_label, "nan": unknown_label})
    station_series = station_series.apply(lambda value: normalize_station_name(value) if value != unknown_label else value)

    counts = station_series.value_counts(dropna=False).reset_index()
    counts.columns = ["站别", "数量"]
    counts["占比"] = counts["数量"] / len(progress_df)

    ordered_labels = BASE_STATIONS + ["工程分析", "已完成", unknown_label]
    order_map = {label: idx for idx, label in enumerate(ordered_labels)}
    counts["排序"] = counts["站别"].map(order_map)

    fallback_order = len(ordered_labels) + counts.index.to_series()
    counts["排序"] = counts["排序"].fillna(fallback_order)
    counts = counts.sort_values(["排序", "站别"]).drop(columns="排序").reset_index(drop=True)
    return counts


def create_progress_table(progress_df: pd.DataFrame) -> pd.DataFrame:
    """创建进度表格"""
    table_data = []

    for _, row in progress_df.iterrows():
        part_number = row.get("料号", "")
        stations = get_stations_for_part(part_number)
        current_station = row.get("当前站点", "")
        is_engineering = row.get("是否工程分析", False)

        if is_engineering and "工程分析" not in stations:
            stations.append("工程分析")

        station_order = -1
        completed_count = 0

        if is_engineering:
            last_station = row.get("上一站", "")
            last_station_normalized = normalize_station_name(last_station)
            if last_station_normalized and last_station_normalized in stations:
                station_order = stations.index(last_station_normalized)
                completed_count = station_order + 1
        elif current_station == "已完成":
            station_order = len(stations) - 1
            completed_count = len(stations)
        elif current_station and current_station in stations:
            station_order = stations.index(current_station)
            completed_count = station_order
        else:
            completed_count = len(row.get("完成站别", []))

        total_count = len(stations)
        progress_pct = (completed_count / total_count * 100) if total_count > 0 else 0

        last_completed_station = ""
        if station_order > 0:
            last_completed_station = stations[station_order - 1]
        elif row.get("完成站别"):
            last_completed_station = row["完成站别"][-1]

        table_data.append(
            {
                "壳体号": row.get("壳体号", ""),
                "料号": part_number,
                "生产订单": row.get("生产订单", ""),
                "当前站点": current_station,
                "已完成站别数": completed_count,
                "总站别数": total_count,
                "完成进度": f"{progress_pct:.1f}%",
                "最新完成站别": last_completed_station,
                "是否工程分析": "是" if is_engineering else "否",
                "站别序号": station_order,
            }
        )

    result_df = pd.DataFrame(table_data)
    if "站别序号" in result_df.columns:
        if not result_df.empty:
            result_df = result_df.sort_values("站别序号", ascending=True)
        result_df = result_df.drop(columns=["站别序号"], errors="ignore")
    return result_df


def get_product_type_service() -> ProductTypeService:
    """Get ProductTypeService instance (refresh if missing new methods)."""
    service = st.session_state.get("progress_product_type_service")
    if service is None or not hasattr(service, "upsert_product_type"):
        service = ProductTypeService()
        st.session_state["progress_product_type_service"] = service
    return service


def prepare_shells_dataframe_for_data_manager(progress_df: pd.DataFrame) -> pd.DataFrame:
    """
    Flatten progress DataFrame to the format expected by Data Manager.
    
    - 展开"站别时间"字典为独立的站别时间列
    - 去重壳体号
    - 生成"更新时间"列（各站别时间的最晚时间）
    """
    if progress_df is None or progress_df.empty:
        raise ValueError("没有可保存的进度数据")

    shells_df = progress_df.copy()

    time_cols: List[str] = []
    if "站别时间" in shells_df.columns:
        time_dicts = shells_df["站别时间"].apply(lambda v: v if isinstance(v, dict) else {})
        all_stations = sorted({s for d in time_dicts for s in d.keys()})
        for station in all_stations:
            col_name = f"{station}时间"
            shells_df[col_name] = time_dicts.apply(lambda d: d.get(station))
            time_cols.append(col_name)

    # 删除字典列，保留展开后的列
    shells_df = shells_df.drop(columns=["站别时间"], errors="ignore")

    if time_cols:
        shells_df[time_cols] = shells_df[time_cols].apply(pd.to_datetime, errors="coerce")
        shells_df["更新时间"] = shells_df[time_cols].apply(
            lambda row: pd.to_datetime(row.dropna()).max() if row.notna().any() else pd.NaT,
            axis=1,
        )

    shell_col = _pick_column(shells_df, ["壳体号", "壳体编码", "壳体", "腔体号", "腔体编号", "Shell ID", "ShellID", "SN", "序列号"])
    if shell_col:
        shells_df = shells_df.drop_duplicates(subset=[shell_col]).reset_index(drop=True)

    return shells_df


# ============================================================================
# Streamlit 页面
# ============================================================================

st.set_page_config(page_title="模块进度", page_icon="📊", layout="wide")
st.title("模块WIP进度")

st.markdown(
    """
<style>
.stMultiSelect div[data-baseweb="select"] > div { flex-wrap: wrap; }
.stMultiSelect [data-baseweb="tag"] {
    max-width: 140px !important;
    min-width: auto !important;
    display: inline-flex !important;
    align-items: center !important;
}
</style>
""",
    unsafe_allow_html=True,
)

SESSION_DEFAULTS = {
    "progress_df": None,
    "progress_raw_df": None,
    "uploaded_filename": None,
    "progress_dir_cache": {},
    "progress_data_cache": {},
    "progress_data_source": "📁 从文件夹选择",
    "progress_folder_path": DEFAULT_DATA_FOLDER,
}
for key, default in SESSION_DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default


def _render_load_saved_progress(as_expander: bool = False, show_details: bool = False) -> None:
    """加载已保存的进度数据（本地缓存）"""
    container = st.expander("📂 加载历史进度数据", expanded=False) if as_expander else st.container()
    with container:
        st.markdown("**📂 加载历史进度数据**")

        def _on_load(df: pd.DataFrame, metadata, extra_data):
            # 恢复到当前页面的缓存
            shell_col = _pick_column(df, ["壳体号", "壳体编码", "壳体", "腔体号", "腔体编号", "Shell ID", "ShellID", "SN", "序列号"])
            if shell_col and shell_col in df.columns:
                df[shell_col] = df[shell_col].fillna("").astype(str).str.strip()

            st.session_state.progress_df = df
            st.session_state.progress_raw_df = df
            st.session_state.uploaded_filename = metadata.name
            st.session_state.progress_data_source = "📁 从文件夹选择"
            st.session_state.progress_loaded_id = metadata.id

        result = render_load_selector(
            category=DataCategory.PROGRESS,
            key="progress_load_inline" if not as_expander else "progress_load_expander",
            show_details=show_details,
            on_load_callback=_on_load,
        )

        if result:
            st.success("已加载历史进度数据")
            st.rerun()


def _load_from_folder() -> None:
    """Render folder selection and load data if requested."""
    st.markdown("**文件夹路径**")
    col_path, col_refresh = st.columns([5, 1], vertical_alignment="center")
    with col_path:
        folder_path = st.text_input(
            "",
            placeholder=f"默认: {DEFAULT_DATA_FOLDER}",
            key="progress_folder_path",
            label_visibility="collapsed",
        )
    with col_refresh:
        refresh_btn = st.button("🔄", use_container_width=True, help="刷新文件列表")

    if not folder_path:
        st.toast("请输入包含 WIP 报表的文件夹路径", icon="ℹ️")
        return

    try:
        search_path = resolve_input_path(folder_path)
    except ValueError as e:
        st.error(str(e))
        return

    if not search_path.exists() or not search_path.is_dir():
        st.error(f"路径不存在或不是文件夹: {search_path}")
        return

    excel_files = list(search_path.glob("*.xlsx")) + list(search_path.glob("*.xls"))
    csv_files = list(search_path.glob("*.csv"))
    all_files = sorted(excel_files + csv_files, key=lambda x: x.stat().st_mtime, reverse=True)

    if not all_files:
        st.warning(f"在 `{search_path}` 中未找到数据文件")
        return

    wip_files = [f for f in all_files if any(keyword in f.name or keyword.lower() in f.name.lower() for keyword in WIP_REPORT_KEYWORDS)]
    display_files = wip_files if wip_files else all_files
    max_display = 200
    display_files = display_files[:max_display]

    _dir_key = str(search_path)
    _dir_cache = st.session_state.progress_dir_cache.get(_dir_key, {})
    file_display_map = {}
    for f in display_files:
        fp = str(f)
        mtime = f.stat().st_mtime
        meta = _dir_cache.get(fp)
        if not meta or meta.get("mtime") != mtime:
            size_kb = f.stat().st_size / 1024.0
            _dir_cache[fp] = {"mtime": mtime, "size_kb": size_kb}
        else:
            size_kb = meta["size_kb"]
        file_display_map[f"{f.name} ({size_kb:.1f} KB)"] = fp
    st.session_state.progress_dir_cache[_dir_key] = _dir_cache

    st.markdown("**选择文件 (已筛选WIP报表)**" if wip_files else "**选择文件**")
    col_select, col_load = st.columns([4, 1], vertical_alignment="center")
    with col_select:
        selected_file_display = st.selectbox(
            "",
            options=list(file_display_map.keys()),
            key="progress_file_select",
            label_visibility="collapsed",
        )
    with col_load:
        load_btn = st.button("📂 加载", type="primary", use_container_width=True)

    if not selected_file_display:
        return

    selected_file_path = file_display_map[selected_file_display]
    auto_load = st.session_state.progress_df is None and bool(wip_files)

    if load_btn or auto_load or refresh_btn:
        p = Path(selected_file_path)
        cache_key = f"{p.resolve()}::{p.stat().st_mtime}"
        cached = st.session_state.progress_data_cache.get(cache_key)

        if cached:
            df, cached_progress_df = cached
            st.session_state.progress_raw_df = df
            st.session_state.progress_df = cached_progress_df
            st.session_state.uploaded_filename = p.name
            st.success(f"已从缓存加载！共 {len(df)} 条记录")
            if auto_load:
                st.rerun()
            return

        with st.spinner(f"正在加载 {p.name}..."):
            try:
                if p.suffix.lower() == ".csv":
                    header_df = pd.read_csv(p, nrows=0)
                    usecols = _compute_usecols(list(header_df.columns))
                    time_cols = [f"{ec}时间" for ec in STATION_MAPPING.keys() if f"{ec}时间" in header_df.columns]
                    dtype_map = {c: "string" for c in ["壳体号", "料号", "生产订单"] if c in usecols}
                    df = pd.read_csv(p, usecols=usecols, dtype=dtype_map, parse_dates=time_cols, low_memory=False)
                else:
                    header_df = pd.read_excel(p, nrows=0)
                    usecols = _compute_usecols(list(header_df.columns))
                    engine = "openpyxl" if p.suffix.lower() == ".xlsx" else None
                    df = pd.read_excel(p, usecols=usecols, engine=engine)
                    time_cols = [f"{ec}时间" for ec in STATION_MAPPING.keys() if f"{ec}时间" in header_df.columns]
                    if time_cols:
                        df[time_cols] = df[time_cols].apply(pd.to_datetime, errors="coerce")

                # 确保壳体号列为字符串，避免 Arrow 序列化失败
                shell_col = _pick_column(df, ["壳体号", "壳体编码", "壳体", "腔体号", "腔体编号", "Shell ID", "ShellID", "SN", "序列号"])
                if shell_col and shell_col in df.columns:
                    df[shell_col] = df[shell_col].fillna("").astype(str).str.strip()

                st.session_state.progress_raw_df = df
                st.session_state.progress_df = extract_progress_data(df)
                st.session_state.uploaded_filename = p.name
                st.success(f"加载成功！共 {len(df)} 条记录")
                st.session_state.progress_data_cache[cache_key] = (df, st.session_state.progress_df)
                if auto_load:
                    st.rerun()
            except Exception as e:
                st.error(f"文件加载失败: {str(e)}")


def _load_from_upload() -> None:
    uploaded_file = st.file_uploader("上传 WIP 文件", type=["csv", "xls", "xlsx"], key="progress_uploader")
    if uploaded_file is None:
        return

    if st.session_state.uploaded_filename == uploaded_file.name:
        return

    with st.spinner("正在解析文件..."):
        df = parse_uploaded_file(uploaded_file)
    if df is not None:
        # 确保壳体号列为字符串，避免 Arrow 序列化失败
        shell_col = _pick_column(df, ["壳体号", "壳体编码", "壳体", "腔体号", "腔体编号", "Shell ID", "ShellID", "SN", "序列号"])
        if shell_col and shell_col in df.columns:
            df[shell_col] = df[shell_col].fillna("").astype(str).str.strip()

        st.session_state.progress_raw_df = df
        st.session_state.progress_df = extract_progress_data(df)
        st.session_state.uploaded_filename = uploaded_file.name
        st.session_state.progress_loaded_id = None
        st.success(f"文件解析成功！共 {len(df)} 条记录")


def _render_filter_section(df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[set]]:
    """Render production order filter and return filtered DataFrame."""
    if df is None or df.empty:
        return df, None

    filtered_df = df.copy()
    selected_order_values = None
    
    if "生产订单" in filtered_df.columns:
        order_series = filtered_df["生产订单"].dropna().astype(str).str.strip()
        order_series = order_series[order_series != ""]
        order_options = sorted(order_series.unique().tolist(), reverse=True)

        if order_options:
            st.markdown("##### 🔍 筛选生产订单")
            col_sel, col_op = st.columns([4, 1], vertical_alignment="bottom")
            
            with col_sel:
                saved_selected = st.session_state.get("progress_production_orders")
                default_selected = saved_selected if saved_selected is not None else []
                
                selected_orders = st.multiselect(
                    "选择生产订单",
                    options=order_options,
                    default=default_selected,
                    key="progress_production_orders",
                    placeholder="可输入搜索并选择订单（支持多选）",
                    label_visibility="collapsed",
                )
            
            with col_op:
                 # 使用 columns 放置小按钮
                 c1, c2 = st.columns(2)
                 
                 def _select_all_orders():
                     st.session_state["progress_production_orders"] = order_options
                     
                 def _clear_all_orders():
                     st.session_state["progress_production_orders"] = []

                 with c1:
                     st.button("全选", key="progress_order_select_all_btn_new", use_container_width=True, on_click=_select_all_orders)
                 with c2:
                     st.button("清空", key="progress_order_clear_btn_new", use_container_width=True, on_click=_clear_all_orders)
            
            if selected_orders:
                selected_order_values = {order.strip() for order in selected_orders}
                filtered_df = filtered_df[
                    filtered_df["生产订单"].fillna("").astype(str).str.strip().isin(selected_order_values)
                ]
            else:
                selected_order_values = None
        else:
            st.caption("未检测到生产订单数据")
    
    return filtered_df, selected_order_values


def _render_save_section(filtered_df: pd.DataFrame):
    if filtered_df.empty:
        st.info("暂无数据可保存")
        return

    # 获取生产订单列表
    production_orders = []
    if "生产订单" in filtered_df.columns:
        production_orders = filtered_df["生产订单"].dropna().astype(str).str.strip().unique().tolist()
        production_orders = [o for o in production_orders if o]
    
    # 产品类型名称输入
    default_name = ""
    if production_orders:
        default_name = production_orders[0] if len(production_orders) == 1 else f"{production_orders[0]} 等{len(production_orders)}个订单"
    
    col_save, col_update = st.columns(2, vertical_alignment="top")
    
    with col_save:
        st.markdown("#### 💾 保存数据")
        product_type_name = st.text_input(
            "数据名称 / 产品类型名称",
            value=default_name,
            placeholder="输入名称（如 M20-AM-C）",
            key="progress_product_type_name",
            help="将作为历史数据集名称和 Data Manager 产品类型名称"
        )
        save_clicked = st.button("💾 保存", key="progress_save_combined_btn", use_container_width=True, type="primary")
    
    with col_update:
        st.markdown("#### 🔄 更新数据")

    if save_clicked:
        if not product_type_name or not product_type_name.strip():
            st.error("❌ 请输入产品类型名称")
            return
        
        save_name = product_type_name.strip()
        source_path = st.session_state.get("uploaded_filename") or st.session_state.get("progress_folder_path")
        dataset_id = None
        product_type_id = None

        # 准备存储服务
        store = st.session_state.get("local_data_store")
        if store is None:
            store = LocalDataStore()
            st.session_state["local_data_store"] = store
        service = get_product_type_service()

        # 检查重名（历史+数据管理器），需要确认覆盖
        existing_history = [ds for ds in store.list_datasets(category=DataCategory.PROGRESS) if ds.name == save_name]
        existing_product_types = [pt for pt in service.list_product_types() if pt.name == save_name]
        overwrite_allowed = (
            st.session_state.get("progress_overwrite_confirmed")
            and st.session_state.get("progress_overwrite_name") == save_name
        )
        if existing_history or existing_product_types:
            if not overwrite_allowed:
                st.warning(f"⚠️ 名称已存在：{save_name}。选择覆盖将删除同名历史并更新产品类型。")
                if st.button("✅ 确认覆盖", key="progress_overwrite_confirm_btn", use_container_width=True, type="primary"):
                    st.session_state["progress_overwrite_confirmed"] = True
                    st.session_state["progress_overwrite_name"] = save_name
                    overwrite_allowed = True
                else:
                    st.info("如需取消覆盖，请修改名称后重新保存。")
                    return

        try:
            # 保存到本地历史（供“加载历史”使用）
            for ds in existing_history:
                try:
                    store.delete(ds.id)
                except Exception:
                    # 忽略单个删除失败，继续尝试保存新数据
                    pass
            dataset_id = store.save(
                df=filtered_df,
                category=DataCategory.PROGRESS,
                name=save_name,
                custom_filename=save_name,
                source_file=source_path,
            )
            st.session_state.progress_loaded_id = dataset_id

            # 准备壳体数据并保存到数据管理器
            shells_df = prepare_shells_dataframe_for_data_manager(filtered_df)
            # 覆盖使用 upsert，避免生成重复产品类型
            if existing_product_types:
                product_type_id = service.upsert_product_type(
                    name=save_name,
                    shells_df=shells_df,
                    production_orders=production_orders,
                    source_file=source_path,
                )
            else:
                product_type_id = service.save_product_type(
                    name=save_name,
                    shells_df=shells_df,
                    production_orders=production_orders,
                    source_file=source_path,
                )

            st.toast(f"✅ 已保存到历史与数据管理器：{save_name}")
            st.caption(f"历史ID: {dataset_id[:8]}... | 产品类型ID: {product_type_id[:8]}...")

            # 覆盖后重置 Data Manager 相关缓存，确保下次加载读取最新壳体数据
            for key in [
                "dm_shells_df",
                "dm_shell_progress_list",
                "dm_shell_cache_key",
                "dm_gantt_data",
                "dm_analysis_df",
            ]:
                st.session_state[key] = None
            st.session_state["dm_thresholds"] = {}
            st.session_state["dm_selected_product_type_id"] = product_type_id
            st.session_state["dm_selected_product_type_ids"] = [product_type_id]
            st.session_state["dm_selected_product_type_name"] = save_name
            st.session_state["dm_selected_orders"] = []
        except ValueError as e:
            st.error(f"❌ 保存失败: {str(e)}")
            if dataset_id and not product_type_id:
                st.info(f"历史数据集已保存 (ID: {dataset_id[:8]}...)，但数据管理器保存未完成")
        except StorageError as e:
            st.error(f"❌ 本地保存失败: {str(e)}")
        except Exception as e:
            st.error(f"❌ 保存时发生错误: {str(e)}")
            if dataset_id and not product_type_id:
                st.info(f"历史数据集已保存 (ID: {dataset_id[:8]}...)，但数据管理器保存未完成")
        finally:
            # 重置覆盖确认状态
            st.session_state["progress_overwrite_confirmed"] = False
            st.session_state["progress_overwrite_name"] = None

    # 更新已有数据集（只更新当前壳体的站别/状态）
    with col_update:
        store = st.session_state.get("local_data_store")
        if store is None:
            store = LocalDataStore()
            st.session_state["local_data_store"] = store

        existing_datasets = store.list_datasets(category=DataCategory.PROGRESS)
        if not existing_datasets:
            st.info("暂无可更新的历史数据集")
            return

        option_map = {
            f"{meta.name}（{meta.row_count}行 | {meta.created_at[:16]}）": meta for meta in existing_datasets
        }
        selected_label = st.selectbox("选择要更新的数据集", list(option_map.keys()), key="progress_update_select")
        update_clicked = st.button("🔄 更新到已有数据集", key="progress_update_btn", use_container_width=True, type="secondary")

        if update_clicked:
            target_meta = option_map.get(selected_label)
            if not target_meta:
                st.error("未找到选中的数据集")
                return

            shell_candidates = ["壳体号", "壳体编码", "壳体", "腔体号", "腔体编号", "Shell ID", "ShellID", "SN", "序列号"]
            shell_col_new = _pick_column(filtered_df, shell_candidates)
            if not shell_col_new:
                st.error("当前数据缺少壳体列，无法更新")
                return

            try:
                df_old, meta_old, extra_old = store.load(target_meta.id)
            except Exception as e:
                st.error(f"加载目标数据集失败: {e}")
                return

            shell_col_old = _pick_column(df_old, shell_candidates)
            target_shell_col = shell_col_old or shell_col_new

            # 归一化壳体列
            df_new = filtered_df.rename(columns={shell_col_new: target_shell_col}) if shell_col_new != target_shell_col else filtered_df.copy()
            df_new[target_shell_col] = df_new[target_shell_col].fillna("").astype(str).str.strip()
            df_old[target_shell_col] = df_old[target_shell_col].fillna("").astype(str).str.strip()

            # 对齐列，保留旧数据中未覆盖的壳体
            all_columns = list({*df_old.columns, *df_new.columns})
            df_old = df_old.reindex(columns=all_columns)
            df_new = df_new.reindex(columns=all_columns)

            new_shells = set(df_new[target_shell_col])
            df_old_kept = df_old[~df_old[target_shell_col].isin(new_shells)]
            combined = pd.concat([df_old_kept, df_new], ignore_index=True)

            try:
                store.delete(target_meta.id)
                updated_id = store.save(
                    df=combined,
                    category=DataCategory.PROGRESS,
                    name=target_meta.name,
                    custom_filename=target_meta.name,
                    note=target_meta.note,
                    extra_data=extra_old,
                    source_file=target_meta.source_file,
                )
                # 同步更新 Data Manager 中的产品类型数据
                shells_df_combined = prepare_shells_dataframe_for_data_manager(combined)
                orders_combined: List[str] = []
                if "生产订单" in combined.columns:
                    orders_combined = (
                        combined["生产订单"].dropna().astype(str).str.strip().unique().tolist()
                    )
                service = get_product_type_service()
                dm_product_type_id = service.upsert_product_type(
                    name=target_meta.name,
                    shells_df=shells_df_combined,
                    production_orders=orders_combined,
                    source_file=target_meta.source_file,
                )
                st.toast(f"✅ 已更新数据集与 Data Manager：{target_meta.name}")
                st.caption(f"新数据集ID: {updated_id[:8]}... | 产品类型ID: {dm_product_type_id[:8]}...")
            except Exception as e:
                st.error(f"更新失败: {e}")


# ============================================================================
# Main Layout
# ============================================================================

with st.container(border=True):
    st.markdown("### 📂 数据管理")

    action_mode = st.radio(
        "操作",
        options=["📁 从文件夹选择", "📤 上传文件", "📜 加载历史", "💾 保存数据"],
        horizontal=True,
        label_visibility="collapsed",
        key="progress_action_mode",
    )
    save_mode_selected = action_mode == "💾 保存数据"

    # 执行选中的操作
    if action_mode == "📁 从文件夹选择":
        _load_from_folder()
    elif action_mode == "📤 上传文件":
        _load_from_upload()
    elif action_mode == "📜 加载历史":
        _render_load_saved_progress(as_expander=False, show_details=True)

    # 加载后统一筛选
    filtered_progress_df = pd.DataFrame()
    selected_order_values = None
    if st.session_state.progress_df is not None:
        filtered_progress_df, selected_order_values = _render_filter_section(st.session_state.progress_df)
    
    # 保存时使用当前筛选后的数据集
    if save_mode_selected:
        if st.session_state.progress_df is not None:
            _render_save_section(filtered_progress_df)
        else:
            st.info("请先加载数据，再保存")

# 使用 session_state 中的数据
if st.session_state.progress_df is not None:
    progress_df = st.session_state.progress_df
    df_raw = st.session_state.progress_raw_df

    # Apply filter to df_raw if needed (Logic preserved from original)
    if selected_order_values is not None and df_raw is not None:
        production_order_column = progress_df.attrs.get("production_order_column")
        if production_order_column and production_order_column in df_raw.columns:
            preview_series = df_raw[production_order_column].fillna("").astype(str).str.strip()
            df_raw = df_raw[preview_series.isin(selected_order_values)] if selected_order_values else df_raw.iloc[0:0]
    elif not selected_order_values and df_raw is not None:
        # If no filter selected, clear raw df (as per original logic)
        df_raw = df_raw.iloc[0:0]

    if filtered_progress_df.empty:
        if st.session_state.progress_df is not None and not st.session_state.progress_df.empty:
             st.warning("筛选条件下没有数据，请调整生产订单选择")
    else:
        # Metrics and Charts
        col1, col2, col3, col4 = st.columns([1, 1.2, 1, 1.5])
        with col1:
            st.metric("壳体总数", len(filtered_progress_df))
        with col2:
            if "完成站别" in filtered_progress_df.columns:
                avg_progress = filtered_progress_df["完成站别"].apply(len).mean()
                st.metric("平均完成站别数", f"{avg_progress:.1f}")
        with col3:
            st.metric("基础站别数", len(BASE_STATIONS))
        with col4:
            latest_time = None
            time_cols = progress_df.attrs.get("time_cols", [])
            if df_raw is not None and time_cols:
                tc = [c for c in time_cols if c in df_raw.columns]
                if tc:
                    parsed = df_raw[tc].apply(pd.to_datetime, errors="coerce")
                    max_val = parsed.max().max()
                    if pd.notna(max_val):
                        latest_time = max_val
            if latest_time:
                st.metric("最新测试时间", latest_time.strftime("%Y-%m-%d %H:%M"))
            else:
                st.metric("最新测试时间", "无数据")

        counts_df = calculate_station_counts(filtered_progress_df)
        if not counts_df.empty:
            st.markdown("---")
            st.markdown("### 各站别当前数量")
            table_col, chart_col = st.columns([2, 3])

            with table_col:
                counts_style = counts_df.style.format({"占比": "{:.1%}"})
                table_height = max(180, min(320, 36 * len(counts_df) + 60))
                st.dataframe(counts_style, use_container_width=True, height=table_height)

            with chart_col:
                station_order = counts_df["站别"].tolist()
                chart_height = max(160, min(360, 28 * len(counts_df)))
                chart = (
                    alt.Chart(counts_df)
                    .mark_bar(cornerRadius=8, opacity=0.9, strokeWidth=1.5)
                    .encode(
                        x=alt.X("数量:Q", title="完成数量", axis=alt.Axis(grid=True, gridOpacity=0.2, tickMinStep=1)),
                        y=alt.Y(
                            "站别:N",
                            sort=station_order,
                            title="站别",
                            axis=alt.Axis(labelFontSize=12, labelFontWeight="bold"),
                        ),
                        color=alt.Color(
                            "数量:Q",
                            scale=alt.Scale(scheme="blues", domain=[counts_df["数量"].min(), counts_df["数量"].max()]),
                            legend=None,
                        ),
                        stroke=alt.value("#ffffff33"),
                        tooltip=["站别", "数量", alt.Tooltip("占比:Q", title="占比", format=".1%")],
                    )
                ).properties(height=chart_height).configure_view(strokeWidth=0).configure_axis(
                    titleFontSize=13, titleFontWeight="bold"
                )
                st.altair_chart(chart, use_container_width=True, theme="streamlit")

        engineering_df = filtered_progress_df[filtered_progress_df["是否工程分析"] == True]
        if not engineering_df.empty:
            st.markdown("---")
            st.markdown("### 🔍 工程分析站别分布")

            engineering_stations = []
            for _, row in engineering_df.iterrows():
                last_station = row.get("上一站", "")
                last_station_normalized = normalize_station_name(last_station)
                if last_station_normalized:
                    engineering_stations.append(last_station_normalized)

            if engineering_stations:
                engineering_counts = pd.Series(engineering_stations).value_counts().reset_index()
                engineering_counts.columns = ["站别", "数量"]
                engineering_counts["占比"] = engineering_counts["数量"] / engineering_counts["数量"].sum()

                eng_table_col, eng_chart_col = st.columns([2, 3])
                with eng_table_col:
                    st.caption(f"工程分析总数: {len(engineering_df)} 个")
                    eng_counts_style = engineering_counts.style.format({"占比": "{:.1%}"})
                    st.dataframe(eng_counts_style, use_container_width=True, hide_index=True)

                with eng_chart_col:
                    st.caption("工程分析站别占比")
                    # 悬停高亮效果
                    hover = alt.selection_point(fields=["站别"], on="pointerover", empty=False)
                    pie_chart = (
                        alt.Chart(engineering_counts)
                        .mark_arc(innerRadius=20, outerRadius=70)
                        .encode(
                            theta=alt.Theta("数量:Q", stack=True),
                            color=alt.Color("站别:N", legend=alt.Legend(title="站别", orient="right"), scale=alt.Scale(scheme="category20")),
                            tooltip=[
                                alt.Tooltip("站别:N", title="站别"),
                                alt.Tooltip("数量:Q", title="数量"),
                                alt.Tooltip("占比:Q", title="占比", format=".1%"),
                            ],
                            opacity=alt.condition(hover, alt.value(1), alt.value(0.6)),
                            stroke=alt.condition(hover, alt.value("#333"), alt.value(None)),
                            strokeWidth=alt.condition(hover, alt.value(2), alt.value(0)),
                        )
                        .add_params(hover)
                        .properties(height=180)
                    )
                    st.altair_chart(pie_chart, use_container_width=True)

        st.markdown("---")
        st.markdown("### 📋 进度表格")
        show_eng_only = st.checkbox("🔍 仅显示工程分析的壳体", value=False, key="progress_show_eng_only")
        source_df = filtered_progress_df[filtered_progress_df["是否工程分析"] == True] if show_eng_only else filtered_progress_df
        table_df = create_progress_table(source_df)
        st.dataframe(table_df, use_container_width=True, height=400)

    with st.expander("📄 查看原始数据"):
        st.dataframe(df_raw.head(20), use_container_width=True)
else:
    st.info(
        """
        ### 📖 使用说明

        1. **上传文件**：点击上方按钮上传包含壳体进度信息的 Excel 或 CSV 文件  
        2. **从文件夹加载**：输入数据文件夹路径，选择并加载 WIP 报表  
        3. **查看结果**：
           - 统计图：展示各站别当前数量和占比  
           - 进度表格：列出每个壳体的完成情况  
        """
    )
