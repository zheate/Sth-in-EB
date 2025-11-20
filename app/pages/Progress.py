# title: 进度追踪
import streamlit as st
import pandas as pd
import altair as alt
from datetime import datetime
from typing import List, Dict, Optional
import io
import sys
from pathlib import Path
import time


# 添加父目录到路径以导入config

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DEFAULT_DATA_FOLDER, WIP_REPORT_KEYWORDS
APP_ROOT = Path(__file__).resolve().parent.parent
def resolve_input_path(path_str: str) -> Path:
    """Resolve user-provided folder path, supporting relative inputs like ./data."""
    normalized = path_str.strip()
    if not normalized:
        raise ValueError("路径不能为空")

    candidate = Path(normalized).expanduser()
    if not candidate.is_absolute():
        candidate = APP_ROOT / candidate
    return candidate.resolve()





# 定义所有站别（按工艺流程顺序）- 基础站别

BASE_STATIONS = [
    "打标", "清洗", "壳体组装", "回流", "fac前备料", "打线", "fac", "fac补胶", 
    "fac补胶后烘烤", "fac测试", "sac组装", "光纤组装", "光纤组装后烘烤", 
    "红光耦合", "装大反", "红光耦合后烘烤", "合束", "合束后烘烤", 
    "NA前镜检", "NA前红光端面检", "NA测试", "耦合测试", "补胶", "温度循环", 
    "Pre测试", "低温存储", "低温存储后测试", "高温存储", "高温存储后测试", 
    "老化前红光端面", "post测试", "红光端面检查", "镜检", "封盖", "封盖测试", 
    "分级", "入库检", "入库", "RMA"
]

def get_stations_for_part(part_number: str) -> list:
    """根据料号返回适用的站别列表"""
    stations = BASE_STATIONS.copy()
    # 如果料号包含V，在合束后烘烤后面插入VBG和VBG后烘烤
    if 'V' in str(part_number).upper():
        hesu_idx = stations.index("合束后烘烤")
        stations.insert(hesu_idx + 1, "VBG")
        stations.insert(hesu_idx + 2, "VBG后烘烤")
    # 添加"已完成"作为最后一个站别
    stations.append("已完成")
    return stations
# 站别映射（Excel列名到标准站别名）

STATION_MAPPING = {
    "机械件打标": "打标",
    "机械件清洗": "清洗",
    "壳体组装": "壳体组装",
    "光耦回流": "回流",
    "FAC前备料": "fac前备料",
    "打线": "打线",
    "FAC": "fac",
    "FAC补胶": "fac补胶",
    "FAC补胶后烘烤": "fac补胶后烘烤",
    "FAC测试": "fac测试",
    "SAC组装": "sac组装",
    "光纤组装": "光纤组装",
    "光纤组装后烘烤": "光纤组装后烘烤",
    "红光耦合": "红光耦合",
    "装大反": "装大反",
    "耦合后烘烤": "红光耦合后烘烤",
    "红光耦合后烘烤": "红光耦合后烘烤",
    "合束": "合束",
    "合束后烘烤": "合束后烘烤",
    "VBG": "VBG",
    "VBG后烘烤": "VBG后烘烤",
    "NA前镜检": "NA前镜检",
    "NA前红光端面检": "NA前红光端面检",
    "NA前红光端面检查": "NA前红光端面检",
    "NA测试": "NA测试",
    "耦合测试": "耦合测试",
    "补胶": "补胶",
    "温度循环": "温度循环",
    "pre测试": "Pre测试",
    "Pre测试": "Pre测试",
    "低温存储": "低温存储",
    "低温储存": "低温存储",
    "低温存储后测试": "低温存储后测试",
    "低温储存后测试": "低温存储后测试",
    "高温存储": "高温存储",
    "高温存储后测试": "高温存储后测试",
    "老化": "老化前红光端面",
    "老化前红光端面": "老化前红光端面",
    "老化前红光端面检查": "老化前红光端面",
    "post测试": "post测试",
    "Post测试": "post测试",
    "红光端面检查": "红光端面检查",
    "镜检": "镜检",
    "封盖": "封盖",
    "封盖测试": "封盖测试",
    "顶盖": "封盖",
    "顶盖测试": "封盖测试",
    "分级": "分级",
    "入库检": "入库检",
    "入库--光耦": "入库",
    "入库": "入库",
    "待入库": "入库",
    "RMA性能测试": "RMA",
    "RMA拆盖检查": "RMA",
    "RMA": "RMA",
    "拆解": "工程分析",
    "未开始": "未开始",
    "已完成": "已完成",
    "complete": "已完成",
    "COMPLETE": "已完成",
    "TERMINATED": "已完成",
    "完成": "已完成"
}

STATION_MAPPING_LOWER = {key.lower(): value for key, value in STATION_MAPPING.items()}
BASE_STATIONS_LOWER = {station.lower(): station for station in BASE_STATIONS}

PRODUCTION_ORDER_CANDIDATES: List[str] = [
    "生产订单",
    "ERP生产订单",
    "SAP生产订单",
    "生产订单号",
    "订单号",
    "工单号",
]
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

def parse_uploaded_file(uploaded_file) -> Optional[pd.DataFrame]:
    """解析上传的文件"""
    try:
        if uploaded_file.name.endswith('.csv'):
            # Try UTF-8 first, then GBK for Chinese files

            try:

                df = pd.read_csv(uploaded_file, encoding='utf-8')

            except UnicodeDecodeError:

                uploaded_file.seek(0)  # Reset file pointer

                df = pd.read_csv(uploaded_file, encoding='gbk')

        elif uploaded_file.name.endswith(('.xls', '.xlsx')):

            df = pd.read_excel(uploaded_file)

        else:

            st.error("不支持的文件格式，请上传 CSV 或 Excel 文件")

            return None

        return df

    except Exception as e:

        st.error(f"文件解析失败: {str(e)}")

        return None



def normalize_station_name(station_name: str) -> str:

    """将Excel中的站别名称标准化（忽略大小写）"""

    if pd.isna(station_name) or station_name == '':

        return ''



    station_name = str(station_name).strip()

    station_name_lower = station_name.lower()



    # 包含 rma 字样的都归到 RMA 站别

    if 'rma' in station_name_lower:

        return 'RMA'



    # 直接映射（忽略大小写）

    if station_name_lower in STATION_MAPPING_LOWER:

        return STATION_MAPPING_LOWER[station_name_lower]



    # 尝试去掉"测试"后缀

    if station_name.endswith('测试') or station_name_lower.endswith('测试'):

        base_name = station_name[:-2]

        base_name_lower = base_name.lower()

        if base_name_lower in STATION_MAPPING_LOWER:

            return STATION_MAPPING_LOWER[base_name_lower]



    # 尝试添加"测试"后缀

    test_name_lower = station_name_lower + '测试'

    if test_name_lower in STATION_MAPPING_LOWER:

        return STATION_MAPPING_LOWER[test_name_lower]



    # 尝试在BASE_STATIONS中查找（忽略大小写）

    if station_name_lower in BASE_STATIONS_LOWER:

        return BASE_STATIONS_LOWER[station_name_lower]



    # 尝试模糊匹配（去除空格、特殊字符）

    clean_name = station_name_lower.replace(' ', '').replace('-', '').replace('_', '')

    for key, value in STATION_MAPPING_LOWER.items():

        clean_key = key.replace(' ', '').replace('-', '').replace('_', '')

        if clean_name == clean_key:

            return value

   

    # 返回原始名称

    return station_name



def extract_progress_data(df: pd.DataFrame, light: bool = False) -> pd.DataFrame:

    """从原始数据中提取进度信息"""

    progress_data = []

    unrecognized_stations = set()

    

    column_lookup = {str(col).strip(): col for col in df.columns}

    production_order_column = next(

        (column_lookup[name] for name in PRODUCTION_ORDER_CANDIDATES if name in column_lookup),

        None,

    )

    

    existing_station_time_cols = [
        (excel_col, STATION_MAPPING[excel_col], f"{excel_col}时间")
        for excel_col in STATION_MAPPING.keys()
        if f"{excel_col}时间" in df.columns
    ]

    for _, row in df.iterrows():

        shell_id = row.get('壳体号', '')

        if pd.isna(shell_id) or shell_id == '':

            continue

        shell_id = str(shell_id).strip()

        

        part_number_value = row.get('料号', '')

        if production_order_column is not None and production_order_column in row.index:

            production_order_value = row.get(production_order_column, '')

        else:

            production_order_value = row.get('生产订单', '')

        part_number = "" if pd.isna(part_number_value) else str(part_number_value).strip()

        production_order = "" if pd.isna(production_order_value) else str(production_order_value).strip()

        

        current_station_raw = row.get('当前站点', '')

        current_station = normalize_station_name(current_station_raw)

        

        # 收集未识别的站别

        if (

            current_station_raw

            and current_station == current_station_raw

            and current_station not in BASE_STATIONS

            and current_station not in {'工程分析', '已完成', '未开始'}

        ):

            unrecognized_stations.add(current_station_raw)

        

        shell_progress = {

            '壳体号': shell_id,

            '料号': part_number,

            '生产订单': production_order,

            '当前站点': current_station,

            '当前站点原始': current_station_raw,

            '上一站': row.get('上一站', ''),

            '完成站别': [],

            '站别时间': {},

            '是否工程分析': current_station == '工程分析'

        }



        station_time_mapping: Dict[str, object] = {}

        


        for _, standard_station, time_col in existing_station_time_cols:

            time_value = row.get(time_col)

            if pd.notna(time_value) and str(time_value).strip():

                shell_progress['完成站别'].append(standard_station)

                station_time_mapping[standard_station] = time_value

        

        shell_progress['站别时间'] = station_time_mapping

        

        progress_data.append(shell_progress)

    

    # 如果有未识别的站别，显示警告

    if unrecognized_stations:

        st.warning(f"⚠️ 发现未识别的站别名称: {', '.join(sorted(unrecognized_stations))}")

    

    result_df = pd.DataFrame(progress_data)

    result_df.attrs["production_order_column"] = production_order_column

    result_df.attrs["time_cols"] = [f"{excel_col}时间" for excel_col in STATION_MAPPING.keys() if f"{excel_col}时间" in df.columns]

    return result_df



def calculate_station_counts(progress_df: pd.DataFrame) -> pd.DataFrame:

    """统计各当前站别的壳体数量与占比"""

    if progress_df.empty:

        return pd.DataFrame(columns=["站别", "数量", "占比"])



    unknown_label = "未识别"

    station_series = (

        progress_df["当前站点"]

        .fillna("")

        .astype(str)

        .str.strip()

    )

    station_series = station_series.replace({"": unknown_label, "nan": unknown_label})

    station_series = station_series.apply(

        lambda value: normalize_station_name(value) if value != unknown_label else value

    )



    counts = station_series.value_counts(dropna=False).reset_index()

    counts.columns = ["站别", "数量"]

    counts["占比"] = counts["数量"] / len(progress_df)



    # RMA 已经在 BASE_STATIONS 中，工程分析放在最后

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

        part_number = row.get('料号', '')

        stations = get_stations_for_part(part_number)

        current_station = row.get('当前站点', '')

        is_engineering = row.get('是否工程分析', False)

        

        # 如果是工程分析，添加工程分析站别

        if is_engineering:

            stations.append('工程分析')

        

        # 获取当前站点的序号（用于排序和计算已完成站别数）

        station_order = -1

        completed_count = 0

        

        if is_engineering:

            # 工程分析：根据上一站计算已完成站别数

            last_station = row.get('上一站', '')

            last_station_normalized = normalize_station_name(last_station)

            if last_station_normalized and last_station_normalized in stations:

                station_order = stations.index(last_station_normalized)

                completed_count = station_order + 1  # 包括上一站

        elif current_station == "已完成":

            # 已完成：所有站别都已完成

            station_order = len(stations) - 1  # 最后一个站别的索引

            completed_count = len(stations)  # 所有站别都完成

        elif current_station and current_station in stations:

            # 正常情况：当前站点之前的都是已完成

            station_order = stations.index(current_station)

            completed_count = station_order  # 当前站点之前的站别数

        else:

            # 没有当前站点信息，使用完成时间记录

            completed_count = len(row['完成站别'])

        

        total_count = len(stations)

        progress_pct = (completed_count / total_count * 100) if total_count > 0 else 0

        

        # 获取最新完成站别（当前站点的前一个）

        last_completed_station = ''

        if station_order > 0:

            last_completed_station = stations[station_order - 1]

        elif row['完成站别']:

            last_completed_station = row['完成站别'][-1]

        

        table_data.append({

            '壳体号': row['壳体号'],

            '料号': part_number,

            '生产订单': row.get('生产订单', ''),

            '当前站点': current_station,

            '已完成站别数': completed_count,

            '总站别数': total_count,

            '完成进度': f"{progress_pct:.1f}%",

            '最新完成站别': last_completed_station,

            '是否工程分析': '是' if is_engineering else '否',

            '站别序号': station_order  # 用于排序的隐藏列

        })

    

    result_df = pd.DataFrame(table_data)
    
    # 只在DataFrame不为空时进行排序和删除列操作
    if '站别序号' in result_df.columns:
        if not result_df.empty:
            # 按站别序号排序（从小到大，进度慢的在前）
            result_df = result_df.sort_values('站别序号', ascending=True)
        # 删除排序用的列；errors='ignore' 防止列不存在时报错
        result_df = result_df.drop(columns=['站别序号'], errors='ignore')

    

    return result_df



# Streamlit 页面

st.set_page_config(page_title="模块进度", page_icon="📊", layout="wide")



st.title("模块WIP进度")



st.markdown(

    """

    <style>

    .stMultiSelect div[data-baseweb="select"] > div {

        flex-wrap: wrap;

    }

    .stMultiSelect [data-baseweb="tag"] {

        max-width: none !important;

        min-width: 260px !important;

        width: fit-content !important;

        display: inline-flex !important;

        align-items: center !important;

        justify-content: space-between !important;

        flex: 0 0 auto !important;

        gap: 6px !important;

    }

    .stMultiSelect [data-baseweb="tag"] > * {

        max-width: none !important;

        flex: 1 1 auto !important;

    }

    .stMultiSelect [data-baseweb="tag-text"] {

        max-width: none !important;

        flex: 1 1 auto !important;

    }

    .stMultiSelect [data-baseweb="tag-text"] span {

        max-width: none !important;

        white-space: nowrap !important;

        overflow: visible !important;

        text-overflow: clip !important;

    }

    .stMultiSelect [data-baseweb="tag"] p {

        white-space: nowrap !important;

        overflow: visible !important;

    }

    .custom-order-tags {

        display: flex;

        flex-wrap: wrap;

        gap: 10px;

        margin-top: 6px;

    }

    .custom-order-tag {

        background-color: #ff5a5f;

        color: #ffffff;

        padding: 4px 14px;

        border-radius: 10px;

        font-weight: 600;

        letter-spacing: 0.5px;

    }

    </style>

    """,

    unsafe_allow_html=True,

)



# 初始化 session_state

if 'progress_df' not in st.session_state:

    st.session_state.progress_df = None

if 'progress_raw_df' not in st.session_state:

    st.session_state.progress_raw_df = None

if 'uploaded_filename' not in st.session_state:

    st.session_state.uploaded_filename = None

if 'progress_dir_cache' not in st.session_state:
    st.session_state.progress_dir_cache = {}
if 'progress_data_cache' not in st.session_state:
    st.session_state.progress_data_cache = {}



# 添加数据源选择（默认从文件夹选择）

if 'progress_data_source' not in st.session_state:

    st.session_state.progress_data_source = "📁 从文件夹选择"



data_source = st.radio(

    "选择数据源",

    options=["📁 从文件夹选择", "📤 上传文件"],

    index=0,  # 默认选择第一个（从文件夹选择）

    horizontal=True,

    key="progress_data_source"

)



uploaded_file = None

selected_file_path = None



if data_source == "📤 上传文件":

    # 文件上传

    uploaded_file = st.file_uploader(

        "上传包含壳体进度信息的文件",

        type=['csv', 'xlsx', 'xls'],

        help="请上传包含壳体号和各站别时间信息的 Excel 或 CSV 文件"

    )

    

    # 如果上传了新文件，更新 session_state

    if uploaded_file is not None and (st.session_state.uploaded_filename != uploaded_file.name):

        # 解析文件并保存到 session_state

        with st.spinner("正在解析文件..."):

            df = parse_uploaded_file(uploaded_file)

        

        if df is not None:

            st.session_state.progress_raw_df = df

            st.session_state.progress_df = extract_progress_data(df, light=st.session_state.get('progress_only_stats', False))

            st.session_state.uploaded_filename = uploaded_file.name

            st.success(f"✅ 文件解析成功！共 {len(df)} 条记录")



else:  # 从文件夹选择



    col_path, col_refresh = st.columns([4, 1])

    with col_path:

        folder_path = st.text_input(

            "文件夹路径",

            value=DEFAULT_DATA_FOLDER,

            placeholder=f"默认: {DEFAULT_DATA_FOLDER}",

            key="progress_folder_path"

        )

    with col_refresh:

        st.markdown("<div style='margin-top: 32px;'></div>", unsafe_allow_html=True)

        refresh_btn = st.button("🔄 刷新", use_container_width=True)

    

    if folder_path:

        try:

            search_path = resolve_input_path(folder_path)

            if search_path.exists() and search_path.is_dir():

                # 查找 Excel 和 CSV 文件

                excel_files = list(search_path.glob("*.xlsx")) + list(search_path.glob("*.xls"))

                csv_files = list(search_path.glob("*.csv"))

                all_files = sorted(excel_files + csv_files, key=lambda x: x.stat().st_mtime, reverse=True)

                

                # 筛选包含"光耦WIP报表"的文件

                wip_files = [f for f in all_files if any(keyword in f.name or keyword.lower() in f.name.lower() for keyword in WIP_REPORT_KEYWORDS)]

                

                if all_files:

                    # 如果找到光耦WIP报表文件，优先显示

                    display_files = wip_files if wip_files else all_files
                    MAX_DISPLAY_FILES = 200
                    display_files = display_files[:MAX_DISPLAY_FILES]

                    

                    # 创建文件选择下拉框

                    _dir_key = str(search_path)
                    _dir_cache = st.session_state.progress_dir_cache.get(_dir_key, {})
                    file_display_map = {}
                    for f in display_files:
                        fp = str(f)
                        mtime = f.stat().st_mtime
                        meta = _dir_cache.get(fp)
                        if not meta or meta.get('mtime') != mtime:
                            size_kb = f.stat().st_size / 1024.0
                            _dir_cache[fp] = {'mtime': mtime, 'size_kb': size_kb}
                        else:
                            size_kb = meta['size_kb']
                        file_display_map[f"{f.name} ({size_kb:.1f} KB)"] = fp
                    st.session_state.progress_dir_cache[_dir_key] = _dir_cache
                    file_options = file_display_map

                    

                    # 默认选择第一个文件（最新的光耦WIP报表）

                    default_index = 0

                    

                    selected_file_display = st.selectbox(

                        "选择文件" + (" (已筛选光耦WIP报表)" if wip_files else ""),

                        options=list(file_options.keys()),

                        index=default_index,

                        key="progress_file_select"

                    )

                    

                    if selected_file_display:

                        selected_file_path = file_options[selected_file_display]

                        

                        # 自动加载最新的光耦WIP报表（仅在首次加载时）

                        auto_load = False

                        if st.session_state.progress_df is None and wip_files and selected_file_display == list(file_options.keys())[0]:

                            auto_load = True

                        

                        # 添加加载按钮

                        load_btn = st.button("📂 加载选中的文件", type="primary")

                        

                        if load_btn or auto_load:

                            with st.spinner(f"正在加载 {Path(selected_file_path).name}..."):

                                try:
                                    p = Path(selected_file_path)
                                    _data_key = f"{p.resolve()}::{p.stat().st_mtime}"
                                    cached = st.session_state.progress_data_cache.get(_data_key)
                                    if cached:
                                        df, cached_progress_df = cached
                                        st.session_state.progress_raw_df = df
                                        st.session_state.progress_df = cached_progress_df
                                        st.session_state.uploaded_filename = Path(selected_file_path).name
                                        st.success(f"✅ 已从缓存加载！共 {len(df)} 条记录")
                                        st.rerun()

                                    read_t0 = time.perf_counter()
                                    if selected_file_path.endswith('.csv'):
                                        header_df = pd.read_csv(selected_file_path, nrows=0)
                                        usecols = _compute_usecols(list(header_df.columns))
                                        time_cols = [f"{excel_col}时间" for excel_col in STATION_MAPPING.keys() if f"{excel_col}时间" in header_df.columns]
                                        dtype_map = {c: "string" for c in ["壳体号", "料号", "生产订单"] if c in usecols}
                                        df = pd.read_csv(selected_file_path, usecols=usecols, dtype=dtype_map, parse_dates=time_cols, infer_datetime_format=True, low_memory=False)
                                    else:
                                        header_df = pd.read_excel(selected_file_path, nrows=0)
                                        usecols = _compute_usecols(list(header_df.columns))
                                        df = pd.read_excel(selected_file_path, usecols=usecols, engine="openpyxl" if selected_file_path.endswith('.xlsx') else None)
                                        time_cols = [f"{excel_col}时间" for excel_col in STATION_MAPPING.keys() if f"{excel_col}时间" in header_df.columns]
                                        if time_cols:
                                            df[time_cols] = df[time_cols].apply(pd.to_datetime, errors='coerce')
                                    read_t1 = time.perf_counter()

                                    

                                    st.session_state.progress_raw_df = df

                                    parse_t0 = time.perf_counter()
                                    st.session_state.progress_df = extract_progress_data(df, light=st.session_state.get('progress_only_stats', False))
                                    parse_t1 = time.perf_counter()
                                    st.info(f"读取耗时: {(read_t1 - read_t0)*1000:.0f} ms，解析耗时: {(parse_t1 - parse_t0)*1000:.0f} ms")

                                    st.session_state.uploaded_filename = Path(selected_file_path).name

                                    st.success(f"✅ 文件加载成功！共 {len(df)} 条记录")
                                    try:
                                        p = Path(selected_file_path)
                                        _data_key = f"{p.resolve()}::{p.stat().st_mtime}"
                                        st.session_state.progress_data_cache[_data_key] = (df, st.session_state.progress_df)
                                    except Exception:
                                        pass

                                    if auto_load:

                                        st.rerun()

                                except Exception as e:

                                    st.error(f"文件加载失败: {str(e)}")

                else:

                    st.warning(f"在 `{search_path}` 中未找到数据文件")

            else:

                st.error(f"路径不存在或不是文件夹: {search_path}")

        except ValueError as value_error:

            st.error(str(value_error))

        except Exception as e:

            st.error(f"读取文件夹时出错: {str(e)}")



# 使用 session_state 中的数据

if st.session_state.progress_df is not None:

    progress_df = st.session_state.progress_df

    df = st.session_state.progress_raw_df

    # 兼容旧版 session_state：如果缺少生产订单列或属性，则重新生成

    if (

        progress_df is not None

        and df is not None

        and (

            '生产订单' not in progress_df.columns

            or "production_order_column" not in progress_df.attrs

        )

    ):

        progress_df = extract_progress_data(df, light=st.session_state.get('progress_only_stats', False))

        st.session_state.progress_df = progress_df

    preview_df = df

    production_order_column = progress_df.attrs.get("production_order_column")

    

    if len(progress_df) > 0:

        filtered_progress_df = progress_df.copy()

        selected_order_values = None

        selected_orders_display: List[str] = []



        with st.container():

            if '生产订单' in filtered_progress_df.columns:

                order_series = (

                    filtered_progress_df['生产订单']

                    .dropna()

                    .astype(str)

                    .str.strip()

                )

                order_series = order_series[order_series != ""]

                order_options = sorted(order_series.unique().tolist())

                

                if order_options:

                    selected_orders = st.multiselect(

                        "生产订单",

                        options=order_options,

                        default=order_options,

                        key="progress_production_orders",

                    )

                    selected_orders_display = selected_orders or []

                    if selected_orders_display:

                        selected_order_values = {

                            order.strip() for order in selected_orders_display

                        }

                        filtered_progress_df = filtered_progress_df[

                            filtered_progress_df['生产订单']

                            .fillna("")

                            .astype(str)

                            .str.strip()

                            .isin(selected_order_values)

                        ]

                    else:

                        selected_order_values = set()

                        filtered_progress_df = filtered_progress_df.iloc[0:0]

                else:

                    st.multiselect(

                        "生产订单",

                        options=[],

                        default=[],

                        key="progress_production_orders",

                    )

                    st.caption("未检测到生产订单数据")

            else:

                st.info("当前数据缺少生产订单列")

        

        if selected_order_values is not None:

            if production_order_column and df is not None and production_order_column in df.columns:

                preview_series = (

                    df[production_order_column]

                    .fillna("")

                    .astype(str)

                    .str.strip()

                )

                if selected_order_values:

                    preview_df = df[preview_series.isin(selected_order_values)]

                else:

                    preview_df = df.iloc[0:0]

            elif not selected_order_values:

                preview_df = df.iloc[0:0]

        

        if filtered_progress_df.empty:

            st.warning("筛选条件下没有数据，请调整生产订单选择。")

        else:

            st.caption('当前筛选结果已缓存，可在“数据分析”页统一保存。')



            # 显示统计信息

            col1, col2, col3, col4 = st.columns([1, 1.2, 1, 1.5])

            with col1:

                st.metric("壳体总数", len(filtered_progress_df))

            with col2:
                if '完成站别' in filtered_progress_df.columns:
                    avg_progress = filtered_progress_df['完成站别'].apply(len).mean()
                    st.metric("平均完成站别数", f"{avg_progress:.1f}")

            with col3:

                total_stations = len(BASE_STATIONS)

                st.metric("基础站别数", total_stations)

            with col4:
                latest_time = None
                time_cols = progress_df.attrs.get("time_cols", [])
                if df is not None and time_cols:
                    tc = [c for c in time_cols if c in df.columns]
                    if tc:
                        parsed = df[tc].apply(pd.to_datetime, errors='coerce')
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

                    # 添加颜色映射以创建渐变效果
                    chart = (

                        alt.Chart(counts_df)

                        .mark_bar(
                            cornerRadius=8,  # 圆角效果
                            opacity=0.9,     # 略微透明增加层次感
                            strokeWidth=1.5  # 描边宽度
                        )

                        .encode(

                            x=alt.X("数量:Q", title="完成数量", 
                                    axis=alt.Axis(grid=True, gridOpacity=0.2, tickMinStep=1)),

                            y=alt.Y("站别:N", sort=station_order, title="站别",
                                    axis=alt.Axis(labelFontSize=12, labelFontWeight='bold')),

                            # 使用渐变色方案创建3D感
                            color=alt.Color('数量:Q',
                                          scale=alt.Scale(
                                              scheme='blues',  # 蓝色渐变方案
                                              domain=[counts_df["数量"].min(), counts_df["数量"].max()]
                                          ),
                                          legend=None),

                            # 添加描边颜色，让条形更立体
                            stroke=alt.value('#ffffff33'),  # 半透明白色描边

                            tooltip=["站别", "数量", alt.Tooltip("占比:Q", title="占比", format=".1%")],

                        )

                    ).properties(
                        height=chart_height
                    ).configure_view(
                        strokeWidth=0  # 移除外边框
                    ).configure_axis(
                        titleFontSize=13,
                        titleFontWeight='bold'
                    )

                    st.altair_chart(chart, use_container_width=True, theme="streamlit")

            

            # 工程分析站别分布

            engineering_df = filtered_progress_df[filtered_progress_df['是否工程分析'] == True]

            if not engineering_df.empty:

                st.markdown("---")

                st.markdown("### 🔍 工程分析站别分布")

                

                # 统计工程分析在各站别的数量

                engineering_stations = []

                for _, row in engineering_df.iterrows():

                    last_station = row.get('上一站', '')

                    last_station_normalized = normalize_station_name(last_station)

                    if last_station_normalized:

                        engineering_stations.append(last_station_normalized)

                

                if engineering_stations:

                    engineering_counts = pd.Series(engineering_stations).value_counts().reset_index()

                    engineering_counts.columns = ['站别', '数量']

                    engineering_counts['占比'] = engineering_counts['数量'] / engineering_counts['数量'].sum()

                    

                    eng_table_col, eng_chart_col = st.columns([2, 3])

                    

                    with eng_table_col:

                        st.caption(f"工程分析总数: {len(engineering_df)} 个")

                        eng_counts_style = engineering_counts.style.format({"占比": "{:.1%}"})

                        st.dataframe(eng_counts_style, use_container_width=True, height=300)

                    

                    with eng_chart_col:

                        # 创建饼图

                        pie_chart = alt.Chart(engineering_counts).mark_arc(innerRadius=40).encode(

                            theta=alt.Theta('数量:Q', stack=True),

                            color=alt.Color('站别:N', 

                                          legend=alt.Legend(title='站别', orient='right'),

                                          scale=alt.Scale(scheme='category20')),

                            tooltip=[

                                alt.Tooltip('站别:N', title='站别'),

                                alt.Tooltip('数量:Q', title='数量'),

                                alt.Tooltip('占比:Q', title='占比', format='.1%')

                            ]

                        ).properties(

                            height=300,

                            title='工程分析站别占比'

                        )

                        st.altair_chart(pie_chart, use_container_width=True)

            

            if not st.session_state.get('progress_only_stats', False):
                st.markdown("---")
                st.markdown("### 📋 进度表格")
                show_eng_only = st.checkbox("🔍 仅显示工程分析的壳体", value=False, key="progress_show_eng_only")
                source_df = filtered_progress_df[filtered_progress_df['是否工程分析'] == True] if show_eng_only else filtered_progress_df
                table_df = create_progress_table(source_df)
                def highlight_engineering(row):
                    return [''] * len(row)
                styled_df = table_df.style.apply(highlight_engineering, axis=1)
                st.dataframe(styled_df, use_container_width=True, height=400)







    else:

        st.warning("⚠️ 未找到有效的壳体进度数据")

        

    # 显示原始数据预览

    with st.expander("📄 查看原始数据"):

        st.dataframe(preview_df.head(20), use_container_width=True)

else:

    # 显示使用说明

    st.info("""

    ### 📖 使用说明

    

    1. **上传文件**：点击上方按钮上传包含壳体进度信息的 Excel 或 CSV 文件

    2. **查看结果**：

       - 甘特图：直观展示所有壳体在各站别的进度

       - 进度表格：详细列出每个壳体的完成情况

    """)

if 'progress_only_stats' not in st.session_state:
    st.session_state.progress_only_stats = False
st.checkbox("仅统计模式", value=st.session_state.progress_only_stats, key="progress_only_stats")

