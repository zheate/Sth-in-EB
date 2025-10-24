# title: 进度追踪

import streamlit as st
import pandas as pd
import altair as alt
from datetime import datetime
from typing import List, Dict, Optional
import io
import sys
from pathlib import Path

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

def parse_uploaded_file(uploaded_file) -> Optional[pd.DataFrame]:
    """解析上传的文件"""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
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

def extract_progress_data(df: pd.DataFrame) -> pd.DataFrame:
    """从原始数据中提取进度信息"""
    progress_data = []
    unrecognized_stations = set()
    
    column_lookup = {str(col).strip(): col for col in df.columns}
    production_order_column = next(
        (column_lookup[name] for name in PRODUCTION_ORDER_CANDIDATES if name in column_lookup),
        None,
    )
    
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
        
        # 检查所有站别的时间列
        for excel_col, standard_station in STATION_MAPPING.items():
            time_col = f"{excel_col}时间"
            if time_col in df.columns:
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

def create_gantt_chart(progress_df: pd.DataFrame) -> alt.Chart:
    """创建甘特图（使用 Altair）"""
    # 过滤掉已完成的壳体
    progress_df = progress_df[progress_df.get('当前站点', '') != '已完成'].copy()
    
    # 准备甘特图数据
    gantt_data = []

    def format_time_value(value: object) -> str:
        """将原始时间值格式化为统一的展示字符串"""
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return "--"
        if isinstance(value, pd.Timestamp):
            return value.strftime("%m-%d %H:%M")
        if isinstance(value, datetime):
            return value.strftime("%m-%d %H:%M")
        if hasattr(value, "strftime"):
            try:
                return value.strftime("%m-%d %H:%M")
            except Exception:  # pragma: no cover - 容错处理
                pass
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return "--"
            parsed = pd.to_datetime(stripped, errors="coerce")
            if pd.notna(parsed):
                return parsed.strftime("%m-%d %H:%M")
            return stripped
        if isinstance(value, (int, float)):
            if pd.isna(value):
                return "--"
            parsed_excel = pd.NaT
            base_date = pd.Timestamp("1899-12-30")
            try:
                parsed_excel = base_date + pd.to_timedelta(float(value), unit="D")
            except Exception:  # pragma: no cover - 容错处理
                parsed_excel = pd.NaT
            if pd.notna(parsed_excel):
                return parsed_excel.strftime("%m-%d %H:%M")
            parsed = pd.NaT
            try:
                parsed = pd.to_datetime(value)
            except Exception:  # pragma: no cover - 容错处理
                parsed = pd.NaT
            if pd.notna(parsed):
                return parsed.strftime("%m-%d %H:%M")
            return str(value)
        return str(value)

    # 收集所有需要显示的站别（包括VBG和工程分析）
    all_stations_set = set()
    
    for idx, row in progress_df.iterrows():
        shell_id = row['壳体号']
        part_number = row.get('料号', '')
        completed_stations = row['完成站别']
        current_station = row.get('当前站点', '')
        is_engineering = row.get('是否工程分析', False)
        station_times = row.get('站别时间', {})
        if not isinstance(station_times, dict):
            station_times = {}
        
        # 获取该料号适用的站别列表
        stations = get_stations_for_part(part_number)
        
        all_stations_set.update(stations)
        
        # 如果是工程分析，找到上一站的索引
        last_station_idx = -1
        if is_engineering:
            # 获取上一站
            last_station = row.get('上一站', '')
            # 标准化上一站名称
            last_station_normalized = normalize_station_name(last_station)
            if last_station_normalized and last_station_normalized in stations:
                last_station_idx = stations.index(last_station_normalized)
        
        # 检查是否为 RMA 站别
        is_rma = (current_station == "RMA")
        
        # 找到当前站点的索引位置（非工程分析和非RMA的情况）
        current_station_idx = -1
        if not is_engineering and not is_rma and current_station and current_station in stations:
            current_station_idx = stations.index(current_station)
        
        # 检查是否到达"已完成"站别
        is_fully_completed = (current_station == "已完成")
        
        for station_idx, station in enumerate(stations):
            # 判断状态的逻辑
            if is_engineering:
                # 工程分析：上一站及之前的都标记为工程分析（红色）
                if last_station_idx >= 0 and station_idx <= last_station_idx:
                    status = "已完成"
                    is_engineering_cell = True
                else:
                    status = "未开始"
                    is_engineering_cell = False
            elif is_rma:
                # RMA：整行都标记为 RMA 状态
                status = "RMA"
                is_engineering_cell = False
            elif is_fully_completed:
                # 如果当前站点是"已完成"，整行都标记为全部完成（全绿）
                status = "全部完成"
                is_engineering_cell = False
            else:
                # 正常情况：
                # 1. 当前站点就是进行中（红色）
                # 2. 当前站点之前的都是已完成（深灰）
                # 3. 当前站点之后的都是未开始（浅灰）
                is_engineering_cell = False
                if current_station_idx >= 0:
                    if station_idx < current_station_idx:
                        status = "已完成"
                    elif station_idx == current_station_idx:
                        status = "进行中"
                    else:
                        status = "未开始"
                else:
                    # 如果没有当前站点信息，根据完成时间判断
                    if station in completed_stations:
                        status = "已完成"
                    else:
                        status = "未开始"

            time_value = station_times.get(station)
            time_source_station = station
            if (time_value is None or (isinstance(time_value, str) and not time_value.strip())) and status == "进行中":
                for prev_idx in range(station_idx - 1, -1, -1):
                    prev_station = stations[prev_idx]
                    prev_value = station_times.get(prev_station)
                    if prev_value is not None:
                        time_value = prev_value
                        time_source_station = prev_station
                        break

            time_display = format_time_value(time_value)
            if time_display != "--" and time_source_station != station:
                time_display = f"{time_display}（上一站：{time_source_station}）"

            gantt_data.append({
                '壳体号': shell_id,
                '站别': station,
                '站别序号': station_idx,
                '状态': status,
                '是否工程分析单元格': is_engineering_cell,
                '站别时间': time_display
            })
    
    gantt_df = pd.DataFrame(gantt_data)
    
    # 创建所有站别的排序列表（不包括工程分析）
    all_stations_sorted = BASE_STATIONS.copy()
    if 'VBG' in all_stations_set:
        hesu_idx = all_stations_sorted.index("合束后烘烤")
        all_stations_sorted.insert(hesu_idx + 1, "VBG")
        all_stations_sorted.insert(hesu_idx + 2, "VBG后烘烤")
    # 添加"已完成"作为最后一个站别
    all_stations_sorted.append("已完成")
    
    # 定义颜色方案
    color_scale = alt.Scale(
        domain=['未开始', '进行中', '已完成'],
        range=['#bdc3c7', '#f1c40f', '#2ecc71']  # 灰色、黄色、绿色
    )
    
    # 为每个壳体计算当前站别的序号（用于排序）
    shell_station_order = {}
    for shell_id in gantt_df['壳体号'].unique():
        shell_data = gantt_df[gantt_df['壳体号'] == shell_id]
        # 检查是否全部完成（优先级最高，排在最上面）
        fully_completed = shell_data[shell_data['状态'] == '全部完成']
        if not fully_completed.empty:
            shell_station_order[shell_id] = 99999  # 使用最大值让全部完成的排在最上面
        else:
            # 检查是否为 RMA 状态
            rma_status = shell_data[shell_data['状态'] == 'RMA']
            if not rma_status.empty:
                # RMA 使用一个较大的序号，排在正常流程之后
                shell_station_order[shell_id] = 90000
            else:
                # 找到进行中的站别序号，如果没有则用已完成的最大序号
                in_progress = shell_data[shell_data['状态'] == '进行中']
                if not in_progress.empty:
                    shell_station_order[shell_id] = in_progress.iloc[0]['站别序号']
                else:
                    completed = shell_data[shell_data['状态'] == '已完成']
                    if not completed.empty:
                        shell_station_order[shell_id] = completed['站别序号'].max()
                    else:
                        shell_station_order[shell_id] = -1
    
    # 添加排序字段到数据框
    gantt_df['壳体排序序号'] = gantt_df['壳体号'].map(shell_station_order)
    
    # 为工程分析单元格添加特殊状态标识
    gantt_df['显示状态'] = gantt_df.apply(
        lambda row: '工程分析' if row['是否工程分析单元格'] else row['状态'],
        axis=1
    )
    
    # 定义包含工程分析、RMA和全部完成的颜色方案
    color_scale_with_engineering = alt.Scale(
        domain=['未开始', '进行中', '已完成', '工程分析', 'RMA', '全部完成'],
        range=['#D3D2D2', '#E84445', '#074166', '#CC011F', '#6FDCB5', '#2ecc71']  # 浅灰、红色、深灰、红色、青绿色、绿色
    )
    
    # 标记测试站别（用于X轴标签颜色标识）
    test_stations = ['耦合测试', 'NA测试', 'Pre测试', '低温存储后测试', 
                     '高温存储后测试', 'post测试', '封盖测试']
    gantt_df['是否测试站别'] = gantt_df['站别'].apply(lambda x: '测试站别' if x in test_stations else '普通站别')
    
    # 创建统一的热力图
    base_chart = alt.Chart(gantt_df).mark_rect(
        stroke='white',
        strokeWidth=0.5
    ).encode(
        x=alt.X('站别:N', 
                title='站别',
                sort=all_stations_sorted,
                axis=alt.Axis(
                    labelAngle=-90, 
                    labelLimit=200,
                    labelColor=alt.expr(
                        f"datum.label == '耦合测试' || datum.label == 'NA测试' || "
                        f"datum.label == 'Pre测试' || datum.label == '低温存储后测试' || datum.label == '高温存储后测试' || "
                        f"datum.label == 'post测试' || datum.label == '封盖测试' ? '#CC011F' : 'black'"
                    )
                )),
        y=alt.Y('壳体号:N', 
                title='壳体号',
                sort=alt.EncodingSortField(field='壳体排序序号', order='descending')),
        color=alt.Color('显示状态:N',
                       scale=color_scale_with_engineering,
                       legend=alt.Legend(title='状态', orient='top')),
        tooltip=[
            alt.Tooltip('壳体号:N', title='壳体号'),
            alt.Tooltip('站别:N', title='站别'),
            alt.Tooltip('站别时间:N', title='时间'),
            alt.Tooltip('显示状态:N', title='状态')
        ]
    ).properties(
        width=1200,
        height=max(400, len(progress_df) * 30),
        title='模块进度甘特图'
    )
    
    chart = base_chart
    
    chart = chart.configure_axis(
        labelFontSize=11,
        titleFontSize=13,
        labelColor='black',
        titleColor='black'
    ).configure_title(
        fontSize=16,
        anchor='start',
        color='black'
    ).configure_view(
        strokeWidth=0
    ).configure_scale(
        bandPaddingInner=0,
        bandPaddingOuter=0
    )
    
    return chart

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
    # 按站别序号排序（从小到大，进度慢的在前）
    result_df = result_df.sort_values('站别序号', ascending=True)
    # 删除排序用的列
    result_df = result_df.drop(columns=['站别序号'])
    
    return result_df

# Streamlit 页面
st.set_page_config(page_title="模块进度", page_icon="📊", layout="wide")

st.title("📊 模块WIP进度")

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
st.markdown("---")

# 初始化 session_state
if 'progress_df' not in st.session_state:
    st.session_state.progress_df = None
if 'progress_raw_df' not in st.session_state:
    st.session_state.progress_raw_df = None
if 'uploaded_filename' not in st.session_state:
    st.session_state.uploaded_filename = None

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
            st.session_state.progress_df = extract_progress_data(df)
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
        refresh_btn = st.button("🔄 刷新", width='stretch')
    
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
                    
                    # 创建文件选择下拉框
                    file_options = {f"{f.name} ({f.stat().st_size / 1024:.1f} KB)": str(f) for f in display_files}
                    
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
                                    if selected_file_path.endswith('.csv'):
                                        df = pd.read_csv(selected_file_path)
                                    else:
                                        df = pd.read_excel(selected_file_path)
                                    
                                    st.session_state.progress_raw_df = df
                                    st.session_state.progress_df = extract_progress_data(df)
                                    st.session_state.uploaded_filename = Path(selected_file_path).name
                                    st.success(f"✅ 文件加载成功！共 {len(df)} 条记录")
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
        progress_df = extract_progress_data(df)
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
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("壳体总数", len(filtered_progress_df))
            with col2:
                avg_progress = filtered_progress_df['完成站别'].apply(len).mean()
                st.metric("平均完成站别数", f"{avg_progress:.1f}")
            with col3:
                total_stations = len(BASE_STATIONS)
                st.metric("基础站别数", total_stations)
            
            counts_df = calculate_station_counts(filtered_progress_df)
            if not counts_df.empty:
                st.markdown("### 各站别当前数量")
                table_col, chart_col = st.columns([2, 3])
                with table_col:
                    counts_style = counts_df.style.format({"占比": "{:.1%}"})
                    st.dataframe(counts_style, width='stretch', height=360)
                with chart_col:
                    station_order = counts_df["站别"].tolist()
                    chart = (
                        alt.Chart(counts_df)
                        .mark_bar()
                        .encode(
                            x=alt.X("数量:Q", title="壳体数量"),
                            y=alt.Y("站别:N", sort=station_order, title="站别"),
                            tooltip=["站别", "数量", alt.Tooltip("占比:Q", title="占比", format=".1%")],
                        )
                    )
                    st.altair_chart(chart, use_container_width=True)
            
            st.markdown("---")
            
            # 选项卡
            tab1, tab2 = st.tabs(["📈 甘特图", "📋 进度表格"])
            
            with tab1:
                with st.spinner("正在生成甘特图..."):
                    chart = create_gantt_chart(filtered_progress_df)
                    st.altair_chart(chart, use_container_width=True)
                        
            with tab2:
                table_df = create_progress_table(filtered_progress_df)
                
                # 使用样式高亮工程分析行
                def highlight_engineering(row):
                    if row['是否工程分析'] == '是':
                        return ['background-color: #ffcccc'] * len(row)
                    return [''] * len(row)
                
                styled_df = table_df.style.apply(highlight_engineering, axis=1)
                st.dataframe(
                    styled_df,
                    width='stretch',
                    height=400
                )
                
                # 下载按钮 - 使用Excel格式避免编码问题
                buffer = io.BytesIO()
                try:
                    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                        table_df.to_excel(writer, index=False, sheet_name='进度表')
                except ImportError:
                    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                        table_df.to_excel(writer, index=False, sheet_name='进度表')
                buffer.seek(0)
                
                st.download_button(
                    label="📥 下载进度表格 (Excel)",
                    data=buffer,
                    file_name=f"壳体进度_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )


    else:
        st.warning("⚠️ 未找到有效的壳体进度数据")
        
    # 显示原始数据预览
    with st.expander("📄 查看原始数据"):
        st.dataframe(preview_df.head(20), width='stretch')
    
    # 添加清除数据按钮
    if st.button("🗑️ 清除已上传的数据"):
        st.session_state.progress_df = None
        st.session_state.progress_raw_df = None
        st.session_state.uploaded_filename = None
        st.rerun()
else:
    # 显示使用说明
    st.info("""
    ### 📖 使用说明
    
    1. **上传文件**：点击上方按钮上传包含壳体进度信息的 Excel 或 CSV 文件
    2. **查看结果**：
       - 甘特图：直观展示所有壳体在各站别的进度
       - 进度表格：详细列出每个壳体的完成情况
    """)
