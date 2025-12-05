"""
Constants for Data Manager (Zh's DataBase).

This module defines:
- Database paths
- Station lists (from Progress.py BASE_STATIONS)
- Station mappings
- Metric columns for analysis
"""

from pathlib import Path
from typing import Dict, List

# ============================================================================
# Database Paths
# ============================================================================

# Base directory for zh_database
DATABASE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "zh_database"

# Subdirectories
ATTACHMENTS_DIR = DATABASE_DIR / "attachments"
SHELLS_DIR = DATABASE_DIR / "shells"
THRESHOLD_CONFIG_DIR = DATABASE_DIR / "thresholds"

# Metadata file
PRODUCT_TYPES_FILE = DATABASE_DIR / "product_types.json"


def ensure_database_dirs() -> None:
    """Ensure all database directories exist."""
    DATABASE_DIR.mkdir(parents=True, exist_ok=True)
    ATTACHMENTS_DIR.mkdir(parents=True, exist_ok=True)
    SHELLS_DIR.mkdir(parents=True, exist_ok=True)
    THRESHOLD_CONFIG_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Station Lists (from Progress.py)
# ============================================================================

# 定义所有站别（按工艺流程顺序）- 基础站别
BASE_STATIONS: List[str] = [
    "打标", "清洗", "壳体组装", "回流", "fac前备料", "打线", "fac", "fac补胶",
    "fac补胶后烘烤", "fac测试", "sac组装", "光纤组装", "光纤组装后烘烤",
    "红光耦合", "装大反", "红光耦合后烘烤", "合束", "合束后烘烤",
    "NA前镜检", "NA前红光端面检", "NA测试", "耦合测试", "补胶", "温度循环",
    "Pre测试", "低温存储", "低温存储后测试", "高温存储", "高温存储后测试",
    "老化前红光端面", "post测试", "红光端面检查", "镜检", "封盖", "封盖测试",
    "分级", "入库检", "入库", "RMA"
]

# VBG 相关站别（用于料号包含 V 的产品）
VBG_STATIONS: List[str] = ["VBG", "VBG后烘烤"]

# 特殊站别
SPECIAL_STATIONS: List[str] = ["工程分析", "已完成", "未开始"]


# ============================================================================
# Station Mapping (Excel column names to standard station names)
# ============================================================================

STATION_MAPPING: Dict[str, str] = {
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

# Lowercase version for case-insensitive lookup
STATION_MAPPING_LOWER: Dict[str, str] = {
    key.lower(): value for key, value in STATION_MAPPING.items()
}

# Lowercase BASE_STATIONS for case-insensitive lookup
BASE_STATIONS_LOWER: Dict[str, str] = {
    station.lower(): station for station in BASE_STATIONS
}


def get_stations_for_part(part_number: str) -> List[str]:
    """
    根据料号返回适用的站别列表。
    
    Args:
        part_number: 料号
        
    Returns:
        站别列表（按工艺流程顺序）
    """
    stations = BASE_STATIONS.copy()
    # 如果料号包含V，在合束后烘烤后面插入VBG和VBG后烘烤
    if 'V' in str(part_number).upper():
        hesu_idx = stations.index("合束后烘烤")
        stations.insert(hesu_idx + 1, "VBG")
        stations.insert(hesu_idx + 2, "VBG后烘烤")
    # 添加"已完成"作为最后一个站别
    stations.append("已完成")
    return stations


def get_station_index(station: str, part_number: str = "") -> int:
    """
    获取站别在工艺流程中的索引。
    
    Args:
        station: 站别名称
        part_number: 料号（用于确定是否包含 VBG 站别）
        
    Returns:
        站别索引，如果不存在返回 -1
    """
    stations = get_stations_for_part(part_number)
    try:
        return stations.index(station)
    except ValueError:
        return -1


# ============================================================================
# Metric Columns for Analysis
# ============================================================================

# 指标设置允许的数值列（仅保留指定的六项）
COMMON_METRIC_COLUMNS: List[str] = [
    "电流(A)", "电流",
    "功率(W)", "功率",
    "电压(V)", "电压",
    "电光效率(%)", "电光效率",
    "波长lambda", "波长",
    "波长shift",
]

# 生产订单列名候选
PRODUCTION_ORDER_CANDIDATES: List[str] = [
    "生产订单",
    "ERP生产订单",
    "SAP生产订单",
    "生产订单号",
    "订单号",
    "工单号",
]

# 壳体号列名候选
SHELL_ID_CANDIDATES: List[str] = [
    "壳体号",
    "壳体编号",
    "Shell ID",
    "ShellID",
    "SN",
    "序列号",
]

# 料号列名候选
PART_NUMBER_CANDIDATES: List[str] = [
    "料号",
    "物料号",
    "Part Number",
    "PartNumber",
    "PN",
    "型号",
]
