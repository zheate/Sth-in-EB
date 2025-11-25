# 常量定义模块
"""
集中管理 Data Fetch 模块的所有常量定义
"""

from pathlib import Path
from typing import Dict, List, Tuple
import sys

# 添加父目录到路径以导入config
parent_dir = str(Path(__file__).parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from config import DATA_FETCH_DEFAULT_FOLDER, DATA_FETCH_CHIP_DEFAULT_FOLDER

# ============================================================================
# 颜色常量
# ============================================================================

PRIMARY_RED = "red"
PRIMARY_DARK = "#262626"
PRIMARY_BLUE = "blue"

# 5个站别的自定义颜色
STATION_COLORS: Dict[str, str] = {
    "耦合": "#000084",      # 深蓝色
    "Pre": "#870A4C",       # 紫红色
    "低温储存后": "#95A8D2", # 浅蓝色
    "Post": "#C3EAB5",      # 浅绿色
    "封盖": "#C5767B"       # 粉红色
}

# 默认调色板（用于多壳体对比等场景）
DEFAULT_PALETTE: List[str] = [
    "#000084",  # 深蓝色
    "#870A4C",  # 紫红色
    "#95A8D2",  # 浅蓝色
    "#C3EAB5",  # 浅绿色
    "#C5767B",  # 粉红色
    "#FF6347",  # 番茄红
    "#4169E1",  # 皇家蓝
    "#32CD32",  # 酸橙绿
    "#FFD700",  # 金色
    "#9370DB",  # 中紫色
]

# ============================================================================
# 测试站别相关
# ============================================================================

PLOT_ORDER: List[str] = ["耦合测试", "Pre测试", "低温储存后测试", "Post测试", "封盖测试"]
SANITIZED_PLOT_ORDER: List[str] = [name.replace("测试", "") for name in PLOT_ORDER]
SANITIZED_ORDER_LOOKUP: Dict[str, int] = {name: index for index, name in enumerate(SANITIZED_PLOT_ORDER)}

TEST_CATEGORY_OPTIONS: List[str] = PLOT_ORDER.copy()

# ============================================================================
# 数据列名常量
# ============================================================================

OUTPUT_COLUMNS: List[str] = [
    "壳体号",
    "测试类型",
    "电流(A)",
    "功率(W)",
    "电压(V)",
    "电光效率(%)",
    "波长lambda",
    "波长shift",
    "2A波长",
    "冷波长",
]

# 解构列名为独立常量
(
    SHELL_COLUMN,
    TEST_TYPE_COLUMN,
    CURRENT_COLUMN,
    POWER_COLUMN,
    VOLTAGE_COLUMN,
    EFFICIENCY_COLUMN,
    LAMBDA_COLUMN,
    SHIFT_COLUMN,
    WAVELENGTH_2A_COLUMN,
    WAVELENGTH_COLD_COLUMN,
) = OUTPUT_COLUMNS

# ============================================================================
# 路径常量
# ============================================================================

DEFAULT_ROOT: Path = Path(DATA_FETCH_DEFAULT_FOLDER)
CHIP_DEFAULT_ROOT: Path = Path(DATA_FETCH_CHIP_DEFAULT_FOLDER)

# ============================================================================
# 模式常量
# ============================================================================

MODULE_MODE: str = "module"
CHIP_MODE: str = "chip"

EXTRACTION_MODE_OPTIONS: Tuple[Tuple[str, str], ...] = (
    ("模块", MODULE_MODE),
    ("芯片", CHIP_MODE),
)

EXTRACTION_MODE_LOOKUP: Dict[str, str] = dict(EXTRACTION_MODE_OPTIONS)

CHIP_SUPPORTED_MEASUREMENTS: Tuple[str, ...] = ("LVI", "Rth")
CHIP_TEST_CATEGORY: str = "芯片测试"

# ============================================================================
# 测量选项
# ============================================================================

MEASUREMENT_OPTIONS: Dict[str, str] = {
    "LVI": "LVI",
    "Rth": "Rth",
    "lambd": "lambd",
}

# ============================================================================
# 数值常量
# ============================================================================

CURRENT_TOLERANCE: float = 1e-6

# Excel 文件解析相关
SUPPORTED_ENGINES: Dict[str, str] = {
    ".xls": "xlrd",
    ".xlsx": "openpyxl",
}

# 文件名时间戳解析模式
DATETIME_PATTERNS: Tuple[Tuple[str, int], ...] = (
    ("%Y%m%d%H%M%S", 14),
    ("%Y%m%d%H%M", 12),
    ("%Y%m%d", 8),
)

# Excel 文件读取的行偏移量（魔法数字文档化）
LVI_SKIP_ROWS: int = 18  # LVI 文件数据起始行
RTH_SKIP_ROWS: int = 8   # Rth 文件数据起始行

# ============================================================================
# Session State 键名
# ============================================================================

EXTRACTION_STATE_KEY: str = "extraction_state"
TEST_SUBDIR_NAME: str = "测试"
