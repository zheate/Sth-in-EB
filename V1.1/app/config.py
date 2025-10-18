"""
应用配置文件
统一管理所有路径和配置项
"""
from pathlib import Path

# ==================== 路径配置 ====================

# 默认数据文件夹路径（用于 Home.py, Progress.py, TestAnalysis.py）
DEFAULT_DATA_FOLDER = r"C:\Users\hao.zhang01\Downloads"

# Data_fetch.py 专用默认路径
DATA_FETCH_DEFAULT_FOLDER = r"D:/"

# 其他可选路径
ALTERNATIVE_PATHS = [
    r"D:/",
    r"./data",
    r"Z:/Ldtd/fcp/",
    r"C:\Users\hao.zhang01\Downloads",
]

# ==================== 文件配置 ====================

# 支持的文件扩展名
SUPPORTED_EXCEL_EXTENSIONS = [".xlsx", ".xls"]
SUPPORTED_CSV_EXTENSIONS = [".csv"]
SUPPORTED_FILE_EXTENSIONS = SUPPORTED_EXCEL_EXTENSIONS + SUPPORTED_CSV_EXTENSIONS

# 光耦WIP报表文件名关键词
WIP_REPORT_KEYWORDS = ["光耦WIP报表", "光耦wip报表"]

# ==================== 应用配置 ====================

# 应用标题
APP_TITLE = "光耦测试数据分析系统"
APP_ICON = "🔬"

# 页面配置
PAGE_LAYOUT = "wide"
SIDEBAR_STATE = "expanded"

# ==================== 辅助函数 ====================

def get_default_folder() -> Path:
    """获取默认数据文件夹路径"""
    return Path(DEFAULT_DATA_FOLDER)

def get_alternative_paths() -> list[Path]:
    """获取备选路径列表"""
    return [Path(p) for p in ALTERNATIVE_PATHS]

def validate_path(path_str: str) -> tuple[bool, str]:
    """
    验证路径是否有效
    
    Args:
        path_str: 路径字符串
        
    Returns:
        (是否有效, 错误信息)
    """
    try:
        path = Path(path_str).expanduser()
        if not path.exists():
            return False, f"路径不存在: {path_str}"
        if not path.is_dir():
            return False, f"路径不是文件夹: {path_str}"
        return True, ""
    except Exception as e:
        return False, f"路径无效: {str(e)}"
