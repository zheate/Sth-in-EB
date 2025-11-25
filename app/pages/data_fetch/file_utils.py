# 文件操作工具模块
"""
包含 Excel 文件读取、路径解析、文件查找等功能
"""

from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import pandas as pd
import streamlit as st

from .constants import (
    DEFAULT_ROOT,
    CHIP_DEFAULT_ROOT,
    SUPPORTED_ENGINES,
    DATETIME_PATTERNS,
    TEST_SUBDIR_NAME,
)


# ============================================================================
# 路径解析函数
# ============================================================================

@lru_cache(maxsize=256)
def _interpret_folder_input_cached(folder_input: str, default_root: str) -> Path:
    """缓存的路径解析实现"""
    folder_input = folder_input.strip()
    if not folder_input:
        raise ValueError("壳体输入不能为空。")
    if any(sep in folder_input for sep in ("\\", "/", ":")):
        return Path(folder_input)
    return Path(default_root).joinpath(*list(folder_input))


def interpret_folder_input(folder_input: str, default_root: Path = DEFAULT_ROOT) -> Path:
    """
    解析壳体号或路径输入。
    
    Args:
        folder_input: 用户输入的壳体号或完整路径
        default_root: 默认根目录
        
    Returns:
        解析后的 Path 对象
        
    Raises:
        ValueError: 输入为空时
    """
    return _interpret_folder_input_cached(folder_input, str(default_root))


def interpret_chip_folder_input(folder_input: str, default_root: Path = CHIP_DEFAULT_ROOT) -> Path:
    """
    解析芯片名称或路径输入。
    
    Args:
        folder_input: 用户输入的芯片名或完整路径
        default_root: 默认根目录
        
    Returns:
        解析后的 Path 对象
        
    Raises:
        ValueError: 输入为空时
        FileNotFoundError: 找不到目录时
    """
    folder_input = folder_input.strip()
    if not folder_input:
        raise ValueError("芯片输入不能为空")
    
    raw_path = Path(folder_input).expanduser()
    candidate_paths: List[Path] = []
    
    if raw_path.is_absolute():
        candidate_paths.append(raw_path)
    else:
        candidate_paths.append(default_root / folder_input)
        candidate_paths.append(raw_path)
    
    # 如果输入不包含路径分隔符，尝试按字符拆分
    if not any(sep in folder_input for sep in ("\\", "/", ":")) and folder_input:
        candidate_paths.append(Path(default_root).joinpath(*list(folder_input)))
    
    seen: set[str] = set()
    for candidate in candidate_paths:
        normalized = str(candidate).lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        if candidate.exists() and candidate.is_dir():
            return candidate
    
    raise FileNotFoundError(f"未找到芯片目录: {folder_input}")


@lru_cache(maxsize=512)
def _resolve_test_folder_cached(base_path_str: str, test_category: str) -> Path:
    """缓存的测试目录解析"""
    base_path = Path(base_path_str)
    candidate = base_path / test_category
    if not candidate.exists():
        raise FileNotFoundError(f"未找到测试目录: {candidate}")
    nested = candidate / TEST_SUBDIR_NAME
    if nested.exists():
        return nested
    return candidate


def resolve_test_folder(base_path: Path, test_category: str) -> Path:
    """
    解析测试目录路径。
    
    Args:
        base_path: 基础路径
        test_category: 测试类型名称
        
    Returns:
        测试目录的 Path 对象
    """
    return _resolve_test_folder_cached(str(base_path), test_category)


# ============================================================================
# Excel 文件读取
# ============================================================================

def read_excel_with_engine(
    file_path: Path,
    sheet_name: Union[int, str] = 0,
    **kwargs: Any
) -> pd.DataFrame:
    """
    智能读取 Excel 文件，自动处理编码和格式问题。
    
    Args:
        file_path: 文件路径
        sheet_name: 工作表名称或索引
        **kwargs: 传递给 pd.read_excel 的其他参数
        
    Returns:
        读取的 DataFrame
        
    Raises:
        ValueError: 无法解析文件时
        ImportError: 缺少必要的引擎时
    """
    last_error: Optional[Exception] = None
    suffix = file_path.suffix.lower()

    # 对于 .xls 文件，尝试不同的编码
    if suffix == ".xls":
        engine = "xlrd"
        for encoding_override in (None, "cp1252", "gbk", "gb18030", "latin1"):
            try:
                if encoding_override:
                    return pd.read_excel(
                        file_path,
                        sheet_name=sheet_name,
                        engine=engine,
                        encoding_override=encoding_override,
                        **kwargs
                    )
                else:
                    return pd.read_excel(
                        file_path,
                        sheet_name=sheet_name,
                        engine=engine,
                        **kwargs
                    )
            except UnicodeDecodeError as exc:
                last_error = exc
                continue
            except ImportError as exc:
                last_error = ImportError(
                    f"读取 {file_path.name} 需要安装 {engine}，请运行 pip install {engine}"
                )
                last_error.__cause__ = exc
                break
            except Exception as exc:
                if encoding_override is None:
                    last_error = exc
                    continue
                else:
                    last_error = exc
                    break

    # 对于其他格式或 .xls 失败后的尝试
    try:
        return pd.read_excel(file_path, sheet_name=sheet_name, **kwargs)
    except ValueError as exc:
        message = str(exc)
        if "Excel file format cannot be determined" not in message and "must specify an engine" not in message:
            last_error = exc
        else:
            engine = SUPPORTED_ENGINES.get(suffix)
            if engine is None:
                last_error = ValueError(f"无法识别的 Excel 后缀: {suffix}")
            else:
                try:
                    return pd.read_excel(
                        file_path,
                        sheet_name=sheet_name,
                        engine=engine,
                        **kwargs
                    )
                except ImportError as engine_exc:
                    last_error = ImportError(
                        f"读取 {file_path.name} 需要安装 {engine}，请运行 pip install {engine}"
                    )
                    last_error.__cause__ = engine_exc
                except Exception as engine_exc:
                    last_error = engine_exc
    except Exception as exc:
        last_error = exc

    # 如果 Excel 读取失败，尝试作为 CSV 读取
    for encoding in ("utf-8", "utf-8-sig", "gbk", "gb18030", "latin1"):
        try:
            return pd.read_csv(
                file_path,
                sep=None,
                engine="python",
                encoding=encoding,
                on_bad_lines='skip',
                **kwargs
            )
        except (UnicodeDecodeError, ValueError) as csv_exc:
            last_error = csv_exc
            continue
        except Exception as csv_exc:
            last_error = csv_exc
            break

    if last_error is not None:
        raise last_error
    raise ValueError(f"无法解析文件 {file_path.name}，请检查格式。")


# ============================================================================
# 文件索引和查找
# ============================================================================

def _extract_timestamp_from_name(path: Path) -> Optional[float]:
    """从文件名中提取时间戳"""
    stem = path.stem
    prefix = stem.split("=", 1)[0]
    digits = "".join(ch for ch in prefix if ch.isdigit())
    
    for fmt, length in DATETIME_PATTERNS:
        if len(digits) >= length:
            snippet = digits[:length]
            try:
                dt = datetime.strptime(snippet, fmt)
                return dt.timestamp()
            except ValueError:
                continue
    return None


def _measurement_file_sort_key(
    item: Tuple[Path, Optional[float], float]
) -> Tuple[int, float, float, str]:
    """文件排序键：优先使用文件名时间戳，其次使用修改时间"""
    path, timestamp, mtime = item
    has_timestamp = 0 if timestamp is not None else 1
    primary_value = timestamp if timestamp is not None else mtime
    return (has_timestamp, primary_value, mtime, path.name)


def _build_measurement_file_index(
    test_folder: Path,
    candidates: Optional[Iterable[Path]] = None,
) -> Dict[str, List[Tuple[Path, Optional[float], float]]]:
    """
    构建测量文件索引。
    
    Args:
        test_folder: 测试目录
        candidates: 候选文件迭代器，默认为目录下所有 .xls* 文件
        
    Returns:
        {token: [(path, timestamp, mtime), ...]} 格式的索引
    """
    index: Dict[str, List[Tuple[Path, Optional[float], float]]] = {}
    iterator = candidates if candidates is not None else test_folder.glob("*.xls*")
    
    for file_path in iterator:
        candidate = Path(file_path)
        if not candidate.is_file():
            continue
        
        suffix = candidate.suffix.lower()
        if suffix not in SUPPORTED_ENGINES:
            continue
        
        stem = candidate.stem
        if "=" not in stem:
            continue
        
        token = stem.rsplit("=", 1)[-1]
        timestamp = _extract_timestamp_from_name(candidate)
        mtime = candidate.stat().st_mtime
        index.setdefault(token, []).append((candidate, timestamp, mtime))
    
    return index


def find_measurement_file(
    test_folder: Path,
    token: str,
    *,
    index: Optional[Dict[str, List[Tuple[Path, Optional[float], float]]]] = None,
) -> Tuple[Path, bool, float]:
    """
    查找匹配的测量文件。
    
    Args:
        test_folder: 测试目录
        token: 文件标识符（如 "LVI", "Rth"）
        index: 预构建的文件索引
        
    Returns:
        (文件路径, 是否有多个匹配, 修改时间)
        
    Raises:
        FileNotFoundError: 找不到匹配文件时
    """
    lookup = index if index is not None else _build_measurement_file_index(test_folder)
    matched = lookup.get(token)
    
    if not matched:
        raise FileNotFoundError(f"未在 {test_folder} 找到匹配 *={token}.xls* 的文件")
    
    selected_path, _, selected_mtime = max(matched, key=_measurement_file_sort_key)
    return selected_path, len(matched) > 1, selected_mtime


# ============================================================================
# 缓存的索引构建函数
# ============================================================================

@st.cache_data(show_spinner=False)
def build_chip_measurement_index_cached(
    chip_root_str: str,
    mtime: float
) -> Dict[str, List[Tuple[Path, Optional[float], float]]]:
    """缓存的芯片测量文件索引构建"""
    chip_root = Path(chip_root_str)
    return _build_measurement_file_index(chip_root, chip_root.rglob("*.xls*"))


@st.cache_data(show_spinner=False)
def build_module_measurement_index_cached(
    folder_str: str,
    mtime: float
) -> Dict[str, List[Tuple[Path, Optional[float], float]]]:
    """缓存的模块测量文件索引构建"""
    test_folder = Path(folder_str)
    return _build_measurement_file_index(test_folder, test_folder.glob("*.xls*"))


def build_chip_measurement_index(
    chip_root: Path
) -> Dict[str, List[Tuple[Path, Optional[float], float]]]:
    """
    构建芯片测量文件索引。
    
    Args:
        chip_root: 芯片根目录
        
    Returns:
        文件索引字典
    """
    if not chip_root.exists():
        raise FileNotFoundError(f"芯片目录不存在: {chip_root}")
    if not chip_root.is_dir():
        raise NotADirectoryError(f"芯片路径不是文件夹: {chip_root}")
    return build_chip_measurement_index_cached(str(chip_root), chip_root.stat().st_mtime)


def find_chip_measurement_file(
    chip_root: Path,
    token: str,
    *,
    index: Optional[Dict[str, List[Tuple[Path, Optional[float], float]]]] = None,
) -> Tuple[Path, bool, float]:
    """
    查找芯片测量文件。
    
    Args:
        chip_root: 芯片根目录
        token: 文件标识符
        index: 预构建的文件索引
        
    Returns:
        (文件路径, 是否有多个匹配, 修改时间)
    """
    lookup_index = index if index is not None else build_chip_measurement_index(chip_root)
    return find_measurement_file(chip_root, token, index=lookup_index)


# ============================================================================
# 辅助函数
# ============================================================================

def ensure_xlsx_suffix(filename: str) -> str:
    """
    确保文件名有 .xlsx 后缀。
    
    Args:
        filename: 原始文件名
        
    Returns:
        带 .xlsx 后缀的文件名
        
    Raises:
        ValueError: 文件名为空时
    """
    filename = filename.strip()
    if not filename:
        raise ValueError("文件名不能为空。")
    
    lower_name = filename.lower()
    if lower_name.endswith(".xlsx"):
        return filename
    
    if lower_name.endswith(".xls"):
        filename = filename[:-4]
    else:
        filename = filename.rstrip(".")
    
    if not filename:
        filename = "combined_subset"
    
    return f"{filename}.xlsx"
