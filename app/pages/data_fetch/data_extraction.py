# 数据提取模块
"""
包含 LVI、Rth 等测试数据的提取逻辑
"""

from pathlib import Path
from typing import Any, List, Optional, Tuple

import pandas as pd
import streamlit as st

from .constants import (
    CURRENT_COLUMN,
    POWER_COLUMN,
    VOLTAGE_COLUMN,
    EFFICIENCY_COLUMN,
    LAMBDA_COLUMN,
    SHIFT_COLUMN,
    WAVELENGTH_2A_COLUMN,
    WAVELENGTH_COLD_COLUMN,
    OUTPUT_COLUMNS,
    CURRENT_TOLERANCE,
    LVI_SKIP_ROWS,
    RTH_SKIP_ROWS,
)
from .file_utils import (
    read_excel_with_engine,
    build_chip_measurement_index_cached,
    build_module_measurement_index_cached,
)

# 导入数据清洗工具
import sys
from pathlib import Path as PathLib
parent_dir = str(PathLib(__file__).parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from utils.data_cleaning import drop_zero_current


# ============================================================================
# 数据对齐和合并函数
# ============================================================================

def align_output_columns(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    对齐 DataFrame 的列到标准输出格式。
    
    Args:
        df: 输入 DataFrame
        columns: 目标列列表，默认使用 OUTPUT_COLUMNS
        
    Returns:
        对齐后的 DataFrame
    """
    target_cols = columns if columns is not None else OUTPUT_COLUMNS
    aligned = df.copy()
    for column in target_cols:
        if column not in aligned.columns:
            aligned[column] = pd.NA
    return aligned[target_cols]


def first_valid_value(series: pd.Series) -> Any:
    """
    获取 Series 中第一个非空值。
    
    Args:
        series: 输入 Series
        
    Returns:
        第一个非空值，如果全为空则返回 pd.NA
    """
    for value in series:
        if pd.notna(value):
            return value
    return pd.NA


def merge_measurement_rows(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    合并相同壳体、测试类型、电流的测量行。
    
    Args:
        df: 输入 DataFrame
        columns: 目标列列表
        
    Returns:
        合并后的 DataFrame
    """
    from .constants import SHELL_COLUMN, TEST_TYPE_COLUMN
    
    target_cols = columns if columns is not None else OUTPUT_COLUMNS
    if df.empty:
        return df
    
    key_columns = [SHELL_COLUMN, TEST_TYPE_COLUMN, CURRENT_COLUMN]
    normalized = df.copy()
    
    for column in target_cols:
        if column not in normalized.columns:
            normalized[column] = pd.NA
    
    normalized[CURRENT_COLUMN] = pd.to_numeric(normalized[CURRENT_COLUMN], errors="coerce")
    agg_dict = {
        column: first_valid_value
        for column in target_cols
        if column not in key_columns
    }
    
    merged = normalized.groupby(key_columns, sort=False, as_index=False).agg(agg_dict)
    return align_output_columns(merged, columns=target_cols)


# ============================================================================
# LVI 数据提取
# ============================================================================

def _extract_lvi_data_impl(
    file_path: Path,
    current_points: Optional[List[float]]
) -> Tuple[pd.DataFrame, List[float], pd.DataFrame]:
    """LVI 数据提取的核心实现"""
    if not file_path.exists():
        raise FileNotFoundError(f"未找到文件: {file_path}")
    
    df = read_excel_with_engine(
        file_path,
        header=None,
        skiprows=LVI_SKIP_ROWS,
        usecols=[0, 1, 2, 3],
        names=[CURRENT_COLUMN, POWER_COLUMN, VOLTAGE_COLUMN, EFFICIENCY_COLUMN],
    )
    
    df = df.dropna(how="all")
    numeric_df = df.apply(pd.to_numeric, errors="coerce").dropna(subset=[CURRENT_COLUMN])
    numeric_df = drop_zero_current(numeric_df, CURRENT_COLUMN, tol=CURRENT_TOLERANCE)
    
    if numeric_df.empty:
        raise ValueError("LVI 数据为空或无法提取有效的电流点。")
    
    if current_points:
        mask = pd.Series(False, index=numeric_df.index)
        missing_points: List[float] = []
        
        for current in current_points:
            current_mask = (numeric_df[CURRENT_COLUMN] - current).abs() <= CURRENT_TOLERANCE
            if current_mask.any():
                mask |= current_mask
            else:
                missing_points.append(current)
        
        filtered = numeric_df.loc[mask]
        if filtered.empty:
            raise ValueError("未找到匹配的电流点，请重新输入。")
        
        return filtered.reset_index(drop=True), missing_points, numeric_df
    
    # 如果 current_points 为 None (全电流模式)，直接返回所有数据
    if current_points is None:
        return numeric_df.reset_index(drop=True), [], numeric_df
    
    # 默认返回最大电流点
    idx = numeric_df[CURRENT_COLUMN].idxmax()
    return numeric_df.loc[[idx]].reset_index(drop=True), [], numeric_df


@st.cache_data(show_spinner=False)
def _extract_lvi_data_cached(
    file_path_str: str,
    mtime: float,
    current_points: Optional[Tuple[float, ...]]
) -> Tuple[pd.DataFrame, List[float], pd.DataFrame]:
    """缓存的 LVI 数据提取"""
    path = Path(file_path_str)
    points_list: Optional[List[float]] = list(current_points) if current_points is not None else None
    return _extract_lvi_data_impl(path, points_list)


def extract_lvi_data(
    file_path: Path,
    current_points: Optional[List[float]],
    *,
    mtime: Optional[float] = None
) -> Tuple[pd.DataFrame, List[float], pd.DataFrame]:
    """
    提取 LVI 测试数据。
    
    Args:
        file_path: LVI 文件路径
        current_points: 要提取的电流点列表，None 表示全部
        mtime: 文件修改时间（用于缓存）
        
    Returns:
        (提取的数据, 未找到的电流点, 完整数据)
    """
    cached_points = tuple(current_points) if current_points is not None else None
    effective_mtime = file_path.stat().st_mtime if mtime is None else mtime
    return _extract_lvi_data_cached(str(file_path), effective_mtime, cached_points)


# ============================================================================
# Rth 数据提取
# ============================================================================

def _extract_rth_data_impl(
    file_path: Path,
    current_points: Optional[List[float]]
) -> Tuple[pd.DataFrame, List[float], pd.DataFrame]:
    """Rth 数据提取的核心实现"""
    if not file_path.exists():
        raise FileNotFoundError(f"未找到文件: {file_path}")
    
    df = read_excel_with_engine(
        file_path,
        header=None,
        skiprows=RTH_SKIP_ROWS,
        usecols=[0, 1, 2],
        names=[LAMBDA_COLUMN, "热量Q", CURRENT_COLUMN],
    )
    
    df = df.dropna(how="all")
    numeric_df = df.apply(pd.to_numeric, errors="coerce").dropna(
        subset=[LAMBDA_COLUMN, "热量Q", CURRENT_COLUMN]
    )
    numeric_df = drop_zero_current(numeric_df, CURRENT_COLUMN, tol=CURRENT_TOLERANCE)
    
    if numeric_df.empty:
        raise ValueError("Rth 数据为空或无法提取有效的电流点。")
    
    # 确定基准电流（优先使用 2A）
    baseline_rows = numeric_df[(numeric_df[CURRENT_COLUMN] - 2.0).abs() <= CURRENT_TOLERANCE]
    if baseline_rows.empty:
        fallback_idx = numeric_df[CURRENT_COLUMN].idxmin()
        baseline_rows = numeric_df.loc[[fallback_idx]]
        baseline_current = float(baseline_rows.iloc[0][CURRENT_COLUMN])
    else:
        baseline_current = 2.0
    
    lambda_reference = float(baseline_rows.iloc[0][LAMBDA_COLUMN])
    
    # 计算完整数据的波长偏移
    full_numeric = numeric_df.copy()
    full_numeric[SHIFT_COLUMN] = full_numeric[LAMBDA_COLUMN] - lambda_reference
    full_numeric.attrs["lambda_baseline_current"] = baseline_current
    
    # 根据电流点过滤
    if current_points:
        mask = pd.Series(False, index=numeric_df.index)
        missing_points: List[float] = []
        
        for current in current_points:
            current_mask = (numeric_df[CURRENT_COLUMN] - current).abs() <= CURRENT_TOLERANCE
            if current_mask.any():
                mask |= current_mask
            else:
                missing_points.append(current)
        
        filtered = numeric_df.loc[mask]
        if filtered.empty:
            raise ValueError("Rth 文件未找到匹配的电流点，请重新输入。")
    elif current_points is None:
        # 全电流模式
        filtered = numeric_df
        missing_points = []
    else:
        # 默认最大电流
        idx = numeric_df[CURRENT_COLUMN].idxmax()
        filtered = numeric_df.loc[[idx]]
        missing_points = []
    
    result = filtered.copy()
    result[SHIFT_COLUMN] = result[LAMBDA_COLUMN] - lambda_reference
    
    # 提取 2A 波长
    rows_2a = numeric_df[(numeric_df[CURRENT_COLUMN] - 2.0).abs() <= CURRENT_TOLERANCE]
    val_2a = float(rows_2a.iloc[0][LAMBDA_COLUMN]) if not rows_2a.empty else pd.NA
    result[WAVELENGTH_2A_COLUMN] = val_2a
    
    # 提取冷波长（最小电流处的波长）
    if not numeric_df.empty:
        idx_min = numeric_df[CURRENT_COLUMN].idxmin()
        val_cold = float(numeric_df.loc[idx_min][LAMBDA_COLUMN])
        result[WAVELENGTH_COLD_COLUMN] = val_cold
    else:
        result[WAVELENGTH_COLD_COLUMN] = pd.NA
    
    result_reset = result.reset_index(drop=True)
    result_reset.attrs["lambda_baseline_current"] = baseline_current
    
    return result_reset, missing_points, full_numeric


@st.cache_data(show_spinner=False)
def _extract_rth_data_cached(
    file_path_str: str,
    mtime: float,
    current_points: Optional[Tuple[float, ...]]
) -> Tuple[pd.DataFrame, List[float], pd.DataFrame]:
    """缓存的 Rth 数据提取"""
    path = Path(file_path_str)
    return _extract_rth_data_impl(
        path,
        list(current_points) if current_points is not None else None
    )


def extract_rth_data(
    file_path: Path,
    current_points: Optional[List[float]],
    *,
    mtime: Optional[float] = None
) -> Tuple[pd.DataFrame, List[float], pd.DataFrame]:
    """
    提取 Rth 测试数据。
    
    Args:
        file_path: Rth 文件路径
        current_points: 要提取的电流点列表，None 表示全部
        mtime: 文件修改时间（用于缓存）
        
    Returns:
        (提取的数据, 未找到的电流点, 完整数据)
    """
    cached_points = tuple(current_points) if current_points is not None else None
    effective_mtime = file_path.stat().st_mtime if mtime is None else mtime
    return _extract_rth_data_cached(str(file_path), effective_mtime, cached_points)


# ============================================================================
# 通用 Excel 提取
# ============================================================================

def _extract_generic_excel_impl(file_path: Path) -> pd.DataFrame:
    """通用 Excel 提取的核心实现"""
    if not file_path.exists():
        raise FileNotFoundError(f"未找到文件: {file_path}")
    
    df = read_excel_with_engine(file_path)
    
    if isinstance(df, pd.Series):
        df = df.to_frame()
    
    df = df.dropna(how="all")
    
    if df.empty:
        raise ValueError(f"文件 {file_path.name} 未能提取有效数据")
    
    return df


@st.cache_data(show_spinner=False)
def _extract_generic_excel_cached(file_path_str: str, mtime: float) -> pd.DataFrame:
    """缓存的通用 Excel 提取"""
    path = Path(file_path_str)
    return _extract_generic_excel_impl(path)


def extract_generic_excel(
    file_path: Path,
    *,
    mtime: Optional[float] = None
) -> pd.DataFrame:
    """
    提取通用 Excel 文件数据。
    
    Args:
        file_path: 文件路径
        mtime: 文件修改时间（用于缓存）
        
    Returns:
        提取的 DataFrame
    """
    effective_mtime = file_path.stat().st_mtime if mtime is None else mtime
    return _extract_generic_excel_cached(str(file_path), effective_mtime)


# ============================================================================
# 缓存清理
# ============================================================================

def clear_extraction_caches() -> None:
    """清除所有提取相关的缓存"""
    _extract_generic_excel_cached.clear()
    _extract_lvi_data_cached.clear()
    _extract_rth_data_cached.clear()
    build_chip_measurement_index_cached.clear()
    build_module_measurement_index_cached.clear()
