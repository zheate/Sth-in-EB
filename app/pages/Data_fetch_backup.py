# title: 数据提取

import io
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
from functools import lru_cache
import importlib
import warnings
import numpy as np
import altair as alt
import sys
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

# 可选依赖：用于拟合预测功能
stats = None
PolynomialFeatures = None
LinearRegression = None
HuberRegressor = None
RANSACRegressor = None
_PREDICTION_LIBS_LOADED = False

HAS_PREDICTION_LIBS = all(
    importlib.util.find_spec(name) is not None
    for name in ("scipy.stats", "sklearn.preprocessing", "sklearn.linear_model")
)


def _ensure_prediction_libs_loaded() -> bool:
    """Import optional prediction dependencies only when needed."""
    global stats, PolynomialFeatures, LinearRegression, HuberRegressor, RANSACRegressor, _PREDICTION_LIBS_LOADED, HAS_PREDICTION_LIBS

    if not HAS_PREDICTION_LIBS:
        return False

    if _PREDICTION_LIBS_LOADED:
        return True

    try:
        stats = importlib.import_module("scipy.stats")
        PolynomialFeatures = importlib.import_module("sklearn.preprocessing").PolynomialFeatures
        linear_model_module = importlib.import_module("sklearn.linear_model")
        LinearRegression = getattr(linear_model_module, "LinearRegression")
        HuberRegressor = getattr(linear_model_module, "HuberRegressor", None)
        RANSACRegressor = getattr(linear_model_module, "RANSACRegressor", None)
    except ImportError:
        HAS_PREDICTION_LIBS = False
        return False

    _PREDICTION_LIBS_LOADED = True
    return True


# 添加父目录到路径以导入config
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from config import DATA_FETCH_DEFAULT_FOLDER, DATA_FETCH_CHIP_DEFAULT_FOLDER

PRIMARY_RED = "red"
PRIMARY_DARK = "#262626"
PRIMARY_BLUE = "blue"

import pandas as pd
import streamlit as st
from utils.data_cleaning import ensure_numeric, drop_zero_current, clean_current_metric

# Bokeh support removed - using Altair for all visualizations

PLOT_ORDER = ["耦合测试", "Pre测试", "低温储存后测试", "Post测试", "封盖测试"]
SANITIZED_PLOT_ORDER = [name.replace("测试", "") for name in PLOT_ORDER]
SANITIZED_ORDER_LOOKUP = {name: index for index, name in enumerate(SANITIZED_PLOT_ORDER)}

# 5个站别的自定义颜色
STATION_COLORS = {
    "耦合": "#000084",      # 深蓝色
    "Pre": "#870A4C",       # 紫红色
    "低温储存后": "#95A8D2", # 浅蓝色
    "Post": "#C3EAB5",      # 浅绿色
    "封盖": "#C5767B"       # 粉红色
}

DEFAULT_ROOT = Path(DATA_FETCH_DEFAULT_FOLDER)
CHIP_DEFAULT_ROOT = Path(DATA_FETCH_CHIP_DEFAULT_FOLDER)

# 备选路径（已注释）
# DEFAULT_ROOT = Path("D:/")
# DEFAULT_ROOT = Path("Z:/Ldtd/fcp/")

TEST_CATEGORY_OPTIONS = PLOT_ORDER.copy()

MEASUREMENT_OPTIONS = {

    "LVI": "LVI",

    "Rth": "Rth",

    "lambd": "lambd",

}

MODULE_MODE = "module"

CHIP_MODE = "chip"

EXTRACTION_MODE_OPTIONS: Tuple[Tuple[str, str], ...] = (

    ("模块", MODULE_MODE),

    ("芯片", CHIP_MODE),

)

EXTRACTION_MODE_LOOKUP: Dict[str, str] = dict(EXTRACTION_MODE_OPTIONS)

CHIP_SUPPORTED_MEASUREMENTS: Tuple[str, ...] = ("LVI", "Rth")

CHIP_TEST_CATEGORY = "芯片测试"

EXTRACTION_STATE_KEY = "extraction_state"

TEST_SUBDIR_NAME = "测试"

_DATETIME_PATTERNS: Tuple[Tuple[str, int], ...] = (

    ("%Y%m%d%H%M%S", 14),

    ("%Y%m%d%H%M", 12),

    ("%Y%m%d", 8),

)

CURRENT_TOLERANCE = 1e-6

_SUPPORTED_ENGINES: Dict[str, str] = {

    ".xls": "xlrd",

    ".xlsx": "openpyxl",

}

OUTPUT_COLUMNS = [
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


def _exclude_zero_current(df: pd.DataFrame) -> pd.DataFrame:
    if CURRENT_COLUMN not in df.columns or df.empty:
        return df
    return drop_zero_current(df, CURRENT_COLUMN, tol=CURRENT_TOLERANCE)

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


@lru_cache(maxsize=512)

def _interpret_folder_input_cached(folder_input: str, default_root: str) -> Path:
    folder_input = folder_input.strip()
    if not folder_input:
        raise ValueError("壳体输入不能为空。")
    if any(sep in folder_input for sep in ("\\", "/", ":")):
        return Path(folder_input)
    return Path(default_root).joinpath(*list(folder_input))

@lru_cache(maxsize=1024)
def _resolve_test_folder_cached(base_path_str: str, test_category: str) -> Path:
    base_path = Path(base_path_str)
    candidate = base_path / test_category
    if not candidate.exists():
        raise FileNotFoundError(f"未找到测试目录: {candidate}")
    nested = candidate / TEST_SUBDIR_NAME
    if nested.exists():
        return nested
    return candidate

def align_output_columns(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    target_cols = columns if columns is not None else OUTPUT_COLUMNS
    aligned = df.copy()
    for column in target_cols:
        if column not in aligned.columns:
            aligned[column] = pd.NA
    return aligned[target_cols]

def first_valid_value(series: pd.Series):
    for value in series:
        if pd.notna(value):
            return value
    return pd.NA

def merge_measurement_rows(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    target_cols = columns if columns is not None else OUTPUT_COLUMNS
    if df.empty:
        return df
    key_columns = [SHELL_COLUMN, TEST_TYPE_COLUMN, CURRENT_COLUMN]
    normalized = df.copy()
    for column in target_cols:
        if column not in normalized.columns:
            normalized[column] = pd.NA
    normalized[CURRENT_COLUMN] = pd.to_numeric(normalized[CURRENT_COLUMN], errors="coerce")
    agg_dict = {column: first_valid_value for column in target_cols if column not in key_columns}
    merged = normalized.groupby(key_columns, sort=False, as_index=False).agg(agg_dict)
    return align_output_columns(merged, columns=target_cols)

# Removed unused Bokeh plot_dual_y_axes function - using Altair instead

# 效率拟合专用模型函数
def _rational_1_2(x, a, b, c):
    return (a*x) / (1.0 + b*x + c*(x**2))

def _hill(x, Emax, K, n):
    with np.errstate(invalid='ignore', over='ignore'):
        return Emax * (x**n) / (K**n + x**n)

def _hill_droop(x, Emax, K, n, d):
    with np.errstate(invalid='ignore', over='ignore'):
        return (Emax * (x**n) / (K**n + x**n)) / (1.0 + d*x)

def _exp_sat(x, Emax, k):
    return Emax * (1 - np.exp(-k*x))

# 效率拟合候选模型（仅专业模型）
EFFICIENCY_MODELS = {
    "hill_droop": (_hill_droop, [60, 5, 2, 0.02], "Hill-Droop"),
    "hill": (_hill, [60, 5, 2], "Hill"),
    "rational_1_2": (_rational_1_2, [5, 0.1, 0.01], "有理函数"),
    "exp_sat": (_exp_sat, [60, 0.2], "指数饱和"),
}

def plot_multi_shell_prediction(
    series_data: List[Tuple[str, pd.DataFrame]],
    metric_column: str,
    metric_label: str,
    test_type: str,
    poly_degree: int = 2,
    fit_mode: str = "global",
    auto_poly: bool = False,
    max_degree: int = 3,
) -> Optional[alt.Chart]:
    """创建多壳体拟合预测图（Altair实现），包含95%预测带
    
    Args:
        series_data: List of (shell_id, dataframe) tuples
        metric_column: Column name for the metric
        metric_label: Display label for the metric
        test_type: Test type name
        poly_degree: Polynomial degree for fitting (default: 2, 仅用于非效率数据)
    """
    if not _ensure_prediction_libs_loaded():
        return None
    
    if not series_data or len(series_data) < 2:
        return None
    
    # 收集所有数据点
    all_x = []
    all_y = []
    shell_points = []  # 用于绘制各壳体散点
    
    for shell_id, df in series_data:
        numeric = (
            df[[CURRENT_COLUMN, metric_column]]
            .apply(pd.to_numeric, errors="coerce")
            .dropna(subset=[CURRENT_COLUMN, metric_column])
        )
        numeric = _exclude_zero_current(numeric)
        if numeric.empty:
            continue

        x_vals = numeric[CURRENT_COLUMN].to_numpy(dtype=float)
        y_vals = numeric[metric_column].to_numpy(dtype=float)
        all_x.extend(x_vals.tolist())
        all_y.extend(y_vals.tolist())

        # 保存各壳体数据用于散点图
        for x, y in zip(x_vals, y_vals):
            shell_points.append(
                {
                    "current": x,
                    "value": y,
                    "shell": shell_id,
                }
            )
    
    all_x = np.array(all_x, dtype=float)
    all_y = np.array(all_y, dtype=float)

    if all_x.size == 0 or all_y.size == 0 or not shell_points:
        return None
    
    if fit_mode == "global":
        if metric_column == EFFICIENCY_COLUMN:
            from scipy.optimize import curve_fit
            from math import log
            best_model_name = None
            best_popt = None
            best_aic = float('inf')
            best_func = None
            model_results = {}
            for model_name, (func, p0, display_name) in EFFICIENCY_MODELS.items():
                try:
                    popt, _ = curve_fit(func, all_x, all_y, p0=p0, maxfev=10000)
                    yhat = func(all_x, *popt)
                    resid = all_y - yhat
                    rss = float(np.sum(resid**2))
                    n = len(all_y)
                    k = len(popt)
                    mse = rss / n
                    rmse = float(np.sqrt(mse))
                    tss = float(np.sum((all_y - np.mean(all_y))**2))
                    r2 = float(1 - rss/tss) if tss > 0 else 0.0
                    aic = float(n*log(rss/n) + 2*k) if rss > 0 else float('inf')
                    model_results[model_name] = {
                        'popt': popt,
                        'rmse': rmse,
                        'r2': r2,
                        'aic': aic,
                        'display_name': display_name
                    }
                    if aic < best_aic:
                        best_aic = aic
                        best_model_name = model_name
                        best_popt = popt
                        best_func = func
                except Exception:
                    continue
            if best_func is None:
                return None
            x_min, x_max = all_x.min(), all_x.max()
            x_range = x_max - x_min
            x_pred = np.linspace(x_min - x_range * 0.05, x_max + x_range * 0.05, 200)
            y_pred = best_func(x_pred, *best_popt)
            y_fitted = best_func(all_x, *best_popt)
            residuals = all_y - y_fitted
            n = len(all_y)
            p = len(best_popt)
            mse = np.sum(residuals**2) / (n - p)
            std_error = np.sqrt(mse)
            r_squared = model_results[best_model_name]['r2']
            model_display_name = model_results[best_model_name]['display_name']
        else:
            degrees = list(range(1, max_degree + 1)) if auto_poly else [poly_degree]
            best = None
            for deg in degrees:
                poly = PolynomialFeatures(degree=deg)
                X_poly = poly.fit_transform(all_x.reshape(-1, 1))
                model = LinearRegression()
                model.fit(X_poly, all_y)
                y_fitted = model.predict(X_poly)
                resid = all_y - y_fitted
                rss = float(np.sum(resid**2))
                n = len(all_y)
                k = deg + 1
                aic = float(n*np.log(rss/n) + 2*k) if rss > 0 else float('inf')
                r2 = float(model.score(X_poly, all_y))
                if best is None or aic < best[0]:
                    best = (aic, deg, poly, model, y_fitted)
            _, chosen_deg, poly, model, y_fitted = best
            x_min, x_max = all_x.min(), all_x.max()
            x_range = x_max - x_min
            x_pred = np.linspace(x_min - x_range * 0.05, x_max + x_range * 0.05, 200)
            X_pred_poly = poly.transform(x_pred.reshape(-1, 1))
            y_pred = model.predict(X_pred_poly)
            residuals = all_y - y_fitted
            n = len(all_y)
            p = chosen_deg + 1
            mse = np.sum(residuals**2) / (n - p)
            std_error = np.sqrt(mse)
            r_squared = model.score(poly.fit_transform(all_x.reshape(-1, 1)), all_y)
            model_display_name = f"{chosen_deg}次多项式" if not auto_poly else f"自动选择{chosen_deg}次多项式"
    else:
        per_shell_frames = []
        shells = [shell_id for shell_id, _ in series_data]
        colors = ['#000084', '#870A4C', '#95A8D2', '#C3EAB5', '#C5767B',
                  '#FF6347', '#4169E1', '#32CD32', '#FFD700', '#9370DB']
        color_range = [colors[i % len(colors)] for i in range(len(shells))]
        for shell_id, df in series_data:
            numeric = (
                df[[CURRENT_COLUMN, metric_column]]
                .apply(pd.to_numeric, errors="coerce")
                .dropna(subset=[CURRENT_COLUMN, metric_column])
            )
            numeric = _exclude_zero_current(numeric)
            if numeric.empty:
                continue
            sx = numeric[CURRENT_COLUMN].to_numpy(dtype=float)
            sy = numeric[metric_column].to_numpy(dtype=float)
            if metric_column == EFFICIENCY_COLUMN:
                from scipy.optimize import curve_fit
                from math import log
                best_model_name = None
                best_popt = None
                best_aic = float('inf')
                best_func = None
                for model_name, (func, p0, display_name) in EFFICIENCY_MODELS.items():
                    try:
                        popt, _ = curve_fit(func, sx, sy, p0=p0, maxfev=10000)
                        yhat = func(sx, *popt)
                        resid = sy - yhat
                        rss = float(np.sum(resid**2))
                        n_local = len(sy)
                        k_local = len(popt)
                        aic = float(n_local*np.log(rss/n_local) + 2*k_local) if rss > 0 else float('inf')
                        if aic < best_aic:
                            best_aic = aic
                            best_model_name = model_name
                            best_popt = popt
                            best_func = func
                    except Exception:
                        continue
                if best_func is None:
                    continue
                xmin, xmax = sx.min(), sx.max()
                xr = xmax - xmin
                x_pred = np.linspace(xmin - xr*0.05, xmax + xr*0.05, 200)
                y_pred = best_func(x_pred, *best_popt)
                y_fit = best_func(sx, *best_popt)
                resid = sy - y_fit
                n_local = len(sy)
                p_local = len(best_popt)
                mse = np.sum(resid**2) / max(1, (n_local - p_local))
                se = np.sqrt(mse)
            else:
                degrees = list(range(1, max_degree + 1)) if auto_poly else [poly_degree]
                best_local = None
                for deg in degrees:
                    poly = PolynomialFeatures(degree=deg)
                    X_poly = poly.fit_transform(sx.reshape(-1, 1))
                    model = LinearRegression()
                    model.fit(X_poly, sy)
                    y_fit = model.predict(X_poly)
                    resid = sy - y_fit
                    rss = float(np.sum(resid**2))
                    n_local = len(sy)
                    k_local = deg + 1
                    aic = float(n_local*np.log(rss/n_local) + 2*k_local) if rss > 0 else float('inf')
                    if best_local is None or aic < best_local[0]:
                        best_local = (aic, deg, poly, model, y_fit)
                _, chosen_deg, poly, model, y_fit = best_local
                xmin, xmax = sx.min(), sx.max()
                xr = xmax - xmin
                x_pred = np.linspace(xmin - xr*0.05, xmax + xr*0.05, 200)
                X_pred_poly = poly.transform(x_pred.reshape(-1, 1))
                y_pred = model.predict(X_pred_poly)
                resid = sy - y_fit
                n_local = len(sy)
                p_local = chosen_deg + 1
                mse = np.sum(resid**2) / max(1, (n_local - p_local))
                se = np.sqrt(mse)
            t_val_local = stats.t.ppf(0.975, max(1, n_local - p_local))
            pred_std = se * np.sqrt(1 + 1/max(1, n_local))
            y_upper = y_pred + t_val_local * pred_std
            y_lower = y_pred - t_val_local * pred_std
            per_shell_frames.append(pd.DataFrame({
                'shell': shell_id,
                'current': x_pred,
                'upper': y_upper,
                'lower': y_lower,
                'fitted': y_pred
            }))
        if not per_shell_frames:
            return None
        band_df = pd.concat(per_shell_frames, ignore_index=True)
        points_df = pd.DataFrame(shell_points)
        shells = [shell_id for shell_id, _ in series_data]
        colors = ['#000084', '#870A4C', '#95A8D2', '#C3EAB5', '#C5767B',
                  '#FF6347', '#4169E1', '#32CD32', '#FFD700', '#9370DB']
        color_range = [colors[i % len(colors)] for i in range(len(shells))]
        band_chart = alt.Chart(band_df).mark_area(opacity=0.15).encode(
            x=alt.X('current:Q', title='电流(A)'),
            y=alt.Y('lower:Q', title=metric_label, scale=alt.Scale(zero=False)),
            y2='upper:Q',
            color=alt.Color('shell:N', scale=alt.Scale(domain=shells, range=color_range), legend=None)
        )
        fit_line = alt.Chart(band_df).mark_line(size=2).encode(
            x='current:Q',
            y='fitted:Q',
            color=alt.Color('shell:N', scale=alt.Scale(domain=shells, range=color_range), title='壳体'),
        )
        points_chart = alt.Chart(points_df).mark_circle(size=80, opacity=0.7).encode(
            x='current:Q',
            y=alt.Y('value:Q', scale=alt.Scale(zero=False)),
            color=alt.Color('shell:N', scale=alt.Scale(domain=shells, range=color_range), title='壳体'),
            tooltip=[
                alt.Tooltip('shell:N', title='壳体'),
                alt.Tooltip('current:Q', title='电流(A)', format='.3f'),
                alt.Tooltip('value:Q', title=metric_label, format='.3f')
            ]
        )
        chart = (band_chart + fit_line + points_chart).properties(
            width=800,
            height=500,
            title={
                'text': f"{test_type.replace('测试', '')}{metric_label}拟合预测（{len(series_data)}个壳体，分壳体拟合）"
            }
        ).configure_axis(
            labelFontSize=11,
            titleFontSize=12,
            grid=True,
            gridOpacity=0.3
        ).configure_legend(
            labelFontSize=11,
            titleFontSize=12,
            orient='right'
        ).configure_title(
            fontSize=14,
            anchor='start'
        )
        return chart
    
    # 计算95%预测带 (使用t分布)
    t_val = stats.t.ppf(0.975, n - p)
    
    # 对每个预测点计算预测区间
    prediction_std = std_error * np.sqrt(1 + 1/n)
    y_upper = y_pred + t_val * prediction_std
    y_lower = y_pred - t_val * prediction_std
    
    # 准备数据框
    # 1. 预测带数据
    band_df = pd.DataFrame({
        'current': x_pred,
        'upper': y_upper,
        'lower': y_lower,
        'fitted': y_pred
    })
    
    # 2. 散点数据
    points_df = pd.DataFrame(shell_points)
    
    # 获取壳体列表和颜色
    shells = [shell_id for shell_id, _ in series_data]
    colors = ['#000084', '#870A4C', '#95A8D2', '#C3EAB5', '#C5767B',
              '#FF6347', '#4169E1', '#32CD32', '#FFD700', '#9370DB']
    color_range = [colors[i % len(colors)] for i in range(len(shells))]
    
    # 创建95%预测带（半透明区域）
    band_chart = alt.Chart(band_df).mark_area(
        opacity=0.2,
        color='#87CEEB'
    ).encode(
        x=alt.X('current:Q', title='电流(A)'),
        y=alt.Y('lower:Q', title=metric_label, scale=alt.Scale(zero=False)),
        y2='upper:Q'
    )
    
    # 创建预测带上下边界线（虚线）
    upper_line = alt.Chart(band_df).mark_line(
        strokeDash=[5, 5],
        color='#4169E1',
        opacity=0.6,
        size=2
    ).encode(
        x='current:Q',
        y='upper:Q'
    )
    
    lower_line = alt.Chart(band_df).mark_line(
        strokeDash=[5, 5],
        color='#4169E1',
        opacity=0.6,
        size=2
    ).encode(
        x='current:Q',
        y='lower:Q'
    )
    
    # 创建拟合曲线
    fit_line = alt.Chart(band_df).mark_line(
        color='#FF4500',
        size=3
    ).encode(
        x='current:Q',
        y='fitted:Q',
        tooltip=[
            alt.Tooltip('current:Q', title='电流(A)', format='.3f'),
            alt.Tooltip('fitted:Q', title=f'{metric_label}(拟合)', format='.3f')
        ]
    )
    
    # 创建各壳体散点图
    points_chart = alt.Chart(points_df).mark_circle(
        size=80,
        opacity=0.7
    ).encode(
        x='current:Q',
        y=alt.Y('value:Q', scale=alt.Scale(zero=False)),
        color=alt.Color(
            'shell:N',
            title='壳体',
            scale=alt.Scale(domain=shells, range=color_range)
        ),
        tooltip=[
            alt.Tooltip('shell:N', title='壳体'),
            alt.Tooltip('current:Q', title='电流(A)', format='.3f'),
            alt.Tooltip('value:Q', title=metric_label, format='.3f')
        ]
    )
    
    # 组合所有图层
    chart = (band_chart + upper_line + lower_line + fit_line + points_chart).properties(
        width=800,
        height=500,
        title={
            "text": f"{test_type.replace('测试', '')}{metric_label}拟合预测（{len(series_data)}个壳体，{model_display_name}）",
            "subtitle": f"R² = {r_squared:.4f}, RMSE = {std_error:.4f}",
            "subtitleColor": "#666666"
        }
    ).configure_axis(
        labelFontSize=11,
        titleFontSize=12,
        grid=True,
        gridOpacity=0.3
    ).configure_legend(
        labelFontSize=11,
        titleFontSize=12,
        orient='right'
    ).configure_title(
        fontSize=14,
        anchor='start'
    )
    
    return chart


def _prepare_metric_series(df: pd.DataFrame, metric_column: str) -> pd.DataFrame:
    if CURRENT_COLUMN not in df.columns or metric_column not in df.columns:
        return pd.DataFrame(columns=[CURRENT_COLUMN, metric_column])
    numeric = ensure_numeric(df[[CURRENT_COLUMN, metric_column]], [CURRENT_COLUMN, metric_column], strict=False)
    numeric = numeric.dropna(subset=[CURRENT_COLUMN, metric_column])
    numeric = _exclude_zero_current(numeric)
    if numeric.empty:
        return numeric
    aggregated = numeric.groupby(CURRENT_COLUMN, as_index=False).mean()
    return aggregated.sort_values(CURRENT_COLUMN)


def _predict_metric_at_current(
    df: pd.DataFrame,
    metric_column: str,
    target_current: float,
    degree: int,
) -> Optional[Dict[str, float]]:
    prepared = _prepare_metric_series(df, metric_column)
    if prepared.empty:
        return None
    x = prepared[CURRENT_COLUMN].to_numpy(dtype=float)
    y = prepared[metric_column].to_numpy(dtype=float)
    min_current = float(np.min(x))
    max_current = float(np.max(x))
    in_range = min_current <= target_current <= max_current
    if len(x) == 1:
        value = float(y[0])
        return {
            "value": value,
            "in_range": in_range,
            "std_error": float("nan"),
            "ci_lower": value,
            "ci_upper": value,
            "min_current": min_current,
            "max_current": max_current,
        }
    effective_degree = max(1, min(degree, len(x) - 1))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", np.RankWarning)
            coeffs, residuals, _, _, _ = np.polyfit(x, y, effective_degree, full=True)
        value = float(np.polyval(coeffs, target_current))
        n = len(x)
        p = effective_degree + 1
        rss = float(residuals[0]) if residuals.size > 0 else 0.0
        mse = rss / max(1, (n - p))
        std_error = float(np.sqrt(mse))
        t_val = stats.t.ppf(0.975, max(1, n - p)) if _ensure_prediction_libs_loaded() else 1.96
        prediction_std = std_error * np.sqrt(1 + 1/max(1, n))
        ci_lower = float(value - t_val * prediction_std)
        ci_upper = float(value + t_val * prediction_std)
    except (np.linalg.LinAlgError, ValueError):
        value = float(np.interp(target_current, x, y))
        std_error = float("nan")
        ci_lower = value
        ci_upper = value
    return {
        "value": value,
        "in_range": in_range,
        "std_error": std_error,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "min_current": min_current,
        "max_current": max_current,
    }


def compute_power_predictions(
    series_data: List[Tuple[str, pd.DataFrame]],
    target_current: float,
    degree: int,
) -> List[Dict[str, object]]:
    predictions: List[Dict[str, object]] = []
    for shell_id, df in series_data:
        current_values = (
            pd.to_numeric(df.get(CURRENT_COLUMN, pd.Series(dtype=float)), errors="coerce")
            .dropna()
        )
        current_values = current_values[current_values.abs() > CURRENT_TOLERANCE]
        if current_values.empty:
            continue
        v_pred = _predict_metric_at_current(df, VOLTAGE_COLUMN, target_current, degree)
        e_pred = _predict_metric_at_current(df, EFFICIENCY_COLUMN, target_current, degree)
        if v_pred is None or e_pred is None:
            continue
        min_current = float(current_values.min())
        max_current = float(current_values.max())
        in_range = min_current <= target_current <= max_current
        efficiency_ratio = float(e_pred["value"])
        if efficiency_ratio > 1.5:
            efficiency_ratio = efficiency_ratio / 100.0
        efficiency_ratio = max(efficiency_ratio, 0.0)
        predicted_power = target_current * float(v_pred["value"]) * efficiency_ratio
        v_lower = float(v_pred["ci_lower"]) if not np.isnan(v_pred["std_error"]) else float(v_pred["value"])
        v_upper = float(v_pred["ci_upper"]) if not np.isnan(v_pred["std_error"]) else float(v_pred["value"])
        e_lower_ratio = float(e_pred["ci_lower"]) if not np.isnan(e_pred["std_error"]) else float(e_pred["value"])
        e_upper_ratio = float(e_pred["ci_upper"]) if not np.isnan(e_pred["std_error"]) else float(e_pred["value"])
        if e_lower_ratio > 1.5:
            e_lower_ratio = e_lower_ratio / 100.0
        if e_upper_ratio > 1.5:
            e_upper_ratio = e_upper_ratio / 100.0
        p_lower = target_current * v_lower * max(0.0, e_lower_ratio)
        p_upper = target_current * v_upper * max(0.0, e_upper_ratio)
        predictions.append(
            {
                "壳体": shell_id,
                "预测电压(V)": round(float(v_pred["value"]), 3),
                "预测效率(%)": round(efficiency_ratio * 100.0, 3),
                "预测功率(W)": round(predicted_power, 3),
                "预测区间(W)": f"{p_lower:.3f}~{p_upper:.3f}",
                "数据范围(A)": f"{min_current:.3f}~{max_current:.3f}",
                "目标电流在范围内": "是" if in_range else "否",
            }
        )
    return predictions

# Removed unused Bokeh plot_multi_shell_dual_y function - using Altair instead


def build_multi_shell_chart(
    series_data: List[Tuple[str, pd.DataFrame]],
    metric_column: str,
    metric_label: str,
    test_type: str,
) -> Optional[alt.Chart]:
    """构建多壳体单Y轴对比图（Altair 实现）。"""
    if not series_data:
        return None

    chart_frames: List[pd.DataFrame] = []
    for shell_id, numeric in series_data:
        if numeric is None or numeric.empty:
            continue
        numeric_clean = numeric.dropna(subset=[CURRENT_COLUMN, metric_column]).copy()
        numeric_clean = _exclude_zero_current(numeric_clean)
        if numeric_clean.empty:
            continue
        numeric_clean[CURRENT_COLUMN] = pd.to_numeric(
            numeric_clean[CURRENT_COLUMN],
            errors="coerce",
        )
        numeric_clean[metric_column] = pd.to_numeric(
            numeric_clean[metric_column],
            errors="coerce",
        )
        numeric_clean = numeric_clean.dropna(subset=[CURRENT_COLUMN, metric_column])
        if numeric_clean.empty:
            continue
        numeric_clean = numeric_clean.assign(series=shell_id)
        renamed = numeric_clean.rename(
            columns={CURRENT_COLUMN: "current", metric_column: "value"}
        )
        chart_frames.append(renamed[["series", "current", "value"]])

    if not chart_frames:
        return None

    chart_data = pd.concat(chart_frames, ignore_index=True)
    present_labels = list(dict.fromkeys(chart_data["series"].tolist()))

    default_colors = [
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
    color_range = [
        default_colors[idx % len(default_colors)]
        for idx, _ in enumerate(present_labels)
    ]

    highlight = alt.selection_point(
        on="mouseover",
        fields=["series"],
        nearest=True,
        empty=True,
    )

    base = (
        alt.Chart(chart_data)
        .encode(
            x=alt.X("current:Q", title="电流(A)"),
            y=alt.Y("value:Q", title=metric_label, scale=alt.Scale(zero=False)),
            color=alt.Color(
                "series:N",
                title="Shell",
                scale=alt.Scale(domain=present_labels, range=color_range),
            ),
            tooltip=[
                alt.Tooltip("series:N", title="Shell"),
                alt.Tooltip("current:Q", title="电流(A)", format=".3f"),
                alt.Tooltip("value:Q", title=metric_label, format=".3f"),
            ],
        )
    )

    points = (
        base.add_params(highlight)
        .mark_circle()
        .encode(
            size=alt.condition(highlight, alt.value(120), alt.value(70)),
            opacity=alt.condition(highlight, alt.value(1.0), alt.value(0.3)),
        )
    )
    lines = base.mark_line().encode(
        size=alt.condition(highlight, alt.value(4), alt.value(2)),
        opacity=alt.condition(highlight, alt.value(1.0), alt.value(0.3)),
    )

    chart = (points + lines).properties(
        title=f"{test_type.replace('测试', '')}{metric_label}对比",
        width=600,
        height=420,
    )

    return chart


def _clean_metric_dataframe(df: pd.DataFrame, metric_column: str) -> pd.DataFrame:
    return clean_current_metric(df, CURRENT_COLUMN, metric_column)


def build_multi_shell_diff_band_charts(
    series_data: List[Tuple[str, pd.DataFrame]],
    metric_column: str,
    metric_label: str,
    test_type: str,
) -> Tuple[Optional[str], List[Tuple[str, alt.Chart]]]:
    """构建多壳体Delta Band图表列表。"""
    if len(series_data) < 2:
        baseline_shell = series_data[0][0] if series_data else None
        return baseline_shell, []

    shell_ids = [shell_id for shell_id, _ in series_data]
    palette = [
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
    color_lookup = {
        shell_id: palette[idx % len(palette)]
        for idx, shell_id in enumerate(shell_ids)
    }

    baseline_shell, baseline_df = series_data[0]
    baseline_clean = _clean_metric_dataframe(baseline_df, metric_column)
    if baseline_clean.empty:
        return baseline_shell, []

    charts: List[Tuple[str, alt.Chart]] = []

    for shell_id, df in series_data[1:]:
        other_clean = _clean_metric_dataframe(df, metric_column)
        if other_clean.empty:
            continue

        currents = sorted(set(baseline_clean["current"]).union(other_clean["current"]))
        if not currents:
            continue

        current_grid = pd.DataFrame({"current": currents})

        base_interp = current_grid.merge(baseline_clean, on="current", how="left")
        comp_interp = current_grid.merge(other_clean, on="current", how="left")
        for frame in (base_interp, comp_interp):
            frame["value"] = frame["value"].interpolate(method="linear")
            frame["value"] = frame["value"].ffill().bfill()

        merged = current_grid.copy()
        merged["value_baseline"] = base_interp["value"]
        merged["value_comparison"] = comp_interp["value"]
        merged = merged.dropna(subset=["value_baseline", "value_comparison"])
        if merged.empty:
            continue

        merged["delta"] = merged["value_comparison"] - merged["value_baseline"]
        merged["delta_abs"] = merged["delta"].abs()

        positive_df = merged[merged["delta"] >= 0].copy()
        negative_df = merged[merged["delta"] < 0].copy()
        band_df = pd.concat(
            [
                positive_df.assign(sign="delta>=0"),
                negative_df.assign(sign="delta<0"),
            ],
            ignore_index=True,
        )

        base_axis = alt.Axis(
            title=metric_label, 
            tickCount=10,
            domain=True,
            domainWidth=2,
            domainColor='black',
            ticks=True,
            tickWidth=2,
            tickSize=6,
            tickColor='black',
            tickMinStep=1
        )
        x_axis = alt.Axis(
            title="电流(A)", 
            tickCount=10,
            domain=True,
            domainWidth=2,
            domainColor='black',
            ticks=True,
            tickWidth=2,
            tickSize=6,
            tickColor='black',
            tickMinStep=1
        )

        if band_df.empty:
            band_chart = (
                alt.Chart(merged.iloc[0:0])
                .mark_area(opacity=0.0)
                .encode(
                    x=alt.X("current:Q", axis=x_axis),
                    y=alt.Y("value_baseline:Q", axis=base_axis),
                    y2="value_comparison:Q",
                )
            )
        else:
            band_chart = (
                alt.Chart(band_df)
                .mark_area(opacity=0.35)
                .encode(
                    x=alt.X("current:Q", axis=x_axis),
                    y=alt.Y("value_baseline:Q", axis=base_axis),
                    y2="value_comparison:Q",
                    color=alt.Color(
                        "sign:N",
                        scale=alt.Scale(
                            domain=["delta>=0", "delta<0"],
                            range=["#F9D7A5", "#B8D7FF"],
                        ),
                        legend=alt.Legend(title="Lead Direction"),
                    ),
                )
            )

        line_data = pd.concat(
            [
                merged[["current", "value_baseline", "delta_abs"]]
                .assign(series=baseline_shell)
                .rename(columns={"value_baseline": "value"}),
                merged[["current", "value_comparison", "delta_abs"]]
                .assign(series=shell_id)
                .rename(columns={"value_comparison": "value"}),
            ],
            ignore_index=True,
        )

        line_chart = (
            alt.Chart(line_data)
            .mark_line()
            .encode(
                x=alt.X("current:Q", axis=x_axis),
                y=alt.Y("value:Q", axis=base_axis),
                color=alt.Color(
                    "series:N",
                    title="Shell",
                    scale=alt.Scale(
                        domain=[baseline_shell, shell_id],
                        range=[color_lookup[baseline_shell], color_lookup[shell_id]],
                    ),
                ),
                strokeWidth=alt.value(3),
                tooltip=[
                    alt.Tooltip("series:N", title="Shell"),
                    alt.Tooltip("current:Q", title="电流(A)", format=".3f"),
                    alt.Tooltip("value:Q", title=metric_label, format=".3f"),
                    alt.Tooltip("delta_abs:Q", title="|Δ|", format=".3f"),
                ],
            )
        )

        point_chart = (
            alt.Chart(line_data)
            .mark_circle(size=110)
            .encode(
                x=alt.X("current:Q", axis=x_axis),
                y=alt.Y("value:Q", axis=base_axis),
                color=alt.Color(
                    "series:N",
                    scale=alt.Scale(
                        domain=[baseline_shell, shell_id],
                        range=[color_lookup[baseline_shell], color_lookup[shell_id]],
                    ),
                    legend=None,
                ),
            )
        )

        end_annotations = pd.DataFrame(
            [
                {
                    "series": baseline_shell,
                    "current": merged.iloc[-1]["current"],
                    "value": merged.iloc[-1]["value_baseline"],
                    "label": f"{baseline_shell}: {merged.iloc[-1]['value_baseline']:.2f}",
                },
                {
                    "series": shell_id,
                    "current": merged.iloc[-1]["current"],
                    "value": merged.iloc[-1]["value_comparison"],
                    "label": f"{shell_id}: {merged.iloc[-1]['value_comparison']:.2f}",
                },
            ]
        )
        end_text = (
            alt.Chart(end_annotations)
            .mark_text(align="left", dx=6, dy=-4, fontSize=12)
            .encode(
                x="current:Q",
                y="value:Q",
                text="label:N",
                color=alt.Color(
                    "series:N",
                    scale=alt.Scale(
                        domain=[baseline_shell, shell_id],
                        range=[color_lookup[baseline_shell], color_lookup[shell_id]],
                    ),
                    legend=None,
                ),
            )
        )

        max_idx = merged["delta_abs"].idxmax()
        max_annotations = merged.loc[[max_idx]].copy()
        max_annotations["label_str"] = max_annotations["delta"].map(lambda v: f"Delta={v:.2f}")
        max_text = (
            alt.Chart(max_annotations)
            .mark_text(
                align="center", dy=-10, fontSize=12, fontStyle="italic"
            )
            .encode(
                x="current:Q",
                y="value_comparison:Q",
                text="label_str:N",
            )
        )
        max_point = alt.Chart(max_annotations).mark_point(size=120, shape="triangle-up").encode(
            x="current:Q",
            y="value_comparison:Q",
            color=alt.value(color_lookup[shell_id]),
        )

        combined = (
            alt.layer(
                band_chart,
                line_chart,
                point_chart,
                end_text,
                max_point,
                max_text,
            )
            .resolve_scale(color="independent")
            .properties(
                width=600,
                height=360,
                title=f"{baseline_shell} vs {shell_id} {metric_label}Delta Band",
            )
            .configure_axis(
                grid=True,
                gridOpacity=0.3,
                tickCount=10,
                labelFontSize=11,
                titleFontSize=12,
            )
            .configure_legend(
                labelFontSize=11,
                titleFontSize=12,
            )
        )

        charts.append((shell_id, combined))

    return baseline_shell, charts


def build_station_metric_chart(
    dataframe: pd.DataFrame,
    metric_column: str,
    metric_label: str,
) -> Optional[alt.Chart]:
    """构建多站别趋势图，按站别区分颜色并支持悬停高亮。"""
    if dataframe is None or dataframe.empty:
        return None

    required_columns = {TEST_TYPE_COLUMN, CURRENT_COLUMN, metric_column}
    if not required_columns.issubset(dataframe.columns):
        return None

    chart_data = dataframe[[TEST_TYPE_COLUMN, CURRENT_COLUMN, metric_column]].dropna()
    if chart_data.empty:
        return None

    chart_data = chart_data.rename(
        columns={
            TEST_TYPE_COLUMN: "站别",
            CURRENT_COLUMN: "current",
            metric_column: "value",
        }
    )

    chart_data["current"] = pd.to_numeric(chart_data["current"], errors="coerce")
    chart_data["value"] = pd.to_numeric(chart_data["value"], errors="coerce")
    chart_data = chart_data.dropna(subset=["current", "value"])
    if chart_data.empty:
        return None

    present_labels = list(dict.fromkeys(chart_data["站别"].tolist()))
    color_domain = [
        label for label in SANITIZED_PLOT_ORDER if label in present_labels
    ]
    extras = [label for label in present_labels if label not in color_domain]
    if color_domain:
        color_domain = color_domain + extras
    else:
        color_domain = present_labels
    base_palette = [
        "#000084",  # 深蓝色
        "#870A4C",  # 紫红色
        "#95A8D2",  # 浅蓝色
        "#C3EAB5",  # 浅绿色
        "#C5767B",  # 粉红色
    ]
    fallback_index = 0
    color_range: List[str] = []
    for label in color_domain:
        color = STATION_COLORS.get(label)
        if color is None:
            color = base_palette[fallback_index % len(base_palette)]
            fallback_index += 1
        color_range.append(color)

    highlight = alt.selection_point(
        on="mouseover",
        fields=["站别"],
        nearest=True,
        empty=False,
    )

    base = (
        alt.Chart(chart_data)
        .encode(
            x=alt.X("current:Q", title="电流(A)"),
            y=alt.Y("value:Q", title=metric_label, scale=alt.Scale(zero=False)),
            color=alt.Color(
                "站别:N",
                title="Station",
                scale=alt.Scale(domain=color_domain, range=color_range),
            ),
            tooltip=[
                alt.Tooltip("站别:N", title="Station"),
                alt.Tooltip("current:Q", title="电流(A)", format=".3f"),
                alt.Tooltip("value:Q", title=metric_label, format=".3f"),
            ],
        )
    )

    points = (
        base.mark_point(filled=True)
        .encode(
            size=alt.condition(highlight, alt.value(120), alt.value(70)),
            opacity=alt.value(0.9),
        )
        .add_params(highlight)
    )
    lines = base.mark_line().encode(
        size=alt.condition(highlight, alt.value(4), alt.value(2))
    )

    chart = (points + lines).properties(width=700, height=420)

    # Remove default chart background so multi-station plots no longer show a filled area
    chart = chart.configure_view(stroke="transparent", fill="transparent").configure(
        background="transparent"
    )

    return chart


def build_single_shell_dual_metric_chart(
    plot_df: pd.DataFrame,
    selected_df: Optional[pd.DataFrame],
    shell_id: str,
    test_type: str,
) -> Optional[alt.Chart]:
    """构建单壳体功率/效率对比的双轴点线图。"""
    if plot_df is None or plot_df.empty:
        return None

    numeric = plot_df[[CURRENT_COLUMN, POWER_COLUMN, EFFICIENCY_COLUMN]].copy()
    numeric[CURRENT_COLUMN] = pd.to_numeric(numeric[CURRENT_COLUMN], errors="coerce")
    numeric[POWER_COLUMN] = pd.to_numeric(numeric[POWER_COLUMN], errors="coerce")
    numeric[EFFICIENCY_COLUMN] = (
        pd.to_numeric(numeric[EFFICIENCY_COLUMN], errors="coerce") * 100.0
    )
    numeric = numeric.dropna(subset=[CURRENT_COLUMN, POWER_COLUMN, EFFICIENCY_COLUMN])
    if numeric.empty:
        return None

    numeric = numeric.rename(
        columns={
            CURRENT_COLUMN: "current",
            POWER_COLUMN: "power",
            EFFICIENCY_COLUMN: "efficiency",
        }
    )
    numeric = numeric.sort_values("current")
    numeric["is_selected"] = False

    if selected_df is not None and not selected_df.empty:
        selected = selected_df[[CURRENT_COLUMN, POWER_COLUMN, EFFICIENCY_COLUMN]].copy()
        selected[CURRENT_COLUMN] = pd.to_numeric(
            selected[CURRENT_COLUMN], errors="coerce"
        )
        selected[POWER_COLUMN] = pd.to_numeric(selected[POWER_COLUMN], errors="coerce")
        selected[EFFICIENCY_COLUMN] = (
            pd.to_numeric(selected[EFFICIENCY_COLUMN], errors="coerce") * 100.0
        )
        selected = selected.dropna(subset=[CURRENT_COLUMN, POWER_COLUMN])
        for _, row in selected.iterrows():
            mask = np.isclose(numeric["current"].to_numpy(), row[CURRENT_COLUMN]) & np.isclose(
                numeric["power"].to_numpy(), row[POWER_COLUMN]
            )
            if mask.any():
                numeric.loc[mask, "is_selected"] = True

    hover = alt.selection_point(
        on="mouseover",
        fields=["current"],
        nearest=True,
        empty=False,
    )

    power_base = alt.Chart(numeric).encode(
        x=alt.X("current:Q", title="电流(A)"),
        y=alt.Y(
            "power:Q",
            title="功率(W)",
            axis=alt.Axis(titleColor=PRIMARY_BLUE, labelColor=PRIMARY_BLUE, orient="left"),
            scale=alt.Scale(zero=False),
        ),
        color=alt.value(PRIMARY_BLUE),
    )

    efficiency_base = alt.Chart(numeric).encode(
        x=alt.X("current:Q", title="电流(A)"),
        y=alt.Y(
            "efficiency:Q",
            title="电光效率(%)",
            axis=alt.Axis(
                titleColor=PRIMARY_RED,
                labelColor=PRIMARY_RED,
                orient="right",
            ),
            scale=alt.Scale(zero=False),
        ),
        color=alt.value(PRIMARY_RED),
    )

    power_line = power_base.mark_line(size=3)
    power_points = power_base.mark_circle(size=90).encode(
        size=alt.condition("datum.is_selected", alt.value(140), alt.value(80)),
        opacity=alt.condition(hover, alt.value(1.0), alt.value(0.85)),
        tooltip=[
            alt.Tooltip("current:Q", title="电流(A)", format=".3f"),
            alt.Tooltip("power:Q", title="功率(W)", format=".3f"),
            alt.Tooltip("efficiency:Q", title="电光效率(%)", format=".3f"),
        ],
    )

    efficiency_line = efficiency_base.mark_line(size=3)
    efficiency_points = efficiency_base.mark_circle(size=90).encode(
        size=alt.condition("datum.is_selected", alt.value(140), alt.value(80)),
        opacity=alt.condition(hover, alt.value(1.0), alt.value(0.85)),
        tooltip=[
            alt.Tooltip("current:Q", title="电流(A)", format=".3f"),
            alt.Tooltip("efficiency:Q", title="电光效率(%)", format=".3f"),
            alt.Tooltip("power:Q", title="功率(W)", format=".3f"),
        ],
    )

    chart = (
        power_line
        + power_points
        + efficiency_line
        + efficiency_points
    ).resolve_scale(y="independent").add_params(hover)

    return chart.properties(
        title=f"{shell_id} {test_type.replace('测试', '')}",
        width=700,
        height=420,
    )


def interpret_folder_input(folder_input: str, default_root: Path = DEFAULT_ROOT) -> Path:
    return _interpret_folder_input_cached(folder_input, str(default_root))


def interpret_chip_folder_input(folder_input: str, default_root: Path = CHIP_DEFAULT_ROOT) -> Path:

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

    raise FileNotFoundError(f"未找到芯片目�? {folder_input}")

@st.cache_data(show_spinner=False)
def build_chip_measurement_index_cached(chip_root_str: str, mtime: float) -> Dict[str, List[Tuple[Path, Optional[float], float]]]:
    chip_root = Path(chip_root_str)
    return _build_measurement_file_index(chip_root, chip_root.rglob("*.xls*"))

@st.cache_data(show_spinner=False)
def build_module_measurement_index_cached(folder_str: str, mtime: float) -> Dict[str, List[Tuple[Path, Optional[float], float]]]:
    test_folder = Path(folder_str)
    return _build_measurement_file_index(test_folder, test_folder.glob("*.xls*"))

def build_chip_measurement_index(chip_root: Path) -> Dict[str, List[Tuple[Path, Optional[float], float]]]:
    if not chip_root.exists():
        raise FileNotFoundError(f"芯片目录不存在: {chip_root}")
    if not chip_root.is_dir():
        raise NotADirectoryError(f"芯片路�?不是文件夹: {chip_root}")
    return build_chip_measurement_index_cached(str(chip_root), chip_root.stat().st_mtime)

def find_chip_measurement_file(

    chip_root: Path,

    token: str,

    *,

    index: Optional[Dict[str, List[Tuple[Path, Optional[float], float]]]] = None,

) -> Tuple[Path, bool, float]:

    lookup_index = index if index is not None else build_chip_measurement_index(chip_root)

    return find_measurement_file(chip_root, token, index=lookup_index)

def ensure_xlsx_suffix(filename: str) -> str:

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

def parse_folder_entries(raw_folders: str) -> List[str]:

    entries: List[str] = []

    for line in raw_folders.replace("，", "\n").splitlines():

        entry = line.strip()

        if entry:

            entries.append(entry)

    return entries

def parse_current_points(raw_points: str) -> Optional[List[float]]:
    """解析电流点输入。输入 'a' 或 'A' 时返回 None 以提取全部电流点；否则返回电流点列表。"""
    text = raw_points.strip()
    if text.lower() == "a":
        return None

    currents: List[float] = []
    cleaned = text.replace("，", ",").replace("～", "~")

    for line in cleaned.splitlines():
        for piece in line.split(","):
            piece = piece.strip()
            if not piece:
                continue

            normalized = piece.replace("～", "~")

            # whitespace-separated individual values, e.g. "2 5 7"
            if "~" not in normalized and "-" not in normalized[1:]:
                space_tokens = [token for token in normalized.split() if token]
                if len(space_tokens) > 1:
                    try:
                        currents.extend(float(token) for token in space_tokens)
                    except ValueError as exc:
                        raise ValueError(f"无法解析电流值: {piece}") from exc
                    continue

            range_tokens: Optional[List[str]] = None

            if "~" in normalized:
                range_tokens = normalized.split("~", 1)
            else:
                hyphen_index = normalized.find("-", 1)
                if hyphen_index != -1:
                    range_tokens = [normalized[:hyphen_index], normalized[hyphen_index + 1 :]]

            if range_tokens:
                start_str, end_str = [token.strip() for token in range_tokens]
                try:
                    start = float(start_str)
                    end = float(end_str)
                except ValueError as exc:
                    raise ValueError(f"无法解析电流范围: {piece}") from exc

                if start.is_integer() and end.is_integer():
                    start_int = int(start)
                    end_int = int(end)
                    step = 1 if end_int >= start_int else -1
                    for value in range(start_int, end_int + step, step):
                        currents.append(float(value))
                else:
                    currents.extend([start, end])
                continue

            try:
                currents.append(float(normalized))
            except ValueError as exc:
                raise ValueError(f"无法解析电流值: {piece}") from exc

    return currents

def resolve_test_folder(base_path: Path, test_category: str) -> Path:

    return _resolve_test_folder_cached(str(base_path), test_category)

def _extract_timestamp_from_name(path: Path) -> Optional[float]:

    stem = path.stem

    prefix = stem.split("=", 1)[0]

    digits = "".join(ch for ch in prefix if ch.isdigit())

    for fmt, length in _DATETIME_PATTERNS:

        if len(digits) >= length:

            snippet = digits[:length]

            try:

                dt = datetime.strptime(snippet, fmt)

                return dt.timestamp()

            except ValueError:

                continue

    return None

def _measurement_file_sort_key(item: Tuple[Path, Optional[float], float]) -> Tuple[int, float, float, str]:

    path, timestamp, mtime = item

    has_timestamp = 0 if timestamp is not None else 1

    primary_value = timestamp if timestamp is not None else mtime

    return (has_timestamp, primary_value, mtime, path.name)

def _build_measurement_file_index(

    test_folder: Path,

    candidates: Optional[Iterable[Path]] = None,

) -> Dict[str, List[Tuple[Path, Optional[float], float]]]:

    index: Dict[str, List[Tuple[Path, Optional[float], float]]] = {}

    iterator = candidates if candidates is not None else test_folder.glob("*.xls*")

    for file_path in iterator:

        candidate = Path(file_path)

        if not candidate.is_file():

            continue

        suffix = candidate.suffix.lower()

        if suffix not in _SUPPORTED_ENGINES:

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
    index: Optional[Dict[str, List[Tuple[Path, Optional[float], float]]]] = None,) -> Tuple[Path, bool, float]:

    lookup = index if index is not None else _build_measurement_file_index(test_folder)

    matched = lookup.get(token)

    if not matched:

        raise FileNotFoundError(f"未在 {test_folder} 找到匹配 *={token}.xls* 的文件")

    selected_path, _, selected_mtime = max(matched, key=_measurement_file_sort_key)

    return selected_path, len(matched) > 1, selected_mtime

def read_excel_with_engine(file_path: Path, sheet_name: Union[int, str] = 0, **kwargs: Any) -> pd.DataFrame:

    last_error: Optional[Exception] = None
    suffix = file_path.suffix.lower()

    # For .xls files, try with xlrd and different encoding_override options
    if suffix == ".xls":
        engine = "xlrd"
        for encoding_override in (None, "cp1252", "gbk", "gb18030", "latin1"):
            try:
                if encoding_override:
                    return pd.read_excel(file_path, sheet_name=sheet_name, engine=engine, encoding_override=encoding_override, **kwargs)
                else:
                    return pd.read_excel(file_path, sheet_name=sheet_name, engine=engine, **kwargs)
            except UnicodeDecodeError as exc:
                last_error = exc
                continue
            except ImportError as exc:
                last_error = ImportError(f"读取 {file_path.name} 需要安装 {engine}，请运行 pip install {engine}")
                last_error.__cause__ = exc
                break
            except Exception as exc:
                if encoding_override is None:
                    last_error = exc
                    continue
                else:
                    last_error = exc
                    break

    # For other Excel formats or if .xls failed
    try:

        return pd.read_excel(file_path, sheet_name=sheet_name, **kwargs)

    except ValueError as exc:

        message = str(exc)

        if "Excel file format cannot be determined" not in message and "must specify an engine" not in message:

            last_error = exc

        else:

            engine = _SUPPORTED_ENGINES.get(suffix)

            if engine is None:

                last_error = ValueError(f"无法识别的 Excel 后缀: {suffix}")

            else:

                try:

                    return pd.read_excel(file_path, sheet_name=sheet_name, engine=engine, **kwargs)

                except ImportError as engine_exc:

                    last_error = ImportError(f"读取 {file_path.name} 需要安装 {engine}，请运行 pip install {engine}")

                    last_error.__cause__ = engine_exc

                except Exception as engine_exc:

                    last_error = engine_exc

    except Exception as exc:

        last_error = exc

    # If Excel reading failed, try as CSV with different encodings
    for encoding in ("utf-8", "utf-8-sig", "gbk", "gb18030", "latin1"):

        try:

            return pd.read_csv(file_path, sep=None, engine="python", encoding=encoding, on_bad_lines='skip', **kwargs)

        except (UnicodeDecodeError, ValueError) as csv_exc:

            last_error = csv_exc

            continue

        except Exception as csv_exc:

            last_error = csv_exc

            break

    if last_error is not None:

        raise last_error

    raise ValueError(f"无法解析文件 {file_path.name}，请检查格式。")

def _extract_generic_excel_impl(file_path: Path) -> pd.DataFrame:

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

    path = Path(file_path_str)

    return _extract_generic_excel_impl(path)

def extract_generic_excel(file_path: Path, *, mtime: Optional[float] = None) -> pd.DataFrame:

    effective_mtime = file_path.stat().st_mtime if mtime is None else mtime

    return _extract_generic_excel_cached(str(file_path), effective_mtime)

def _extract_lvi_data_impl(file_path: Path, current_points: Optional[List[float]]) -> Tuple[pd.DataFrame, List[float], pd.DataFrame]:

    if not file_path.exists():

        raise FileNotFoundError(f"未找到文件: {file_path}")

    df = read_excel_with_engine(

        file_path,

        header=None,

        skiprows=18,

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

    idx = numeric_df[CURRENT_COLUMN].idxmax()

    return numeric_df.loc[[idx]].reset_index(drop=True), [], numeric_df

@st.cache_data(show_spinner=False)

def _extract_lvi_data_cached(file_path_str: str, mtime: float, current_points: Optional[Tuple[float, ...]]) -> Tuple[pd.DataFrame, List[float], pd.DataFrame]:

    path = Path(file_path_str)
    points_list: Optional[List[float]] = list(current_points) if current_points is not None else None
    return _extract_lvi_data_impl(path, points_list)

def extract_lvi_data(file_path: Path, current_points: Optional[List[float]], *, mtime: Optional[float] = None) -> Tuple[pd.DataFrame, List[float], pd.DataFrame]:
    cached_points = tuple(current_points) if current_points is not None else None

    effective_mtime = file_path.stat().st_mtime if mtime is None else mtime

    return _extract_lvi_data_cached(str(file_path), effective_mtime, cached_points)

def _extract_rth_data_impl(file_path: Path, current_points: Optional[List[float]]) -> Tuple[pd.DataFrame, List[float], pd.DataFrame]:

    if not file_path.exists():

        raise FileNotFoundError(f"未找到文件: {file_path}")

    df = read_excel_with_engine(

        file_path,

        header=None,

        skiprows=8,

        usecols=[0, 1, 2],

        names=[LAMBDA_COLUMN, "热量Q", CURRENT_COLUMN],

    )

    df = df.dropna(how="all")

    numeric_df = df.apply(pd.to_numeric, errors="coerce").dropna(subset=[LAMBDA_COLUMN, "热量Q", CURRENT_COLUMN])
    numeric_df = drop_zero_current(numeric_df, CURRENT_COLUMN, tol=CURRENT_TOLERANCE)

    if numeric_df.empty:

        raise ValueError("Rth 数据为空或无法提取有效的电流点。")

    baseline_rows = numeric_df[(numeric_df[CURRENT_COLUMN] - 2.0).abs() <= CURRENT_TOLERANCE]

    if baseline_rows.empty:

        fallback_idx = numeric_df[CURRENT_COLUMN].idxmin()

        baseline_rows = numeric_df.loc[[fallback_idx]]

        baseline_current = float(baseline_rows.iloc[0][CURRENT_COLUMN])

    else:

        baseline_current = 2.0

    lambda_reference = float(baseline_rows.iloc[0][LAMBDA_COLUMN])

    full_numeric = numeric_df.copy()

    full_numeric[SHIFT_COLUMN] = full_numeric[LAMBDA_COLUMN] - lambda_reference

    full_numeric.attrs["lambda_baseline_current"] = baseline_current

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

    # 提取2A波长
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

def _extract_rth_data_cached(file_path_str: str, mtime: float, current_points: Optional[Tuple[float, ...]]) -> Tuple[pd.DataFrame, List[float], pd.DataFrame]:
    path = Path(file_path_str)
    return _extract_rth_data_impl(path, list(current_points) if current_points is not None else None)

def extract_rth_data(file_path: Path, current_points: Optional[List[float]], *, mtime: Optional[float] = None) -> Tuple[pd.DataFrame, List[float], pd.DataFrame]:
    cached_points = tuple(current_points) if current_points is not None else None
    effective_mtime = file_path.stat().st_mtime if mtime is None else mtime
    return _extract_rth_data_cached(str(file_path), effective_mtime, cached_points)


def clear_extraction_caches() -> None:
    """清除提取流程使用的缓存，确保后续操作能够强制重新读取文件。"""
    _extract_generic_excel_cached.clear()
    _extract_lvi_data_cached.clear()
    _extract_rth_data_cached.clear()
    build_chip_measurement_index_cached.clear()
    build_module_measurement_index_cached.clear()

def show_toast(message: str, icon: str = "ℹ️", duration: int = 2000) -> None:
    """显示自定义持续时间的toast消息
    Args:
        message: 消息内容
        icon: 图标
        duration: 持续时间（毫秒），默认2000ms（2秒）
    """
    st.toast(message, icon=icon)
    # 注入JavaScript来控制toast消失时间
    st.markdown(
        f"""
        <script>
        setTimeout(function() {{
            const toasts = document.querySelectorAll('[data-testid="stToast"]');
            if (toasts.length > 0) {{
                const lastToast = toasts[toasts.length - 1];
                lastToast.style.transition = 'opacity 0.3s ease-out';
                lastToast.style.opacity = '0';
                setTimeout(function() {{
                    lastToast.remove();
                }}, 300);
            }}
        }}, {duration});
        </script>
        """,
        unsafe_allow_html=True
    )

def trigger_scroll_if_needed(anchor_id: str) -> None:
    """将页面滚动到指定锚点（如果需要）。"""
    pending = st.session_state.get("pending_scroll_target")
    if pending != anchor_id:
        return

    st.markdown(
        f"""
        <script>
        const anchor = document.getElementById("{anchor_id}");
        if (anchor) {{
            anchor.scrollIntoView({{ behavior: "smooth", block: "start" }});
        }}
        </script>
        """,
        unsafe_allow_html=True,
    )
    st.session_state.pending_scroll_target = None


def render_extraction_results_section(
    container,
    result_df: Optional[pd.DataFrame],
    error_messages: Optional[Iterable[str]],
    info_messages: Optional[Iterable[str]],
    *,
    entity_label: str = "壳体",
) -> None:
    """渲染抽取结果展示区段"""
    if result_df is None:
        return

    errors = list(error_messages or [])
    infos = list(info_messages or [])

    with container:
        st.markdown('<div id="results"></div>', unsafe_allow_html=True)
        st.markdown("---")
        st.subheader("📊 抽取结果概览")

        overview_cols = st.columns(3)
        shell_series = (
            result_df[SHELL_COLUMN]
            if SHELL_COLUMN in result_df.columns
            else pd.Series(dtype=str)
        )
        test_series = (
            result_df[TEST_TYPE_COLUMN]
            if TEST_TYPE_COLUMN in result_df.columns
            else pd.Series(dtype=str)
        )
        with overview_cols[0]:
            st.metric("记录数", len(result_df))
        with overview_cols[1]:
            st.metric(f"{entity_label}数量", int(shell_series.nunique()))
        with overview_cols[2]:
            st.metric("站别数量", int(test_series.nunique()))

        with st.expander("查看抽取结果明细", expanded=True):
            row_count = len(result_df)
            table_height = max(140, min(600, row_count * 34 + 60))
            st.dataframe(
                result_df,
                use_container_width=True,
                hide_index=False,
                height=table_height,
            )

        st.markdown("---")
        st.subheader("💾 导出数据")

        col_name, col_btn = st.columns([3, 1])
        with col_name:
            download_name_input = st.text_input(
                "文件名称",
                value="combined_subset",
                help="输入文件名（无需扩展名，自动添加.xlsx)",
                key="download_name_input",
            )
        with col_btn:
            st.markdown("<div style='margin-top: 32px;'></div>", unsafe_allow_html=True)
            download_requested = st.button(
                "💾 生成下载文件", key="download_btn"
            )

        if download_requested:
            default_download_name = "combined_subset.xlsx"
            requested_name = (download_name_input or "").strip()
            try:
                download_filename = ensure_xlsx_suffix(
                    requested_name or default_download_name
                )
            except ValueError:
                show_toast("请输入有效的文件名�?", icon="⚠️")
            else:
                buffer = io.BytesIO()
                try:
                    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
                        result_df.to_excel(writer, index=False, sheet_name="Sheet1")
                except ImportError:
                    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                        result_df.to_excel(writer, index=False, sheet_name="Sheet1")
                buffer.seek(0)

                st.session_state.download_payload = buffer.getvalue()
                st.session_state.download_filename = download_filename
                st.session_state.download_request_counter = (
                    st.session_state.get("download_request_counter", 0) + 1
                )
                show_toast(
                    f"数据已准备，请点击下方按钮下载：{download_filename}", icon="📁"
                )

        download_payload = st.session_state.get("download_payload")
        download_counter = st.session_state.get("download_request_counter", 0)
        if download_payload and download_counter:
            st.download_button(
                "📥 点击下载保存文件",
                data=download_payload,
                file_name=st.session_state.get(
                    "download_filename", "combined_subset.xlsx"
                ),
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key=f"download_button_{download_counter}",
                use_container_width=True,
            )

        if errors or infos:
            col1, col2 = st.columns(2)

            if errors:
                with col1:
                    with st.expander(
                        f"展开查看失败详情（{len(errors)} 条）", expanded=False
                    ):
                        for message in errors:
                            st.markdown(f"- {message}")

            if infos:
                with col2:
                    with st.expander(
                        f"处理提示（{len(infos)} 条）", expanded=False
                    ):
                        for message in infos:
                            st.markdown(f"- {message}")


def main() -> None:
    st.set_page_config(page_title="Excel 数据列提取", layout="wide")
    # 侧边栏目录
    with st.sidebar:
        st.title("📑 功能导航")
        st.markdown("---")
        
        # 分析功能区
        st.markdown("### 📊 数据分析")
        
        if st.button("📈 单壳体分析", use_container_width=True):
            st.session_state.show_single_analysis = True
            st.session_state.show_multi_power = False
            st.session_state.show_multi_station = False
            st.session_state.show_boxplot = False
            st.query_params.update({"section": "single"})
            st.session_state.pending_scroll_target = "single"
        
        if st.button("📉 多壳体分析", use_container_width=True):
            st.session_state.show_multi_power = True
            st.session_state.show_single_analysis = False
            st.session_state.show_multi_station = False
            st.session_state.show_boxplot = False
            st.query_params.update({"section": "multi_power"})
            st.session_state.pending_scroll_target = "multi_power"
        
        if st.button("🔄 多站别分析", use_container_width=True):
            st.session_state.show_multi_station = True
            st.session_state.show_single_analysis = False
            st.session_state.show_multi_power = False
            st.session_state.show_boxplot = False
            st.query_params.update({"section": "multi_station"})
            st.session_state.pending_scroll_target = "multi_station"
        
        if st.button("📦 箱线图分析", use_container_width=True):
            st.session_state.show_boxplot = True
            st.session_state.show_single_analysis = False
            st.session_state.show_multi_power = False
            st.session_state.show_multi_station = False
            st.query_params.update({"section": "boxplot"})
            st.session_state.pending_scroll_target = "boxplot"
        
        st.markdown("---")
        # 快速统计信息
        if EXTRACTION_STATE_KEY in st.session_state and st.session_state[EXTRACTION_STATE_KEY]:
            extraction_state = st.session_state[EXTRACTION_STATE_KEY]
            if extraction_state and "result_df" in extraction_state and extraction_state["result_df"] is not None:
                result_df = extraction_state["result_df"]
                st.markdown("### 📌 当前状态")
                col1, col2, col3 = st.columns(3)
                state_mode = extraction_state.get("form_mode", MODULE_MODE)
                sidebar_label = "壳体" if state_mode == MODULE_MODE else "芯片"
                with col1:
                    st.metric(f"{sidebar_label}数", len(extraction_state.get("folder_entries", [])), label_visibility="visible")
                with col2:
                    st.metric("数据量", len(result_df), label_visibility="visible")
                with col3:
                    if TEST_TYPE_COLUMN in result_df.columns:
                        st.metric("站别数", result_df[TEST_TYPE_COLUMN].nunique(), label_visibility="visible")
                st.markdown("---")
        

    st.title("壳体测试数据查询")
    st.caption("支持输入多个壳体号，按测试类型与测试文件批量提取数据。")
    st.markdown('<div id="input"></div>', unsafe_allow_html=True)

    mode_labels = [label for label, _ in EXTRACTION_MODE_OPTIONS]
    mode_label = st.radio(
        "数据提取模式",
        mode_labels,
        index=0,
        horizontal=True,
        key="data_fetch_mode",
    )
    extraction_mode = EXTRACTION_MODE_LOOKUP.get(mode_label, MODULE_MODE)

    folder_label = "壳体号或Ldtd路径" if extraction_mode == MODULE_MODE else "芯片名称或路径"
    folder_help = (
        "可输入一个或多个壳体号，每行一个，例如 HHD550048。也支持直接粘贴完整路径。"
        if extraction_mode == MODULE_MODE
        else "可输入一个或多个芯片名或完整路径，每行一个，例如 2019-12-120240。"
    )
    measurement_options = (
        [label for label in MEASUREMENT_OPTIONS.keys() if label in CHIP_SUPPORTED_MEASUREMENTS]
        if extraction_mode == CHIP_MODE
        else list(MEASUREMENT_OPTIONS.keys())
    )

    with st.form("input_form"):
        folder_input = st.text_area(
            folder_label,
            help=folder_help,
            key=f"folder_input_{extraction_mode}",
        )

        if extraction_mode == MODULE_MODE:
            selected_tests = st.multiselect(
                "选择测试类型",
                options=TEST_CATEGORY_OPTIONS,
                default=TEST_CATEGORY_OPTIONS,
                key="module_test_select",
            )
        else:
            selected_tests = [CHIP_TEST_CATEGORY]
            st.info("芯片模式会自动递归查找最新的 LVI / Rth 测试文件。", icon="ℹ️")

        selected_measurements = st.multiselect(
            "选择测试文件",
            options=measurement_options,
            default=measurement_options,
            key=f"measurement_select_{extraction_mode}",
        )

        current_input = st.text_input(
            "电流点",
            help="可选，默认最高电流点。输入 'a' 或 'A' 提取所有电流点。也可输入单值或范围（如 12~19）。",
            key=f"current_input_{extraction_mode}",
        )
        submit_col, refresh_col = st.columns(2)

        with submit_col:
            submitted = st.form_submit_button("🚀 开始抽取", use_container_width=True)

        with refresh_col:
            force_refresh = st.form_submit_button("♻️ 强制刷新缓存", use_container_width=True)

    action_requested = submitted or force_refresh
    entry_label = "壳体" if extraction_mode == MODULE_MODE else "芯片"
    entry_prompt = "壳体号" if extraction_mode == MODULE_MODE else "芯片名或路径"

    # 保存分析状态到session state
    if 'pending_scroll_target' not in st.session_state:
        st.session_state.pending_scroll_target = None
    if 'show_multi_station' not in st.session_state:
        st.session_state.show_multi_station = False
    if 'show_boxplot' not in st.session_state:
        st.session_state.show_boxplot = False
    if 'show_single_analysis' not in st.session_state:
        st.session_state.show_single_analysis = False
    if 'show_multi_power' not in st.session_state:
        st.session_state.show_multi_power = False
    if 'download_payload' not in st.session_state:
        st.session_state.download_payload = None
    if 'download_filename' not in st.session_state:
        st.session_state.download_filename = "combined_subset.xlsx"
    if 'download_request_counter' not in st.session_state:
        st.session_state.download_request_counter = 0

    extraction_state = st.session_state.get(EXTRACTION_STATE_KEY)

    previous_inputs_match = False
    if extraction_state and "form_folder_input" in extraction_state:
        previous_inputs_match = (
            folder_input == extraction_state.get("form_folder_input", "")
            and selected_tests == extraction_state.get("form_selected_tests", [])
            and selected_measurements == extraction_state.get("form_selected_measurements", [])
            and current_input == extraction_state.get("form_current_input", "")
            and extraction_mode == extraction_state.get("form_mode", MODULE_MODE)
        )

    if extraction_state is not None and not action_requested and not previous_inputs_match:
        st.session_state[EXTRACTION_STATE_KEY] = None
        extraction_state = None

    if force_refresh:
        clear_extraction_caches()
        st.session_state.pop(EXTRACTION_STATE_KEY, None)
        st.session_state.pop("lvi_plot_sources", None)
        st.session_state.pop("rth_plot_sources", None)
        extraction_state = None
        previous_inputs_match = False

    should_recompute = (
        force_refresh
        or extraction_state is None
        or (action_requested and not previous_inputs_match)
    )

    result_df: Optional[pd.DataFrame] = None

    if not action_requested and extraction_state is None:

        st.info("填写参数后点击“开始提取”或“生成电流-功率-电光效率图”按钮")

        return

    # 重置所有分析状态
    if action_requested:
        st.session_state.show_multi_station = False
        st.session_state.show_boxplot = False
        st.session_state.show_single_analysis = False
        st.session_state.show_multi_power = False
        st.session_state.pending_scroll_target = None

        if not folder_input:

            st.toast(f"⚠️请填写{entry_prompt}", icon="⚠️")

            return

        if extraction_mode == MODULE_MODE and not selected_tests:

            st.toast("⚠️请至少选择一个测试类型", icon="⚠️")

            return

        if not selected_measurements:

            st.toast("⚠️请至少选择一个测试文件", icon="⚠️")

            return

        folder_entries = parse_folder_entries(folder_input)

        if not folder_entries:

            st.toast(f"⚠️未识别到有效的{entry_label}输入，请检查格式", icon="⚠️")

            return

        if current_input.strip():

            try:

                current_points = parse_current_points(current_input)

            except ValueError as exc:

                st.toast(f"⚠️{str(exc)}", icon="⚠️")

                return

        else:

            current_points = []
        if should_recompute:
            combined_frames = []
            error_messages = []
            info_messages = []
        else:
            cached_state = extraction_state or {}
            folder_entries = cached_state.get("folder_entries", folder_entries)
            combined_frames = cached_state.get("combined_frames", [])
            error_messages = cached_state.get("error_messages", [])
            info_messages = cached_state.get("info_messages", [])
            result_df = cached_state.get("result_df")
            current_points = cached_state.get("current_points", current_points)
            st.info("输入未变化，复用上次提取结果。如需重新加载，请点击“强制刷新缓存”。")

    else:

        folder_entries = extraction_state["folder_entries"]

        combined_frames = extraction_state["combined_frames"]

        error_messages = extraction_state["error_messages"]

        info_messages = extraction_state["info_messages"]


        result_df = extraction_state["result_df"]

        current_points = extraction_state.get("current_points", [])



    extraction_results_container = st.container()

    # 使用session_state保存数据，避免切换时丢失
    if 'lvi_plot_sources' not in st.session_state:
        st.session_state.lvi_plot_sources = {}
    if 'rth_plot_sources' not in st.session_state:
        st.session_state.rth_plot_sources = {}
    if action_requested and should_recompute:

        st.session_state.lvi_plot_sources = {}
        st.session_state.rth_plot_sources = {}

        
        lvi_plot_sources: Dict[Tuple[str, str], Tuple[pd.DataFrame, Optional[pd.DataFrame]]] = st.session_state.lvi_plot_sources
        rth_plot_sources: Dict[Tuple[str, str], pd.DataFrame] = st.session_state.rth_plot_sources
        
        total_entries = len(folder_entries)
        
        if total_entries >= 20:
            st.info(f"{entry_label}数量较多，可考虑分批处理以缩短等待时间。")
        
        effective_output_columns = list(OUTPUT_COLUMNS)
        if extraction_mode == MODULE_MODE:
            if WAVELENGTH_COLD_COLUMN in effective_output_columns:
                effective_output_columns.remove(WAVELENGTH_COLD_COLUMN)

        # 创建进度显示区域
        progress_container = st.container()
        with progress_container:
            progress_text = st.empty()
            progress_bar = st.progress(0.0)
            status_text = st.empty()
        
        progress_text.markdown(f"**正在处理 {total_entries} 个壳体...**")

        def do_measurement(entry_id: str, test_category: str, measurement_label: str, file_path: Path, file_mtime: float, multiple_found: bool, context_label: str):
            try:
                if measurement_label == "LVI":
                    extracted, missing_currents, lvi_full = extract_lvi_data(
                        file_path=file_path,
                        current_points=current_points,
                        mtime=file_mtime,
                    )
                    extracted = drop_zero_current(extracted, CURRENT_COLUMN, tol=CURRENT_TOLERANCE)
                    lvi_full = drop_zero_current(lvi_full, CURRENT_COLUMN, tol=CURRENT_TOLERANCE)
                    selected_subset = extracted if current_points else None
                    info_parts = []
                    if missing_currents:
                        info_parts.append(f"{context_label}: 未找到电流点 {missing_currents}")
                    tagged = extracted.copy()
                    tagged.insert(0, TEST_TYPE_COLUMN, test_category)
                    tagged.insert(0, SHELL_COLUMN, entry_id)
                    tagged = align_output_columns(tagged, columns=effective_output_columns)
                    return {
                        "tagged": tagged,
                        "lvi": (entry_id, test_category, lvi_full, selected_subset),
                        "rth": None,
                        "info": [f"找到文件: {context_label} -> {file_path.name}"] + info_parts,
                        "multiple": multiple_found,
                        "context": context_label,
                        "error": None,
                    }
                elif measurement_label == "Rth":
                    extracted, missing_currents, rth_full = extract_rth_data(
                        file_path=file_path,
                        current_points=current_points,
                        mtime=file_mtime,
                    )
                    extracted = drop_zero_current(extracted, CURRENT_COLUMN, tol=CURRENT_TOLERANCE)
                    rth_full = drop_zero_current(rth_full, CURRENT_COLUMN, tol=CURRENT_TOLERANCE)
                    info_parts = []
                    if missing_currents:
                        info_parts.append(f"{context_label}: 未找到电流点 {missing_currents}")
                    baseline_current = extracted.attrs.get("lambda_baseline_current")
                    if baseline_current is not None and abs(baseline_current - 2.0) > CURRENT_TOLERANCE:
                        info_parts.append(f"{context_label}: 波长shift基准使用 {baseline_current:.3f}A")
                    tagged = extracted.copy()
                    tagged.insert(0, TEST_TYPE_COLUMN, test_category)
                    tagged.insert(0, SHELL_COLUMN, entry_id)
                    tagged = align_output_columns(tagged, columns=effective_output_columns)
                    return {
                        "tagged": tagged,
                        "lvi": None,
                        "rth": (entry_id, test_category, rth_full),
                        "info": [f"找到文件: {context_label} -> {file_path.name}"] + info_parts,
                        "multiple": multiple_found,
                        "context": context_label,
                        "error": None,
                    }
                else:
                    extracted = extract_generic_excel(file_path, mtime=file_mtime)
                    tagged = extracted.copy()
                    tagged.insert(0, TEST_TYPE_COLUMN, test_category)
                    tagged.insert(0, SHELL_COLUMN, entry_id)
                    tagged = align_output_columns(tagged, columns=effective_output_columns)
                    return {
                        "tagged": tagged,
                        "lvi": None,
                        "rth": None,
                        "info": [f"找到文件: {context_label} -> {file_path.name}"],
                        "multiple": multiple_found,
                        "context": context_label,
                        "error": None,
                    }
            except Exception as exc:
                return {
                    "tagged": None,
                    "lvi": None,
                    "rth": None,
                    "info": [],
                    "multiple": multiple_found,
                    "context": context_label,
                    "error": f"{context_label}: {exc}",
                }
        
        def handle_measurement(entry_id: str, test_category: str, measurement_label: str, file_path: Path, file_mtime: float, multiple_found: bool, context_label: str) -> None:
            info_messages.append(f"找到文件: {context_label} -> {file_path.name}")
            try:
                if measurement_label == "LVI":
                    extracted, missing_currents, lvi_full = extract_lvi_data(
                        file_path=file_path,
                        current_points=current_points,
                        mtime=file_mtime,
                    )

                    if missing_currents:
                        info_messages.append(
                            f"{context_label}: 未找到电流点 {missing_currents}"
                        )

                    selected_subset = extracted if current_points else None
                    lvi_plot_sources[(entry_id, test_category)] = (lvi_full, selected_subset)

                elif measurement_label == "Rth":
                    extracted, missing_currents, rth_full_numeric = extract_rth_data(
                        file_path=file_path,
                        current_points=current_points,
                        mtime=file_mtime,
                    )
                    extracted = drop_zero_current(extracted, CURRENT_COLUMN, tol=CURRENT_TOLERANCE)
                    rth_full_numeric = drop_zero_current(rth_full_numeric, CURRENT_COLUMN, tol=CURRENT_TOLERANCE)

                    if missing_currents:
                        info_messages.append(
                            f"{context_label}: 未找到电流点 {missing_currents}"
                        )

                    rth_plot_sources[(entry_id, test_category)] = rth_full_numeric

                    rth_summary_parts: List[str] = []

                    for _, row in extracted.iterrows():
                        try:
                            current_val = float(row["Current (A)"])
                            lambda_val = float(row["波长lambda"])
                            shift_val = float(row["波长shift"])
                            rth_summary_parts.append(
                                f"I={current_val:.3f}A -> λ={lambda_val:.3f}, shift={shift_val:.3f}"
                            )
                        except Exception:
                            continue

                    if rth_summary_parts:
                        info_messages.append(
                            f"{context_label}: {', '.join(rth_summary_parts)}"
                        )

                    baseline_current = extracted.attrs.get("lambda_baseline_current")
                    if baseline_current is not None and abs(baseline_current - 2.0) > CURRENT_TOLERANCE:
                        info_messages.append(
                            f"{context_label}: 波长shift基准使用 {baseline_current:.3f}A（未找到 2A 数据）"
                        )

                else:
                    extracted = extract_generic_excel(file_path, mtime=file_mtime)
            except (FileNotFoundError, KeyError, ValueError) as exc:

                error_messages.append(f"{context_label}: {exc}")

                return

            except ImportError as exc:

                error_messages.append(str(exc))

                return

            except Exception as exc:

                error_messages.append(f"{context_label}: {exc}")

                return

            tagged = extracted.copy()

            tagged.insert(0, TEST_TYPE_COLUMN, test_category)

            tagged.insert(0, SHELL_COLUMN, entry_id)

            tagged = align_output_columns(tagged, columns=effective_output_columns)

            combined_frames.append(tagged)

            if multiple_found:
                info_messages.append(
                    f"{context_label}: 使用最新文件 {file_path.name}"
                )

        executor_workers = max(4, min(16, (os.cpu_count() or 4) * 2))
        futures = []
        total_tasks = 0
        with ThreadPoolExecutor(max_workers=executor_workers) as executor:
            for idx, entry in enumerate(folder_entries, start=1):
                if extraction_mode == MODULE_MODE:
                    try:
                        base_path = interpret_folder_input(entry)
                        info_messages.append(f"解析路径: {entry} -> {base_path} (存在: {base_path.exists()})")
                    except ValueError as exc:
                        error_messages.append(f"{entry}: {exc}")
                        progress_bar.progress(min(idx / total_entries, 1.0))
                        status_text.text(f"处理中: {idx}/{total_entries} - {entry_label}{entry} (路径错误)")
                        continue

                    for test_category in selected_tests:
                        try:
                            test_folder = resolve_test_folder(base_path, test_category)
                            files_in_folder = list(test_folder.glob("*.xls*"))
                            info_messages.append(f"测试文件夹: {test_folder}, 包含 {len(files_in_folder)} 个Excel文件")
                            if files_in_folder:
                                info_messages.append(f"  文件列表: {', '.join([f.name for f in files_in_folder[:5]])}")
                            measurement_index = build_module_measurement_index_cached(str(test_folder), test_folder.stat().st_mtime)
                        except FileNotFoundError as exc:
                            error_messages.append(f"{entry}/{test_category}: {exc}")
                            continue

                        for measurement_label in selected_measurements:
                            token = MEASUREMENT_OPTIONS[measurement_label]
                            try:
                                file_path, multiple_found, file_mtime = find_measurement_file(
                                    test_folder,
                                    token,
                                    index=measurement_index,
                                )
                            except (FileNotFoundError, KeyError, ValueError) as exc:
                                error_messages.append(f"{entry}/{test_category}/{measurement_label}: {exc}")
                                continue
                            except ImportError as exc:
                                error_messages.append(str(exc))
                                continue
                            except Exception as exc:
                                error_messages.append(f"{entry}/{test_category}/{measurement_label}: {exc}")
                                continue

                            futures.append(executor.submit(
                                do_measurement,
                                entry,
                                test_category,
                                measurement_label,
                                file_path,
                                file_mtime,
                                multiple_found,
                                f"{entry}/{test_category}/{measurement_label}",
                            ))
                            total_tasks += 1
                else:
                    try:
                        chip_folder = interpret_chip_folder_input(entry)
                        info_messages.append(f"解析芯片路径: {entry} -> {chip_folder} (存在: {chip_folder.exists()})")
                    except (ValueError, FileNotFoundError, NotADirectoryError) as exc:
                        error_messages.append(f"{entry}: {exc}")
                        progress_bar.progress(min(idx / total_entries, 1.0))
                        status_text.text(f"处理中: {idx}/{total_entries} - {entry_label}{entry} (路径错误)")
                        continue

                    try:
                        measurement_index = build_chip_measurement_index(chip_folder)
                    except (FileNotFoundError, NotADirectoryError) as exc:
                        error_messages.append(f"{entry}: {exc}")
                        continue

                    for measurement_label in selected_measurements:
                        token = MEASUREMENT_OPTIONS[measurement_label]
                        try:
                            file_path, multiple_found, file_mtime = find_chip_measurement_file(
                                chip_folder,
                                token,
                                index=measurement_index,
                            )
                        except FileNotFoundError as exc:
                            error_messages.append(f"{entry}/{measurement_label}: {exc}")
                            continue

                        futures.append(executor.submit(
                            do_measurement,
                            entry,
                            CHIP_TEST_CATEGORY,
                            measurement_label,
                            file_path,
                            file_mtime,
                            multiple_found,
                            f"{entry}/{measurement_label}",
                        ))
                        total_tasks += 1
        completed = 0
        for fut in as_completed(futures):
            res = fut.result()
            if res.get("error"):
                error_messages.append(res["error"])
            else:
                tagged = res.get("tagged")
                if tagged is not None:
                    combined_frames.append(tagged)
                info_messages.extend(res.get("info", []))
                if res.get("multiple"):
                    info_messages.append(f"{res.get('context')}: 使用最新文件")
                lvi_tuple = res.get("lvi")
                if lvi_tuple:
                    e_id, t_cat, lvi_full, selected_subset = lvi_tuple
                    lvi_plot_sources[(e_id, t_cat)] = (lvi_full, selected_subset)
                rth_tuple = res.get("rth")
                if rth_tuple:
                    e_id, t_cat, rth_full = rth_tuple
                    rth_plot_sources[(e_id, t_cat)] = rth_full
            completed += 1
            progress_bar.progress(min(completed / max(1, total_tasks), 1.0))
            status_text.text(f"已完成任务 {completed}/{total_tasks}")

        progress_bar.empty()
        progress_text.empty()
        status_text.empty()

        st.session_state.lvi_plot_sources = lvi_plot_sources
        st.session_state.rth_plot_sources = rth_plot_sources
        

    else:

        lvi_plot_sources: Dict[Tuple[str, str], Tuple[pd.DataFrame, Optional[pd.DataFrame]]] = st.session_state.lvi_plot_sources
        rth_plot_sources: Dict[Tuple[str, str], pd.DataFrame] = st.session_state.rth_plot_sources

    if action_requested and should_recompute:

        if not combined_frames:
        
            st.toast("❌ 未能汇总出任何数据", icon="❌")
        
            if error_messages:
        
                with st.expander(f"失败详情（{len(error_messages)} 条）", expanded=False):
        
                    for message in error_messages:
        
                        st.markdown(f"- {message}")
        
            st.session_state[EXTRACTION_STATE_KEY] = None
            return
        
        valid_frames: List[pd.DataFrame] = []
        
        for combined_frame in combined_frames:
        
            if combined_frame.empty:
        
                continue
        
            non_na_frame = combined_frame.dropna(how="all")
        
            if non_na_frame.empty:
        
                continue
        
            non_na_frame = non_na_frame.loc[:, ~non_na_frame.isna().all()]
        
            if non_na_frame.empty:
        
                continue
        
            valid_frames.append(non_na_frame)
        
        if not valid_frames:
        
            st.toast("❌ 无法整理出有效的数据", icon="❌")
        
            if error_messages:
        
                with st.expander(f"失败详情（{len(error_messages)} 条）", expanded=False):
        
                    for message in error_messages:
        
                        st.markdown(f"- {message}")
        
            st.session_state[EXTRACTION_STATE_KEY] = None
            return
        
        result_df = pd.concat(valid_frames, ignore_index=True)
        
        if "电光效率(%)" in result_df.columns:
        
            result_df[EFFICIENCY_COLUMN] = pd.to_numeric(result_df[EFFICIENCY_COLUMN], errors="coerce")
        
            result_df[EFFICIENCY_COLUMN] = result_df[EFFICIENCY_COLUMN].multiply(100).round(3)
        
        result_df = merge_measurement_rows(result_df, columns=effective_output_columns)
        
        # 对所有数值列保留三位小数
        numeric_columns = [CURRENT_COLUMN, POWER_COLUMN, VOLTAGE_COLUMN, EFFICIENCY_COLUMN, LAMBDA_COLUMN, SHIFT_COLUMN]
        for col in numeric_columns:
            if col in result_df.columns:
                result_df[col] = pd.to_numeric(result_df[col], errors="coerce").round(3)
        
        if TEST_TYPE_COLUMN in result_df.columns:
        
            result_df[TEST_TYPE_COLUMN] = pd.Categorical(result_df[TEST_TYPE_COLUMN], categories=PLOT_ORDER, ordered=True)
        
            if "Current (A)" in result_df.columns:
        
                result_df[CURRENT_COLUMN] = pd.to_numeric(result_df[CURRENT_COLUMN], errors="coerce")
        
                result_df = result_df.sort_values(by=[TEST_TYPE_COLUMN, CURRENT_COLUMN], kind="stable")
        
            else:
        
                result_df = result_df.sort_values(by=[TEST_TYPE_COLUMN], kind="stable")
        
            result_df[TEST_TYPE_COLUMN] = result_df[TEST_TYPE_COLUMN].astype("object").str.replace("测试", "", regex=False)
        

        st.session_state[EXTRACTION_STATE_KEY] = {
            "folder_entries": folder_entries,
            "combined_frames": combined_frames,
            "error_messages": error_messages,
            "info_messages": info_messages,
            "result_df": result_df,
            "current_points": current_points,
            # 保存表单输入用于检测变化
            "form_folder_input": folder_input,
            "form_selected_tests": selected_tests,
            "form_selected_measurements": selected_measurements,
            "form_current_input": current_input,
            "form_mode": extraction_mode,
        }

    else:

        result_df = extraction_state["result_df"]
        combined_frames = extraction_state["combined_frames"]
        error_messages = extraction_state["error_messages"]
        info_messages = extraction_state["info_messages"]

    render_extraction_results_section(
        extraction_results_container,
        result_df,
        error_messages,
        info_messages,
        entity_label=entry_label,
    )

    # 多壳体功率分析
    if st.session_state.get('show_multi_power', False):
        st.markdown('<div id="multi_power"></div>', unsafe_allow_html=True)
        trigger_scroll_if_needed("multi_power")
        st.subheader("多壳体分析")
        shells = sorted({shell_id for shell_id, _ in lvi_plot_sources.keys()})
        if len(shells) == 0:
            show_toast("请先抽取数据", icon="⚠️")
        elif len(shells) > 10:
            show_toast("多壳体分析最多支持10个壳体，当前有{}个壳体".format(len(shells)), icon="⚠️")
        else:
            # 收集功率、效率和波长数据
            power_tab_entries = []
            efficiency_tab_entries = []
            lambda_tab_entries = []
            
            # 获取rth数据源
            rth_plot_sources = st.session_state.get('rth_plot_sources', {})
            
            for test_type in PLOT_ORDER:
                power_series = []
                efficiency_series = []
                lambda_series = []
                
                for shell_id in shells:
                    data_tuple = lvi_plot_sources.get((shell_id, test_type))
                    if not data_tuple:
                        continue
                    df_full, _ = data_tuple
                    if df_full is None or df_full.empty:
                        continue
                    
                    # 功率数据
                    power_df = df_full.dropna(subset=[CURRENT_COLUMN, POWER_COLUMN])
                    if not power_df.empty:
                        power_numeric = power_df[[CURRENT_COLUMN, POWER_COLUMN]].copy()
                        power_numeric[CURRENT_COLUMN] = pd.to_numeric(power_numeric[CURRENT_COLUMN], errors="coerce")
                        power_numeric[POWER_COLUMN] = pd.to_numeric(power_numeric[POWER_COLUMN], errors="coerce")
                        power_numeric = power_numeric.dropna(subset=[CURRENT_COLUMN, POWER_COLUMN])
                        if not power_numeric.empty:
                            power_series.append((shell_id, power_numeric))
                    
                    # 效率数据
                    efficiency_df = df_full.dropna(subset=[CURRENT_COLUMN, EFFICIENCY_COLUMN])
                    if not efficiency_df.empty:
                        efficiency_numeric = efficiency_df[[CURRENT_COLUMN, EFFICIENCY_COLUMN]].copy()
                        efficiency_numeric[CURRENT_COLUMN] = pd.to_numeric(efficiency_numeric[CURRENT_COLUMN], errors="coerce")
                        efficiency_numeric[EFFICIENCY_COLUMN] = pd.to_numeric(efficiency_numeric[EFFICIENCY_COLUMN], errors="coerce")
                        efficiency_numeric = efficiency_numeric.dropna(subset=[CURRENT_COLUMN, EFFICIENCY_COLUMN])
                        if not efficiency_numeric.empty:
                            efficiency_series.append((shell_id, efficiency_numeric))
                    
                    # 波长数据
                    rth_df = rth_plot_sources.get((shell_id, test_type))
                    if rth_df is not None and not rth_df.empty:
                        lambda_df = rth_df.dropna(subset=[CURRENT_COLUMN, LAMBDA_COLUMN])
                        if not lambda_df.empty:
                            lambda_numeric = lambda_df[[CURRENT_COLUMN, LAMBDA_COLUMN]].copy()
                            lambda_numeric[CURRENT_COLUMN] = pd.to_numeric(lambda_numeric[CURRENT_COLUMN], errors="coerce")
                            lambda_numeric[LAMBDA_COLUMN] = pd.to_numeric(lambda_numeric[LAMBDA_COLUMN], errors="coerce")
                            lambda_numeric = lambda_numeric.dropna(subset=[CURRENT_COLUMN, LAMBDA_COLUMN])
                            if not lambda_numeric.empty:
                                lambda_series.append((shell_id, lambda_numeric))
                
                if power_series:
                    power_tab_entries.append((test_type, power_series))
                if efficiency_series:
                    efficiency_tab_entries.append((test_type, efficiency_series))
                if lambda_series:
                    lambda_tab_entries.append((test_type, lambda_series))
            
            if not power_tab_entries and not efficiency_tab_entries and not lambda_tab_entries:
                st.info("所选壳体在功率、效率和波长数据上缺少可对比的站别。")
            else:
                # 创建功率、效率和波长的主标签页
                tab_names = ["功率对比", "效率对比"]
                if lambda_tab_entries:
                    tab_names.append("波长对比")
                main_tabs = st.tabs(tab_names)
                
                # 功率对比标签页
                with main_tabs[0]:
                    if not power_tab_entries:
                        st.info("所选壳体在功率数据上缺少可对比的站别。")
                    else:
                        tab_labels = [test_type.replace("测试", "") for test_type, _ in power_tab_entries]
                        tabs = st.tabs(tab_labels)
                        for tab, (test_type, series) in zip(tabs, power_tab_entries):
                            with tab:
                                # 创建子标签页：对比图和预测图
                                sub_tabs = st.tabs(["📊 对比图", "📈 拟合预测"])
                                
                                with sub_tabs[0]:
                                    # 显示多壳体单轴折线对比图
                                    chart = build_multi_shell_chart(
                                        series,
                                        POWER_COLUMN,
                                        "功率(W)",
                                        test_type,
                                    )
                                    if chart is not None:
                                        st.altair_chart(
                                            chart,
                                            theme="streamlit",
                                            use_container_width=True,
                                        )
                                    else:
                                        st.info("无法生成对比图表")
                                
                                with sub_tabs[1]:
                                    combined_currents: list[float] = []
                                    for _, df_metric in series:
                                        if CURRENT_COLUMN not in df_metric.columns:
                                            continue
                                        current_series = pd.to_numeric(
                                            df_metric[CURRENT_COLUMN],
                                            errors="coerce",
                                        ).dropna()
                                        if not current_series.empty:
                                            combined_currents.extend(current_series.tolist())

                                    target_current_value: Optional[float] = None
                                    col1, col2 = st.columns([3, 1])
                                    with col1:
                                        fit_mode = st.radio(
                                            "拟合模式",
                                            ["全局", "分壳体"],
                                            index=0,
                                            horizontal=True,
                                            key=f"power_fit_mode_{test_type}"
                                        )
                                        auto_poly = st.checkbox(
                                            "自动选择阶数",
                                            value=False,
                                            key=f"power_auto_poly_{test_type}"
                                        )
                                        poly_degree = st.slider(
                                            "多项式阶数",
                                            min_value=1,
                                            max_value=5,
                                            value=1,
                                            key=f"power_poly_{test_type}",
                                            help="选择拟合多项式的阶数（1=线性，2=二次，3=三次...）"
                                        )
                                    with col2:
                                        if combined_currents:
                                            currents_array = np.array(combined_currents, dtype=float)
                                            min_current = float(np.min(currents_array))
                                            max_current = float(np.max(currents_array))
                                            default_current = float(np.median(currents_array))
                                            target_current_input = st.number_input(
                                                "目标电流(A)",
                                                min_value=0.0,
                                                value=default_current,
                                                step=0.1,
                                                key=f"power_target_current_{test_type}",
                                                format="%.3f",
                                            )
                                            target_current_value = float(target_current_input)
                                            st.caption(f"数据范围约 {min_current:.3f}~{max_current:.3f} A")
                                        else:
                                            st.caption("暂无可用于拟合的电流数据")

                                    libs_available = _ensure_prediction_libs_loaded()
                                    if not libs_available:
                                        st.warning("⚠️ 拟合预测功能需要安装以下依赖：")
                                        st.code("pip install scipy scikit-learn", language="bash")
                                        st.info("安装后重启应用即可使用拟合预测功能")
                                    else:
                                        prediction_chart = plot_multi_shell_prediction(
                                            series,
                                            POWER_COLUMN,
                                            "功率(W)",
                                            test_type,
                                            poly_degree=poly_degree,
                                            fit_mode="global" if fit_mode == "全局" else "per_shell",
                                            auto_poly=auto_poly,
                                        )
                                        if prediction_chart is not None:
                                            st.altair_chart(prediction_chart, use_container_width=True)
                                        else:
                                            st.info("无法生成预测图表")

                                    if target_current_value is not None:
                                        power_predictions = compute_power_predictions(
                                            series,
                                            target_current_value,
                                            poly_degree,
                                        )
                                        if power_predictions:
                                            st.markdown("#### 目标电流功率预测")
                                            prediction_df = pd.DataFrame(power_predictions)
                                            st.dataframe(
                                                prediction_df,
                                                width="stretch",
                                                hide_index=True,
                                            )
                                        else:
                                            st.info("未能生成目标电流的功率预测，请检查数据质量或调整电流值。")
                
                # 效率对比标签页
                with main_tabs[1]:
                    if not efficiency_tab_entries:
                        st.info("所选壳体在效率数据上缺少可对比的站别。")
                    else:
                        tab_labels = [test_type.replace("测试", "") for test_type, _ in efficiency_tab_entries]
                        tabs = st.tabs(tab_labels)
                        for tab, (test_type, series) in zip(tabs, efficiency_tab_entries):
                            with tab:
                                # 需要将效率转换为百分比
                                series_percent = []
                                for shell_id, numeric in series:
                                    numeric_copy = numeric.copy()
                                    numeric_copy[EFFICIENCY_COLUMN] = numeric_copy[EFFICIENCY_COLUMN] * 100
                                    series_percent.append((shell_id, numeric_copy))
                                
                                # 创建子标签页：对比图和预测图
                                sub_tabs = st.tabs(["📊 对比图", "📈 拟合预测"])
                                
                                with sub_tabs[0]:
                                    # 显示多壳体单轴折线对比图
                                    chart = build_multi_shell_chart(
                                        series_percent,
                                        EFFICIENCY_COLUMN,
                                        "电光效率(%)",
                                        test_type,
                                    )
                                    if chart is not None:
                                        st.altair_chart(
                                            chart,
                                            theme="streamlit",
                                            use_container_width=True,
                                        )
                                    else:
                                        st.info("无法生成对比图表")
                                
                                with sub_tabs[1]:
                                    if not HAS_PREDICTION_LIBS:
                                        st.warning("⚠️ 拟合预测功能需要安装以下依赖：")
                                        st.code("pip install scipy scikit-learn", language="bash")
                                        st.info("安装后重启应用即可使用拟合预测功能")
                                    else:
                                        fit_mode = st.radio(
                                            "拟合模式",
                                            ["全局", "分壳体"],
                                            index=0,
                                            horizontal=True,
                                            key=f"eff_fit_mode_{test_type}"
                                        )
                                        prediction_chart = plot_multi_shell_prediction(
                                            series_percent,
                                            EFFICIENCY_COLUMN,
                                            "电光效率(%)",
                                            test_type,
                                            fit_mode="global" if fit_mode == "全局" else "per_shell",
                                        )
                                        if prediction_chart is not None:
                                            st.altair_chart(prediction_chart, use_container_width=True)
                                        else:
                                            st.error("❌ 所有效率模型拟合失败，请检查数据质量")
                
                # 波长对比标签页
                if lambda_tab_entries:
                    with main_tabs[2]:
                        tab_labels = [test_type.replace("测试", "") for test_type, _ in lambda_tab_entries]
                        tabs = st.tabs(tab_labels)
                        for tab, (test_type, series) in zip(tabs, lambda_tab_entries):
                            with tab:
                                # 创建子标签页：对比图和预测图
                                sub_tabs = st.tabs(["📊 对比图", "📈 拟合预测"])
                                
                                with sub_tabs[0]:
                                    # 显示多壳体单轴折线对比图
                                    chart = build_multi_shell_chart(
                                        series,
                                        LAMBDA_COLUMN,
                                        "波长(nm)",
                                        test_type,
                                    )
                                    if chart is not None:
                                        st.altair_chart(
                                            chart,
                                            theme="streamlit",
                                            use_container_width=True,
                                        )
                                    else:
                                        st.info("无法生成对比图表")
                                
                                with sub_tabs[1]:
                                    if not HAS_PREDICTION_LIBS:
                                        st.warning("⚠️ 拟合预测功能需要安装以下依赖：")
                                        st.code("pip install scipy scikit-learn", language="bash")
                                        st.info("安装后重启应用即可使用拟合预测功能")
                                    else:
                                        col1, col2 = st.columns([3, 1])
                                        with col1:
                                            fit_mode = st.radio(
                                                "拟合模式",
                                                ["全局", "分壳体"],
                                                index=0,
                                                horizontal=True,
                                                key=f"lambda_fit_mode_{test_type}"
                                            )
                                            auto_poly = st.checkbox(
                                                "自动选择阶数",
                                                value=False,
                                                key=f"lambda_auto_poly_{test_type}"
                                            )
                                            poly_degree = st.slider(
                                                "多项式阶数",
                                                min_value=1,
                                                max_value=5,
                                                value=2,
                                                key=f"lambda_poly_{test_type}",
                                                help="选择拟合多项式的阶数（1=线性，2=二次，3=三次...）"
                                            )
                                        prediction_chart = plot_multi_shell_prediction(
                                            series,
                                            LAMBDA_COLUMN,
                                            "波长(nm)",
                                            test_type,
                                            poly_degree=poly_degree,
                                            fit_mode="global" if fit_mode == "全局" else "per_shell",
                                            auto_poly=auto_poly,
                                        )
                                        if prediction_chart is not None:
                                            st.altair_chart(prediction_chart, use_container_width=True)
                                        else:
                                            st.info("无法生成预测图表")

    # 多站别分析
    if st.session_state.get('show_multi_station', False):
        lvi_plot_sources = st.session_state.get('lvi_plot_sources')
        if lvi_plot_sources:
            st.markdown('---')
            st.markdown('<div id="multi_station"></div>', unsafe_allow_html=True)
            trigger_scroll_if_needed("multi_station")
            st.subheader("📊 多站别分析")

            available_shells = sorted({shell_id for (shell_id, _) in lvi_plot_sources.keys()})
            rth_plot_sources = st.session_state.get('rth_plot_sources', {})

            if len(available_shells) > 1:
                st.markdown("**📊 所有壳体平均值变化分析**")
                all_shells_data: List[pd.DataFrame] = []

                for shell_id in available_shells:
                    for (sid, test_type), (df_full, _) in lvi_plot_sources.items():
                        if sid == shell_id and df_full is not None and not df_full.empty:
                            temp_df = df_full.copy()
                            temp_df[TEST_TYPE_COLUMN] = test_type.replace("测试", "")
                            temp_df[SHELL_COLUMN] = shell_id
                            all_shells_data.append(temp_df)

                    if rth_plot_sources:
                        for (sid, test_type), rth_df in rth_plot_sources.items():
                            if sid == shell_id and rth_df is not None and not rth_df.empty:
                                for idx, lvi_df in enumerate(all_shells_data):
                                    if (
                                        lvi_df[SHELL_COLUMN].iloc[0] == shell_id
                                        and lvi_df[TEST_TYPE_COLUMN].iloc[0] == test_type.replace("测试", "")
                                    ):
                                        rth_temp = rth_df.copy()
                                        rth_temp[TEST_TYPE_COLUMN] = test_type.replace("测试", "")
                                        rth_temp[SHELL_COLUMN] = shell_id
                                        merged = pd.merge(
                                            lvi_df,
                                            rth_temp[
                                                [
                                                    CURRENT_COLUMN,
                                                    LAMBDA_COLUMN,
                                                    SHIFT_COLUMN,
                                                    TEST_TYPE_COLUMN,
                                                    SHELL_COLUMN,
                                                ]
                                            ],
                                            on=[CURRENT_COLUMN, TEST_TYPE_COLUMN, SHELL_COLUMN],
                                            how="outer",
                                        )
                                        all_shells_data[idx] = merged
                                        break

                if all_shells_data:
                    all_shells_df = pd.concat(all_shells_data, ignore_index=True)
                    agg_dict: Dict[str, str] = {
                        POWER_COLUMN: 'mean',
                        EFFICIENCY_COLUMN: 'mean',
                        VOLTAGE_COLUMN: 'mean',
                    }
                    if LAMBDA_COLUMN in all_shells_df.columns:
                        agg_dict[LAMBDA_COLUMN] = 'mean'
                    if SHIFT_COLUMN in all_shells_df.columns:
                        agg_dict[SHIFT_COLUMN] = 'mean'

                    avg_by_station = all_shells_df.groupby(TEST_TYPE_COLUMN).agg(agg_dict).reset_index()
                    ordered_avg_types = [
                        t for t in SANITIZED_PLOT_ORDER if t in avg_by_station[TEST_TYPE_COLUMN].unique()
                    ]

                    avg_change_data: List[Dict[str, Union[str, float]]] = []
                    for idx in range(len(ordered_avg_types) - 1):
                        from_type = ordered_avg_types[idx]
                        to_type = ordered_avg_types[idx + 1]
                        from_row = avg_by_station[avg_by_station[TEST_TYPE_COLUMN] == from_type]
                        to_row = avg_by_station[avg_by_station[TEST_TYPE_COLUMN] == to_type]
                        if from_row.empty or to_row.empty:
                            continue

                        avg_change_row: Dict[str, Union[str, float]] = {"变化": f"{from_type} -> {to_type}"}
                        power_from = from_row[POWER_COLUMN].iloc[0]
                        power_to = to_row[POWER_COLUMN].iloc[0]
                        if pd.notna(power_from) and pd.notna(power_to):
                            avg_change_row["功率变化(W)"] = power_to - power_from

                        eff_from = from_row[EFFICIENCY_COLUMN].iloc[0] * 100
                        eff_to = to_row[EFFICIENCY_COLUMN].iloc[0] * 100
                        if pd.notna(eff_from) and pd.notna(eff_to):
                            avg_change_row["效率变化(%)"] = eff_to - eff_from

                        voltage_from = from_row[VOLTAGE_COLUMN].iloc[0]
                        voltage_to = to_row[VOLTAGE_COLUMN].iloc[0]
                        if pd.notna(voltage_from) and pd.notna(voltage_to):
                            avg_change_row["电压变化(V)"] = voltage_to - voltage_from

                        if LAMBDA_COLUMN in avg_by_station.columns:
                            lambda_from = from_row[LAMBDA_COLUMN].iloc[0]
                            lambda_to = to_row[LAMBDA_COLUMN].iloc[0]
                            if pd.notna(lambda_from) and pd.notna(lambda_to):
                                avg_change_row["波长变化(nm)"] = lambda_to - lambda_from

                        if SHIFT_COLUMN in avg_by_station.columns:
                            shift_from = from_row[SHIFT_COLUMN].iloc[0]
                            shift_to = to_row[SHIFT_COLUMN].iloc[0]
                            if pd.notna(shift_from) and pd.notna(shift_to):
                                avg_change_row["Shift变化(nm)"] = shift_to - shift_from

                        avg_change_data.append(avg_change_row)

                    if avg_change_data:
                        avg_change_df = pd.DataFrame(avg_change_data)
                        numeric_cols = [col for col in avg_change_df.columns if col != "变化"]
                        for column in numeric_cols:
                            avg_change_df[column] = avg_change_df[column].apply(
                                lambda value: 0.0
                                if pd.notna(value) and abs(round(value, 3)) < 0.001
                                else round(value, 3) if pd.notna(value) else value
                            )

                        for _, row in avg_change_df.iterrows():
                            st.markdown(f"**{row['变化']}**")
                            if not numeric_cols:
                                continue
                            cols = st.columns(len(numeric_cols))
                            for idx, column in enumerate(numeric_cols):
                                if column not in row or pd.isna(row[column]):
                                    continue
                                value = row[column]
                                if "(W)" in column:
                                    unit = "W"
                                    label = column.replace("(W)", "").strip()
                                elif "(%)" in column:
                                    unit = "%"
                                    label = column.replace("(%)", "").strip()
                                elif "(V)" in column:
                                    unit = "V"
                                    label = column.replace("(V)", "").strip()
                                elif "(nm)" in column:
                                    unit = "nm"
                                    label = column.replace("(nm)", "").strip()
                                else:
                                    unit = ""
                                    label = column
                                with cols[idx]:
                                    st.metric(
                                        label=label,
                                        value=f"{abs(value):.3f}{unit}",
                                        delta=f"{value:+.3f}{unit}",
                                        delta_color="normal",
                                    )
                            st.markdown("---")

                st.markdown("---")

            extraction_state = st.session_state.get(EXTRACTION_STATE_KEY)
            result_df_for_analysis = None
            if extraction_state and extraction_state.get("result_df") is not None:
                result_df_for_analysis = extraction_state["result_df"]

            analysis_columns = [
                POWER_COLUMN,
                VOLTAGE_COLUMN,
                EFFICIENCY_COLUMN,
                LAMBDA_COLUMN,
                SHIFT_COLUMN,
            ]
            available_metrics = (
                [column for column in analysis_columns if column in result_df_for_analysis.columns]
                if result_df_for_analysis is not None
                else []
            )
            per_type_records: List[Dict[str, Any]] = []

            if result_df_for_analysis is not None and not result_df_for_analysis.empty:
                if available_metrics and TEST_TYPE_COLUMN in result_df_for_analysis.columns:
                    for test_type, group in result_df_for_analysis.groupby(TEST_TYPE_COLUMN):
                        for column in available_metrics:
                            series = pd.to_numeric(group[column], errors="coerce").dropna()
                            if series.empty:
                                continue
                            per_type_records.append(
                                {
                                    "站别": test_type,
                                    "指标": column,
                                    "数量": int(series.count()),
                                    "均值": round(series.mean(), 3),
                                    "中位数": round(series.median(), 3),
                                    "标准差": round(series.std(ddof=1), 3) if series.count() > 1 else 0.0,
                                    "最小值": round(series.min(), 3),
                                    "最大值": round(series.max(), 3),
                                }
                            )

                if available_metrics:
                    with st.expander("📊 指标分析", expanded=True):
                        if TEST_TYPE_COLUMN in result_df_for_analysis.columns:
                            available_test_types = [
                                t for t in SANITIZED_PLOT_ORDER if t in result_df_for_analysis[TEST_TYPE_COLUMN].unique()
                            ]
                            if available_test_types:
                                test_type_options = ["全部"] + available_test_types
                                default_index = len(test_type_options) - 1
                                selected_test_type = st.selectbox(
                                    "选择站别进行统计",
                                    options=test_type_options,
                                    index=default_index,
                                    key="stats_test_type_select",
                                )
                                if selected_test_type == "全部":
                                    numeric_data = result_df_for_analysis[available_metrics].apply(
                                        pd.to_numeric, errors="coerce"
                                    )
                                    st.markdown("### 📈 全部数据统计")
                                else:
                                    selected_test_df = result_df_for_analysis[
                                        result_df_for_analysis[TEST_TYPE_COLUMN] == selected_test_type
                                    ]
                                    numeric_data = selected_test_df[available_metrics].apply(pd.to_numeric, errors="coerce")
                                    st.markdown(f"### 📈 {selected_test_type} 站数据统计")
                            else:
                                numeric_data = result_df_for_analysis[available_metrics].apply(
                                    pd.to_numeric, errors="coerce"
                                )
                                st.markdown("### 📈 全部数据统计")
                        else:
                            numeric_data = result_df_for_analysis[available_metrics].apply(pd.to_numeric, errors="coerce")
                            st.markdown("### 📈 全部数据统计")

                        counts = numeric_data.notna().sum()
                        overall_summary = pd.DataFrame(
                            {
                                "数量": counts,
                                "均值": numeric_data.mean(),
                                "中位数": numeric_data.median(),
                                "标准差": numeric_data.std(ddof=1),
                                "最小值": numeric_data.min(),
                                "最大值": numeric_data.max(),
                            }
                        )
                        overall_summary["数量"] = overall_summary["数量"].astype("Int64")
                        overall_summary["标准差"] = overall_summary["标准差"].fillna(0.0)
                        summary_cols = ["均值", "中位数", "标准差", "最小值", "最大值"]
                        overall_summary[summary_cols] = overall_summary[summary_cols].round(3)
                        overall_summary.index.name = "指标"
                        styled_summary = overall_summary.style.format(
                            {col: "{:.3f}" for col in summary_cols}
                        )
                        st.dataframe(styled_summary, use_container_width=True)
                else:
                    st.info("按站别统计缺少有效的数值列")

                if per_type_records:
                    with st.expander("📋 按站别详细统计", expanded=False):
                        ordered_cols = ["站别", "指标", "数量", "均值", "中位数", "标准差", "最小值", "最大值"]
                        per_type_df = pd.DataFrame(per_type_records)[ordered_cols]
                        unique_metrics = per_type_df["指标"].unique()

                        for metric in unique_metrics:
                            metric_data = per_type_df[per_type_df["指标"] == metric].copy()
                            metric_data = metric_data.drop(columns=["指标"])
                            metric_data["__order"] = metric_data["站别"].map(SANITIZED_ORDER_LOOKUP)
                            metric_data = metric_data.sort_values("__order").drop(columns=["__order"])
                            metric_data = metric_data.set_index("站别")

                            st.markdown(f"#### 🔹 {metric}")
                            styled_metric = metric_data.style.format(
                                {
                                    "均值": "{:.3f}",
                                    "中位数": "{:.3f}",
                                    "标准差": "{:.3f}",
                                    "最小值": "{:.3f}",
                                    "最大值": "{:.3f}",
                                }
                            )
                            st.dataframe(styled_metric, use_container_width=True)

                            if len(metric_data) > 1:
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.caption("均值对比")
                                    st.bar_chart(metric_data["均值"], use_container_width=True)
                                with col2:
                                    st.caption("标准差对比")
                                    st.bar_chart(metric_data["标准差"], use_container_width=True)
            else:
                st.info("无可用数据")
        else:
            st.info("请先抽取数据")

    # 箱线图分析
    if st.session_state.get('show_boxplot', False):
        lvi_plot_sources = st.session_state.get('lvi_plot_sources')
        if lvi_plot_sources:
            st.markdown('---')
            st.markdown('<div id="boxplot"></div>', unsafe_allow_html=True)
            trigger_scroll_if_needed("boxplot")
            st.subheader("📊 箱线图分析")

            extraction_state = st.session_state.get(EXTRACTION_STATE_KEY)
            selected_currents: List[float] = []
            if extraction_state:
                selected_currents = extraction_state.get("current_points", []) or []

            all_data_for_boxplot: List[pd.DataFrame] = []
            for (shell_id, test_type), (df_full, df_selected) in lvi_plot_sources.items():
                if df_full is None or df_full.empty or CURRENT_COLUMN not in df_full.columns:
                    continue

                if df_selected is not None and not df_selected.empty:
                    base_df = df_selected.copy()
                else:
                    base_df = df_full.copy()
                    if selected_currents:
                        filtered_mask = pd.Series(False, index=base_df.index)
                        for current in selected_currents:
                            filtered_mask |= (base_df[CURRENT_COLUMN] - current).abs() <= CURRENT_TOLERANCE
                        filtered_df = base_df.loc[filtered_mask]
                        if not filtered_df.empty:
                            base_df = filtered_df.copy()
                        else:
                            max_current = base_df[CURRENT_COLUMN].max()
                            if pd.notna(max_current):
                                base_df = base_df.loc[(base_df[CURRENT_COLUMN] - max_current).abs() <= CURRENT_TOLERANCE]
                    else:
                        max_current = base_df[CURRENT_COLUMN].max()
                        if pd.notna(max_current):
                            base_df = base_df.loc[(base_df[CURRENT_COLUMN] - max_current).abs() <= CURRENT_TOLERANCE]

                if base_df.empty:
                    continue

                tagged = base_df.copy()
                tagged[TEST_TYPE_COLUMN] = test_type.replace("测试", "")
                tagged[SHELL_COLUMN] = shell_id
                all_data_for_boxplot.append(tagged)

            if all_data_for_boxplot:
                combined_boxplot_df = pd.concat(all_data_for_boxplot, ignore_index=True)
                rth_plot_sources = st.session_state.get('rth_plot_sources', {})
                if rth_plot_sources:
                    rth_data_list: List[pd.DataFrame] = []
                    for (shell_id, test_type), rth_df in rth_plot_sources.items():
                        if rth_df is None or rth_df.empty or CURRENT_COLUMN not in rth_df.columns:
                            continue
                        rth_temp = rth_df.copy()
                        if selected_currents:
                            mask = pd.Series(False, index=rth_temp.index)
                            for current in selected_currents:
                                mask |= (rth_temp[CURRENT_COLUMN] - current).abs() <= CURRENT_TOLERANCE
                            filtered_rth = rth_temp.loc[mask]
                            if filtered_rth.empty:
                                rth_max = rth_temp[CURRENT_COLUMN].max()
                                if pd.notna(rth_max):
                                    filtered_rth = rth_temp.loc[(rth_temp[CURRENT_COLUMN] - rth_max).abs() <= CURRENT_TOLERANCE]
                            rth_temp = filtered_rth
                        else:
                            rth_max = rth_temp[CURRENT_COLUMN].max()
                            if pd.notna(rth_max):
                                rth_temp = rth_temp.loc[(rth_temp[CURRENT_COLUMN] - rth_max).abs() <= CURRENT_TOLERANCE]

                        if rth_temp.empty:
                            continue

                        rth_temp[TEST_TYPE_COLUMN] = test_type.replace("测试", "")
                        rth_temp[SHELL_COLUMN] = shell_id
                        keep_cols = [SHELL_COLUMN, TEST_TYPE_COLUMN, CURRENT_COLUMN]
                        if LAMBDA_COLUMN in rth_temp.columns:
                            keep_cols.append(LAMBDA_COLUMN)
                        if SHIFT_COLUMN in rth_temp.columns:
                            keep_cols.append(SHIFT_COLUMN)
                        rth_data_list.append(rth_temp[keep_cols])

                    if rth_data_list:
                        rth_combined = pd.concat(rth_data_list, ignore_index=True)
                        cols_to_drop = [col for col in (LAMBDA_COLUMN, SHIFT_COLUMN) if col in combined_boxplot_df.columns]
                        if cols_to_drop:
                            combined_boxplot_df = combined_boxplot_df.drop(columns=cols_to_drop)
                        combined_boxplot_df = pd.merge(
                            combined_boxplot_df,
                            rth_combined,
                            on=[SHELL_COLUMN, TEST_TYPE_COLUMN, CURRENT_COLUMN],
                            how="outer",
                        )

                has_lambda = (
                    LAMBDA_COLUMN in combined_boxplot_df.columns
                    and combined_boxplot_df[LAMBDA_COLUMN].notna().any()
                )
                has_shift = (
                    SHIFT_COLUMN in combined_boxplot_df.columns
                    and combined_boxplot_df[SHIFT_COLUMN].notna().any()
                )

                tab_names = ["功率", "效率", "电压"]
                if has_lambda:
                    tab_names.append("波长")
                if has_shift:
                    tab_names.append("波长Shift")

                boxplot_tabs = st.tabs(tab_names)

                def _render_boxplot(data: pd.DataFrame, value_col: str, value_label: str, transform=None):
                    if transform:
                        data = data.copy()
                        data[value_col] = transform(data[value_col])
                    data = data.dropna()
                    if data.empty:
                        st.info(f"无{value_label}数据")
                        return

                    station_counts = data.groupby(TEST_TYPE_COLUMN).size()
                    stations_with_enough = station_counts[station_counts >= 2].index.tolist()
                    variance_threshold = 1e-10
                    stations_with_data: List[str] = []
                    stations_no_variance: List[str] = []
                    for station in stations_with_enough:
                        std_val = data[data[TEST_TYPE_COLUMN] == station][value_col].std()
                        if std_val > variance_threshold:
                            stations_with_data.append(station)
                        else:
                            stations_no_variance.append(station)

                    if stations_with_data:
                        filtered = data[data[TEST_TYPE_COLUMN].isin(stations_with_data)].copy()
                        filtered["__order"] = filtered[TEST_TYPE_COLUMN].map(SANITIZED_ORDER_LOOKUP)
                        filtered = filtered.sort_values("__order").drop(columns=["__order"])
                        present_stations = [s for s in SANITIZED_PLOT_ORDER if s in stations_with_data]
                        extras = [s for s in stations_with_data if s not in present_stations]
                        present_stations.extend(extras)
                        present_colors = [STATION_COLORS.get(s, "#000084") for s in present_stations]

                        chart = (
                            alt.Chart(filtered)
                            .mark_boxplot(extent="min-max", size=50)
                            .encode(
                                x=alt.X(
                                    f"{TEST_TYPE_COLUMN}:N",
                                    title="Station",
                                    sort=present_stations,
                                    axis=alt.Axis(labelAngle=-45),
                                ),
                                y=alt.Y(f"{value_col}:Q", title=value_label, scale=alt.Scale(zero=False)),
                                color=alt.Color(
                                    f"{TEST_TYPE_COLUMN}:N",
                                    legend=None,
                                    scale=alt.Scale(domain=present_stations, range=present_colors),
                                ),
                                tooltip=[
                                    alt.Tooltip(f"{TEST_TYPE_COLUMN}:N", title="Station"),
                                    alt.Tooltip("lower_whisker:Q", title="最小值", format=".3f"),
                                    alt.Tooltip("lower_box:Q", title="下四分位数", format=".3f"),
                                    alt.Tooltip("middle:Q", title="中位数", format=".3f"),
                                    alt.Tooltip("upper_box:Q", title="上四分位数", format=".3f"),
                                    alt.Tooltip("upper_whisker:Q", title="最大值", format=".3f"),
                                ],
                            )
                            .properties(height=500, title=f"各站别{value_label}分布箱线图")
                            .configure_title(fontSize=16, anchor="middle")
                        )
                        st.altair_chart(chart, use_container_width=True)

                        # ---------------------------------------------------------
                        # 新增：统计分析 (衰减百分比 & T-test)
                        # ---------------------------------------------------------
                        if len(present_stations) > 1:
                            # 确保统计库已加载
                            _ensure_prediction_libs_loaded()
                            
                            stats_results = []
                            
                            # 按顺序两两比较
                            for i in range(1, len(present_stations)):
                                curr_name = present_stations[i]
                                prev_name = present_stations[i-1]
                                
                                curr_series = filtered[filtered[TEST_TYPE_COLUMN] == curr_name][value_col]
                                prev_series = filtered[filtered[TEST_TYPE_COLUMN] == prev_name][value_col]
                                
                                if curr_series.empty or prev_series.empty:
                                    continue
                                
                                curr_mean = curr_series.mean()
                                prev_mean = prev_series.mean()
                                
                                # 1. 计算变化百分比
                                if prev_mean != 0:
                                    pct_change = (curr_mean - prev_mean) / abs(prev_mean) * 100
                                else:
                                    pct_change = np.nan
                                
                                # 2. T-test (Welch's t-test, 不假设方差相等)
                                p_value = np.nan
                                sig_label = "N/A"
                                if HAS_PREDICTION_LIBS and stats is not None:
                                    try:
                                        # nan_policy='omit' to be safe, though we dropped na earlier
                                        t_stat, p_val = stats.ttest_ind(
                                            curr_series, 
                                            prev_series, 
                                            equal_var=False, 
                                            nan_policy='omit'
                                        )
                                        p_value = p_val
                                        
                                        if p_val < 0.001:
                                            sig_label = "***"
                                        elif p_val < 0.01:
                                            sig_label = "**"
                                        elif p_val < 0.05:
                                            sig_label = "*"
                                        else:
                                            sig_label = "ns" # not significant
                                    except Exception:
                                        pass
                                
                                stats_results.append({
                                    "比较项": f"{curr_name} vs {prev_name}",
                                    "前序均值": prev_mean,
                                    "当前均值": curr_mean,
                                    "变化幅度(%)": pct_change,
                                    "P值": p_value,
                                    "显著性": sig_label
                                })
                            
                            if stats_results:
                                st.write("#### 📉 统计分析 (T-test)")
                                st.caption("注：显著性标记 ***(p<0.001), **(p<0.01), *(p<0.05), ns(无显著差异)")
                                
                                df_stats = pd.DataFrame(stats_results)
                                
                                # 格式化显示
                                # 使用 Styler 进行格式化，或者直接处理数据
                                # 这里为了简单直接处理数据为字符串用于展示，保留原始值用于计算（如果需要）
                                display_df = df_stats.copy()
                                
                                display_df["前序均值"] = display_df["前序均值"].apply(lambda x: f"{x:.4f}")
                                display_df["当前均值"] = display_df["当前均值"].apply(lambda x: f"{x:.4f}")
                                display_df["变化幅度(%)"] = display_df["变化幅度(%)"].apply(
                                    lambda x: f"{x:+.2f}%" if pd.notnull(x) else "N/A"
                                )
                                display_df["P值"] = display_df["P值"].apply(
                                    lambda x: f"{x:.4e}" if pd.notnull(x) else "N/A"
                                )
                                
                                st.table(display_df)
                        # ---------------------------------------------------------

                        stations_insufficient = station_counts[station_counts < 2].index.tolist()
                        warnings: List[str] = []
                        if stations_insufficient:
                            warnings.append(f"数据点不足（至少需要 2 个壳体）：{', '.join(stations_insufficient)}")
                        if stations_no_variance:
                            warnings.append(f"数据无变化：{', '.join(stations_no_variance)}")
                        if warnings:
                            st.caption("⚠️ " + "；".join(warnings))
                    else:
                        if stations_no_variance:
                            st.info(f"以下站别数据无变化：{', '.join(stations_no_variance)}")
                        else:
                            st.info("各站别数据点不足（至少需要 2 个壳体的数据）")

                tab_idx = 0
                with boxplot_tabs[tab_idx]:
                    tab_idx += 1
                    _render_boxplot(
                        combined_boxplot_df[[TEST_TYPE_COLUMN, POWER_COLUMN]].copy(),
                        POWER_COLUMN,
                        "功率(W)",
                    )

                with boxplot_tabs[tab_idx]:
                    tab_idx += 1
                    efficiency_data = combined_boxplot_df[[TEST_TYPE_COLUMN, EFFICIENCY_COLUMN]].copy()
                    _render_boxplot(
                        efficiency_data,
                        EFFICIENCY_COLUMN,
                        "效率(%)",
                        transform=lambda series: pd.to_numeric(series, errors="coerce") * 100,
                    )

                with boxplot_tabs[tab_idx]:
                    tab_idx += 1
                    _render_boxplot(
                        combined_boxplot_df[[TEST_TYPE_COLUMN, VOLTAGE_COLUMN]].copy(),
                        VOLTAGE_COLUMN,
                        "电压(V)",
                    )

                if has_lambda:
                    with boxplot_tabs[tab_idx]:
                        tab_idx += 1
                        _render_boxplot(
                            combined_boxplot_df[[TEST_TYPE_COLUMN, LAMBDA_COLUMN]].copy(),
                            LAMBDA_COLUMN,
                            "波长(nm)",
                        )

                if has_shift:
                    with boxplot_tabs[tab_idx]:
                        _render_boxplot(
                            combined_boxplot_df[[TEST_TYPE_COLUMN, SHIFT_COLUMN]].copy(),
                            SHIFT_COLUMN,
                            "波长Shift(nm)",
                        )
            else:
                st.info("无可用数据")
            st.markdown('---')
        else:
            st.info("请先抽取数据")

    if st.session_state.get('show_single_analysis', False):
        st.markdown('<div id="single"></div>', unsafe_allow_html=True)
        trigger_scroll_if_needed("single")
        extraction_state = st.session_state.get(EXTRACTION_STATE_KEY)
        if not extraction_state:
            show_toast("请先抽取数据后再进行分析", icon="⚠️")
        else:
            folder_entries = extraction_state["folder_entries"]
            lvi_plot_sources = st.session_state.get('lvi_plot_sources', {})
            
            if len(folder_entries) != 1:
                show_toast("单壳体分析仅支持单个壳体号，请调整输入", icon="⚠️")
            else:
                shell_id = folder_entries[0]
                plotted_any = False
                st.subheader("电流-功率-电光效率曲线")

                available_entries = []
                for test_type in PLOT_ORDER:
                    data_tuple = lvi_plot_sources.get((shell_id, test_type))
                    if data_tuple is None:
                        continue
                    df_full, df_selected = data_tuple
                    if df_full is None or df_full.empty:
                        continue
                    plot_df = df_full.dropna(subset=[CURRENT_COLUMN, POWER_COLUMN, EFFICIENCY_COLUMN])
                    if plot_df.empty:
                        continue
                    available_entries.append((test_type, df_full, df_selected, plot_df))

                if available_entries:
                    tab_labels = [entry[0].replace("测试", "") for entry in available_entries]
                    tabs = st.tabs(tab_labels)
                    for tab, (test_type, df_full, df_selected, plot_df) in zip(tabs, available_entries):
                        with tab:
                            chart = build_single_shell_dual_metric_chart(
                                plot_df,
                                df_selected,
                                shell_id,
                                test_type,
                            )
                            if chart is not None:
                                st.altair_chart(
                                    chart,
                                    theme="streamlit",
                                    use_container_width=True,
                                )
                                plotted_any = True
                            else:
                                st.info("无法生成趋势图表")
                else:
                    show_toast("未找到可用于绘制的站别数据", icon="⚠️")

                if not plotted_any:
                    show_toast("未找到可绘制的 LVI 数据", icon="⚠️")

main()
