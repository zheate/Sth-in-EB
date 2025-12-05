# 拟合模型模块
"""
包含效率拟合等数学模型函数和预测库的延迟加载逻辑
"""

import importlib
import importlib.util
from typing import Callable, Dict, List, Optional, Tuple, Any
import numpy as np

# ============================================================================
# 可选依赖的延迟加载
# ============================================================================

# 全局变量用于存储延迟加载的模块
stats: Optional[Any] = None
PolynomialFeatures: Optional[Any] = None
LinearRegression: Optional[Any] = None
HuberRegressor: Optional[Any] = None
RANSACRegressor: Optional[Any] = None
_PREDICTION_LIBS_LOADED: bool = False

# 检查预测库是否可用
HAS_PREDICTION_LIBS: bool = all(
    importlib.util.find_spec(name) is not None
    for name in ("scipy.stats", "sklearn.preprocessing", "sklearn.linear_model")
)


def ensure_prediction_libs_loaded() -> bool:
    """
    延迟导入可选的预测依赖库。
    
    Returns:
        bool: 如果所有依赖都成功加载则返回 True
    """
    global stats, PolynomialFeatures, LinearRegression, HuberRegressor, RANSACRegressor
    global _PREDICTION_LIBS_LOADED, HAS_PREDICTION_LIBS

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


def get_stats_module():
    """获取 scipy.stats 模块（需先调用 ensure_prediction_libs_loaded）"""
    return stats


def get_polynomial_features():
    """获取 PolynomialFeatures 类"""
    return PolynomialFeatures


def get_linear_regression():
    """获取 LinearRegression 类"""
    return LinearRegression


# ============================================================================
# 效率拟合模型函数
# ============================================================================

def rational_1_2(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """
    有理函数模型: (a*x) / (1 + b*x + c*x²)
    适用于效率随电流先升后降的曲线
    """
    return (a * x) / (1.0 + b * x + c * (x ** 2))


def hill(x: np.ndarray, Emax: float, K: float, n: float) -> np.ndarray:
    """
    Hill 方程模型: Emax * x^n / (K^n + x^n)
    经典的饱和曲线模型
    """
    with np.errstate(invalid='ignore', over='ignore'):
        return Emax * (x ** n) / (K ** n + x ** n)


def hill_droop(x: np.ndarray, Emax: float, K: float, n: float, d: float) -> np.ndarray:
    """
    带衰减的 Hill 方程: Hill(x) / (1 + d*x)
    适用于高电流时效率下降的情况
    """
    with np.errstate(invalid='ignore', over='ignore'):
        return (Emax * (x ** n) / (K ** n + x ** n)) / (1.0 + d * x)


def exp_sat(x: np.ndarray, Emax: float, k: float) -> np.ndarray:
    """
    指数饱和模型: Emax * (1 - exp(-k*x))
    简单的饱和曲线
    """
    return Emax * (1 - np.exp(-k * x))


# ============================================================================
# 效率拟合候选模型配置
# ============================================================================

# 格式: {模型名: (函数, 初始参数, 显示名称)}
EFFICIENCY_MODELS: Dict[str, Tuple[Callable, List[float], str]] = {
    "hill_droop": (hill_droop, [60, 5, 2, 0.02], "Hill-Droop"),
    "hill": (hill, [60, 5, 2], "Hill"),
    "rational_1_2": (rational_1_2, [5, 0.1, 0.01], "有理函数"),
    "exp_sat": (exp_sat, [60, 0.2], "指数饱和"),
}


# ============================================================================
# 拟合辅助函数
# ============================================================================

def fit_efficiency_model(
    x: np.ndarray,
    y: np.ndarray,
) -> Optional[Tuple[str, Callable, np.ndarray, Dict[str, float]]]:
    """
    自动选择最佳效率拟合模型。
    
    Args:
        x: 电流数据
        y: 效率数据
        
    Returns:
        Tuple of (模型名, 模型函数, 拟合参数, 统计指标) 或 None
    """
    if not ensure_prediction_libs_loaded():
        return None
    
    from scipy.optimize import curve_fit
    from math import log
    
    best_model_name: Optional[str] = None
    best_popt: Optional[np.ndarray] = None
    best_aic: float = float('inf')
    best_func: Optional[Callable] = None
    model_results: Dict[str, Dict] = {}
    
    for model_name, (func, p0, display_name) in EFFICIENCY_MODELS.items():
        try:
            popt, _ = curve_fit(func, x, y, p0=p0, maxfev=10000)
            yhat = func(x, *popt)
            resid = y - yhat
            rss = float(np.sum(resid ** 2))
            n = len(y)
            k = len(popt)
            mse = rss / n
            rmse = float(np.sqrt(mse))
            tss = float(np.sum((y - np.mean(y)) ** 2))
            r2 = float(1 - rss / tss) if tss > 0 else 0.0
            aic = float(n * log(rss / n) + 2 * k) if rss > 0 else float('inf')
            
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
    
    if best_func is None or best_model_name is None or best_popt is None:
        return None
    
    return (
        best_model_name,
        best_func,
        best_popt,
        model_results[best_model_name]
    )


def fit_polynomial_model(
    x: np.ndarray,
    y: np.ndarray,
    degree: int = 2,
    auto_select: bool = False,
    max_degree: int = 3,
) -> Optional[Tuple[Any, Any, int, float]]:
    """
    多项式拟合模型。
    
    Args:
        x: 自变量数据
        y: 因变量数据
        degree: 多项式阶数
        auto_select: 是否自动选择最佳阶数
        max_degree: 自动选择时的最大阶数
        
    Returns:
        Tuple of (PolynomialFeatures, LinearRegression, 选择的阶数, R²) 或 None
    """
    if not ensure_prediction_libs_loaded():
        return None
    
    PolyFeatures = get_polynomial_features()
    LinReg = get_linear_regression()
    
    if PolyFeatures is None or LinReg is None:
        return None
    
    degrees = list(range(1, max_degree + 1)) if auto_select else [degree]
    best: Optional[Tuple[float, int, Any, Any, float]] = None
    
    for deg in degrees:
        poly = PolyFeatures(degree=deg)
        X_poly = poly.fit_transform(x.reshape(-1, 1))
        model = LinReg()
        model.fit(X_poly, y)
        y_fitted = model.predict(X_poly)
        resid = y - y_fitted
        rss = float(np.sum(resid ** 2))
        n = len(y)
        k = deg + 1
        aic = float(n * np.log(rss / n) + 2 * k) if rss > 0 else float('inf')
        r2 = float(model.score(X_poly, y))
        
        if best is None or aic < best[0]:
            best = (aic, deg, poly, model, r2)
    
    if best is None:
        return None
    
    _, chosen_deg, poly, model, r2 = best
    return (poly, model, chosen_deg, r2)
