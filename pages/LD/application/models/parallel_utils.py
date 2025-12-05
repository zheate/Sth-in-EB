"""并行计算工具模块

利用多核 CPU 加速计算密集型任务
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from functools import partial
from typing import Any, Callable, List, Optional, Tuple, TypeVar

import numpy as np

# 获取 CPU 核心数，默认使用逻辑处理器数量的 75%
CPU_COUNT = os.cpu_count() or 4
DEFAULT_WORKERS = max(1, int(CPU_COUNT * 0.75))

T = TypeVar('T')


def parallel_map(
    func: Callable[..., T],
    items: List[Any],
    *args,
    max_workers: Optional[int] = None,
    use_threads: bool = True,
    **kwargs
) -> List[T]:
    """并行执行函数映射
    
    Args:
        func: 要执行的函数
        items: 要处理的项目列表
        *args: 传递给 func 的额外位置参数
        max_workers: 最大工作线程/进程数，默认为 CPU 核心数的 75%
        use_threads: True 使用线程池，False 使用进程池
        **kwargs: 传递给 func 的额外关键字参数
        
    Returns:
        结果列表，顺序与输入 items 相同
    """
    if not items:
        return []
    
    workers = max_workers or DEFAULT_WORKERS
    workers = min(workers, len(items))  # 不需要比任务数更多的工作器
    
    # 如果只有一个任务或一个工作器，直接顺序执行
    if len(items) == 1 or workers == 1:
        return [func(item, *args, **kwargs) for item in items]
    
    # 创建偏函数，绑定额外参数
    if args or kwargs:
        worker_func = partial(func, *args, **kwargs) if not args else lambda x: func(x, *args, **kwargs)
    else:
        worker_func = func
    
    Executor = ThreadPoolExecutor if use_threads else ProcessPoolExecutor
    results = [None] * len(items)
    
    with Executor(max_workers=workers) as executor:
        # 提交所有任务并记录索引
        future_to_idx = {
            executor.submit(worker_func, item): idx 
            for idx, item in enumerate(items)
        }
        
        # 收集结果
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            results[idx] = future.result()
    
    return results


def parallel_compute_intensity(
    gaussian_beams_f: List,
    gaussian_beams_s: List,
    lens_f_list: List,
    lens_s_list: List,
    source_intensity_f: List,
    source_intensity_s: List,
    w_f: float,
    w_s: float,
    center_f: float,
    center_s: float,
    z_list_f: List[float],
    z_list_s: List[float],
    max_workers: Optional[int] = None,
) -> List[np.ndarray]:
    """并行计算光强分布
    
    Args:
        gaussian_beams_f: 快轴高斯光束列表
        gaussian_beams_s: 慢轴高斯光束列表
        lens_f_list: 快轴透镜列表
        lens_s_list: 慢轴透镜列表
        source_intensity_f: 快轴光源强度列表
        source_intensity_s: 慢轴光源强度列表
        w_f: 快轴宽度
        w_s: 慢轴宽度
        center_f: 快轴中心
        center_s: 慢轴中心
        z_list_f: 快轴 z 位置列表
        z_list_s: 慢轴 z 位置列表
        max_workers: 最大工作线程数
        
    Returns:
        光强分布列表
    """
    def compute_single(idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        relationship_tuple_f = (lens_f_list[idx], (w_f, (center_f, z_list_f[idx])))
        y, intensity_f = gaussian_beams_f[idx].intensity_distribution_calculate(relationship_tuple_f)
        intensity_f = intensity_f * source_intensity_f[idx]
        
        relationship_tuple_s = (lens_s_list[idx], (w_s, (center_s, z_list_s[idx])))
        x, intensity_s = gaussian_beams_s[idx].intensity_distribution_calculate(relationship_tuple_s)
        intensity_s = intensity_s * source_intensity_s[idx]
        
        intensity_s_grid, intensity_f_grid = np.meshgrid(intensity_s, intensity_f)
        return x, y, intensity_f_grid * intensity_s_grid
    
    workers = max_workers or DEFAULT_WORKERS
    n = len(gaussian_beams_f)
    
    if n <= 2 or workers == 1:
        # 小任务量时顺序执行更快
        results = [compute_single(i) for i in range(n)]
    else:
        results = parallel_map(compute_single, list(range(n)), max_workers=workers)
    
    return results


def parallel_beam_calculations(
    gaussian_beams: List,
    lens_list: List,
    calc_func: Callable,
    max_workers: Optional[int] = None,
) -> List:
    """并行执行光束计算
    
    Args:
        gaussian_beams: 高斯光束列表
        lens_list: 透镜列表
        calc_func: 计算函数，接受 (gaussian_beam, lens, idx) 参数
        max_workers: 最大工作线程数
        
    Returns:
        计算结果列表
    """
    def compute_single(idx: int):
        return calc_func(gaussian_beams[idx], lens_list[idx], idx)
    
    return parallel_map(compute_single, list(range(len(gaussian_beams))), max_workers=max_workers)


def vectorized_energy_ratio(
    x: np.ndarray,
    y: np.ndarray,
    intensity: np.ndarray,
    x_center: float,
    y_center: float,
    radii: np.ndarray,
) -> np.ndarray:
    """向量化计算多个半径的能量比例
    
    Args:
        x: x 坐标网格
        y: y 坐标网格
        intensity: 光强分布
        x_center: 圆心 x 坐标
        y_center: 圆心 y 坐标
        radii: 半径数组
        
    Returns:
        各半径对应的能量比例数组
    """
    # 计算到圆心的距离平方
    dist_sq = (x - x_center)**2 + (y - y_center)**2
    total_intensity = intensity.sum()
    
    if total_intensity == 0:
        return np.zeros(len(radii))
    
    # 向量化计算所有半径的能量比例
    results = np.zeros(len(radii))
    for i, r in enumerate(radii):
        mask = dist_sq <= r**2
        results[i] = intensity[mask].sum() / total_intensity
    
    return results


class ParallelConfig:
    """并行计算配置"""
    
    def __init__(
        self,
        enabled: bool = True,
        max_workers: Optional[int] = None,
        use_threads: bool = True,
    ):
        """
        Args:
            enabled: 是否启用并行计算
            max_workers: 最大工作线程/进程数
            use_threads: True 使用线程池，False 使用进程池
        """
        self.enabled = enabled
        self.max_workers = max_workers or DEFAULT_WORKERS
        self.use_threads = use_threads
    
    @classmethod
    def auto(cls) -> 'ParallelConfig':
        """自动配置，根据 CPU 核心数优化"""
        return cls(
            enabled=CPU_COUNT >= 4,
            max_workers=DEFAULT_WORKERS,
            use_threads=True,
        )
    
    @classmethod
    def disabled(cls) -> 'ParallelConfig':
        """禁用并行计算"""
        return cls(enabled=False, max_workers=1)
    
    @classmethod
    def max_performance(cls) -> 'ParallelConfig':
        """最大性能配置，使用所有 CPU 核心"""
        return cls(
            enabled=True,
            max_workers=CPU_COUNT,
            use_threads=True,
        )


# 全局默认配置
_default_config = ParallelConfig.auto()


def get_default_config() -> ParallelConfig:
    """获取默认并行配置"""
    return _default_config


def set_default_config(config: ParallelConfig) -> None:
    """设置默认并行配置"""
    global _default_config
    _default_config = config

