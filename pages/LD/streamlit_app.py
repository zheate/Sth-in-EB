from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from application.models.laser_diode_calculation import LaserDiodeCalculation
from application.models.parallel_utils import ParallelConfig
from application.models.parameters_conversion import (
    load_config_from_json,
    save_config_to_json,
    migrate_excel_to_json,
    parameters_convert,
    PARAM_DEFINITIONS,
    CONFIG_JSON_PATH,
)

ROOT = Path(__file__).parent
ASSETS_PATH = ROOT / 'assets'
CONFIG_JSON = Path(CONFIG_JSON_PATH)
SOURCE_EXCEL = ASSETS_PATH / 'LD光纤耦合参数.xlsx'

# 自定义颜色映射（类似 jet）
CUSTOM_COLORSCALE = [
    [0.0, 'rgb(255,255,255)'],
    [0.05, 'rgb(255,255,255)'],
    [0.1, 'rgb(0,0,255)'],
    [0.3, 'rgb(0,255,255)'],
    [0.5, 'rgb(0,255,0)'],
    [0.7, 'rgb(255,255,0)'],
    [0.85, 'rgb(255,128,0)'],
    [1.0, 'rgb(128,0,0)'],
]

# 参数标签映射
def load_param_labels_from_excel() -> dict:
    """从 Excel 文件读取参数标签，与 QT 版本保持一致"""
    try:
        df = pd.read_excel(SOURCE_EXCEL, header=None)
        labels = {}
        
        # 根据 PARAM_DEFINITIONS 的定义读取标签
        # 参数定义格式: (row, value_col, key, unit_col)
        param_defs = [
            # 第一列参数 (row, value_col=1, key, unit_col=2, label_col=0)
            (0, 1, 'wavelength', 2, 0),
            (1, 1, 'waist_f', 2, 0),
            (2, 1, 'divergence_angle_f', 2, 0),
            (3, 1, 'near_field_order_f', 2, 0),
            (4, 1, 'far_field_order_f', 2, 0),
            (5, 1, 'number_f', 2, 0),
            (6, 1, 'interval_f', 2, 0),
            (7, 1, 'astigmatism', 2, 0),
            (8, 1, 'waist_s', 2, 0),
            (9, 1, 'divergence_angle_s', 2, 0),
            (10, 1, 'near_field_order_s', 2, 0),
            (11, 1, 'far_field_order_s', 2, 0),
            (12, 1, 'number_s', 2, 0),
            (13, 1, 'interval_s', 2, 0),
            (14, 1, 'z_spatial_beam_combining_f', 2, 0),
            # 第二列参数 (row, value_col=4, key, unit_col=5, label_col=3)
            (0, 4, 'collimation_lens_effective_focal_length_f', 5, 3),
            (1, 4, 'collimation_lens_effective_focal_length_s', 5, 3),
            (2, 4, 'z_mirror_and_chip', 5, 3),
            (3, 4, 'z_polarized_beam_combining', 5, 3),
            (4, 4, 'z_spatial_beam_combining_s', 5, 3),
            (5, 4, 'coupling_lens_effective_focal_length_f', 5, 3),
            (6, 4, 'coupling_lens_effective_focal_length_s', 5, 3),
            (7, 4, 'z_coupling_lens_f_and_mirror', 5, 3),
            (8, 4, 'fiber_core_diameter', 5, 3),
            (9, 4, 'fiber_cladding_diameter', 5, 3),
            (10, 4, 'fiber_na', 5, 3),
            (11, 4, 'index_fiber_core', 5, 3),
            (12, 4, 'index_environment', 5, 3),
            (13, 4, 'fiber_coiling_radius', 5, 3),
        ]
        
        for row, value_col, key, unit_col, label_col in param_defs:
            try:
                label = str(df.iloc[row, label_col]) if pd.notna(df.iloc[row, label_col]) else key
                unit = str(df.iloc[row, unit_col]) if pd.notna(df.iloc[row, unit_col]) else ''
                labels[key] = (label, unit)
            except (IndexError, KeyError):
                # 如果读取失败，使用默认标签
                labels[key] = (key, '')
        
        return labels
    except Exception:
        # 如果无法读取 Excel，返回默认标签
        return None


# 默认标签（作为备用，与 Excel/QT 版本一致）
DEFAULT_PARAM_LABELS = {
    'wavelength': ('波长', 'um'),
    'waist_f': ('快轴束腰半径', 'um'),
    'divergence_angle_f': ('快轴发散半角', '°'),
    'near_field_order_f': ('快轴近场阶数', ''),
    'far_field_order_f': ('快轴远场阶数', ''),
    'number_f': ('COS数量', ''),
    'interval_f': ('芯片高度', 'mm'),
    'astigmatism': ('像散', 'um'),
    'waist_s': ('慢轴束腰半径', 'um'),
    'divergence_angle_s': ('慢轴发散半角', '°'),
    'near_field_order_s': ('慢轴近场阶数', ''),
    'far_field_order_s': ('慢轴远场阶数', ''),
    'number_s': ('慢轴堆叠数量', ''),
    'interval_s': ('慢轴堆叠间隔', 'mm'),
    'z_spatial_beam_combining_f': ('台阶间距', 'mm'),
    'collimation_lens_effective_focal_length_f': ('快轴准直镜焦距', 'mm'),
    'collimation_lens_effective_focal_length_s': ('慢轴准直镜焦距', 'mm'),
    'z_mirror_and_chip': ('反射镜距芯片距离', 'mm'),
    'z_polarized_beam_combining': ('偏振合束光程差', 'mm'),
    'z_spatial_beam_combining_s': ('慢轴空间合束光程差', 'mm'),
    'coupling_lens_effective_focal_length_f': ('快轴耦合镜焦距', 'mm'),
    'coupling_lens_effective_focal_length_s': ('慢轴耦合镜焦距', 'mm'),
    'z_coupling_lens_f_and_mirror': ('快轴耦合镜距第一反射镜', 'mm'),
    'fiber_core_diameter': ('光纤纤芯直径', 'um'),
    'fiber_cladding_diameter': ('光纤包层直径', 'um'),
    'fiber_na': ('光纤NA', ''),
    'index_fiber_core': ('纤芯折射率', ''),
    'index_environment': ('环境折射率', ''),
    'fiber_coiling_radius': ('光纤盘绕直径', 'mm'),
}

# 尝试从 Excel 读取标签，失败则使用默认值
PARAM_LABELS = load_param_labels_from_excel() or DEFAULT_PARAM_LABELS

PARAM_GROUPS = {
    'left': [
        'wavelength', 'waist_f', 'divergence_angle_f', 'near_field_order_f', 
        'far_field_order_f', 'number_f', 'interval_f', 'astigmatism',
        'waist_s', 'divergence_angle_s', 'near_field_order_s', 'far_field_order_s',
        'number_s', 'interval_s', 'z_spatial_beam_combining_f'
    ],
    'right': [
        'collimation_lens_effective_focal_length_f', 'collimation_lens_effective_focal_length_s',
        'z_mirror_and_chip', 'z_polarized_beam_combining', 'z_spatial_beam_combining_s',
        'coupling_lens_effective_focal_length_f', 'coupling_lens_effective_focal_length_s',
        'z_coupling_lens_f_and_mirror', 'fiber_core_diameter', 'fiber_cladding_diameter',
        'fiber_na', 'index_fiber_core', 'index_environment', 'fiber_coiling_radius'
    ]
}

INTEGER_PARAMS = {'number_f', 'number_s'}
HIGH_PRECISION_PARAMS = {'wavelength', 'index_fiber_core'}


@dataclass(frozen=True)
class ParameterField:
    key: str
    label: str
    unit: str
    decimals: int
    is_integer: bool


def ensure_config_exists() -> None:
    if not CONFIG_JSON.exists():
        if SOURCE_EXCEL.exists():
            try:
                migrate_excel_to_json(str(SOURCE_EXCEL), str(CONFIG_JSON))
                st.info('已从 Excel 文件迁移配置到 JSON 格式。')
            except Exception as e:
                st.error(f'无法迁移配置文件：{e}')
        else:
            st.error('配置文件不存在，请先创建 ld_config.json 或 LD光纤耦合参数.xlsx')


def load_config() -> Dict[str, Any] | None:
    if not CONFIG_JSON.exists():
        st.error('配置文件不存在，无法显示参数。')
        return None
    try:
        return load_config_from_json(str(CONFIG_JSON))
    except Exception as exc:
        st.error(f'加载配置文件失败：{exc}')
        return None


def generate_parameter_fields() -> Dict[str, ParameterField]:
    fields: Dict[str, ParameterField] = {}
    for _, _, key, _ in PARAM_DEFINITIONS:
        if key not in PARAM_LABELS:
            continue
        label, unit = PARAM_LABELS[key]
        display_label = f'{label} ({unit})' if unit else label
        decimals = 4 if key in HIGH_PRECISION_PARAMS else 3
        is_integer = key in INTEGER_PARAMS
        fields[key] = ParameterField(key=key, label=display_label, unit=unit,
                                      decimals=decimals, is_integer=is_integer)
    return fields


def render_parameter_inputs(config: Dict[str, Any]) -> Dict[str, float | int]:
    field_definitions = generate_parameter_fields()
    values: Dict[str, float | int] = {}
    
    left_params = PARAM_GROUPS['left']
    right_params = PARAM_GROUPS['right']
    
    # 使用两个主列布局，视觉上更整齐紧凑
    col1, col2 = st.columns(2, gap='small')
    
    def render_column_params(column, params):
        with column:
            for key in params:
                field = field_definitions.get(key)
                if field and key in config:
                    input_key = f'param_{key}'
                    default_value = config[key]['value']
                    
                    if input_key not in st.session_state:
                        st.session_state[input_key] = float(default_value)
                    
                    # 使用更紧凑的标签格式
                    if field.is_integer:
                        value = st.number_input(
                            field.label, value=int(st.session_state[input_key]),
                            step=1, min_value=1, max_value=1_000_000_000, key=input_key
                        )
                    else:
                        step = 0.0001 if field.decimals == 4 else 0.001
                        value = st.number_input(
                            field.label, value=float(st.session_state[input_key]),
                            format=f'%.{field.decimals}f', step=step, key=input_key
                        )
                    values[key] = value

    render_column_params(col1, left_params)
    render_column_params(col2, right_params)
    
    return values


def persist_config(config: Dict[str, Any], values: Dict[str, float | int]) -> bool:
    for key, value in values.items():
        if key in config:
            config[key]['value'] = value
    try:
        save_config_to_json(config, str(CONFIG_JSON))
        return True
    except Exception as e:
        st.error(f'保存配置失败：{e}')
        return False


def sync_config(config: Dict[str, Any], values: Dict[str, float | int]) -> Dict[str, Any] | None:
    updated_config = {k: dict(v) for k, v in config.items()}
    for key, value in values.items():
        if key in updated_config:
            updated_config[key]['value'] = value
    if persist_config(updated_config, values):
        return updated_config
    return None


def mask_intensity(data: np.ndarray, threshold_ratio: float = 0.05) -> np.ndarray:
    """掩码处理低强度区域"""
    data_max = data.max()
    if data_max == 0:
        return data
    masked = data.copy()
    masked[masked < threshold_ratio * data_max] = np.nan
    return masked


def create_circle_points(center_x: float, center_y: float, radius: float, n_points: int = 100):
    """创建圆形轨迹点"""
    theta = np.linspace(0, 2 * np.pi, n_points)
    x = center_x + radius * np.cos(theta)
    y = center_y + radius * np.sin(theta)
    return x, y


def plot_far_near_fields_matplotlib(data: Tuple) -> plt.Figure:
    """使用 Matplotlib 绘制远场和近场分布图"""
    x_far = data[0] * 1000  # 转换为 mm
    y_far = data[1] * 1000
    intensity_far = data[2]
    center_x_far = data[3] * 1000
    center_y_far = data[4] * 1000
    na_fiber_diameter = data[5] * 1000
    
    x_near = data[8] * 1000
    y_near = data[9] * 1000
    intensity_near = data[10]
    center_x_near = data[11] * 1000
    center_y_near = data[12] * 1000
    fiber_core_diameter = data[13] * 1000
    fiber_cladding_diameter = data[14] * 1000
    
    # 创建自定义颜色映射（与 CUSTOM_COLORSCALE 一致）
    colors = ['white', 'white', 'blue', 'cyan', 'green', 'yellow', 'orange', 'darkred']
    positions = [0.0, 0.05, 0.1, 0.3, 0.5, 0.7, 0.85, 1.0]
    cmap = LinearSegmentedColormap.from_list('custom', list(zip(positions, colors)))
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 远场热图
    ax1 = axes[0]
    masked_far = mask_intensity(intensity_far)
    ax1.pcolormesh(x_far, y_far, masked_far, cmap=cmap, shading='auto')
    
    # 远场 NA 圆
    circle_x, circle_y = create_circle_points(center_x_far, center_y_far, na_fiber_diameter / 2)
    ax1.plot(circle_x, circle_y, 'k-', linewidth=2)
    
    radius_far = na_fiber_diameter / 1.8
    ax1.set_xlim(center_x_far - radius_far, center_x_far + radius_far)
    ax1.set_ylim(center_y_far - radius_far, center_y_far + radius_far)
    ax1.set_xlabel('x (mm)')
    ax1.set_ylabel('y (mm)')
    ax1.set_title('Far Field')
    ax1.set_aspect('equal')
    
    # 近场热图 - 坐标归零（以中心为原点）
    ax2 = axes[1]
    masked_near = mask_intensity(intensity_near)
    # 将坐标偏移至以 (0, 0) 为中心
    x_near_centered = x_near - center_x_near
    y_near_centered = y_near - center_y_near
    ax2.pcolormesh(x_near_centered, y_near_centered, masked_near, cmap=cmap, shading='auto')
    
    # 近场纤芯圆（以原点为中心）
    circle_x, circle_y = create_circle_points(0, 0, fiber_core_diameter / 2)
    ax2.plot(circle_x, circle_y, 'k-', linewidth=2)
    
    # 近场包层圆（以原点为中心）
    circle_x, circle_y = create_circle_points(0, 0, fiber_cladding_diameter / 2)
    ax2.plot(circle_x, circle_y, 'k--', linewidth=2)
    
    radius_near = fiber_cladding_diameter / 1.8
    ax2.set_xlim(-radius_near, radius_near)
    ax2.set_ylim(-radius_near, radius_near)
    ax2.set_xlabel('x (mm)')
    ax2.set_ylabel('y (mm)')
    ax2.set_title('Near Field')
    ax2.set_aspect('equal')
    
    plt.tight_layout()
    return fig


def plot_trace_plotly(trace_f_list: List[List], trace_s_list: List[List]) -> go.Figure:
    """使用 Plotly 绘制光线追迹图"""
    fig = make_subplots(rows=1, cols=2, subplot_titles=['Fast Axis Trace', 'Slow Axis Trace'],
                        horizontal_spacing=0.1)
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    # 快轴追迹
    for idx, trace in enumerate(trace_f_list):
        z, center, outline1, outline2 = trace
        color = colors[idx % len(colors)]
        # 中心线
        fig.add_trace(go.Scatter(
            x=z, y=center, mode='lines', name=f'Ch{idx}',
            line=dict(color=color, width=1.5),
            legendgroup=f'fast_{idx}', showlegend=True
        ), row=1, col=1)
        # 上轮廓
        fig.add_trace(go.Scatter(
            x=z, y=outline1, mode='lines',
            line=dict(color=color, width=1, dash='dot'),
            legendgroup=f'fast_{idx}', showlegend=False
        ), row=1, col=1)
        # 下轮廓
        fig.add_trace(go.Scatter(
            x=z, y=outline2, mode='lines',
            line=dict(color=color, width=1, dash='dot'),
            legendgroup=f'fast_{idx}', showlegend=False
        ), row=1, col=1)
    
    # 慢轴追迹
    for idx, trace in enumerate(trace_s_list):
        z, center, outline1, outline2 = trace
        color = colors[idx % len(colors)]
        fig.add_trace(go.Scatter(
            x=z, y=center, mode='lines', name=f'Ch{idx}',
            line=dict(color=color, width=1.5),
            legendgroup=f'slow_{idx}', showlegend=False
        ), row=1, col=2)
        fig.add_trace(go.Scatter(
            x=z, y=outline1, mode='lines',
            line=dict(color=color, width=1, dash='dot'),
            legendgroup=f'slow_{idx}', showlegend=False
        ), row=1, col=2)
        fig.add_trace(go.Scatter(
            x=z, y=outline2, mode='lines',
            line=dict(color=color, width=1, dash='dot'),
            legendgroup=f'slow_{idx}', showlegend=False
        ), row=1, col=2)
    
    fig.update_xaxes(title_text='Position z (m)', row=1, col=1)
    fig.update_yaxes(title_text='Position t (m)', row=1, col=1)
    fig.update_xaxes(title_text='Position z (m)', row=1, col=2)
    fig.update_yaxes(title_text='Position t (m)', row=1, col=2)
    
    fig.update_layout(height=350, margin=dict(l=60, r=20, t=50, b=50),
                      legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5))
    return fig


def plot_lens_heatmap_plotly(
    x_list: List[np.ndarray], y_list: List[np.ndarray],
    intensity_list: List[np.ndarray], titles: List[str]
) -> go.Figure:
    """使用 Plotly 绘制透镜热图"""
    n = len(x_list)
    fig = make_subplots(rows=1, cols=n, subplot_titles=titles[:n], horizontal_spacing=0.08)
    
    for i, (x, y, intensity) in enumerate(zip(x_list, y_list, intensity_list)):
        # 热图
        fig.add_trace(go.Heatmap(
            x=x[0, :], y=y[:, 0], z=mask_intensity(intensity),
            colorscale=CUSTOM_COLORSCALE, showscale=(i == n - 1),
            colorbar=dict(title='强度', len=0.8) if i == n - 1 else None,
            hovertemplate='x: %{x:.4f}<br>y: %{y:.4f}<br>强度: %{z:.2e}<extra></extra>'
        ), row=1, col=i+1)
        
        # X 方向积分曲线
        x_1d = x[0, :]
        y_1d = y[:, 0]
        intensity_x = intensity.sum(axis=0)
        intensity_y = intensity.sum(axis=1)
        
        if intensity_x.max() > 0:
            intensity_x_norm = intensity_x / intensity_x.max() * (y_1d.max() - y_1d.min()) * 0.2 + y_1d.min()
            fig.add_trace(go.Scatter(
                x=x_1d, y=intensity_x_norm, mode='lines',
                line=dict(color='black', width=2), showlegend=False
            ), row=1, col=i+1)
        
        if intensity_y.max() > 0:
            intensity_y_norm = intensity_y / intensity_y.max() * (x_1d.max() - x_1d.min()) * 0.2 + x_1d.min()
            fig.add_trace(go.Scatter(
                x=intensity_y_norm, y=y_1d, mode='lines',
                line=dict(color='black', width=2), showlegend=False
            ), row=1, col=i+1)
        
        fig.update_xaxes(title_text='x', row=1, col=i+1)
        fig.update_yaxes(title_text='y', row=1, col=i+1)
    
    fig.update_layout(height=350, margin=dict(l=60, r=80, t=50, b=50))
    return fig


def run_full_calculation(config: Dict[str, Any]) -> Dict:
    """执行完整计算"""
    timing: Dict[str, float] = {}
    total_start = time.perf_counter()
    
    step_start = time.perf_counter()
    calc_data = parameters_convert(config)
    timing['参数转换'] = time.perf_counter() - step_start
    
    step_start = time.perf_counter()
    calculation = LaserDiodeCalculation(calc_data, parallel_config=ParallelConfig.max_performance())
    timing['初始化'] = time.perf_counter() - step_start

    results: Dict[str, object] = {
        'beam_spreading': calculation.m2_ratio_list,
        'beam_cutting': calculation.beam_cutting_energy_ratio_list,
        'fiber_na': config.get('fiber_na', {}).get('value', 0.22),  # 保存光纤NA用于过滤显示
    }
    
    step_start = time.perf_counter()
    results['far_near'] = calculation.na_and_coupling_calculate()
    timing['NA和耦合计算'] = time.perf_counter() - step_start
    
    step_start = time.perf_counter()
    results['trace'] = calculation.trace_calculate()
    timing['光线追迹'] = time.perf_counter() - step_start
    
    step_start = time.perf_counter()
    results['lens_f'] = calculation.beam_e2_width_on_lens_f_calculate(0)
    timing['快轴镜面光斑'] = time.perf_counter() - step_start
    
    step_start = time.perf_counter()
    results['lens_s'] = calculation.beam_e2_width_on_lens_s_calculate(0)
    timing['慢轴镜面光斑'] = time.perf_counter() - step_start
    
    step_start = time.perf_counter()
    results['coupling_lens_f'] = calculation.beam_e2_width_on_coupling_lens_f_calculate()
    timing['快轴耦合镜光斑'] = time.perf_counter() - step_start
    
    step_start = time.perf_counter()
    results['coupling_lens_s'] = calculation.beam_e2_width_on_coupling_lens_s_calculate()
    timing['慢轴耦合镜光斑'] = time.perf_counter() - step_start

    for gb in calculation.gaussian_beam_f_list:
        gb.clear_lru_cache()
    for gb in calculation.gaussian_beam_s_list:
        gb.clear_lru_cache()

    step_start = time.perf_counter()
    results['divergence_f'] = calculation.divergence_angle_f_calculate(0)
    timing['快轴发散角'] = time.perf_counter() - step_start
    
    step_start = time.perf_counter()
    results['divergence_s'] = calculation.divergence_angle_s_calculate(0)
    timing['慢轴发散角'] = time.perf_counter() - step_start
    
    timing['总计'] = time.perf_counter() - total_start
    results['timing'] = timing
    
    return results


def summarize_text(results: Dict):
    lines: List[str] = []
    beam_spreading = results.get('beam_spreading') or []
    if beam_spreading:
        formatted = ', '.join(str(round(val, 3)) for val in beam_spreading)
        lines.append(f'快轴方向光斑切割展宽情况：{formatted}')
    beam_cutting = results.get('beam_cutting') or []
    if beam_cutting:
        formatted = ', '.join(f'{round(val * 100, 3)}%' for val in beam_cutting)
        lines.append(f'快轴方向光斑切割后能量占比：{formatted}')

    far_near = results.get('far_near')
    if far_near:
        na = far_near[6]
        na_ratio = far_near[7]
        coupling_efficiency = far_near[15]
        cladding_light_energy_ratio = far_near[16]
        e2_width_near_field = far_near[17]
        
        # 获取光纤NA值，用于过滤显示范围
        fiber_na_value = results.get('fiber_na', 0.22)
        # 确保 fiber_na_value 是有效数值
        if not fiber_na_value or fiber_na_value <= 0:
            fiber_na_value = 0.22
        na_min = fiber_na_value / 2  # 下限为光纤NA的一半
        na_max = fiber_na_value      # 上限为光纤NA
        
        highlight_lines = []
        for value, ratio in zip(na, na_ratio):
            # 只显示 fiber_na/2 到 fiber_na 范围内的 NA 数据
            if na_min <= value <= na_max:
                text = f'{round(value, 3)}NA的能量占比：{round(ratio * 100, 2)}%'
                # 高亮接近光纤NA的值
                highlight = abs(value - fiber_na_value) < 0.001
                if highlight:
                    highlight_lines.append(f'<span style="color:red">{text}</span>')
                else:
                    highlight_lines.append(text)
        highlight_lines.append(f'<span style="color:red">耦合效率：{round(coupling_efficiency * 100, 2)}%</span>')
        highlight_lines.append(f'<span style="color:red">包层光占比：{round(cladding_light_energy_ratio * 100, 2)}%</span>')
        highlight_lines.append(
            f'近场光斑尺寸(1/e²)：{round(e2_width_near_field[0] * 1e6, 2)}µm × '
            f'{round(e2_width_near_field[1] * 1e6, 2)}µm'
        )
        st.markdown('<br>'.join(highlight_lines), unsafe_allow_html=True)

    divergence_f = results.get('divergence_f') or []
    if divergence_f:
        text = ', '.join(f'{round(val * 1000, 3)}mrad' for val in divergence_f)
        lines.append(f'快轴方向不同镜子后的发散角：{text}')
    divergence_s = results.get('divergence_s') or []
    if divergence_s:
        text = ', '.join(f'{round(val * 1000, 3)}mrad' for val in divergence_s)
        lines.append(f'慢轴方向不同镜子后的发散角：{text}')

    if lines:
        st.info('\n'.join(lines))


def format_e2_text(label: str, entries: List[Tuple[float, float]], unit_scale: float, unit: str) -> str:
    parts = []
    for idx, value in enumerate(entries):
        parts.append(
            f'{label}第{idx}个光斑尺寸(1/e²)：{round(value[0] * unit_scale, 2)}{unit} × '
            f'{round(value[1] * unit_scale, 2)}{unit}'
        )
    return '\n'.join(parts)




def render_calculation_results(results: Dict):
    st.subheader('计算结果')
    if not results:
        st.info('点击"计算"按钮后将在此显示结果。')
        return

    # 使用 Tab 标签页组织结果，更加整洁
    tab1, tab2, tab3 = st.tabs(["📊 核心结果", "🌈 光线追迹", "🔍 透镜分析"])
    
    with tab1:
        summarize_text(results)
        far_near = results.get('far_near')
        if far_near:
            st.markdown('##### 远场 / 近场光斑')
            fig = plot_far_near_fields_matplotlib(far_near)
            st.pyplot(fig)
            plt.close(fig)  # 释放内存

    with tab2:
        trace_data = results.get('trace')
        if trace_data:
            st.markdown('##### 光线追迹')
            fig = plot_trace_plotly(trace_data[0], trace_data[1])
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("无光线追迹数据")

    with tab3:
        # 透镜分析使用网格布局
        lens_f = results.get('lens_f')
        lens_s = results.get('lens_s')
        coupling_f = results.get('coupling_lens_f')
        coupling_s = results.get('coupling_lens_s')
        
        col1, col2 = st.columns(2)
        
        with col1:
            if lens_f:
                st.markdown('**快轴镜面光斑**')
                titles = ['FAC', '镜面'] + [''] * (len(lens_f[0]) - 2)
                fig = plot_lens_heatmap_plotly(lens_f[0], lens_f[1], lens_f[2], titles)
                st.plotly_chart(fig, use_container_width=True)
                st.caption(format_e2_text('快轴方向', lens_f[3], 1e3, 'mm'))
            
            if coupling_f:
                st.markdown('**快轴耦合镜光斑**')
                fig = plot_lens_heatmap_plotly(coupling_f[0], coupling_f[1], coupling_f[2], ['FOC'])
                st.plotly_chart(fig, use_container_width=True)
                st.caption(format_e2_text('快轴耦合镜', coupling_f[3], 1e3, 'mm'))

        with col2:
            if lens_s:
                st.markdown('**慢轴镜面光斑**')
                fig = plot_lens_heatmap_plotly(lens_s[0], lens_s[1], lens_s[2], ['SAC'])
                st.plotly_chart(fig, use_container_width=True)
                st.caption(format_e2_text('慢轴方向', lens_s[3], 1e3, 'mm'))

            if coupling_s:
                st.markdown('**慢轴耦合镜光斑**')
                fig = plot_lens_heatmap_plotly(coupling_s[0], coupling_s[1], coupling_s[2], ['SOC'])
                st.plotly_chart(fig, use_container_width=True)
                st.caption(format_e2_text('慢轴耦合镜', coupling_s[3], 1e3, 'mm'))


def main():
    st.set_page_config(page_title='光纤耦合模块设计', layout='wide')
    
    # 注入自定义 CSS 以优化排版
    st.markdown("""
        <style>
            .block-container {padding-top: 1rem; padding-bottom: 2rem;}
            div[data-testid="stVerticalBlock"] > div {padding-bottom: 0.5rem;}
            .stButton button {width: 100%; border-radius: 8px; font-weight: bold;}
            /* 紧凑的数字输入框 */
            div[data-testid="stNumberInput"] label {font-size: 0.85rem; margin-bottom: 0px;}
            div[data-testid="stNumberInput"] input {min-height: 0px; padding: 0px 8px; height: 32px;}
            /* 调整 Tab 样式 */
            .stTabs [data-baseweb="tab-list"] {gap: 24px;}
            .stTabs [data-baseweb="tab"] {height: 50px; white-space: pre-wrap; background-color: transparent; border-radius: 4px 4px 0px 0px; gap: 1px; padding-top: 10px; padding-bottom: 10px;}
        </style>
    """, unsafe_allow_html=True)
    
    st.title('光纤耦合模块设计（Streamlit）')

    ensure_config_exists()
    config = load_config()

    if config is None:
        return

    left_col, right_col = st.columns([4, 6])

    with left_col:
        st.subheader('参数配置')
        parameter_values = render_parameter_inputs(config)

        calculate = st.button('计算', type='primary', use_container_width=True)
        if calculate:
            updated_config = sync_config(config, parameter_values)
            if updated_config:
                with st.spinner('正在计算...'):
                    try:
                        results = run_full_calculation(updated_config)
                    except Exception as exc:
                        import traceback
                        st.error(f'计算失败：{exc}')
                        st.code(traceback.format_exc(), language='python')
                    else:
                        st.session_state['calc_results'] = results
                        # 使用 toast 显示计算耗时
                        timing = results.get('timing', {})
                        total_time = timing.get('总计', 0)
                        st.toast(f'✅ 计算完成，耗时 {total_time:.2f} 秒', icon='⏱️')

    with right_col:
        render_calculation_results(st.session_state.get('calc_results'))


if __name__ == '__main__':
    main()
