"""LD光纤耦合计算器 - 页面入口

该模块作为LD光纤耦合计算功能的入口点，集成到主应用的工具页面中。
包含完整功能：远场/近场光斑、NA计算、光线追迹、镜面光斑分析。
界面与 LD/streamlit_app.py 保持一致。
"""
from __future__ import annotations

import sys
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

from auth import enforce_login

# 配置页面（仅在独立运行时使用）
try:
    st.set_page_config(page_title="LD光纤耦合", page_icon="💡", layout="wide")
except st.errors.StreamlitAPIException:
    pass

enforce_login()

# 将 LD 模块目录添加到 sys.path 以支持其内部导入
LD_MODULE_PATH = Path(__file__).parent / "LD"
if str(LD_MODULE_PATH) not in sys.path:
    sys.path.insert(0, str(LD_MODULE_PATH))

# 导入 LD 模块的核心组件
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

ROOT = LD_MODULE_PATH
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
    'number_f': ('单侧COS数量', ''),
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

# 参数分组（按功能模块分组）
PARAM_GROUPS = {
    '光源配置': [
        'wavelength', 'waist_f', 'divergence_angle_f', 'near_field_order_f',
        'far_field_order_f', 'number_f', 'interval_f',
        'waist_s', 'divergence_angle_s', 'near_field_order_s', 'far_field_order_s',
        'number_s', 'interval_s', 'z_spatial_beam_combining_f', 'z_polarized_beam_combining'
    ],
    'FAC配置': [
        'collimation_lens_effective_focal_length_f'
    ],
    'SAC配置': [
        'collimation_lens_effective_focal_length_s'
    ],
    '小反配置': [
        'z_mirror_and_chip'
    ],
    'FOC配置': [
        'coupling_lens_effective_focal_length_f', 'z_coupling_lens_f_and_mirror'
    ],
    'SOC配置': [
        'coupling_lens_effective_focal_length_s'
    ],
    '光纤配置': [
        'fiber_core_diameter', 'fiber_cladding_diameter', 'fiber_na'
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
    """确保配置文件存在，如果不存在则从 Excel 迁移"""
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
    """使用卡片式分组布局渲染参数输入"""
    field_definitions = generate_parameter_fields()
    values: Dict[str, float | int] = {}
    
    def render_param_input(key: str):
        """渲染单个参数输入框"""
        field = field_definitions.get(key)
        if field and key in config:
            input_key = f'ld_param_{key}'
            default_value = config[key]['value']
            
            if field.is_integer:
                st.number_input(
                    field.label, 
                    value=int(default_value),
                    step=1, min_value=1, max_value=1_000_000_000, 
                    key=input_key
                )
            else:
                step = 0.0001 if field.decimals == 4 else 0.001
                st.number_input(
                    field.label, 
                    value=float(default_value),
                    format=f'%.{field.decimals}f', 
                    step=step, 
                    key=input_key
                )
            # 从 session_state 读取当前值（Streamlit 自动管理）
            values[key] = st.session_state.get(input_key, default_value)
    
    # 按功能模块分组渲染，使用卡片样式
    # 将所有配置组分为多列展示
    group_items = list(PARAM_GROUPS.items())
    
    # 第一行：光源配置（占满整行）
    st.markdown('<div class="param-card">', unsafe_allow_html=True)
    st.markdown(f'<div class="param-card-header">💡 光源配置</div>', unsafe_allow_html=True)
    # 光源配置参数较多，使用5列布局
    light_params = PARAM_GROUPS['光源配置']
    cols = st.columns(5, gap='small')
    for idx, key in enumerate(light_params):
        with cols[idx % 5]:
            render_param_input(key)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 第二行：光学元件配置（6个小卡片）
    optical_groups = ['FAC配置', 'SAC配置', '小反配置', 'FOC配置', 'SOC配置', '光纤配置']
    icons = {'FAC配置': '🔷', 'SAC配置': '🔶', '小反配置': '🪞', 'FOC配置': '🎯', 'SOC配置': '⭕', '光纤配置': '🔌'}
    
    cols = st.columns(6, gap='small')
    for col_idx, group_name in enumerate(optical_groups):
        with cols[col_idx]:
            st.markdown(f'<div class="param-card-mini">', unsafe_allow_html=True)
            st.markdown(f'<div class="param-card-header-mini">{icons.get(group_name, "📦")} {group_name}</div>', unsafe_allow_html=True)
            for key in PARAM_GROUPS.get(group_name, []):
                render_param_input(key)
            st.markdown('</div>', unsafe_allow_html=True)
    
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


def plot_far_field_matplotlib(data: Tuple) -> plt.Figure:
    """使用 Matplotlib 绘制远场分布图"""
    x_far = data[0] * 1000  # 转换为 mm
    y_far = data[1] * 1000
    intensity_far = data[2]
    center_x_far = data[3] * 1000
    center_y_far = data[4] * 1000
    na_fiber_diameter = data[5] * 1000
    
    # 创建自定义颜色映射
    colors = ['white', 'white', 'blue', 'cyan', 'green', 'yellow', 'orange', 'darkred']
    positions = [0.0, 0.05, 0.1, 0.3, 0.5, 0.7, 0.85, 1.0]
    cmap = LinearSegmentedColormap.from_list('custom', list(zip(positions, colors)))
    
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    
    masked_far = mask_intensity(intensity_far)
    ax.pcolormesh(x_far, y_far, masked_far, cmap=cmap, shading='auto')
    
    # 远场 NA 圆
    circle_x, circle_y = create_circle_points(center_x_far, center_y_far, na_fiber_diameter / 2)
    ax.plot(circle_x, circle_y, 'k-', linewidth=2)
    
    radius_far = na_fiber_diameter / 1.8
    ax.set_xlim(center_x_far - radius_far, center_x_far + radius_far)
    ax.set_ylim(center_y_far - radius_far, center_y_far + radius_far)
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.set_title('Far Field')
    ax.set_aspect('equal')
    
    fig.subplots_adjust(left=0.22, right=0.95, top=0.92, bottom=0.12)
    return fig


def plot_near_field_matplotlib(data: Tuple) -> plt.Figure:
    """使用 Matplotlib 绘制近场分布图"""
    x_near = data[8] * 1000
    y_near = data[9] * 1000
    intensity_near = data[10]
    center_x_near = data[11] * 1000
    center_y_near = data[12] * 1000
    fiber_core_diameter = data[13] * 1000
    fiber_cladding_diameter = data[14] * 1000
    
    # 创建自定义颜色映射
    colors = ['white', 'white', 'blue', 'cyan', 'green', 'yellow', 'orange', 'darkred']
    positions = [0.0, 0.05, 0.1, 0.3, 0.5, 0.7, 0.85, 1.0]
    cmap = LinearSegmentedColormap.from_list('custom', list(zip(positions, colors)))
    
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    
    masked_near = mask_intensity(intensity_near)
    # 将坐标偏移至以 (0, 0) 为中心
    x_near_centered = x_near - center_x_near
    y_near_centered = y_near - center_y_near
    ax.pcolormesh(x_near_centered, y_near_centered, masked_near, cmap=cmap, shading='auto')
    
    # 近场纤芯圆（以原点为中心）
    circle_x, circle_y = create_circle_points(0, 0, fiber_core_diameter / 2)
    ax.plot(circle_x, circle_y, 'k-', linewidth=2)
    
    # 近场包层圆（以原点为中心）
    circle_x, circle_y = create_circle_points(0, 0, fiber_cladding_diameter / 2)
    ax.plot(circle_x, circle_y, 'k--', linewidth=2)
    
    radius_near = fiber_cladding_diameter / 1.8
    ax.set_xlim(-radius_near, radius_near)
    ax.set_ylim(-radius_near, radius_near)
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.set_title('Near Field')
    ax.set_aspect('equal')
    
    fig.subplots_adjust(left=0.22, right=0.95, top=0.92, bottom=0.12)
    return fig


def plot_far_near_fields_matplotlib(data: Tuple) -> plt.Figure:
    """使用 Matplotlib 绘制远场和近场分布图（保留用于兼容）"""
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
    """显示 NA 能量占比等摘要信息"""
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
    """渲染计算结果，使用三列布局：近场光斑、远场光斑、NA数据"""
    st.subheader('📊 计算结果')
    if not results:
        st.info('点击"计算"按钮后将在此显示结果。')
        return

    far_near = results.get('far_near')
    
    # 使用两栏布局：左侧显示图表和指标（2/3），右侧显示NA数据（1/3）
    left_main, right_main = st.columns([2, 1], gap='medium')
    
    with left_main:
        # 左侧内部再分两列显示近场/远场图
        plot_col1, plot_col2 = st.columns(2)
        with plot_col1:
            st.markdown('##### 🔵 近场光斑')
            if far_near:
                fig = plot_near_field_matplotlib(far_near)
                st.pyplot(fig, use_container_width=True, bbox_inches=None)
                plt.close(fig)
            else:
                st.info('无近场数据')
        
        with plot_col2:
            st.markdown('##### 🔴 远场光斑')
            if far_near:
                fig = plot_far_field_matplotlib(far_near)
                st.pyplot(fig, use_container_width=True, bbox_inches=None)
                plt.close(fig)
            else:
                st.info('无远场数据')

        # 图表下方直接显示指标（消除垂直间隙）
        if far_near:
            e2_width_near_field = far_near[17]
            coupling_efficiency = far_near[15]
            cladding_light_energy_ratio = far_near[16]
            
            st.markdown(
                f'**光斑尺寸(1/e²):** {round(e2_width_near_field[0] * 1e6, 2)}µm（慢轴） × '
                f'{round(e2_width_near_field[1] * 1e6, 2)}µm（快轴）'
            )
            st.markdown(f'**🎯 耦合效率:** <span style="color:green;font-size:1.2em;font-weight:bold">{round(coupling_efficiency * 100, 2)}%</span>', unsafe_allow_html=True)
            st.markdown(f'**💡 包层光占比:** <span style="color:orange">{round(cladding_light_energy_ratio * 100, 2)}%</span>', unsafe_allow_html=True)
            
            # 发散角信息（带光学元件标签）
            divergence_f = results.get('divergence_f') or []
            divergence_s = results.get('divergence_s') or []
            # 快轴光学元件标签：FAC, 小反, FOC
            fast_axis_labels = ['FAC', '小反', 'FOC']
            # 慢轴光学元件标签：SAC, SOC
            slow_axis_labels = ['SAC', 'SOC']
            
            if divergence_f:
                labeled_f = [f"{round(v*1000, 2)}mrad（{fast_axis_labels[i] if i < len(fast_axis_labels) else f'镜{i+1}'}）" 
                             for i, v in enumerate(divergence_f)]
                st.markdown(f'**快轴发散角:** {", ".join(labeled_f)}')
            if divergence_s:
                labeled_s = [f"{round(v*1000, 2)}mrad（{slow_axis_labels[i] if i < len(slow_axis_labels) else f'镜{i+1}'}）" 
                             for i, v in enumerate(divergence_s)]
                st.markdown(f'**慢轴发散角:** {", ".join(labeled_s)}')

            # 光斑切割信息（放在指标下方）
            beam_spreading = results.get('beam_spreading') or []
            beam_cutting = results.get('beam_cutting') or []
            if beam_spreading or beam_cutting:
                with st.expander('📐 光斑切割详情', expanded=False):
                    if beam_spreading:
                        st.markdown(f'**光斑展宽:** {", ".join(f"{round(v, 3)}" for v in beam_spreading)}')
                    if beam_cutting:
                        st.markdown(f'**切割能量占比:** {", ".join(f"{round(v*100, 2)}%" for v in beam_cutting)}')

    with right_main:
        st.markdown('##### 📈 NA数据')
        if far_near:
            na = far_near[6]
            na_ratio = far_near[7]
            
            # 直接从当前输入读取光纤NA值（确保使用最新值）
            fiber_na_value = st.session_state.get('ld_param_fiber_na', 0.22)
            if not fiber_na_value or fiber_na_value <= 0:
                fiber_na_value = 0.22
            na_min = fiber_na_value / 2
            na_max = fiber_na_value
            
            # NA能量占比表格
            na_data = []
            for value, ratio in zip(na, na_ratio):
                if na_min <= value <= na_max:
                    na_data.append({
                        'NA': f'{round(value, 3)}',
                        '能量占比': f'{round(ratio * 100, 2)}%',
                    })
            
            if na_data:
                na_data.reverse()  # 倒序排列（从大到小）
                st.dataframe(pd.DataFrame(na_data), hide_index=True, use_container_width=True, height=600)
        else:
            st.info('无NA数据')




def main():
    # 注入自定义 CSS 以优化排版 - 卡片样式
    st.markdown("""
        <style>
            .block-container {padding-top: 0.5rem; padding-bottom: 1rem;}
            div[data-testid="stVerticalBlock"] > div {padding-bottom: 0.2rem;}
            .stButton button {border-radius: 8px; font-weight: bold;}
            /* 减少图表刷新时的布局跳动 */
            img, canvas, .stPlotlyChart, [data-testid="stImage"] {
                transition: opacity 0.15s ease;
            }
            /* 超紧凑的数字输入框 */
            div[data-testid="stNumberInput"] {margin-bottom: -10px;}
            div[data-testid="stNumberInput"] label {font-size: 0.75rem; margin-bottom: 0px; line-height: 1.2;}
            div[data-testid="stNumberInput"] input {min-height: 0px; padding: 2px 6px; height: 28px; font-size: 0.8rem;}
            /* 隐藏数字输入框的加减按钮 */
            div[data-testid="stNumberInput"] button {display: none;}
            /* 调整 Tab 样式 */
            .stTabs [data-baseweb="tab-list"] {gap: 16px;}
            .stTabs [data-baseweb="tab"] {height: 40px; padding: 8px 12px;}
            /* 卡片样式 */
            .param-card {
                background: linear-gradient(135deg, rgba(100,149,237,0.1) 0%, rgba(70,130,180,0.05) 100%);
                border: 1px solid rgba(100,149,237,0.3);
                border-radius: 10px;
                padding: 12px;
                margin-bottom: 10px;
            }
            .param-card-header {
                font-weight: 600;
                font-size: 0.9rem;
                color: #4a90d9;
                margin-bottom: 8px;
                border-bottom: 1px solid rgba(100,149,237,0.2);
                padding-bottom: 6px;
            }
            .param-card-mini {
                background: linear-gradient(135deg, rgba(150,150,150,0.08) 0%, rgba(100,100,100,0.03) 100%);
                border: 1px solid rgba(150,150,150,0.25);
                border-radius: 8px;
                padding: 8px;
                margin-bottom: 8px;
            }
            .param-card-header-mini {
                font-weight: 600;
                font-size: 0.8rem;
                color: #666;
                margin-bottom: 6px;
                text-align: center;
            }
            /* 减少列间距 */
            div[data-testid="column"] {padding: 0 4px;}
        </style>
    """, unsafe_allow_html=True)
    
    st.title('光纤耦合模块设计')

    ensure_config_exists()
    config = load_config()

    if config is None:
        return

    # 参数配置区域（上方）
    st.subheader('⚙️ 参数配置')
    parameter_values = render_parameter_inputs(config)

    def do_calculation():
        """回调函数：在按钮点击时执行计算"""
        # 从 session_state 重新收集所有参数值
        current_values = {}
        for group_params in PARAM_GROUPS.values():
            for key in group_params:
                input_key = f'ld_param_{key}'
                if input_key in st.session_state:
                    current_values[key] = st.session_state[input_key]
        
        # 更新配置（仅在内存中，不保存到文件）
        updated_config = {k: dict(v) for k, v in config.items()}
        for key, value in current_values.items():
            if key in updated_config:
                updated_config[key]['value'] = value
        
        # 执行计算
        try:
            results = run_full_calculation(updated_config)
            st.session_state['ld_calc_results'] = results
            st.session_state['ld_calc_success'] = True
        except Exception as exc:
            st.session_state['ld_calc_error'] = str(exc)
            st.session_state['ld_calc_success'] = False

    # 计算按钮
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        st.button('🚀 开始计算', type='primary', use_container_width=True, on_click=do_calculation)
    
    # 显示计算结果或错误
    if st.session_state.get('ld_calc_success') == False:
        st.error(f'计算失败：{st.session_state.get("ld_calc_error", "未知错误")}')
    elif st.session_state.get('ld_calc_results'):
        timing = st.session_state['ld_calc_results'].get('timing', {})
        total_time = timing.get('总计', 0)
        if total_time > 0:
            st.toast(f'✅ 计算完成，耗时 {total_time:.2f} 秒', icon='⏱️')

    # 分隔线
    st.divider()
    
    # 结果显示区域（使用容器固定高度，防止布局跳动）
    results_container = st.container()
    with results_container:
        render_calculation_results(st.session_state.get('ld_calc_results'))


if __name__ == '__main__':
    main()
else:
    main()
