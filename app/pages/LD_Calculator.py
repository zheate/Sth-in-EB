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
from copy import deepcopy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from scipy.optimize import minimize

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
    parameters_convert,
    PARAM_DEFINITIONS,
    CONFIG_JSON_PATH,
    list_presets,
    load_preset,
    save_preset,
    delete_preset,
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
    'number_f': ('⭐️ 单侧COS数量', ''),
    'interval_f': ('⭐️ 芯片高度', 'mm'),
    'astigmatism': ('像散', 'um'),
    'waist_s': ('⭐️ 半条宽', 'um'),
    'divergence_angle_s': ('慢轴发散半角', '°'),
    'near_field_order_s': ('慢轴近场阶数', ''),
    'far_field_order_s': ('慢轴远场阶数', ''),
    'number_s': ('慢轴堆叠数量', ''),
    'interval_s': ('慢轴堆叠间隔', 'mm'),
    'z_spatial_beam_combining_f': ('⭐️ 台阶间距', 'mm'),
    'collimation_lens_effective_focal_length_f': ('⭐️ FAC焦距', 'mm'),
    'collimation_lens_effective_focal_length_s': ('⭐️ SAC焦距', 'mm'),
    'z_mirror_and_chip': ('小反到芯片距离', 'mm'),
    'z_polarized_beam_combining': ('偏振合束光程差', 'mm'),
    'z_spatial_beam_combining_s': ('慢轴空间合束光程差', 'mm'),
    'coupling_lens_effective_focal_length_f': ('⭐️ 快轴耦合镜焦距', 'mm'),
    'coupling_lens_effective_focal_length_s': ('⭐️ 慢轴耦合镜焦距', 'mm'),
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
        'wavelength', 'waist_f', 'divergence_angle_f', 'near_field_order_f', 'far_field_order_f',
        'divergence_angle_s', 'near_field_order_s', 'far_field_order_s', 'number_s', 'interval_s',
        'number_f', 'waist_s', 'interval_f', 'z_spatial_beam_combining_f', 'z_polarized_beam_combining'
    ],
    '准直配置': [
        'collimation_lens_effective_focal_length_f',  # FAC
        'collimation_lens_effective_focal_length_s',  # SAC
        'z_mirror_and_chip'  # 小反
    ],
    '耦合配置': [
        'coupling_lens_effective_focal_length_f',  # FOC
        'coupling_lens_effective_focal_length_s',  # SOC
        'z_coupling_lens_f_and_mirror'
    ],
    '光纤配置': [
        'fiber_core_diameter', 'fiber_cladding_diameter', 'fiber_na'
    ]
}

INTEGER_PARAMS = {'number_f', 'number_s', 'waist_s'}  # 整数参数
HIGH_PRECISION_PARAMS = {'wavelength', 'index_fiber_core'}  # 4位小数
CUSTOM_DECIMALS = {  # 自定义小数位数
    'interval_f': 3,  # 芯片高度: 3位小数
    'z_spatial_beam_combining_f': 2,  # 台阶间距: 2位小数
}


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
        # 确定小数位数：优先自定义 > 高精度(4位) > 默认(3位)
        if key in CUSTOM_DECIMALS:
            decimals = CUSTOM_DECIMALS[key]
        elif key in HIGH_PRECISION_PARAMS:
            decimals = 4
        else:
            decimals = 3
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
            
            # 避免 Streamlit 警告：如果 key 已在 session_state 中，不要通过 value 参数设置默认值
            if input_key not in st.session_state:
                st.session_state[input_key] = int(default_value) if field.is_integer else float(default_value)

            if field.is_integer:
                st.number_input(
                    field.label, 
                    step=1, min_value=1, max_value=1_000_000_000, 
                    key=input_key
                )
            else:
                step = 0.0001 if field.decimals == 4 else 0.001
                st.number_input(
                    field.label, 
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
    st.divider()
    st.markdown(f'<div class="param-card-header">💡 光源配置</div>', unsafe_allow_html=True)
    # 光源配置参数较多，使用5列布局
    light_params = PARAM_GROUPS['光源配置']
    cols = st.columns(5, gap='small')
    for idx, key in enumerate(light_params):
        with cols[idx % 5]:
            render_param_input(key)
    
    # 第二行：光学元件配置（3个卡片：准直、耦合、光纤）
    optical_groups = ['准直配置', '耦合配置', '光纤配置']
    icons = {'准直配置': '💠', '耦合配置': '🎯', '光纤配置': '🔌'}
    
    cols = st.columns(3, gap='medium')
    for col_idx, group_name in enumerate(optical_groups):
        with cols[col_idx]:
            st.divider()
            st.markdown(f'<div class="param-card-header-mini">{icons.get(group_name, "📦")} {group_name}</div>', unsafe_allow_html=True)
            for key in PARAM_GROUPS.get(group_name, []):
                render_param_input(key)
    
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
    # 远场图像上下颠倒
    masked_far = np.flipud(masked_far)
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
    # 转换为 um
    x_near = data[8] * 1e6
    y_near = data[9] * 1e6
    intensity_near = data[10]
    center_x_near = data[11] * 1e6
    center_y_near = data[12] * 1e6
    fiber_core_diameter = data[13] * 1e6
    fiber_cladding_diameter = data[14] * 1e6
    
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
    ax.set_xlabel('x (um)')
    ax.set_ylabel('y (um)')
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
            coupling_eff_val = round(coupling_efficiency * 100, 2)
            eff_color = "red" if coupling_eff_val < 90 else "green"
            st.markdown(f'**🎯 耦合效率:** <span style="color:{eff_color};font-size:1.2em;font-weight:bold">{coupling_eff_val}%</span>', unsafe_allow_html=True)
            cladding_ratio_val = round(cladding_light_energy_ratio * 100, 2)
            if cladding_ratio_val < 1:
                cladding_color = "green"
            elif cladding_ratio_val > 2:
                cladding_color = "red"
            else:
                cladding_color = "orange"
            st.markdown(f'**💡 包层光占比:** <span style="color:{cladding_color};font-size:1.2em;font-weight:bold">{cladding_ratio_val}%</span>', unsafe_allow_html=True)
            
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
                    st.caption('说明：下表数据对应每一个子光束（如每个COS芯片或台阶）的计算结果。')
                    # 构造数据表格
                    data = []
                    max_len = max(len(beam_spreading), len(beam_cutting))
                    for i in range(max_len):
                        spread = beam_spreading[i] if i < len(beam_spreading) else None
                        cut = beam_cutting[i] if i < len(beam_cutting) else None
                        row = {'序号': i + 1}
                        if spread is not None:
                            row['光斑展宽'] = f'{round(spread, 3)}'
                        if cut is not None:
                            row['切割能量占比'] = f'{round(cut * 100, 2)}%'
                        data.append(row)
                    
                    if data:
                        st.dataframe(pd.DataFrame(data), hide_index=True, use_container_width=True)

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
                        'NA': value,  # Keep numeric for styling logic
                        '能量占比': ratio, # Keep numeric for styling logic
                    })
            
            if na_data:
                df = pd.DataFrame(na_data)
                df = df.iloc[::-1] # 倒序排列
                
                # 定义样式函数
                def highlight_row(row):
                    styles = [''] * len(row)
                    # 检查是否满足条件：光纤NA约为0.22 且 当前行NA约为0.18
                    if abs(fiber_na_value - 0.22) < 0.001 and abs(row['NA'] - 0.18) < 0.001:
                        # NA列加粗
                        styles[0] = 'font-weight: bold; color: black;'
                        # 能量占比列：大于95%绿色加粗，否则红色加粗
                        ratio_val = row['能量占比'] * 100
                        color = 'green' if ratio_val > 95 else 'red'
                        styles[1] = f'font-weight: bold; color: {color};'
                    return styles

                # 应用样式并格式化显示
                styled_df = df.style.apply(highlight_row, axis=1)\
                    .format({'NA': '{:.3f}', '能量占比': '{:.2%}'})
                
                st.dataframe(styled_df, hide_index=True, use_container_width=True, height=600)
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
            /* 关键参数标签使用金黄色 */
            div[data-testid="stNumberInput"] label p {color: inherit;}
            div[data-testid="stNumberInput"]:has(label:first-child) label:first-child {
                color: #333;
            }
            div[data-testid="stNumberInput"] input {
                min-height: 0px; 
                padding: 2px 6px; 
                height: 28px; 
                font-size: 0.8rem;
                background-color: white; /* 强制白色背景 */
            }
            /* 隐藏数字输入框的加减按钮 */
            div[data-testid="stNumberInput"] button {display: none;}
            /* 调整 Tab 样式 */
            .stTabs [data-baseweb="tab-list"] {gap: 16px;}
            .stTabs [data-baseweb="tab"] {height: 40px; padding: 8px 12px;}
            /* 卡片样式 */
            /* 卡片样式 - 毛玻璃效果 */
            .param-card {
                background: rgba(255, 255, 255, 0.4);
                backdrop-filter: blur(12px);
                -webkit-backdrop-filter: blur(12px);
                border: 1px solid rgba(255, 255, 255, 0.6);
                box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.07);
                border-radius: 12px;
                padding: 15px;
                margin-bottom: 5px;
            }
            /* 调整分割线间距 */
            hr {margin-top: 5px !important; margin-bottom: 15px !important;}
            .param-card-header {
                font-weight: 700;
                font-size: 0.95rem;
                color: #2c3e50;
                margin-bottom: 10px;
                border-bottom: 1px solid rgba(0, 0, 0, 0.05);
                padding-bottom: 8px;
            }
            .param-card-mini {
                background: rgba(255, 255, 255, 0.3);
                backdrop-filter: blur(8px);
                -webkit-backdrop-filter: blur(8px);
                border: 1px solid rgba(255, 255, 255, 0.5);
                box-shadow: 0 4px 16px 0 rgba(31, 38, 135, 0.05);
                border-radius: 10px;
                padding: 10px;
                margin-bottom: 10px;
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
            /* 修复标题被遮挡 */
            h1 {
                padding-top: 2rem !important;
            }
        </style>
    """, unsafe_allow_html=True)
    
    st.title('光纤耦合模块设计')

    ensure_config_exists()
    config = load_config()

    if config is None:
        return

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
        
        # 保存配置到文件 (现在文件在 config 目录，不会触发 rerun)
        try:
            save_config_to_json(updated_config)
        except Exception as e:
            st.session_state['ld_calc_error'] = f"保存配置失败: {e}"
            # 继续执行计算，不中断
        
        # 执行计算
        try:
            results = run_full_calculation(updated_config)
            st.session_state['ld_calc_results'] = results
            st.session_state['ld_calc_success'] = True
        except Exception as exc:
            st.session_state['ld_calc_error'] = str(exc)
            st.session_state['ld_calc_success'] = False

    # 参数配置区域（上方）
    # 使用列布局放置标题和预设选择
    header_col1, header_col2 = st.columns([1, 2], vertical_alignment="center")
    with header_col1:
        st.markdown('<h3 style="margin: 0; padding: 0;">⚙️ 参数配置</h3>', unsafe_allow_html=True)
    
    with header_col2:
        # 使用四列布局：下拉菜单，搜索按钮，保存按钮，删除按钮
        # 调整比例，给下拉菜单更多空间
        sel_col, search_col, save_col, del_col = st.columns([4, 0.5, 0.5, 0.5], vertical_alignment="center")
        
        # 获取预设列表
        presets = list_presets()
        # 添加默认选项
        preset_options = ['当前配置'] + presets
        
        def on_preset_change():
            """预设改变时的回调"""
            selected = st.session_state.get('preset_selector')
            if selected and selected != '当前配置':
                # 加载预设
                preset_config = load_preset(selected)
                # 更新 session_state 中的参数值
                for key, param in preset_config.items():
                    input_key = f'ld_param_{key}'
                    st.session_state[input_key] = param['value']
                # 更新当前配置对象 (用于本次渲染)
                config.update(preset_config)
                st.toast(f'已加载预设: {selected}')
                # 自动执行计算
                do_calculation()

        with sel_col:
            st.selectbox(
                '选择预设', 
                options=preset_options, 
                key='preset_selector', 
                label_visibility='collapsed',
                on_change=on_preset_change
            )
        
        with search_col:
            with st.popover("🔍", use_container_width=True):
                search_query = st.text_input("搜索预设", placeholder="输入名称...")
                if search_query:
                    filtered_presets = [p for p in presets if search_query.lower() in p.lower()]
                    if filtered_presets:
                        st.markdown("---")
                        for p in filtered_presets:
                            def select_preset_callback(preset_name):
                                st.session_state['preset_selector'] = preset_name
                                # 手动触发预设加载逻辑 (因为 on_change 可能不会在代码修改 session_state 时触发)
                                preset_config = load_preset(preset_name)
                                for key, param in preset_config.items():
                                    input_key = f'ld_param_{key}'
                                    st.session_state[input_key] = param['value']
                                config.update(preset_config)
                                st.toast(f'已加载预设: {preset_name}')
                                # 设置标志位以便在重新运行后执行计算
                                st.session_state['do_calc_next_run'] = True

                            if st.button(p, key=f"search_res_{p}", use_container_width=True, on_click=select_preset_callback, args=(p,)):
                                pass # Callback handles logic
                    else:
                        st.caption("未找到匹配的预设")
                else:
                    st.caption("请输入关键词进行搜索")

        with save_col:
            # 保存预设按钮 (仅图标)
            with st.popover("💾", use_container_width=True):
                preset_name = st.text_input("预设名称", placeholder="请按功率-波长-模块命名")
                if st.button("确认保存", type="primary", use_container_width=True):
                    if preset_name:
                        # 收集当前参数
                        current_values = {}
                        for group_params in PARAM_GROUPS.values():
                            for key in group_params:
                                input_key = f'ld_param_{key}'
                                if input_key in st.session_state:
                                    current_values[key] = st.session_state[input_key]
                        
                        # 更新配置并保存
                        preset_config = {k: dict(v) for k, v in config.items()}
                        for key, value in current_values.items():
                            if key in preset_config:
                                preset_config[key]['value'] = value
                        
                        try:
                            save_preset(preset_name, preset_config)
                            st.toast(f"预设 '{preset_name}' 保存成功！")
                            # 强制刷新以更新下拉列表
                            st.rerun()
                        except Exception as e:
                            st.error(f"保存失败: {e}")
                    else:
                        st.warning("请输入预设名称")

        # 仅当选择了非默认预设时显示删除按钮
        selected_preset = st.session_state.get('preset_selector')
        if selected_preset and selected_preset != '当前配置':
            with del_col:
                with st.popover("🗑️", use_container_width=True):
                    st.markdown(f"确定删除预设 **{selected_preset}** 吗？")
                    
                    def delete_preset_callback(preset_name):
                        if delete_preset(preset_name):
                            st.toast(f"预设 '{preset_name}' 已删除")
                            st.session_state['preset_selector'] = '当前配置'
                        else:
                            st.error("删除失败")
                            
                    st.button("确认删除", type="primary", use_container_width=True, on_click=delete_preset_callback, args=(selected_preset,))

    parameter_values = render_parameter_inputs(config)

    # 计算按钮 (居中且加宽)
    # 计算按钮 (居中且加宽)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        c1, c2 = st.columns([1, 1])
        with c1:
            st.button('🚀 开始计算', type='primary', use_container_width=True, on_click=do_calculation)
        with c2:
            do_optimize = st.button('✨ 优化', use_container_width=True, help="自动寻找最佳的快轴耦合镜和慢轴耦合镜焦距")
        
        # 优化结果显示区域 (放在按钮下方，宽度与 col2 一致)
        optimization_container = st.container()
        
        if do_optimize:
            run_optimization(config, optimization_container)
    
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


def run_optimization(config: Dict[str, Any], container):
    """运行优化算法"""
    with container:
        st.markdown("#### 🚀 正在进行优化计算...")
        status_text = st.empty()
        progress_bar = st.progress(0)
    
    # 初始参数 [FOC, SOC]
    initial_params = [
        config['coupling_lens_effective_focal_length_f']['value'],     # FOC
        config['coupling_lens_effective_focal_length_f']['value']      # SOC (Initial guess same as FOC if missing, or use actual SOC)
    ]
    # Correct SOC key if it was wrong in my thought process, checking file...
    # Line 106: 'coupling_lens_effective_focal_length_s'
    initial_params[1] = config['coupling_lens_effective_focal_length_s']['value']

    # 优化目标函数已移动到 optimization_logic.py 以支持多进程

    # 边界条件 (当前值 +/- 50%, 且 > 0)
    bounds = [
        (max(0.1, p * 0.5), p * 1.5) for p in initial_params
    ]

    status_text.text("正在优化中，请稍候... (这可能需要几分钟)")
    
    # 回调函数更新进度 (scipy minimize callback is limited, just simple spinner)
    
    # 使用差分进化算法 (Differential Evolution) 进行全局优化
    # 为了防止系统卡顿，保留 2 个核心：os.cpu_count() - 2
    import os
    max_workers = max(1, (os.cpu_count() or 1) - 2)
    
    from scipy.optimize import differential_evolution
    from optimization_logic import optimization_objective
    
    start_time = time.time()
    
    res = differential_evolution(
        optimization_objective, 
        bounds=bounds,
        args=(config,),  # Pass config as argument
        strategy='best1bin',
        maxiter=20,
        popsize=10,
        tol=0.01,
        workers=max_workers,
        disp=True,
        polish=True
    )
    
    end_time = time.time()
    duration = end_time - start_time
    st.toast(f"优化完成，耗时 {duration:.2f} 秒", icon="⏱️")
    
    progress_bar.progress(100)
    status_text.empty()
    
    # 即使达到最大迭代次数，通常也找到了较好的解，因此也显示结果
    if res.success or "Maximum number of iterations has been exceeded" in str(res.message):
        if res.success:
            st.success("优化成功！")
        else:
            st.warning("已达到最大计算次数，显示当前找到的最佳结果。")
        
        # 显示优化结果
        opt_foc = res.x[0]
        opt_soc = res.x[1]
        
        st.markdown("### 🏆 优化结果")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("最佳 快轴耦合镜焦距", f"{opt_foc:.3f} mm", delta=f"{opt_foc - initial_params[0]:.3f} mm")
        with col2:
            st.metric("最佳 慢轴耦合镜焦距", f"{opt_soc:.3f} mm", delta=f"{opt_soc - initial_params[1]:.3f} mm")
            
        # 应用按钮
        def apply_optimized():
            st.session_state['ld_param_coupling_lens_effective_focal_length_f'] = float(opt_foc)
            st.session_state['ld_param_coupling_lens_effective_focal_length_s'] = float(opt_soc)
            st.toast("已应用优化参数，请点击“开始计算”查看详细结果")
            
        st.button("应用优化参数", on_click=apply_optimized, type="primary")
        
    else:
        st.error(f"优化失败: {res.message}")


if __name__ == '__main__':
    main()
else:
    main()
