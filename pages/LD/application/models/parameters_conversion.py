from __future__ import annotations

import json
from os.path import dirname, join, exists
from typing import Dict, Any, Optional

from numpy import pi, full, nan, inf
from pandas import read_excel, DataFrame, ExcelWriter


assets_path = join(dirname(dirname(dirname(__file__))), 'assets')

# JSON 配置文件路径 (位于根目录的 config 文件夹)
# __file__ -> models -> application -> LD -> pages -> Sth-in-EB (根目录)
PROJECT_ROOT = dirname(dirname(dirname(dirname(dirname(__file__)))))
CONFIG_JSON_PATH = join(PROJECT_ROOT, 'config', 'ld_config.json')

# 参数定义：(row, col, key, unit_col)
PARAM_DEFINITIONS = [
    # 第一列参数 (row, value_col=1, key, unit_col=2)
    (0, 1, 'wavelength', 2),
    (1, 1, 'waist_f', 2),
    (2, 1, 'divergence_angle_f', 2),
    (3, 1, 'near_field_order_f', 2),
    (4, 1, 'far_field_order_f', 2),
    (5, 1, 'number_f', 2),
    (6, 1, 'interval_f', 2),
    (7, 1, 'astigmatism', 2),
    (8, 1, 'waist_s', 2),
    (9, 1, 'divergence_angle_s', 2),
    (10, 1, 'near_field_order_s', 2),
    (11, 1, 'far_field_order_s', 2),
    (12, 1, 'number_s', 2),
    (13, 1, 'interval_s', 2),
    (14, 1, 'z_spatial_beam_combining_f', 2),
    # 第二列参数 (row, value_col=4, key, unit_col=5)
    (0, 4, 'collimation_lens_effective_focal_length_f', 5),
    (1, 4, 'collimation_lens_effective_focal_length_s', 5),
    (2, 4, 'z_mirror_and_chip', 5),
    (3, 4, 'z_polarized_beam_combining', 5),
    (4, 4, 'z_spatial_beam_combining_s', 5),
    (5, 4, 'coupling_lens_effective_focal_length_f', 5),
    (6, 4, 'coupling_lens_effective_focal_length_s', 5),
    (7, 4, 'z_coupling_lens_f_and_mirror', 5),
    (8, 4, 'fiber_core_diameter', 5),
    (9, 4, 'fiber_cladding_diameter', 5),
    (10, 4, 'fiber_na', 5),
    (11, 4, 'index_fiber_core', 5),
    (12, 4, 'index_environment', 5),
    (13, 4, 'fiber_coiling_radius', 5),
]


def value_convert(value, unit):
    """将值按单位转换为标准单位（SI）"""
    if unit == 'um':
        return value / 1e6
    elif unit == 'mm':
        return value / 1e3
    elif unit == '°':
        return value / 180 * pi
    else:
        return value


def load_config_from_json(json_path: Optional[str] = None) -> Dict[str, Any]:
    """从 JSON 文件加载配置参数
    
    Args:
        json_path: JSON 文件路径，默认使用 CONFIG_JSON_PATH
        
    Returns:
        配置字典，包含所有参数的 value 和 unit
    """
    path = json_path or CONFIG_JSON_PATH
    if not exists(path):
        raise FileNotFoundError(f'配置文件不存在: {path}')
    
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_config_to_json(config: Dict[str, Any], json_path: Optional[str] = None) -> None:
    """将配置参数保存到 JSON 文件
    
    Args:
        config: 配置字典
        json_path: JSON 文件路径，默认使用 CONFIG_JSON_PATH
    """
    path = json_path or CONFIG_JSON_PATH
    # 确保目录存在
    dirname_path = dirname(path)
    if not exists(dirname_path):
        import os
        os.makedirs(dirname_path, exist_ok=True)
        
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


# 预设配置目录
PRESETS_DIR = join(PROJECT_ROOT, 'config', 'presets')

def list_presets() -> list[str]:
    """列出所有可用的预设名称"""
    if not exists(PRESETS_DIR):
        return []
    
    presets = []
    import os
    for filename in os.listdir(PRESETS_DIR):
        if filename.endswith('.json'):
            presets.append(filename[:-5])  # 移除 .json 后缀
    return sorted(presets)

def load_preset(name: str) -> Dict[str, Any]:
    """加载指定名称的预设"""
    preset_path = join(PRESETS_DIR, f'{name}.json')
    return load_config_from_json(preset_path)

def save_preset(name: str, config: Dict[str, Any]) -> None:
    """保存配置为预设"""
    preset_path = join(PRESETS_DIR, f'{name}.json')
    save_config_to_json(config, preset_path)

def delete_preset(name: str) -> bool:
    """删除指定名称的预设"""
    preset_path = join(PRESETS_DIR, f'{name}.json')
    if exists(preset_path):
        import os
        os.remove(preset_path)
        return True
    return False


def load_config_from_excel(excel_path: Optional[str] = None) -> Dict[str, Any]:
    """从 Excel 文件加载配置参数并转换为标准格式
    
    Args:
        excel_path: Excel 文件路径，默认使用 'LD光纤耦合参数.xlsx'
        
    Returns:
        配置字典
    """
    path = excel_path or join(assets_path, 'LD光纤耦合参数.xlsx')
    df = read_excel(path, header=None)
    
    config = {}
    for row, value_col, key, unit_col in PARAM_DEFINITIONS:
        value = df.iloc[row, value_col]
        unit = df.iloc[row, unit_col] if unit_col < df.shape[1] else ''
        if isinstance(unit, float) and unit != unit:  # NaN check
            unit = ''
        config[key] = {
            'value': float(value) if not isinstance(value, (int, float)) else value,
            'unit': str(unit) if unit else ''
        }
    
    return config


def migrate_excel_to_json(excel_path: Optional[str] = None, json_path: Optional[str] = None) -> Dict[str, Any]:
    """将 Excel 配置文件迁移到 JSON 格式
    
    Args:
        excel_path: 源 Excel 文件路径
        json_path: 目标 JSON 文件路径
        
    Returns:
        迁移后的配置字典
    """
    config = load_config_from_excel(excel_path)
    save_config_to_json(config, json_path)
    return config


def parameters_convert(config: Optional[Dict[str, Any]] = None) -> Dict[str, DataFrame]:
    """将配置参数转换为计算所需的 DataFrame 数据结构
    
    Args:
        config: 配置字典。如果为 None，则尝试从 JSON 文件加载
        
    Returns:
        包含所有计算数据的字典：
        - 'source_data_f': 快轴光源数据 DataFrame
        - 'source_data_s': 慢轴光源数据 DataFrame  
        - 'lens_data_f': 快轴透镜数据 DataFrame
        - 'lens_data_s': 慢轴透镜数据 DataFrame
        - 'fiber_data': 光纤数据 DataFrame
    """
    if config is None:
        config = load_config_from_json()
    
    # 提取并转换参数
    def get_param(key: str) -> float:
        param = config[key]
        return value_convert(param['value'], param['unit'])
    
    wavelength = get_param('wavelength')
    waist_f = get_param('waist_f')
    divergence_angle_f = get_param('divergence_angle_f')
    near_field_order_f = get_param('near_field_order_f')
    far_field_order_f = get_param('far_field_order_f')
    number_f = int(get_param('number_f'))
    interval_f = get_param('interval_f')
    astigmatism = get_param('astigmatism')
    
    waist_s = get_param('waist_s')
    divergence_angle_s = get_param('divergence_angle_s')
    near_field_order_s = get_param('near_field_order_s')
    far_field_order_s = get_param('far_field_order_s')
    number_s = int(get_param('number_s'))
    interval_s = get_param('interval_s')
    z_spatial_beam_combining_f = get_param('z_spatial_beam_combining_f')
    
    collimation_lens_effective_focal_length_f = get_param('collimation_lens_effective_focal_length_f')
    collimation_lens_effective_focal_length_s = get_param('collimation_lens_effective_focal_length_s')
    z_mirror_and_chip = get_param('z_mirror_and_chip')
    z_polarized_beam_combining = get_param('z_polarized_beam_combining')
    z_spatial_beam_combining_s = get_param('z_spatial_beam_combining_s')
    coupling_lens_effective_focal_length_f = get_param('coupling_lens_effective_focal_length_f')
    coupling_lens_effective_focal_length_s = get_param('coupling_lens_effective_focal_length_s')
    z_coupling_lens_f_and_mirror = get_param('z_coupling_lens_f_and_mirror')
    fiber_core_diameter = get_param('fiber_core_diameter')
    fiber_cladding_diameter = get_param('fiber_cladding_diameter')
    fiber_na = get_param('fiber_na')
    index_fiber_core = get_param('index_fiber_core')
    index_environment = get_param('index_environment')
    fiber_coiling_radius = get_param('fiber_coiling_radius')
    
    # 计算总数
    total_number = number_f * number_s
    if z_polarized_beam_combining:
        total_number *= 2
    
    # 构建快轴光源数据
    dataframe_source_data_f = DataFrame(full((total_number, 9), nan), columns=[
        '波长/m', '束腰半宽/m', '发散半角/rad', '近场阶数', '远场阶数', 
        '位置t/m', '位置z/m', '切光情况', '光源光强'
    ])
    dataframe_source_data_f.iloc[:, 0] = wavelength
    dataframe_source_data_f.iloc[:, 1] = waist_f
    dataframe_source_data_f.iloc[:, 2] = divergence_angle_f
    dataframe_source_data_f.iloc[:, 3] = near_field_order_f
    dataframe_source_data_f.iloc[:, 4] = far_field_order_f
    dataframe_source_data_f.iloc[:, 8] = 1
    
    for ii in range(number_f):
        dataframe_source_data_f.iloc[ii, 5] = ii * interval_f
        dataframe_source_data_f.iloc[ii, 6] = -ii * z_spatial_beam_combining_f
        if number_f == 1:
            dataframe_source_data_f.iloc[ii, 7] = 0
        else:
            if ii == 0:
                dataframe_source_data_f.iloc[ii, 7] = 1
            elif ii == number_f - 1:
                dataframe_source_data_f.iloc[ii, 7] = -1
            else:
                dataframe_source_data_f.iloc[ii, 7] = 2
    
    base_number = number_f
    if z_polarized_beam_combining:
        base_number *= 2
        dataframe_source_data_f.iloc[number_f: base_number, 5] = dataframe_source_data_f.iloc[: number_f, 5]
        dataframe_source_data_f.iloc[number_f: base_number, 6] = (
            dataframe_source_data_f.iloc[: number_f, 6] - z_polarized_beam_combining
        )
        dataframe_source_data_f.iloc[number_f: base_number, 7] = dataframe_source_data_f.iloc[: number_f, 7]
    
    for ii in range(number_s - 1):
        dataframe_source_data_f.iloc[base_number * (ii + 1): base_number * (ii + 2), 5] = (
            dataframe_source_data_f.iloc[: base_number, 5]
        )
        dataframe_source_data_f.iloc[base_number * (ii + 1): base_number * (ii + 2), 6] = (
            dataframe_source_data_f.iloc[: base_number, 6] - (ii + 1) * z_spatial_beam_combining_s
        )
        dataframe_source_data_f.iloc[base_number * (ii + 1): base_number * (ii + 2), 7] = (
            dataframe_source_data_f.iloc[: base_number, 7]
        )
    
    # 构建慢轴光源数据
    dataframe_source_data_s = DataFrame(full((total_number, 8), nan), columns=[
        '波长/m', '束腰半宽/m', '发散半角/rad', '近场阶数', '远场阶数', 
        '位置t/m', '位置z/m', '光源光强'
    ])
    dataframe_source_data_s.iloc[:, 0] = wavelength
    dataframe_source_data_s.iloc[:, 1] = waist_s
    dataframe_source_data_s.iloc[:, 2] = divergence_angle_s
    dataframe_source_data_s.iloc[:, 3] = near_field_order_s
    dataframe_source_data_s.iloc[:, 4] = far_field_order_s
    dataframe_source_data_s.iloc[:, 7] = 1
    dataframe_source_data_s.iloc[: number_f, 5] = 0
    
    for ii in range(number_f):
        dataframe_source_data_s.iloc[ii, 6] = -astigmatism - ii * z_spatial_beam_combining_f
    
    base_number = number_f
    if z_polarized_beam_combining:
        base_number *= 2
        dataframe_source_data_s.iloc[number_f: base_number, 5] = dataframe_source_data_s.iloc[: number_f, 5]
        dataframe_source_data_s.iloc[number_f: base_number, 6] = (
            dataframe_source_data_s.iloc[: number_f, 6] - z_polarized_beam_combining
        )
    
    for ii in range(number_s - 1):
        dataframe_source_data_s.iloc[base_number * (ii + 1): base_number * (ii + 2), 5] = (
            dataframe_source_data_s.iloc[: base_number, 5] + (ii + 1) * interval_s
        )
        dataframe_source_data_s.iloc[base_number * (ii + 1): base_number * (ii + 2), 6] = (
            dataframe_source_data_s.iloc[: base_number, 6] - (ii + 1) * z_spatial_beam_combining_s
        )
    
    # 构建快轴透镜数据
    # 注意：M2改变比例列需要支持 'auto' 字符串，因此使用 object 类型
    lens_f_columns = [
        '焦距/m', '位置t/m', '位置z/m', 'M2改变比例',
        '焦距/m', '位置t/m', '位置z/m', 'M2改变比例',
        '焦距/m', '位置t/m', '位置z/m', 'M2改变比例'
    ]
    dataframe_lens_data_f = DataFrame(full((total_number, 12), nan), columns=lens_f_columns)
    # 将 M2改变比例 列（索引 3, 7, 11）转换为 object 类型以支持混合类型
    for col_idx in [3, 7, 11]:
        dataframe_lens_data_f[lens_f_columns[col_idx]] = dataframe_lens_data_f[lens_f_columns[col_idx]].astype(object)
    
    dataframe_lens_data_f.iloc[:, 0] = collimation_lens_effective_focal_length_f
    dataframe_lens_data_f.iloc[:, 3] = 1.0  # 浮点数
    dataframe_lens_data_f.iloc[:, 4] = inf
    dataframe_lens_data_f.iloc[:, 7] = 'auto'  # 字符串
    dataframe_lens_data_f.iloc[:, 8] = coupling_lens_effective_focal_length_f
    dataframe_lens_data_f.iloc[:, 9] = (number_f - 1) / 2 * interval_f
    dataframe_lens_data_f.iloc[:, 10] = z_mirror_and_chip + z_coupling_lens_f_and_mirror
    dataframe_lens_data_f.iloc[:, 11] = 1.0  # 浮点数
    
    for ii in range(number_f):
        dataframe_lens_data_f.iloc[ii, 1] = ii * interval_f
        dataframe_lens_data_f.iloc[ii, 5] = ii * interval_f
        dataframe_lens_data_f.iloc[ii, 2] = collimation_lens_effective_focal_length_f - ii * z_spatial_beam_combining_f
        dataframe_lens_data_f.iloc[ii, 6] = z_mirror_and_chip - ii * z_spatial_beam_combining_f
    
    base_number = number_f
    if z_polarized_beam_combining:
        base_number *= 2
        dataframe_lens_data_f.iloc[number_f: base_number, 1] = dataframe_lens_data_f.iloc[: number_f, 1]
        dataframe_lens_data_f.iloc[number_f: base_number, 5] = dataframe_lens_data_f.iloc[: number_f, 5]
        dataframe_lens_data_f.iloc[number_f: base_number, 2] = (
            dataframe_lens_data_f.iloc[: number_f, 2] - z_polarized_beam_combining
        )
        dataframe_lens_data_f.iloc[number_f: base_number, 6] = (
            dataframe_lens_data_f.iloc[: number_f, 6] - z_polarized_beam_combining
        )
    
    for ii in range(number_s - 1):
        dataframe_lens_data_f.iloc[base_number * (ii + 1): base_number * (ii + 2), 1] = (
            dataframe_lens_data_f.iloc[: base_number, 1]
        )
        dataframe_lens_data_f.iloc[base_number * (ii + 1): base_number * (ii + 2), 5] = (
            dataframe_lens_data_f.iloc[: base_number, 5]
        )
        dataframe_lens_data_f.iloc[base_number * (ii + 1): base_number * (ii + 2), 2] = (
            dataframe_lens_data_f.iloc[: base_number, 2] - (ii + 1) * z_spatial_beam_combining_s
        )
        dataframe_lens_data_f.iloc[base_number * (ii + 1): base_number * (ii + 2), 6] = (
            dataframe_lens_data_f.iloc[: base_number, 6] - (ii + 1) * z_spatial_beam_combining_s
        )
    
    # 构建慢轴透镜数据
    dataframe_lens_data_s = DataFrame(full((total_number, 8), nan), columns=[
        '焦距/m', '位置t/m', '位置z/m', 'M2改变比例',
        '焦距/m', '位置t/m', '位置z/m', 'M2改变比例'
    ])
    dataframe_lens_data_s.iloc[:, 0] = collimation_lens_effective_focal_length_s
    dataframe_lens_data_s.iloc[:, 3] = 1
    dataframe_lens_data_s.iloc[:, 4] = coupling_lens_effective_focal_length_s
    dataframe_lens_data_s.iloc[:, 5] = (number_s - 1) / 2 * interval_s
    dataframe_lens_data_s.iloc[:, 6] = (
        z_mirror_and_chip + z_coupling_lens_f_and_mirror +
        coupling_lens_effective_focal_length_f - coupling_lens_effective_focal_length_s
    )
    dataframe_lens_data_s.iloc[:, 7] = 1
    dataframe_lens_data_s.iloc[: number_f, 1] = 0
    
    for ii in range(number_f):
        dataframe_lens_data_s.iloc[ii, 2] = (
            collimation_lens_effective_focal_length_s - astigmatism - ii * z_spatial_beam_combining_f
        )
    
    base_number = number_f
    if z_polarized_beam_combining:
        base_number *= 2
        dataframe_lens_data_s.iloc[number_f: base_number, 1] = dataframe_lens_data_s.iloc[: number_f, 1]
        dataframe_lens_data_s.iloc[number_f: base_number, 2] = (
            dataframe_lens_data_s.iloc[: number_f, 2] - z_polarized_beam_combining
        )
    
    for ii in range(number_s - 1):
        dataframe_lens_data_s.iloc[base_number * (ii + 1): base_number * (ii + 2), 1] = (
            dataframe_lens_data_s.iloc[: base_number, 1] + (ii + 1) * interval_s
        )
        dataframe_lens_data_s.iloc[base_number * (ii + 1): base_number * (ii + 2), 2] = (
            dataframe_lens_data_s.iloc[: base_number, 2] - (ii + 1) * z_spatial_beam_combining_s
        )
    
    # 构建光纤数据
    dataframe_fiber_data = DataFrame(
        [[
            fiber_core_diameter, fiber_cladding_diameter, fiber_na, index_fiber_core, index_environment,
            fiber_coiling_radius, (number_s - 1) / 2 * interval_s, (number_f - 1) / 2 * interval_f,
            z_mirror_and_chip + z_coupling_lens_f_and_mirror + coupling_lens_effective_focal_length_f
        ]],
        columns=[
            '纤芯直径/m', '包层直径/m', 'NA', '纤芯折射率', '外部环境折射率', 
            '光纤盘绕直径/m', '位置x/m', '位置y/m', '位置z/m'
        ]
    )
    
    return {
        'source_data_f': dataframe_source_data_f,
        'source_data_s': dataframe_source_data_s,
        'lens_data_f': dataframe_lens_data_f,
        'lens_data_s': dataframe_lens_data_s,
        'fiber_data': dataframe_fiber_data,
    }


def parameters_convert_to_excel(config: Optional[Dict[str, Any]] = None) -> None:
    """将配置参数转换并写入 Excel 文件（向后兼容）
    
    Args:
        config: 配置字典。如果为 None，则尝试从 JSON 文件加载
    """
    data = parameters_convert(config)
    
    with ExcelWriter(join(assets_path, '__LD光纤耦合参数.xlsx')) as excel_writer:
        data['source_data_f'].to_excel(excel_writer, sheet_name='光源数据_快轴', index=False)
        data['source_data_s'].to_excel(excel_writer, sheet_name='光源数据_慢轴', index=False)
        data['lens_data_f'].to_excel(excel_writer, sheet_name='透镜数据_快轴', index=False)
        data['lens_data_s'].to_excel(excel_writer, sheet_name='透镜数据_慢轴', index=False)
        data['fiber_data'].to_excel(excel_writer, sheet_name='光纤数据', index=False)
