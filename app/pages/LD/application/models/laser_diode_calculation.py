from __future__ import annotations

from copy import deepcopy
from os.path import exists
from typing import Dict, List, Optional, Tuple, Union

from numpy import inf, meshgrid, array, arange, arcsin, tan, sqrt, exp, where, equal, nan, ndarray
from pandas import read_excel, DataFrame
from scipy.integrate import quad

from .fresnel_equation import fresnel_equation_calculate
from .gaussian_beam import GaussianBeam
from .parallel_utils import (
    ParallelConfig,
    get_default_config,
    parallel_map,
    vectorized_energy_ratio,
)
from .width_calculate_one_dimension import width_calculate_one_dimension


class LaserDiodeCalculation:
    __slots__ = ('gaussian_beam_f_list', 'beam_cutting_list', 'beam_cutting_energy_ratio_list', 'm2_ratio_list',
                 'steps_f_list', 'source_intensity_f', 'gaussian_beam_s_list', 'source_intensity_s', 'lens_f_list',
                 'lens_s_list', 'fiber', 'parallel_config')

    def __init__(
        self,
        data: Optional[Union[str, Dict[str, DataFrame]]] = None,
        parallel_config: Optional[ParallelConfig] = None,
    ):
        """初始化激光二极管计算
        
        Args:
            data: 可以是以下类型之一：
                - str: Excel 文件路径（向后兼容）
                - Dict[str, DataFrame]: 包含计算数据的字典（来自 parameters_convert）
                - None: 仅初始化空列表，需要后续调用 get_from_dict 或 get_from_excel
            parallel_config: 并行计算配置，默认使用自动配置
        """
        self.gaussian_beam_f_list: list[GaussianBeam] = []
        self.beam_cutting_list = []
        self.beam_cutting_energy_ratio_list = []
        self.m2_ratio_list = []
        self.steps_f_list = []
        self.source_intensity_f = []
        self.gaussian_beam_s_list: list[GaussianBeam] = []
        self.source_intensity_s = []
        self.lens_f_list: list[tuple] = []
        self.lens_s_list: list[tuple] = []
        self.fiber = []
        self.parallel_config = parallel_config or get_default_config()

        if data is not None:
            if isinstance(data, str):
                # 向后兼容：Excel 文件路径
                self.get_from_excel(data)
            elif isinstance(data, dict):
                # 新方式：字典数据
                self.get_from_dict(data)
            self.judge_list_length()
            self.beam_spreading_calculate()

    def get_from_excel(self, excel_data_path):
        if not exists(excel_data_path):
            raise ValueError('数据不存在')

        dataframe_light_source_f = read_excel(excel_data_path, sheet_name='光源数据_快轴')
        self.gaussian_beam_f_list = []
        self.beam_cutting_list = []
        self.steps_f_list = []
        self.source_intensity_f = []
        for ii in range(dataframe_light_source_f.shape[0]):
            value = dataframe_light_source_f.iloc[ii, :5].values.tolist()
            value.append((dataframe_light_source_f.iloc[ii, 5], dataframe_light_source_f.iloc[ii, 6]))
            self.gaussian_beam_f_list.append(GaussianBeam(*value))
            self.beam_cutting_list.append(dataframe_light_source_f.iloc[ii, 7])
            if dataframe_light_source_f.shape[0] == 1:
                self.steps_f_list.append(None)
            else:
                if dataframe_light_source_f.iloc[ii, 7] == 1:
                    self.steps_f_list.append(dataframe_light_source_f.iloc[ii + 1, 5] - dataframe_light_source_f.iloc[ii, 5])
                elif dataframe_light_source_f.iloc[ii, 7] == 2:
                    self.steps_f_list.append(
                        ((dataframe_light_source_f.iloc[ii + 1, 5] - dataframe_light_source_f.iloc[ii, 5]) +
                         (dataframe_light_source_f.iloc[ii, 5] - dataframe_light_source_f.iloc[ii - 1, 5])) / 2)
                elif dataframe_light_source_f.iloc[ii, 7] == -1:
                    self.steps_f_list.append(dataframe_light_source_f.iloc[ii, 5] - dataframe_light_source_f.iloc[ii - 1, 5])
                else:
                    self.steps_f_list.append(None)
            self.source_intensity_f.append(dataframe_light_source_f.iloc[ii, 8])

        dataframe_light_source_s = read_excel(excel_data_path, sheet_name='光源数据_慢轴')
        self.gaussian_beam_s_list = []
        self.source_intensity_s = []
        for ii in range(dataframe_light_source_s.shape[0]):
            value = dataframe_light_source_s.iloc[ii, :5].values.tolist()
            value.append((dataframe_light_source_s.iloc[ii, 5], dataframe_light_source_s.iloc[ii, 6]))
            self.gaussian_beam_s_list.append(GaussianBeam(*value))
            self.source_intensity_s.append(dataframe_light_source_s.iloc[ii, 7])

        dataframe_lens_f = read_excel(excel_data_path, sheet_name='透镜数据_快轴')
        self.lens_f_list = []
        for ii in range(dataframe_lens_f.shape[0]):
            value = []
            for jj in range(int(dataframe_lens_f.shape[1] / 4)):
                value.append((dataframe_lens_f.iloc[ii, jj * 4], (dataframe_lens_f.iloc[ii, jj * 4 + 1],
                                                                  dataframe_lens_f.iloc[ii, jj * 4 + 2]),
                              dataframe_lens_f.iloc[ii, jj * 4 + 3]))
            self.lens_f_list.append(tuple(value))

        dataframe_lens_s = read_excel(excel_data_path, sheet_name='透镜数据_慢轴')
        self.lens_s_list = []
        for ii in range(dataframe_lens_s.shape[0]):
            value = []
            for jj in range(int(dataframe_lens_s.shape[1] / 4)):
                value.append((dataframe_lens_s.iloc[ii, jj * 4], (dataframe_lens_s.iloc[ii, jj * 4 + 1],
                                                                  dataframe_lens_s.iloc[ii, jj * 4 + 2]),
                              dataframe_lens_s.iloc[ii, jj * 4 + 3]))
            self.lens_s_list.append(tuple(value))

        dataframe_fiber = read_excel(excel_data_path, sheet_name='光纤数据')
        self.fiber = dataframe_fiber.iloc[0, :].values.tolist()

    def get_from_dict(self, data: Dict[str, DataFrame]) -> None:
        """从字典数据结构加载计算参数
        
        Args:
            data: 包含以下键的字典：
                - 'source_data_f': 快轴光源数据 DataFrame
                - 'source_data_s': 慢轴光源数据 DataFrame
                - 'lens_data_f': 快轴透镜数据 DataFrame
                - 'lens_data_s': 慢轴透镜数据 DataFrame
                - 'fiber_data': 光纤数据 DataFrame
        """
        # 快轴光源数据
        dataframe_light_source_f = data['source_data_f']
        self.gaussian_beam_f_list = []
        self.beam_cutting_list = []
        self.steps_f_list = []
        self.source_intensity_f = []
        for ii in range(dataframe_light_source_f.shape[0]):
            value = dataframe_light_source_f.iloc[ii, :5].values.tolist()
            value.append((dataframe_light_source_f.iloc[ii, 5], dataframe_light_source_f.iloc[ii, 6]))
            self.gaussian_beam_f_list.append(GaussianBeam(*value))
            self.beam_cutting_list.append(dataframe_light_source_f.iloc[ii, 7])
            if dataframe_light_source_f.shape[0] == 1:
                self.steps_f_list.append(None)
            else:
                if dataframe_light_source_f.iloc[ii, 7] == 1:
                    self.steps_f_list.append(dataframe_light_source_f.iloc[ii + 1, 5] - dataframe_light_source_f.iloc[ii, 5])
                elif dataframe_light_source_f.iloc[ii, 7] == 2:
                    self.steps_f_list.append(
                        ((dataframe_light_source_f.iloc[ii + 1, 5] - dataframe_light_source_f.iloc[ii, 5]) +
                         (dataframe_light_source_f.iloc[ii, 5] - dataframe_light_source_f.iloc[ii - 1, 5])) / 2)
                elif dataframe_light_source_f.iloc[ii, 7] == -1:
                    self.steps_f_list.append(dataframe_light_source_f.iloc[ii, 5] - dataframe_light_source_f.iloc[ii - 1, 5])
                else:
                    self.steps_f_list.append(None)
            self.source_intensity_f.append(dataframe_light_source_f.iloc[ii, 8])

        # 慢轴光源数据
        dataframe_light_source_s = data['source_data_s']
        self.gaussian_beam_s_list = []
        self.source_intensity_s = []
        for ii in range(dataframe_light_source_s.shape[0]):
            value = dataframe_light_source_s.iloc[ii, :5].values.tolist()
            value.append((dataframe_light_source_s.iloc[ii, 5], dataframe_light_source_s.iloc[ii, 6]))
            self.gaussian_beam_s_list.append(GaussianBeam(*value))
            self.source_intensity_s.append(dataframe_light_source_s.iloc[ii, 7])

        # 快轴透镜数据
        dataframe_lens_f = data['lens_data_f']
        self.lens_f_list = []
        for ii in range(dataframe_lens_f.shape[0]):
            value = []
            for jj in range(int(dataframe_lens_f.shape[1] / 4)):
                focal_length = float(dataframe_lens_f.iloc[ii, jj * 4])
                pos_t = float(dataframe_lens_f.iloc[ii, jj * 4 + 1])
                pos_z = float(dataframe_lens_f.iloc[ii, jj * 4 + 2])
                m2_ratio_raw = dataframe_lens_f.iloc[ii, jj * 4 + 3]
                # M2 比例可能是 'auto' 字符串或数值
                if isinstance(m2_ratio_raw, str) and m2_ratio_raw == 'auto':
                    m2_ratio = 'auto'
                else:
                    m2_ratio = float(m2_ratio_raw)
                value.append((focal_length, (pos_t, pos_z), m2_ratio))
            self.lens_f_list.append(tuple(value))

        # 慢轴透镜数据
        dataframe_lens_s = data['lens_data_s']
        self.lens_s_list = []
        for ii in range(dataframe_lens_s.shape[0]):
            value = []
            for jj in range(int(dataframe_lens_s.shape[1] / 4)):
                focal_length = float(dataframe_lens_s.iloc[ii, jj * 4])
                pos_t = float(dataframe_lens_s.iloc[ii, jj * 4 + 1])
                pos_z = float(dataframe_lens_s.iloc[ii, jj * 4 + 2])
                m2_ratio_raw = dataframe_lens_s.iloc[ii, jj * 4 + 3]
                # M2 比例可能是 'auto' 字符串或数值
                if isinstance(m2_ratio_raw, str) and m2_ratio_raw == 'auto':
                    m2_ratio = 'auto'
                else:
                    m2_ratio = float(m2_ratio_raw)
                value.append((focal_length, (pos_t, pos_z), m2_ratio))
            self.lens_s_list.append(tuple(value))

        # 光纤数据
        dataframe_fiber = data['fiber_data']
        self.fiber = dataframe_fiber.iloc[0, :].values.tolist()

    def judge_list_length(self):
        if (len(self.gaussian_beam_f_list) != len(self.gaussian_beam_s_list)
                and len(self.gaussian_beam_f_list) != len(self.beam_cutting_list)
                and len(self.gaussian_beam_f_list) != len(self.steps_f_list)
                and len(self.gaussian_beam_f_list) != len(self.source_intensity_f)
                and len(self.gaussian_beam_f_list) != len(self.source_intensity_s)
                and len(self.gaussian_beam_f_list) != len(self.lens_f_list)
                and len(self.gaussian_beam_f_list) != len(self.lens_s_list)):
            raise ValueError('光纤耦合参数设置错误')

    @staticmethod
    def _beam_spreading_single(
        gaussian_beam_f: GaussianBeam, 
        lens_f_: tuple, 
        beam_cutting: int, 
        steps_f: float
    ) -> Tuple[tuple, float, float]:
        """单个光束的展宽计算（无副作用，可并行）
        
        Returns:
            (updated_lens_f, beam_cutting_energy_ratio, m2_ratio)
        """
        def intensity_calculate(x: float | ndarray, w: float, order__: float):
            return exp(-2 * ((x / w)**2)**order__)

        lens_f = deepcopy(lens_f_)
        beam_cutting_energy_ratio = 1.0
        m2_ratio = 1.0
        
        for ii in range(len(lens_f)):
            if float(lens_f[ii][0]) == inf and str(lens_f[ii][2]) == 'auto':
                relationship_tuple = (lens_f[:ii], (0, (0, lens_f[ii][1][1])))
                wz = gaussian_beam_f.beam_radius_calculate(relationship_tuple)
                order_ = gaussian_beam_f.order_calculate(relationship_tuple)
                if beam_cutting == 1:
                    ratio1 = steps_f / wz / 2
                    ratio2 = None
                    integral_result1 = quad(intensity_calculate, -4 * wz, steps_f / 2, args=(wz, order_))[0]
                    integral_result2 = quad(intensity_calculate, -4 * wz, 4 * wz, args=(wz, order_))[0]
                    beam_cutting_energy_ratio = integral_result1 / integral_result2
                elif beam_cutting == -1:
                    ratio1 = None
                    ratio2 = steps_f / wz / 2
                    integral_result1 = quad(intensity_calculate, -steps_f / 2, 4 * wz, args=(wz, order_))[0]
                    integral_result2 = quad(intensity_calculate, -4 * wz, 4 * wz, args=(wz, order_))[0]
                    beam_cutting_energy_ratio = integral_result1 / integral_result2
                elif beam_cutting == 2:
                    ratio1 = steps_f / wz / 2
                    ratio2 = steps_f / wz / 2
                    integral_result1 = quad(intensity_calculate, -steps_f / 2, steps_f / 2, args=(wz, order_))[0]
                    integral_result2 = quad(intensity_calculate, -4 * wz, 4 * wz, args=(wz, order_))[0]
                    beam_cutting_energy_ratio = integral_result1 / integral_result2
                else:
                    ratio1 = None
                    ratio2 = None
                    beam_cutting_energy_ratio = 1
                m2_ratio = gaussian_beam_f.beam_spreading_calculate(ratio1, ratio2)
                lens_f = list(lens_f)
                lens_f[ii] = list(lens_f[ii])
                lens_f[ii][2] = m2_ratio
                lens_f[ii] = tuple(lens_f[ii])
                lens_f = tuple(lens_f)
        return lens_f, beam_cutting_energy_ratio, m2_ratio

    def beam_spreading_calculate(self):
        """计算光束展宽（并行优化版本）"""
        n_beams = len(self.gaussian_beam_f_list)
        use_parallel = self.parallel_config.enabled and n_beams >= 4
        
        if use_parallel:
            # 准备并行计算的参数
            def compute_single(idx: int) -> Tuple[tuple, float, float]:
                return self._beam_spreading_single(
                    self.gaussian_beam_f_list[idx],
                    self.lens_f_list[idx],
                    self.beam_cutting_list[idx],
                    self.steps_f_list[idx]
                )
            
            # 并行执行
            results = parallel_map(
                compute_single,
                list(range(n_beams)),
                max_workers=self.parallel_config.max_workers
            )
            
            # 收集结果
            for ii, (lens_f, energy_ratio, m2_ratio) in enumerate(results):
                self.lens_f_list[ii] = lens_f
                self.beam_cutting_energy_ratio_list.append(energy_ratio)
                self.m2_ratio_list.append(m2_ratio)
        else:
            # 串行执行（向后兼容）
            for ii in range(n_beams):
                lens_f, energy_ratio, m2_ratio = self._beam_spreading_single(
                    self.gaussian_beam_f_list[ii],
                    self.lens_f_list[ii],
                    self.beam_cutting_list[ii],
                    self.steps_f_list[ii]
                )
                self.lens_f_list[ii] = lens_f
                self.beam_cutting_energy_ratio_list.append(energy_ratio)
                self.m2_ratio_list.append(m2_ratio)

    @staticmethod
    def energy_ratio_in_circle_calculate(x_, y_, intensity_, x_center, y_center, radius):
        if x_.ndim == 1 and y_.ndim == 1:
            x, y = meshgrid(x_, y_)
        else:
            x = array(x_)
            y = array(y_)
        intensity1 = array(intensity_)
        intensity2 = array(intensity_)
        condition = ((x - x_center)**2 + (y - y_center)**2) > radius**2
        intensity2[condition] = 0
        return intensity2.sum() / intensity1.sum()

    def _compute_waist_and_rayleigh(self, idx: int, axis: str = 'f') -> Tuple[float, float]:
        """并行计算束腰和瑞利距离（辅助方法）"""
        if axis == 'f':
            gb = self.gaussian_beam_f_list[idx]
            lens = self.lens_f_list[idx]
        else:
            gb = self.gaussian_beam_s_list[idx]
            lens = self.lens_s_list[idx]
        relationship_tuple = (lens, (0, (0, 0)))
        w0, z, m2 = gb.waist_calculate(relationship_tuple)
        return z, gb.rayleigh_distance_calculate(w0, m2)

    def _compute_position_and_width(self, idx: int, z_values: List[float], axis: str = 'f') -> Tuple[float, float]:
        """并行计算位置和光束半径（辅助方法）"""
        if axis == 'f':
            gb = self.gaussian_beam_f_list[idx]
            lens = self.lens_f_list[idx]
        else:
            gb = self.gaussian_beam_s_list[idx]
            lens = self.lens_s_list[idx]
        relationship_tuple = (lens, (0, (0, z_values[idx])))
        position = gb.beam_position_calculate(relationship_tuple)
        wz = gb.beam_radius_calculate(relationship_tuple)
        return position, wz

    def _compute_intensity(self, idx: int, w_f: float, w_s: float, center_f: float, center_s: float,
                           z_f: float, z_s: float) -> Tuple[ndarray, ndarray, ndarray]:
        """并行计算光强分布（辅助方法）"""
        relationship_f = (self.lens_f_list[idx], (w_f, (center_f, z_f)))
        y, intensity_f = self.gaussian_beam_f_list[idx].intensity_distribution_calculate(relationship_f)
        intensity_f = intensity_f * self.source_intensity_f[idx]
        
        relationship_s = (self.lens_s_list[idx], (w_s, (center_s, z_s)))
        x, intensity_s = self.gaussian_beam_s_list[idx].intensity_distribution_calculate(relationship_s)
        intensity_s = intensity_s * self.source_intensity_s[idx]
        
        intensity_s_grid, intensity_f_grid = meshgrid(intensity_s, intensity_f)
        return x, y, intensity_f_grid * intensity_s_grid

    def na_and_coupling_calculate(self):
        """计算 NA 和耦合效率（并行优化版本）"""
        n_beams = len(self.gaussian_beam_f_list)
        use_parallel = self.parallel_config.enabled and n_beams >= 4
        max_workers = self.parallel_config.max_workers if use_parallel else 1
        
        # === 阶段 1: 计算 z 位置和瑞利距离 ===
        if use_parallel:
            # 并行计算快轴
            results_f = parallel_map(
                lambda i: self._compute_waist_and_rayleigh(i, 'f'),
                list(range(n_beams)),
                max_workers=max_workers
            )
            z_list_f = [r[0] for r in results_f]
            rayleigh_f = [r[1] for r in results_f]
            
            # 并行计算慢轴
            results_s = parallel_map(
                lambda i: self._compute_waist_and_rayleigh(i, 's'),
                list(range(n_beams)),
                max_workers=max_workers
            )
            z_list_s = [r[0] for r in results_s]
            rayleigh_s = [r[1] for r in results_s]
            rayleigh_distance_list = rayleigh_f + rayleigh_s
        else:
            z_list_f = []
            rayleigh_distance_list = []
            for ii in range(n_beams):
                relationship_tuple = (self.lens_f_list[ii], (0, (0, 0)))
                w0, z, m2 = self.gaussian_beam_f_list[ii].waist_calculate(relationship_tuple)
                z_list_f.append(z)
                rayleigh_distance_list.append(self.gaussian_beam_f_list[ii].rayleigh_distance_calculate(w0, m2))
            z_list_s = []
            for ii in range(n_beams):
                relationship_tuple = (self.lens_s_list[ii], (0, (0, 0)))
                w0, z, m2 = self.gaussian_beam_s_list[ii].waist_calculate(relationship_tuple)
                z_list_s.append(z)
                rayleigh_distance_list.append(self.gaussian_beam_s_list[ii].rayleigh_distance_calculate(w0, m2))
        
        maximum_rayleigh_distance = max(rayleigh_distance_list)
        ll = 8 * maximum_rayleigh_distance
        z_fiber_far_field_f = [z + ll for z in z_list_f]
        z_fiber_far_field_s = [z + ll for z in z_list_s]

        # === 阶段 2: 计算远场位置和光束宽度 ===
        if use_parallel:
            results_f = parallel_map(
                lambda i: self._compute_position_and_width(i, z_fiber_far_field_f, 'f'),
                list(range(n_beams)),
                max_workers=max_workers
            )
            position_f = [r[0] for r in results_f]
            wz_f = [r[1] for r in results_f]
            
            results_s = parallel_map(
                lambda i: self._compute_position_and_width(i, z_fiber_far_field_s, 's'),
                list(range(n_beams)),
                max_workers=max_workers
            )
            position_s = [r[0] for r in results_s]
            wz_s = [r[1] for r in results_s]
        else:
            position_f, wz_f = [], []
            for ii in range(n_beams):
                relationship_tuple = (self.lens_f_list[ii], (0, (0, z_fiber_far_field_f[ii])))
                pos = self.gaussian_beam_f_list[ii].beam_position_calculate(relationship_tuple)
                wz = self.gaussian_beam_f_list[ii].beam_radius_calculate(relationship_tuple)
                position_f.append(pos)
                wz_f.append(wz)
            
            position_s, wz_s = [], []
            for ii in range(n_beams):
                relationship_tuple = (self.lens_s_list[ii], (0, (0, z_fiber_far_field_s[ii])))
                pos = self.gaussian_beam_s_list[ii].beam_position_calculate(relationship_tuple)
                wz = self.gaussian_beam_s_list[ii].beam_radius_calculate(relationship_tuple)
                position_s.append(pos)
                wz_s.append(wz)
        
        center_far_field_f = sum(position_f) / len(position_f)
        wz_and_position_f = []
        for p, w in zip(position_f, wz_f):
            wz_and_position_f.extend([p + 4 * w, p - 4 * w])
        w_far_field_f = max(wz_and_position_f) - min(wz_and_position_f)
        
        center_far_field_s = sum(position_s) / len(position_s)
        wz_and_position_s = []
        for p, w in zip(position_s, wz_s):
            wz_and_position_s.extend([p + 4 * w, p - 4 * w])
        w_far_field_s = max(wz_and_position_s) - min(wz_and_position_s)

        # === 阶段 3: 计算远场光强分布 ===
        if use_parallel:
            intensity_results = parallel_map(
                lambda i: self._compute_intensity(
                    i, w_far_field_f, w_far_field_s, center_far_field_f, center_far_field_s,
                    z_fiber_far_field_f[i], z_fiber_far_field_s[i]
                ),
                list(range(n_beams)),
                max_workers=max_workers
            )
            x_far_field = intensity_results[0][0]
            y_far_field = intensity_results[0][1]
            intensity_far_field = [r[2] for r in intensity_results]
        else:
            x_far_field = None
            y_far_field = None
            intensity_far_field = []
            for ii in range(n_beams):
                relationship_tuple = (self.lens_f_list[ii], (w_far_field_f, (center_far_field_f, z_fiber_far_field_f[ii])))
                y_far_field, intensity_f = self.gaussian_beam_f_list[ii].intensity_distribution_calculate(relationship_tuple)
                intensity_f *= self.source_intensity_f[ii]
                relationship_tuple = (self.lens_s_list[ii], (w_far_field_s, (center_far_field_s, z_fiber_far_field_s[ii])))
                x_far_field, intensity_s = self.gaussian_beam_s_list[ii].intensity_distribution_calculate(relationship_tuple)
                intensity_s *= self.source_intensity_s[ii]
                intensity_s, intensity_f = meshgrid(intensity_s, intensity_f)
                intensity_far_field.append(intensity_f * intensity_s)
        
        if not intensity_far_field:
            raise ValueError('光斑数量设置错误')
        x_far_field, y_far_field = meshgrid(x_far_field, y_far_field)
        intensity_far_field = array(sum(intensity_far_field))

        # === 阶段 4: 计算 NA 比例（向量化优化）===
        na = arange(0, self.fiber[2] + 0.005, 0.005)
        radii = tan(arcsin(na / self.fiber[4])) * ll
        na_ratio = vectorized_energy_ratio(
            x_far_field, y_far_field, intensity_far_field,
            self.fiber[6], self.fiber[7], radii
        ).tolist()
        
        na_fiber_diameter = 2 * tan(arcsin(self.fiber[2] / self.fiber[4])) * ll
        na_ratio_fiber = na_ratio[-1]
        if na_ratio_fiber > 0:
            na_ratio = [r / na_ratio_fiber for r in na_ratio]

        index_cladding_square = self.fiber[3]**2 - self.fiber[2]**2
        na_cladding = sqrt(index_cladding_square - self.fiber[4]**2)
        if na_cladding >= 1:
            na_ratio_cladding = 1
        else:
            na_ratio_cladding = self.energy_ratio_in_circle_calculate(
                x_far_field, y_far_field, intensity_far_field,
                self.fiber[6], self.fiber[7],
                tan(arcsin(na_cladding / self.fiber[4])) * ll
            )

        # === 阶段 5: 计算近场 ===
        z_fiber_near_field = self.fiber[8]

        if use_parallel:
            # 并行计算近场位置
            results_f = parallel_map(
                lambda i: self._compute_position_and_width(i, [z_fiber_near_field] * n_beams, 'f'),
                list(range(n_beams)),
                max_workers=max_workers
            )
            position_near_f = [r[0] for r in results_f]
            wz_near_f = [r[1] for r in results_f]
            
            results_s = parallel_map(
                lambda i: self._compute_position_and_width(i, [z_fiber_near_field] * n_beams, 's'),
                list(range(n_beams)),
                max_workers=max_workers
            )
            position_near_s = [r[0] for r in results_s]
            wz_near_s = [r[1] for r in results_s]
        else:
            position_near_f, wz_near_f = [], []
            for ii in range(n_beams):
                relationship_tuple = (self.lens_f_list[ii], (0, (0, z_fiber_near_field)))
                pos = self.gaussian_beam_f_list[ii].beam_position_calculate(relationship_tuple)
                wz = self.gaussian_beam_f_list[ii].beam_radius_calculate(relationship_tuple)
                position_near_f.append(pos)
                wz_near_f.append(wz)
            
            position_near_s, wz_near_s = [], []
            for ii in range(n_beams):
                relationship_tuple = (self.lens_s_list[ii], (0, (0, z_fiber_near_field)))
                pos = self.gaussian_beam_s_list[ii].beam_position_calculate(relationship_tuple)
                wz = self.gaussian_beam_s_list[ii].beam_radius_calculate(relationship_tuple)
                position_near_s.append(pos)
                wz_near_s.append(wz)
        
        center_near_field_f = sum(position_near_f) / len(position_near_f)
        wz_and_position_near_f = []
        for p, w in zip(position_near_f, wz_near_f):
            wz_and_position_near_f.extend([p + 4 * w, p - 4 * w])
        w_near_field_f = max(wz_and_position_near_f) - min(wz_and_position_near_f)
        
        center_near_field_s = sum(position_near_s) / len(position_near_s)
        wz_and_position_near_s = []
        for p, w in zip(position_near_s, wz_near_s):
            wz_and_position_near_s.extend([p + 4 * w, p - 4 * w])
        w_near_field_s = max(wz_and_position_near_s) - min(wz_and_position_near_s)

        # === 阶段 6: 计算近场光强分布 ===
        z_near_list = [z_fiber_near_field] * n_beams
        if use_parallel:
            intensity_results = parallel_map(
                lambda i: self._compute_intensity(
                    i, w_near_field_f, w_near_field_s, center_near_field_f, center_near_field_s,
                    z_near_list[i], z_near_list[i]
                ),
                list(range(n_beams)),
                max_workers=max_workers
            )
            x_near_field = intensity_results[0][0]
            y_near_field = intensity_results[0][1]
            intensity_near_field = [r[2] for r in intensity_results]
        else:
            x_near_field = None
            y_near_field = None
            intensity_near_field = []
            for ii in range(n_beams):
                relationship_tuple = (self.lens_f_list[ii], (w_near_field_f, (center_near_field_f, z_fiber_near_field)))
                y_near_field, intensity_f = self.gaussian_beam_f_list[ii].intensity_distribution_calculate(relationship_tuple)
                intensity_f *= self.source_intensity_f[ii]
                relationship_tuple = (self.lens_s_list[ii], (w_near_field_s, (center_near_field_s, z_fiber_near_field)))
                x_near_field, intensity_s = self.gaussian_beam_s_list[ii].intensity_distribution_calculate(relationship_tuple)
                intensity_s *= self.source_intensity_s[ii]
                intensity_s, intensity_f = meshgrid(intensity_s, intensity_f)
                intensity_near_field.append(intensity_f * intensity_s)
        
        if not intensity_near_field:
            raise ValueError('光斑数量设置错误')
        x_near_field, y_near_field = meshgrid(x_near_field, y_near_field)
        intensity_near_field = array(sum(intensity_near_field))

        # === 阶段 7: 计算耦合效率 ===
        energy_ratio_in_fiber_core = self.energy_ratio_in_circle_calculate(
            x_near_field, y_near_field, intensity_near_field,
            self.fiber[6], self.fiber[7], self.fiber[0] / 2
        )
        fiber_output_transmittance = 1 - fresnel_equation_calculate(self.fiber[3], self.fiber[4], 0)
        coupling_efficiency = energy_ratio_in_fiber_core * na_ratio_fiber * 0.995**5 * fiber_output_transmittance

        energy_ratio_in_fiber = self.energy_ratio_in_circle_calculate(
            x_near_field, y_near_field, intensity_near_field,
            self.fiber[6], self.fiber[7], self.fiber[1] / 2
        )
        energy_ratio_in_fiber_cladding = energy_ratio_in_fiber - energy_ratio_in_fiber_core
        cladding_light_energy_ratio = energy_ratio_in_fiber_cladding * na_ratio_cladding + 1 - na_ratio_fiber

        e2_width_near_field = self.width_calculate(x_near_field, y_near_field, intensity_near_field, 1 / exp(2))

        return (x_far_field, y_far_field, intensity_far_field, self.fiber[6], self.fiber[7], na_fiber_diameter, na, na_ratio,
                x_near_field, y_near_field, intensity_near_field, self.fiber[6], self.fiber[7], self.fiber[0], self.fiber[1],
                coupling_efficiency, cladding_light_energy_ratio, e2_width_near_field)

    def _compute_trace(self, idx: int, z: float) -> Tuple[List, List]:
        """并行计算光线追迹（辅助方法）"""
        relationship_f = (self.lens_f_list[idx], (0, (0, z)))
        trace_f = self.gaussian_beam_f_list[idx].beam_outline_calculate(relationship_f)
        relationship_s = (self.lens_s_list[idx], (0, (0, z)))
        trace_s = self.gaussian_beam_s_list[idx].beam_outline_calculate(relationship_s)
        return trace_f, trace_s

    def trace_calculate(self):
        """计算光线追迹（并行优化版本）"""
        z_fiber_near_field_plus_1mm = self.fiber[8] + 1e-3
        n_beams = len(self.gaussian_beam_f_list)
        use_parallel = self.parallel_config.enabled and n_beams >= 4
        
        if use_parallel:
            results = parallel_map(
                lambda i: self._compute_trace(i, z_fiber_near_field_plus_1mm),
                list(range(n_beams)),
                max_workers=self.parallel_config.max_workers
            )
            trace_f_list = [r[0] for r in results]
            trace_s_list = [r[1] for r in results]
        else:
            trace_f_list = []
            trace_s_list = []
            for ii in range(n_beams):
                relationship_tuple = (self.lens_f_list[ii], (0, (0, z_fiber_near_field_plus_1mm)))
                trace_f_list.append(self.gaussian_beam_f_list[ii].beam_outline_calculate(relationship_tuple))
                relationship_tuple = (self.lens_s_list[ii], (0, (0, z_fiber_near_field_plus_1mm)))
                trace_s_list.append(self.gaussian_beam_s_list[ii].beam_outline_calculate(relationship_tuple))
        
        return trace_f_list, trace_s_list

    @staticmethod
    def width_calculate(x_, y_, intensity_, ratio):
        if x_.ndim == 1 and y_.ndim == 1:
            x, y = meshgrid(x_, y_)
        else:
            x = array(x_)
            y = array(y_)
        intensity = array(intensity_)

        intensity_maximum = intensity.max()
        center_y, center_x = where(intensity == intensity_maximum)
        center_x = round(center_x.sum() / center_x.size)
        center_y = round(center_y.sum() / center_y.size)

        x_x = x[center_y, :]
        y_y = y[:, center_x]

        intensity_x = intensity[center_y, :]
        intensity_y = intensity[:, center_x]

        width_x = width_calculate_one_dimension(x_x, intensity_x, ratio)
        width_y = width_calculate_one_dimension(y_y, intensity_y, ratio)
        return width_x, width_y

    @staticmethod
    def judge_coupling_lens(lens_list):
        coupling_lens_columns = [[] for _ in range(len(lens_list))]
        if len(lens_list) > 1:
            dataframe_lens_list = DataFrame(lens_list)
            for ii in range(dataframe_lens_list.shape[0]):
                for jj in range(dataframe_lens_list.shape[1]):
                    dataframe_lens_list.iloc[ii,
                                             jj] = (dataframe_lens_list.iloc[ii,
                                                                             jj][0], dataframe_lens_list.iloc[ii,
                                                                                                              jj][1], nan)
            for jj in range(dataframe_lens_list.shape[1]):
                dataframe_judge = dataframe_lens_list.isin((dataframe_lens_list.iloc[0, jj], ))
                if dataframe_judge.sum(axis=1).prod():
                    judge_state = where(equal(dataframe_judge.values, True))
                    [
                        coupling_lens_columns[judge_state[0][ii]].append(judge_state[1][ii])
                        for ii in range(judge_state[0].size)
                    ]
        return coupling_lens_columns

    def beam_e2_width_on_lens_f_calculate(self, channel_number):
        if channel_number > len(self.lens_f_list) - 1 or channel_number > len(self.lens_s_list) - 1:
            raise ValueError('通道数设置错误')

        coupling_lens_columns_f = self.judge_coupling_lens(self.lens_f_list)[channel_number]

        lens_f = self.lens_f_list[channel_number]
        lens_s = self.lens_s_list[channel_number]

        x_list = []
        y_list = []
        intensity_list = []
        e2_width_list = []
        for ii in range(len(lens_f)):
            if ii not in coupling_lens_columns_f:
                lens_f_ = lens_f[:ii]
                lens_s_ = []
                for jj in lens_s:
                    if jj[1][1] < lens_f[ii][1][1]:
                        lens_s_.append(jj)
                lens_s_ = tuple(lens_s_)

                z = lens_f[ii][1][1]

                relationship = (lens_f_, (0, (0, z)))
                t = self.gaussian_beam_f_list[channel_number].beam_position_calculate(relationship)
                wz = self.gaussian_beam_f_list[channel_number].beam_radius_calculate(relationship)
                w = 8 * wz
                relationship = (lens_f_, (w, (t, z)))
                y, intensity_f = self.gaussian_beam_f_list[channel_number].intensity_distribution_calculate(relationship)
                intensity_f *= self.source_intensity_f[channel_number]

                relationship = (lens_s_, (0, (0, z)))
                t = self.gaussian_beam_s_list[channel_number].beam_position_calculate(relationship)
                wz = self.gaussian_beam_s_list[channel_number].beam_radius_calculate(relationship)
                w = 8 * wz
                relationship = (lens_s_, (w, (t, z)))
                x, intensity_s = self.gaussian_beam_s_list[channel_number].intensity_distribution_calculate(relationship)
                intensity_s *= self.source_intensity_s[channel_number]

                x, y = meshgrid(x, y)
                intensity_s, intensity_f = meshgrid(intensity_s, intensity_f)
                intensity = intensity_f * intensity_s

                e2_width = self.width_calculate(x, y, intensity, 1 / exp(2))

                x_list.append(x)
                y_list.append(y)
                intensity_list.append(intensity)
                e2_width_list.append(e2_width)

        return x_list, y_list, intensity_list, e2_width_list

    def beam_e2_width_on_lens_s_calculate(self, channel_number):
        if channel_number > len(self.lens_f_list) - 1 or channel_number > len(self.lens_s_list) - 1:
            raise ValueError('通道数设置错误')

        coupling_lens_columns_s = self.judge_coupling_lens(self.lens_s_list)[channel_number]

        lens_f = self.lens_f_list[channel_number]
        lens_s = self.lens_s_list[channel_number]

        x_list = []
        y_list = []
        intensity_list = []
        e2_width_list = []
        for ii in range(len(lens_s)):
            if ii not in coupling_lens_columns_s:
                lens_f_ = []
                lens_s_ = lens_s[:ii]
                for jj in lens_f:
                    if jj[1][1] < lens_s[ii][1][1]:
                        lens_f_.append(jj)
                lens_f_ = tuple(lens_f_)

                z = lens_s[ii][1][1]

                relationship = (lens_f_, (0, (0, z)))
                t = self.gaussian_beam_f_list[channel_number].beam_position_calculate(relationship)
                wz = self.gaussian_beam_f_list[channel_number].beam_radius_calculate(relationship)
                w = 8 * wz
                relationship = (lens_f_, (w, (t, z)))
                y, intensity_f = self.gaussian_beam_f_list[channel_number].intensity_distribution_calculate(relationship)
                intensity_f *= self.source_intensity_f[channel_number]

                relationship = (lens_s_, (0, (0, z)))
                t = self.gaussian_beam_s_list[channel_number].beam_position_calculate(relationship)
                wz = self.gaussian_beam_s_list[channel_number].beam_radius_calculate(relationship)
                w = 8 * wz
                relationship = (lens_s_, (w, (t, z)))
                x, intensity_s = self.gaussian_beam_s_list[channel_number].intensity_distribution_calculate(relationship)
                intensity_s *= self.source_intensity_s[channel_number]

                x, y = meshgrid(x, y)
                intensity_s, intensity_f = meshgrid(intensity_s, intensity_f)
                intensity = intensity_f * intensity_s

                e2_width = self.width_calculate(x, y, intensity, 1 / exp(2))

                x_list.append(x)
                y_list.append(y)
                intensity_list.append(intensity)
                e2_width_list.append(e2_width)

        return x_list, y_list, intensity_list, e2_width_list

    def beam_e2_width_on_coupling_lens_f_calculate(self):
        coupling_lens_columns_f = self.judge_coupling_lens(self.lens_f_list)

        x_list = []
        y_list = []
        intensity_list = []
        e2_width_list = []
        for ii in range(len(coupling_lens_columns_f[0])):
            position = []
            wz_and_position = []
            for jj in range(len(self.gaussian_beam_f_list)):
                z = self.lens_f_list[jj][coupling_lens_columns_f[jj][ii]][1][1]

                relationship_tuple = (self.lens_f_list[jj][:coupling_lens_columns_f[jj][ii]], (0, (0, z)))
                position_ = self.gaussian_beam_f_list[jj].beam_position_calculate(relationship_tuple)
                position.append(position_)
                wz = self.gaussian_beam_f_list[jj].beam_radius_calculate(relationship_tuple)
                wz_and_position.extend([position_ + 4 * wz, position_ - 4 * wz])
            center_f = sum(position) / len(position)
            w_f = max(wz_and_position) - min(wz_and_position)

            position = []
            wz_and_position = []
            for jj in range(len(self.gaussian_beam_s_list)):
                z = self.lens_f_list[jj][coupling_lens_columns_f[jj][ii]][1][1]
                lens_s = self.lens_s_list[jj]
                lens_s_ = []
                for kk in lens_s:
                    if kk[1][1] <= z:
                        lens_s_.append(kk)
                lens_s_ = tuple(lens_s_)

                relationship_tuple = (lens_s_, (0, (0, z)))
                position_ = self.gaussian_beam_s_list[jj].beam_position_calculate(relationship_tuple)
                position.append(position_)
                wz = self.gaussian_beam_s_list[jj].beam_radius_calculate(relationship_tuple)
                wz_and_position.extend([position_ + 4 * wz, position_ - 4 * wz])
            center_s = sum(position) / len(position)
            w_s = max(wz_and_position) - min(wz_and_position)

            x = None
            y = None
            intensity = []
            for jj in range(len(self.gaussian_beam_f_list)):
                z = self.lens_f_list[jj][coupling_lens_columns_f[jj][ii]][1][1]
                lens_s = self.lens_s_list[jj]
                lens_s_ = []
                for kk in lens_s:
                    if kk[1][1] <= z:
                        lens_s_.append(kk)
                lens_s_ = tuple(lens_s_)

                relationship_tuple = (self.lens_f_list[jj][:coupling_lens_columns_f[jj][ii]], (w_f, (center_f, z)))
                y, intensity_f = self.gaussian_beam_f_list[jj].intensity_distribution_calculate(relationship_tuple)
                intensity_f *= self.source_intensity_f[jj]
                relationship_tuple = (lens_s_, (w_s, (center_s, z)))
                x, intensity_s = self.gaussian_beam_s_list[jj].intensity_distribution_calculate(relationship_tuple)
                intensity_s *= self.source_intensity_s[jj]
                intensity_s, intensity_f = meshgrid(intensity_s, intensity_f)
                intensity.append(intensity_f * intensity_s)
            if not (x is not None and y is not None and intensity):
                raise ValueError('光斑数量为0')
            x, y = meshgrid(x, y)
            intensity = array(sum(intensity))

            e2_width = self.width_calculate(x, y, intensity, 1 / exp(2))

            x_list.append(x)
            y_list.append(y)
            intensity_list.append(intensity)
            e2_width_list.append(e2_width)

        return x_list, y_list, intensity_list, e2_width_list

    def beam_e2_width_on_coupling_lens_s_calculate(self):
        coupling_lens_columns_s = self.judge_coupling_lens(self.lens_s_list)

        x_list = []
        y_list = []
        intensity_list = []
        e2_width_list = []
        for ii in range(len(coupling_lens_columns_s[0])):
            position = []
            wz_and_position = []
            for jj in range(len(self.gaussian_beam_f_list)):
                z = self.lens_s_list[jj][coupling_lens_columns_s[jj][ii]][1][1]
                lens_f = self.lens_f_list[jj]
                lens_f_ = []
                for kk in lens_f:
                    if kk[1][1] <= z:
                        lens_f_.append(kk)
                lens_f_ = tuple(lens_f_)

                relationship_tuple = (lens_f_, (0, (0, z)))
                position_ = self.gaussian_beam_f_list[jj].beam_position_calculate(relationship_tuple)
                position.append(position_)
                wz = self.gaussian_beam_f_list[jj].beam_radius_calculate(relationship_tuple)
                wz_and_position.extend([position_ + 4 * wz, position_ - 4 * wz])
            center_f = sum(position) / len(position)
            w_f = max(wz_and_position) - min(wz_and_position)

            position = []
            wz_and_position = []
            for jj in range(len(self.gaussian_beam_s_list)):
                z = self.lens_s_list[jj][coupling_lens_columns_s[jj][ii]][1][1]

                relationship_tuple = (self.lens_s_list[jj][:coupling_lens_columns_s[jj][ii]], (0, (0, z)))
                position_ = self.gaussian_beam_s_list[jj].beam_position_calculate(relationship_tuple)
                position.append(position_)
                wz = self.gaussian_beam_s_list[jj].beam_radius_calculate(relationship_tuple)
                wz_and_position.extend([position_ + 4 * wz, position_ - 4 * wz])
            center_s = sum(position) / len(position)
            w_s = max(wz_and_position) - min(wz_and_position)

            x = None
            y = None
            intensity = []
            for jj in range(len(self.gaussian_beam_f_list)):
                z = self.lens_s_list[jj][coupling_lens_columns_s[jj][ii]][1][1]
                lens_f = self.lens_f_list[jj]
                lens_f_ = []
                for kk in lens_f:
                    if kk[1][1] <= z:
                        lens_f_.append(kk)
                lens_f_ = tuple(lens_f_)

                relationship_tuple = (lens_f_, (w_f, (center_f, z)))
                y, intensity_f = self.gaussian_beam_f_list[jj].intensity_distribution_calculate(relationship_tuple)
                intensity_f *= self.source_intensity_f[jj]
                relationship_tuple = (self.lens_s_list[jj][:coupling_lens_columns_s[jj][ii]], (w_s, (center_s, z)))
                x, intensity_s = self.gaussian_beam_s_list[jj].intensity_distribution_calculate(relationship_tuple)
                intensity_s *= self.source_intensity_s[jj]
                intensity_s, intensity_f = meshgrid(intensity_s, intensity_f)
                intensity.append(intensity_f * intensity_s)
            if not (x is not None and y is not None and intensity):
                raise ValueError('光斑数量为0')
            x, y = meshgrid(x, y)
            intensity = array(sum(intensity))

            e2_width = self.width_calculate(x, y, intensity, 1 / exp(2))

            x_list.append(x)
            y_list.append(y)
            intensity_list.append(intensity)
            e2_width_list.append(e2_width)

        return x_list, y_list, intensity_list, e2_width_list

    def divergence_angle_f_calculate(self, channel_number):
        lens_f = self.lens_f_list[channel_number]
        divergence_angle_list = []
        for ii in range(len(lens_f)):
            relationship_tuple = (lens_f[:ii + 1], (0, (0, 1.01 * lens_f[ii][1][1])))
            w0, _, m2 = self.gaussian_beam_f_list[channel_number].waist_calculate(relationship_tuple)
            divergence_angle = self.gaussian_beam_f_list[channel_number].divergence_angle_calculate(w0, m2)
            divergence_angle_list.append(divergence_angle)
        return divergence_angle_list

    def divergence_angle_s_calculate(self, channel_number):
        lens_s = self.lens_s_list[channel_number]
        divergence_angle_list = []
        for ii in range(len(lens_s)):
            relationship_tuple = (lens_s[:ii + 1], (0, (0, 1.01 * lens_s[ii][1][1])))
            w0, _, m2 = self.gaussian_beam_s_list[channel_number].waist_calculate(relationship_tuple)
            divergence_angle = self.gaussian_beam_s_list[channel_number].divergence_angle_calculate(w0, m2)
            divergence_angle_list.append(divergence_angle)
        return divergence_angle_list
