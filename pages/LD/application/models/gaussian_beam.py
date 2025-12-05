from numpy import pi, exp, ndarray, inf, sqrt, linspace, array, ones, conj, real
from scipy.integrate import quad
from scipy.signal import convolve
from .cross_point import cross_point_kb
from functools import lru_cache
from .width_calculate_one_dimension import width_calculate_one_dimension


class GaussianBeam:
    """
        标准relationship格式:
        (光学镜信息((f, (t, z), m2_ratio), ...), 探测器信息(w, (t, z)))
        f: 透镜焦距，若为inf,则表示无穷焦距或其他不影响传输的光学镜
        t, z坐标系: t为纵坐标, z为横坐标
        m2_ratio: 光学镜前后M2因子比
        w: 探测器宽度
    """

    __slots__ = ('lambda_', 'w0', 'theta', 'near_field_order', 'far_field_order', 'initial_position', 'm2')

    # 所有参数单位均为国际单位
    def __init__(self, lambda_: float, w0: float, theta: float, near_field_order: float, far_field_order: float,
                 initial_position: tuple):
        self.lambda_ = lambda_
        self.w0 = w0
        self.theta = theta
        self.near_field_order = near_field_order
        self.far_field_order = far_field_order
        self.initial_position = initial_position
        self.m2 = pi * self.w0 * self.theta / self.lambda_

    @lru_cache
    def rayleigh_distance_calculate(self, w0, m2):
        return pi * w0**2 / self.lambda_ / m2

    @lru_cache
    def order_calculate(self, relationship_tuple: tuple):
        if len(relationship_tuple[1]) == 0:
            raise ValueError('未设置探测器信息')
        if len(relationship_tuple) != 2:
            raise ValueError('relationship设置错误')
        if len(relationship_tuple[0]) == 0:
            z0 = self.rayleigh_distance_calculate(self.w0, self.m2)
            if relationship_tuple[1][1][1] == inf:
                order_ = self.far_field_order
            else:
                z = relationship_tuple[1][1][1] - self.initial_position[1]
                order_ = (self.near_field_order - self.far_field_order) * exp(-(z / z0)**2) + self.far_field_order
        else:
            if relationship_tuple[0][-1][0] == inf:
                order_ = self.order_calculate((relationship_tuple[0][:-1], (0, (0, relationship_tuple[1][1][1]))))
            elif abs((relationship_tuple[1][1][1] - relationship_tuple[0][-1][1][1] - relationship_tuple[0][-1][0]) /
                     relationship_tuple[0][-1][0]) < 1e-9:
                order_ = self.order_calculate((relationship_tuple[0][:-1], (0, (0, inf))))
            elif relationship_tuple[1][1][1] == inf:
                order_ = self.order_calculate(
                    (relationship_tuple[0][:-1], (0, (0, relationship_tuple[0][-1][1][1] - relationship_tuple[0][-1][0]))))
            else:
                n = relationship_tuple[1][1][1] - relationship_tuple[0][-1][1][1]
                f = relationship_tuple[0][-1][0]
                # 避免除零错误：当 n 接近 0 时，使用一个极小值
                if abs(n) < 1e-12:
                    n = 1e-12 if n >= 0 else -1e-12
                m = 1 / (1 / f - 1 / n)
                z = relationship_tuple[0][-1][1][1] - m
                order_ = self.order_calculate((relationship_tuple[0][:-1], (0, (0, z))))
        return order_

    @lru_cache
    def waist_calculate(self, relationship_tuple: tuple):
        if len(relationship_tuple[1]) == 0:
            raise ValueError('未设置探测器信息')
        if len(relationship_tuple) != 2:
            raise ValueError('relationship设置错误')
        if len(relationship_tuple[0]) == 0:
            return self.w0, self.initial_position[1], self.m2
        else:
            w0, z, m2 = self.waist_calculate((relationship_tuple[0][:-1], (0, (0, 0))))
            p = relationship_tuple[0][-1][1][1] - z
            m2_ratio = relationship_tuple[0][-1][2]
            z0 = self.rayleigh_distance_calculate(w0, m2)
            if relationship_tuple[0][-1][0] == inf:
                w0_ = w0 * m2_ratio / sqrt(1 + (m2_ratio**2 - 1) * z0**2 / (p**2 + z0**2))
                return w0_, z, m2 * m2_ratio
            else:
                f = relationship_tuple[0][-1][0]
                w0_ = w0 * m2_ratio / sqrt((p / f - 1)**2 + (z0 / f)**2 + (m2_ratio**2 - 1) * (z0 / f)**2 / ((p / f)**2 +
                                                                                                             (z0 / f)**2))
                if abs((p - f) / f) < 1e-9:
                    q_ = 1 + (m2_ratio**2 - 1) * (z0 / f)**2 / ((p / f)**2 + (z0 / f)**2) / ((p / f)**2 - p / f +
                                                                                             (z0 / f)**2)
                else:
                    q_ = (1 + (m2_ratio**2 - 1) * (z0 / f)**2 / ((p / f)**2 + (z0 / f)**2) /
                          ((p / f)**2 - p / f + (z0 / f)**2) - 1 / ((p / f) + (z0 / f)**2 / (p / f - 1)))
                q = 1 / q_ * f
                z_ = relationship_tuple[0][-1][1][1] + q
                return w0_, z_, m2 * m2_ratio

    @lru_cache
    def divergence_angle_calculate(self, w0, m2):
        theta = self.lambda_ * m2 / pi / w0
        return theta

    @lru_cache
    def beam_radius_calculate(self, relationship_tuple: tuple):
        if len(relationship_tuple[1]) == 0:
            raise ValueError('未设置探测器信息')
        if len(relationship_tuple) != 2:
            raise ValueError('relationship设置错误')
        w0, z, m2 = self.waist_calculate(relationship_tuple)
        p = relationship_tuple[1][1][1] - z
        z0 = self.rayleigh_distance_calculate(w0, m2)
        wz = w0 * sqrt(1 + (p / z0)**2)
        return wz

    @lru_cache
    def _beam_position_calculate(self, relationship_tuple: tuple):
        if len(relationship_tuple[1]) == 0:
            raise ValueError('未设置探测器信息')
        if len(relationship_tuple) != 2:
            raise ValueError('relationship设置错误')
        if len(relationship_tuple[0]) == 0:
            return 0, self.initial_position[0]
        else:
            k, b = self._beam_position_calculate((relationship_tuple[0][:-1], (0, (0, 0))))
            if relationship_tuple[0][-1][0] == inf:
                return k, b
            else:
                point1_x, point1_y = cross_point_kb((k, b), (None, relationship_tuple[0][-1][1][1]))
                b_parallel = relationship_tuple[0][-1][1][0] - k * relationship_tuple[0][-1][1][1]
                point2_x, point2_y = cross_point_kb((k, b_parallel),
                                                    (None, relationship_tuple[0][-1][1][1] + relationship_tuple[0][-1][0]))
                k_out = (point2_y - point1_y) / (point2_x - point1_x)
                b_out = point2_y - point2_x * k_out
                return k_out, b_out

    @lru_cache
    def beam_position_calculate(self, relationship_tuple: tuple):
        if len(relationship_tuple[1]) == 0:
            raise ValueError('未设置探测器信息')
        if len(relationship_tuple) != 2:
            raise ValueError('relationship设置错误')
        k, b = self._beam_position_calculate(relationship_tuple)
        t = k * relationship_tuple[1][1][1] + b
        return t

    @staticmethod
    def intensity_calculate(x: float | ndarray, w: float, x0: float, order_: float):

        def _intensity_calculate(x_):
            intensity_ = exp(-2 * ((x_ / w)**2)**order_)
            return intensity_

        integral_result = quad(_intensity_calculate, -4 * w, 4 * w)[0]
        intensity = exp(-2 * (((x - x0) / w)**2)**order_) / integral_result
        return intensity

    @lru_cache
    def intensity_distribution_calculate(self, relationship_tuple: tuple):
        if len(relationship_tuple[1]) == 0:
            raise ValueError('未设置探测器信息')
        if len(relationship_tuple) != 2:
            raise ValueError('relationship设置错误')
        wz = self.beam_radius_calculate(relationship_tuple)
        t = self.beam_position_calculate(relationship_tuple)
        order_ = self.order_calculate(relationship_tuple)
        # 采样点数优化：从1000降至500
        x = linspace(relationship_tuple[1][1][0] - relationship_tuple[1][0] / 2,
                     relationship_tuple[1][1][0] + relationship_tuple[1][0] / 2, 500)
        y = self.intensity_calculate(x, wz, t, order_)
        return x, y

    @lru_cache
    def _beam_outline_calculate(self, relationship_tuple: tuple):
        # 采样点数优化：降至100（光线追迹仅用于可视化，无需高精度）
        n_outline_samples = 100
        if len(relationship_tuple[1]) == 0:
            raise ValueError('未设置探测器信息')
        if len(relationship_tuple) != 2:
            raise ValueError('relationship设置错误')
        if len(relationship_tuple[0]) == 0:
            z_arr = linspace(self.initial_position[1], relationship_tuple[1][1][1], n_outline_samples, endpoint=False)
            z = z_arr.tolist()
            # 批量计算：使用列表推导式（lru_cache 会自动缓存中间结果）
            lens_tuple = relationship_tuple[0]
            center = [self.beam_position_calculate((lens_tuple, (0, (0, zi)))) for zi in z]
            radius = [self.beam_radius_calculate((lens_tuple, (0, (0, zi)))) for zi in z]
            # 使用 NumPy 向量化计算轮廓
            center_arr = array(center)
            radius_arr = array(radius)
            outline1 = (center_arr + radius_arr).tolist()
            outline2 = (center_arr - radius_arr).tolist()
            return z, center, outline1, outline2
        else:
            z_, center_, outline1_, outline2_ = self._beam_outline_calculate(
                (relationship_tuple[0][:-1], (0, (0, relationship_tuple[0][-1][1][1]))))
            z_arr = linspace(relationship_tuple[0][-1][1][1], relationship_tuple[1][1][1], n_outline_samples, endpoint=False)
            z = z_arr.tolist()
            lens_tuple = relationship_tuple[0]
            center = [self.beam_position_calculate((lens_tuple, (0, (0, zi)))) for zi in z]
            radius = [self.beam_radius_calculate((lens_tuple, (0, (0, zi)))) for zi in z]
            center_arr = array(center)
            radius_arr = array(radius)
            outline1 = (center_arr + radius_arr).tolist()
            outline2 = (center_arr - radius_arr).tolist()
            z_.extend(z)
            center_.extend(center)
            outline1_.extend(outline1)
            outline2_.extend(outline2)
            return z_, center_, outline1_, outline2_

    @lru_cache
    def beam_outline_calculate(self, relationship_tuple: tuple):
        if len(relationship_tuple[1]) == 0:
            raise ValueError('未设置探测器信息')
        if len(relationship_tuple) != 2:
            raise ValueError('relationship设置错误')
        z_, center_, outline1_, outline2_ = self._beam_outline_calculate(relationship_tuple)
        center = self.beam_position_calculate(relationship_tuple)
        radius = self.beam_radius_calculate(relationship_tuple)
        outline1 = center + radius
        outline2 = center - radius
        z_.append(relationship_tuple[1][1][1])
        center_.append(center)
        outline1_.append(outline1)
        outline2_.append(outline2)
        return z_, center_, outline1_, outline2_

    def light_transfer_1d(self, e1: ndarray, x1: ndarray, x2: ndarray, ll: float):
        """向量化光学传输计算（优化版本）"""
        # 预处理：只计算一次
        convolve_coeff_diff = array([1, -1])
        convolve_coeff_avg = ones(2) * 0.5
        dx1 = convolve(x1, convolve_coeff_diff, mode='valid')
        x1_mid = convolve(x1, convolve_coeff_avg, mode='valid')
        e1_mid = convolve(e1, convolve_coeff_avg, mode='valid')
        
        # 向量化：x2 扩展为列向量，x1_mid 为行向量
        x2_col = x2[:, None]  # (N, 1)
        x1_row = x1_mid[None, :]  # (1, M)
        
        # 计算距离矩阵
        r = sqrt(ll**2 + (x2_col - x1_row)**2)  # (N, M)
        
        # 向量化传输计算
        phase = exp(1j * 2 * pi / self.lambda_ * r)
        factor = (ll / r + 1) / 2 * dx1
        e2 = (1 / 1j / self.lambda_) * (e1_mid * phase / r * factor).sum(axis=1)
        
        return e2

    @lru_cache
    def beam_spreading_calculate_(self, ratio1: float | None, ratio2: float | None):
        # 采样点数优化：从1000降至400，在精度和速度间平衡
        n_samples = 400

        def e(x):
            return exp(-(x / self.w0)**2)

        if ratio1 is None:
            range1 = -4 * self.w0
        else:
            range1 = -ratio1 * self.w0
        if ratio2 is None:
            range2 = 4 * self.w0
        else:
            range2 = ratio2 * self.w0
        z0 = self.rayleigh_distance_calculate(self.w0, self.m2)
        ll = 4 * z0
        wz = self.beam_radius_calculate(((), (0, (0, self.initial_position[1] + ll))))
        x2 = linspace(-4 * wz, 4 * wz, n_samples)

        x1 = linspace(range1, range2, n_samples)
        e1 = e(x1)
        e2 = self.light_transfer_1d(e1, x1, x2, ll)

        x1_ = linspace(-4 * self.w0, 4 * self.w0, n_samples)
        e1_ = e(x1_)
        e2_ = self.light_transfer_1d(e1_, x1_, x2, ll)

        return x2, real(e2 * conj(e2)), real(e2_ * conj(e2_))

    @lru_cache
    def beam_spreading_calculate(self, ratio1: float | None, ratio2: float | None):
        x2, intensity2, intensity2_ = self.beam_spreading_calculate_(ratio1, ratio2)
        e2width = width_calculate_one_dimension(x2, intensity2, 1 / exp(2))
        e2width_ = width_calculate_one_dimension(x2, intensity2_, 1 / exp(2))
        return e2width / e2width_

    def clear_lru_cache(self):
        self.order_calculate.cache_clear()
        self.rayleigh_distance_calculate.cache_clear()
        self.waist_calculate.cache_clear()
        self.divergence_angle_calculate.cache_clear()
        self.beam_radius_calculate.cache_clear()
        self._beam_position_calculate.cache_clear()
        self.beam_position_calculate.cache_clear()
        self.intensity_distribution_calculate.cache_clear()
        self._beam_outline_calculate.cache_clear()
        self.beam_outline_calculate.cache_clear()
        self.beam_spreading_calculate_.cache_clear()
        self.beam_spreading_calculate.cache_clear()
        cross_point_kb.cache_clear()
