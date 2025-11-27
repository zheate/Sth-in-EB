import os
import pandas as pd
import numpy as np
from concurrent import futures as fut
import datetime as dt
from scipy import stats as stt
import copy as cop
import itertools as it
import warnings as war

war.simplefilter('ignore', np.RankWarning)


class Data_Fetch():

    def __init__(self, pull_state, forecast_mode):    #pull_state: 'chip' or 'moudle'
        self.pull_state = pull_state
        self.forecast_mode = forecast_mode

    @staticmethod
    def str2float(x):
        try:
            return float(x)
        except:
            return np.nan

    @staticmethod
    def np_find_nearest(array, value):
        try:
            array = np.array(array)
            diff = array-value
            mul = diff[0: -1] * diff[1:]
            bound = np.where(mul<=0)[0]
            return bound
        except:
            return np.array([0, -1])

    def e2width_cal(self, x_, y_):
        try:
            x = np.array(x_)
            y = np.array(y_)
            ymax = y.max()
            bound = self.np_find_nearest(y, ymax/np.exp(2))
            width_left = (x[bound[0]] + x[bound[0]+1])/2
            width_right = (x[bound[-1]] + x[bound[-1]+1])/2
            width = width_right - width_left
            return width
        except:
            return np.nan
    
    @staticmethod
    def central_wavelength_cal(wavelength, intensity):
        try:
            wavelength = np.array(wavelength)
            intensity = np.array(intensity)
            pos_max = np.argmax(intensity)
            wavelength_left = wavelength[0: pos_max]
            wavelength_right = wavelength[pos_max:]
            if wavelength_left.size and wavelength_right.size:
                intensity_left = intensity[0: pos_max]
                intensity_right = intensity[pos_max:]
                half_intensity = intensity.max() / 2
                abs_left = np.abs(intensity_left - half_intensity)
                abs_right = np.abs(intensity_right - half_intensity)
                pos_min_left = np.argmin(abs_left)
                pos_min_right = np.argmin(abs_right)
                central_wavelength = (wavelength_left[pos_min_left]+\
                    wavelength_right[pos_min_right])/2
                return central_wavelength
            else:
                return np.nan
        except:
            return np.nan
    
    @staticmethod
    def denoise(intensity):
        try:
            intensity = np.array(intensity)
            if len(intensity) >= 20:
                for ii in range(20, len(intensity)):
                    if ii < 1000:
                        p = stt.shapiro(intensity[: ii])[1]
                        if p <= 0.05:
                            break
                    else:
                        p = stt.shapiro(intensity[ii-1000: ii])[1]
                        if p <= 0.05:
                            break
                for jj in range(-21, -len(intensity)-1, -1):
                    if jj >= -1000:
                        p = stt.shapiro(intensity[: jj: -1])[1]
                        if p <= 0.05:
                            break
                    else:
                        p = stt.shapiro(intensity[jj+1000: jj: -1])[1]
                        if p <= 0.05:
                            break
                data = np.hstack([intensity[: (ii-10)], \
                    intensity[: (jj+10): -1]])
                mean = data.mean()
                std = data.std()
                intensity = intensity - mean
                intensity[intensity<=4*std] = 0
                return intensity
            else:
                return np.full(intensity.shape, np.nan)
        except:
            return np.full(intensity.shape, np.nan)

    def centroid_wavelength_cal(self, wavelength, intensity, PIB_derange):
        try:
            wavelength = np.array(wavelength)
            intensity = np.array(intensity)
            central_wavelength = self.central_wavelength_cal(wavelength, intensity)
            intensity = self.denoise(intensity)
            if PIB_derange > 0:
                up_limit = central_wavelength + PIB_derange/2
                down_limit = central_wavelength - PIB_derange/2
                if (wavelength>up_limit).sum():
                    intensity[wavelength>up_limit] = 0
                if (wavelength<down_limit).sum():
                    intensity[wavelength<down_limit] = 0
            if intensity.sum():
                centroid_wavelength = (intensity*wavelength).sum() / intensity.sum()
                return centroid_wavelength
            else:
                return np.nan
        except:
           return np.nan

    def PIB_cal(self, wavelength, intensity, PIB_morange, PIB_derange):
        try:
            wavelength = np.array(wavelength)
            intensity = np.array(intensity)
            central_wavelength = self.central_wavelength_cal(wavelength, intensity)
            intensity = self.denoise(intensity)
            if central_wavelength:
                if PIB_morange > 0:
                    up_limit = central_wavelength + PIB_morange/2
                    down_limit = central_wavelength - PIB_morange/2
                else:
                    _1_e2 = self.e2width_cal(wavelength, intensity) / 2
                    up_limit = central_wavelength + 4* _1_e2
                    down_limit = central_wavelength - 4* _1_e2
                intensity_ = cop.deepcopy(intensity)
                judge_intensity_ = wavelength > up_limit
                if judge_intensity_.sum():
                    intensity_[judge_intensity_] = 0
                judge_intensity_ = wavelength < down_limit
                if judge_intensity_.sum():
                    intensity_[judge_intensity_] = 0
                PIB_molecule = (wavelength*intensity_).sum()
                if PIB_derange > 0:
                    up_limit = central_wavelength + PIB_derange/2
                    down_limit = central_wavelength - PIB_derange/2
                    intensity_ = cop.deepcopy(intensity)
                    judge_intensity_ = wavelength > up_limit
                    if judge_intensity_.sum():
                        intensity_[judge_intensity_] = 0
                    judge_intensity_ = wavelength < down_limit
                    if judge_intensity_.sum():
                        intensity_[judge_intensity_] = 0
                    PIB_denominator = (wavelength*intensity_).sum()
                else:
                    PIB_denominator = (wavelength*intensity).sum()
                if PIB_denominator:
                    PIB = PIB_molecule / PIB_denominator
                    return PIB
                else:
                    return np.nan
            else:
               return np.nan
        except:
            return np.nan

    def FWHM_cal(self, wavelength, intensity):
        try:
            wavelength = np.array(wavelength)
            intensity = np.array(intensity)
            half_intensity = intensity.max() / 2
            pos_min_left = self.np_find_nearest(intensity, half_intensity)[0]
            pos_min_right = self.np_find_nearest(intensity, half_intensity)[-1]
            FWHM = (wavelength[pos_min_right] + wavelength[pos_min_right+1] - \
                wavelength[pos_min_left] - wavelength[pos_min_left+1]) / 2
            return FWHM
        except:
            return np.nan
    
    def spectra_shift_cal(self, wavelength, intensity, current, PIB_derange, n):
        try:
            central_wavelength = np.array([self.central_wavelength_cal(\
                wavelength[:, ii], intensity[:, ii]) \
                for ii in range(wavelength.shape[1])])
            centroid_wavelength = np.array([self.centroid_wavelength_cal(\
                wavelength[:, ii], intensity[:, ii], PIB_derange) \
                for ii in range(wavelength.shape[1])])
            current = np.array(current)
            state = (central_wavelength-centroid_wavelength) < 0.5
            current = current[state]
            centroid_wavelength = centroid_wavelength[state]
            shift = np.polyfit(current, centroid_wavelength, n)
            return shift
        except:
            return np.nan
    
    def wavelength_forecast(self, wavelength, intensity, current, PIB_derange, \
        n, cur_req):
        try:
            central_wavelength = np.array([self.central_wavelength_cal(\
                wavelength[:, ii], intensity[:, ii]) \
                for ii in range(wavelength.shape[1])])
            centroid_wavelength = np.array([self.centroid_wavelength_cal(\
                wavelength[:, ii], intensity[:, ii], PIB_derange) \
                for ii in range(wavelength.shape[1])])
            current = np.array(current)
            state = (central_wavelength-centroid_wavelength) < 1
            current = current[state]
            central_wavelength = central_wavelength[state]
            centroid_wavelength = centroid_wavelength[state]
            coefficient = np.polyfit(current, central_wavelength, n)
            f = np.poly1d(coefficient)
            central_wavelength_req = f(cur_req)
            coefficient = np.polyfit(current, centroid_wavelength, n)
            f = np.poly1d(coefficient)
            centroid_wavelength_req = f(cur_req)
            return central_wavelength_req, centroid_wavelength_req
        except:
            return np.nan, np.nan
    
    def PIB_forecast(self, wavelength, intensity, current, PIB_morange, \
        PIB_derange, cur_req):
        try:
            central_wavelength = np.array([self.central_wavelength_cal(\
                wavelength[:, ii], intensity[:, ii]) \
                for ii in range(wavelength.shape[1])])
            centroid_wavelength = np.array([self.centroid_wavelength_cal(\
                wavelength[:, ii], intensity[:, ii], PIB_derange) \
                for ii in range(wavelength.shape[1])])
            current = np.array(current)
            state = (central_wavelength-centroid_wavelength) < 1
            current = current[state]
            PIB = np.array([self.PIB_cal(wavelength[:, ii], intensity[:, ii], \
                PIB_morange, PIB_derange) for ii in range(wavelength.shape[1])])
            PIB = PIB[state]
            right_ = np.where(current>cur_req)[0]
            left_ = np.where(current<cur_req)[0]
            if right_.any() and left_.any():
                right = right_[0]
                left = left_[-1]
                PIB_require = (PIB[right] + PIB[left]) / 2
            elif right_.any() and not left_.any():
                right = right_[0]
                PIB_require = PIB[right]
            elif not right_.any() and left_.any():
                left = left_[-1]
                PIB_require = PIB[left]
            else:
                PIB_require = 0
            return PIB_require
        except:
            return np.nan
    
    def FWHM_forecast(self, wavelength, intensity, current, PIB_derange, cur_req):
        try:
            central_wavelength = np.array([self.central_wavelength_cal(\
                wavelength[:, ii], intensity[:, ii]) \
                for ii in range(wavelength.shape[1])])
            centroid_wavelength = np.array([self.centroid_wavelength_cal(\
                wavelength[:, ii], intensity[:, ii], PIB_derange) \
                for ii in range(wavelength.shape[1])])
            current = np.array(current)
            state = (central_wavelength-centroid_wavelength) < 1
            current = current[state]
            FWHM = np.array([self.FWHM_cal(wavelength[:, ii], intensity[:, ii]) \
                for ii in range(wavelength.shape[1])])
            FWHM = FWHM[state]
            right_ = np.where(current>cur_req)[0]
            left_ = np.where(current<cur_req)[0]
            if right_.any() and left_.any():
                right = right_[0]
                left = left_[-1]
                FWHM_require = (FWHM[right] + FWHM[left]) / 2
            elif right_.any() and not left_.any():
                right = right_[0]
                FWHM_require = FWHM[right]
            elif not right_.any() and left_.any():
                left = left_[-1]
                FWHM_require = FWHM[left]
            else:
                FWHM_require = 0
            return FWHM_require
        except:
            return np.nan
    
    @staticmethod
    def lvi_forecast(current, voltage, power, n, cur_req):
        try:
            coefficient = np.polyfit(current, voltage, n)
            f = np.poly1d(coefficient)
            voltage_require = f(cur_req)
            coefficient = np.polyfit(current, power, n)
            f = np.poly1d(coefficient)
            power_require = f(cur_req)
            eff_req = power_require / voltage_require / cur_req
            return voltage_require, power_require, eff_req
        except:
            return np.nan, np.nan, np.nan
    
    @staticmethod
    def threshold_cal(current, power, n):
        try:
            coefficient = np.polyfit(power, current, n)
            f = np.poly1d(coefficient)
            return f(0)
        except:
            return np.nan

    def data_pull(self, number, current, stations, PIB_morange, PIB_derange, \
        mtime, method):
        if not isinstance(current, (list, tuple, np.ndarray)):
            current = [current]
        current = [cur for cur in current if (cur>=0 or cur==-1)]
        current_ = []
        for cur in current:
            if cur not in current_:
                current_.append(cur)
        current = current_
        if not isinstance(stations, (list, tuple, np.ndarray)):
            stations = [stations]
        stations = [station.upper() for station in stations]
        if self.pull_state == 'chip':
            if current == [-1]:
                columns1 = [str(station)+'功率' for station in stations]
                columns2 = [str(station)+'电压' for station in stations]
                columns3 = [str(station)+'电光效率' for station in stations]
                columns4 = [str(station)+'中心波长' for station in stations]
                columns5 = [str(station)+'质心波长' for station in stations]
                columns6 = [str(station)+' FWHM' for station in stations]
                columns7 = [str(station)+' PIB' for station in stations]
                columns11 = ['最终测试功率']
                columns12 = ['最终测试电压']
                columns13 = ['最终测试电光效率']
                columns14 = ['最终测试中心波长']
                columns15 = ['最终测试质心波长']
                columns16 = ['最终测试FWHM']
                columns17 = ['最终测试PIB']
            else:
                columns1 = [str(station)+str(cur)+'A功率' for station in stations for cur in current]
                columns2 = [str(station)+str(cur)+'A电压' for station in stations for cur in current]
                columns3 = [str(station)+str(cur)+'A电光效率' for station in stations for cur in current]
                columns4 = [str(station)+str(cur)+'A中心波长' for station in stations for cur in current]
                columns5 = [str(station)+str(cur)+'A质心波长' for station in stations for cur in current]
                columns6 = [str(station)+str(cur)+'A FWHM' for station in stations for cur in current]
                columns7 = [str(station)+str(cur)+'A PIB' for station in stations for cur in current]
                columns11 = [str(cur)+'A最终测试功率' for cur in current]
                columns12 = [str(cur)+'A最终测试电压' for cur in current]
                columns13 = [str(cur)+'A最终测试电光效率' for cur in current]
                columns14 = [str(cur)+'A最终测试中心波长' for cur in current]
                columns15 = [str(cur)+'A最终测试质心波长' for cur in current]
                columns16 = [str(cur)+'A最终测试FWHM' for cur in current]
                columns17 = [str(cur)+'A最终测试PIB' for cur in current]
            columns8 = [str(station)+ '阈值' for station in stations]
            columns9 = [str(station)+' SHIFT' for station in stations]
            columns10 = [str(station)+'峰值波长' for station in stations]
            columns18 = ['最终测试阈值']
            columns19 = ['最终测试SHIFT']
            columns20 = ['最终测试峰值波长']
            result_columns = ['器件号'] + columns1 + columns2 + columns3 + columns4 +columns5 + \
                columns6 + columns7 + columns8 + columns9 + columns10 + columns11 + columns12 + \
                columns13 + columns14 + columns15 + columns16 + columns17 + columns18 + columns19 + \
                columns20
            result = pd.DataFrame(np.full([1, len(result_columns)], ''), \
                columns=result_columns)
            result.loc[0, '器件号'] = number
            path = 'Z:\\Ldtd'
            for num in str(number):
                path += f'\\{num}'
            if os.path.exists(path):
                walk = list(os.walk(path))
                final_test = {}
                if current == [-1]:
                    final_test['最终测试功率'] = np.nan
                    final_test['最终测试电压'] = np.nan
                    final_test['最终测试电光效率'] = np.nan
                    final_test['最终测试中心波长'] = np.nan
                    final_test['最终测试质心波长'] = np.nan
                    final_test['最终测试FWHM'] = np.nan
                    final_test['最终测试PIB'] = np.nan
                else:
                    for cur in current:
                        final_test[str(cur)+'A最终测试功率'] = np.nan
                        final_test[str(cur)+'A最终测试电压'] = np.nan
                        final_test[str(cur)+'A最终测试电光效率'] = np.nan
                        final_test[str(cur)+'A最终测试中心波长'] = np.nan
                        final_test[str(cur)+'A最终测试质心波长'] = np.nan
                        final_test[str(cur)+'A最终测试FWHM'] = np.nan
                        final_test[str(cur)+'A最终测试PIB'] = np.nan
                final_test['最终测试阈值'] = np.nan
                final_test['最终测试SHIFT'] = np.nan
                final_test['最终测试峰值波长'] = np.nan
                try:
                    for station in stations:
                        lvi_mtime = []
                        lvi_path = []
                        rth_mtime = []
                        rth_path = []
                        sp_mtime = []
                        sp_path = []
                        for content in walk:
                            basename = os.path.basename(content[0]).upper()
                            dirname = os.path.dirname(content[0]).upper()
                            if basename == station and dirname == path.upper():
                                for file in content[2]:
                                    if '=LVI' in file.upper():
                                        abspath = os.path.join(content[0], file)
                                        lvi_path.append(abspath)
                                        lvi_mtime.append(os.path.getmtime(abspath))
                                    if '=RTH' in file.upper():
                                        abspath = os.path.join(content[0], file)
                                        rth_path.append(abspath)
                                        rth_mtime.append(os.path.getmtime(abspath))
                                    if '=SP' in file.upper():
                                        abspath = os.path.join(content[0], file)
                                        sp_path.append(abspath)
                                        sp_mtime.append(os.path.getmtime(abspath))
                        if lvi_mtime:
                            if mtime:
                                lvi_mtime_ = np.array(lvi_mtime) - \
                                    dt.datetime.timestamp(mtime)
                                if method == '>=' or method == '>':
                                    state = lvi_mtime_ < 0
                                    if state.all():
                                        index = None
                                    else:
                                        lvi_mtime_[state] = lvi_mtime_.max() + 1
                                        index = np.argmin(lvi_mtime_)
                                if method == '<=' or method == '<':
                                    state = lvi_mtime_ > 0
                                    if state.all():
                                        index = None
                                    else:
                                        lvi_mtime_[state] = lvi_mtime_.min() - 1
                                        index = np.argmax(lvi_mtime_)
                            else:
                                lvi_mtime_max = max(lvi_mtime)
                                index = lvi_mtime.index(lvi_mtime_max)
                            if index != None:
                                path_require = lvi_path[index]
                                try:
                                    data = pd.read_table(path_require, encoding='gbk', \
                                        header=None)
                                except:
                                    with open(path_require, mode='r') as f:
                                        data_ = f.readlines()
                                    data = pd.DataFrame([dat.split() for dat in data_])
                                data.iloc[:, 0] = data.iloc[:, 0].apply(self.str2float)
                                current_forecast = data.iloc[19:, 0].values.\
                                    astype(np.float64)
                                voltage_forecast = data.iloc[19:, 2].values.\
                                    astype(np.float64)
                                power_forecast = data.iloc[19:, 1].values.\
                                    astype(np.float64)
                                state = power_forecast > 1
                                current_forecast = current_forecast[state]
                                voltage_forecast = voltage_forecast[state]
                                power_forecast = power_forecast[state]
                                if current_forecast.size > 2:
                                    threshold = self.threshold_cal(current_forecast, power_forecast, 2)
                                    result.loc[0, station+'阈值'] = threshold
                                    final_test['最终测试阈值'] = threshold
                                if current == [-1]:
                                    power = self.str2float(data.iloc[-1, 1])
                                    voltage = self.str2float(data.iloc[-1, 2])
                                    eff = self.str2float(data.iloc[-1, 3])
                                    result.loc[0, station+'功率'] = power
                                    result.loc[0, station+'电压'] = voltage
                                    result.loc[0, station+'电光效率'] = eff
                                    final_test['最终测试功率'] = power
                                    final_test['最终测试电压'] = voltage
                                    final_test['最终测试电光效率'] = eff
                                else:
                                    for cur in current:
                                        data_require = data[data.iloc[:, 0].isin([cur])]
                                        if data_require.empty:
                                            if self.forecast_mode and len(current_forecast) > 2:
                                                voltage, power, eff = self.lvi_forecast(current_forecast, \
                                                    voltage_forecast, power_forecast, 2, cur)
                                                result.loc[0, station+str(cur)+'A功率'] = power
                                                result.loc[0, station+str(cur)+'A电压'] = voltage
                                                result.loc[0, station+str(cur)+'A电光效率'] = eff
                                                final_test[str(cur)+'A最终测试功率'] = power
                                                final_test[str(cur)+'A最终测试电压'] = voltage
                                                final_test[str(cur)+'A最终测试电光效率'] = eff
                                        else:
                                            power = self.str2float(data_require.iloc[-1, 1])
                                            voltage = self.str2float(data_require.iloc[-1, 2])
                                            result.loc[0, station+str(cur)+'A功率'] = power
                                            result.loc[0, station+str(cur)+'A电压'] = voltage
                                            if voltage and cur:
                                                result.loc[0, station+str(cur)+'A电光效率'] = power\
                                                    /voltage/cur
                                            final_test[str(cur)+'A最终测试功率'] = power
                                            final_test[str(cur)+'A最终测试电压'] = voltage
                                            if voltage and cur:
                                                final_test[str(cur)+'A最终测试电光效率'] = power\
                                                    /voltage/cur
                        if rth_mtime:
                            if mtime:
                                rth_mtime_ = np.array(rth_mtime) - \
                                    dt.datetime.timestamp(mtime)
                                if method == '>=' or method == '>':
                                    state = rth_mtime_ < 0
                                    if state.all():
                                        index = None
                                    else:
                                        rth_mtime_[state] = rth_mtime_.max() + 1
                                        index = np.argmin(rth_mtime_)
                                if method == '<=' or method == '<':
                                    state = rth_mtime_ > 0
                                    if state.all():
                                        index = None
                                    else:
                                        rth_mtime_[rth_mtime_>0] = rth_mtime_.min() - 1
                                        index = np.argmax(rth_mtime_)
                            else:
                                rth_mtime_max = max(rth_mtime)
                                index = rth_mtime.index(rth_mtime_max)
                            if index != None:
                                path_require = rth_path[index]
                                try:
                                    data = pd.read_table(path_require, encoding='gbk')
                                except:
                                    with open(path_require, mode='r') as f:
                                        data_ = f.readlines()
                                    data__ = [dat.split() for dat in data_]
                                    data = pd.DataFrame(data__[1:], header=data__[0])
                                data.dropna(axis=1, how='all', inplace=True)
                                data.dropna(axis=0, how='all', inplace=True)
                                spectra = data.iloc[8:, 3:]
                                if not spectra.empty:
                                    try:
                                        threshold_ = threshold
                                    except:
                                        threshold_ = None
                                    spectra_columns = spectra.columns.tolist()
                                    spectra_columns = [spec_col.strip('a') for spec_col in spectra_columns]
                                    spectra_columns = [spec_col.strip('A') for spec_col in spectra_columns]
                                    spectra_columns = list(map(self.str2float, spectra_columns))
                                    if threshold_:
                                        current_forecast = [spectra_columns[2*ii] for ii in \
                                            range(int(len(spectra_columns)/2))]
                                        compress = [True if cur>threshold else False for cur in current_forecast]
                                        current_forecast = list(it.compress(current_forecast, compress))
                                        wavelength_columns = [2*ii for ii in range(int(len(spectra_columns)/2))]
                                        intensity_columns = [2*ii+1 for ii in range(int(len(spectra_columns)/2))]
                                        wavelength_forecast_ = spectra.iloc[2:-1, wavelength_columns].values.\
                                            astype(np.float64)
                                        intensity_forecast = spectra.iloc[2:-1, intensity_columns].values.\
                                            astype(np.float64)
                                        wavelength_forecast_ = wavelength_forecast_[:, compress]
                                        intensity_forecast = intensity_forecast[:, compress]
                                    if current == [-1]:
                                        wavelength = spectra.iloc[:, -2].dropna().iloc[2:-1].values.astype(np.float64)
                                        intensity = spectra.iloc[:, -1].dropna().iloc[2:-1].values.astype(np.float64)
                                        central_wavelength = self.central_wavelength_cal(wavelength, intensity)
                                        FWHM = self.FWHM_cal(wavelength, intensity)
                                        centroid_wavelength = self.centroid_wavelength_cal(wavelength, intensity, \
                                            PIB_derange)
                                        PIB = self.PIB_cal(wavelength, intensity, PIB_morange, PIB_derange)
                                        result.loc[0, station+'中心波长'] = central_wavelength
                                        result.loc[0, station+'质心波长'] = centroid_wavelength
                                        result.loc[0, station+' FWHM'] = FWHM
                                        result.loc[0, station+' PIB'] = PIB
                                        final_test['最终测试中心波长'] = central_wavelength
                                        final_test['最终测试质心波长'] = centroid_wavelength
                                        final_test['最终测试FWHM'] = FWHM
                                        final_test['最终测试PIB'] = PIB
                                    else:
                                        for cur in current:
                                            if cur in spectra_columns:
                                                column_index = spectra_columns.index(cur)
                                                wavelength = spectra.iloc[:, column_index].dropna().\
                                                    iloc[2:-1].values.astype(np.float64)
                                                intensity = spectra.iloc[:, column_index+1].dropna().\
                                                    iloc[2:-1].values.astype(np.float64)
                                                central_wavelength = self.central_wavelength_cal(wavelength, intensity)
                                                FWHM = self.FWHM_cal(wavelength, intensity)
                                                centroid_wavelength = self.centroid_wavelength_cal(\
                                                    wavelength, intensity, PIB_derange)
                                                PIB = self.PIB_cal(wavelength, intensity, PIB_morange, PIB_derange)
                                                result.loc[0, station+str(cur)+'A中心波长'] = central_wavelength
                                                result.loc[0, station+str(cur)+'A质心波长'] = centroid_wavelength
                                                result.loc[0, station+str(cur)+'A FWHM'] = FWHM
                                                result.loc[0, station+str(cur)+'A PIB'] = PIB
                                                final_test[str(cur)+'A最终测试中心波长'] = central_wavelength
                                                final_test[str(cur)+'A最终测试质心波长'] = centroid_wavelength
                                                final_test[str(cur)+'A最终测试FWHM'] = FWHM
                                                final_test[str(cur)+'A最终测试PIB'] = PIB
                                            else:
                                                if self.forecast_mode and threshold_ and len(wavelength_forecast_) > 2:
                                                    central_wavelength, centroid_wavelength = self.wavelength_forecast(\
                                                        wavelength_forecast_, intensity_forecast, current_forecast, \
                                                        PIB_derange, 2, cur)
                                                    FWHM = self.FWHM_forecast(wavelength_forecast_, intensity_forecast, \
                                                        current_forecast, PIB_derange, cur)
                                                    PIB = self.PIB_forecast(wavelength_forecast_, intensity_forecast, \
                                                        current_forecast, PIB_morange, PIB_derange, cur)
                                                    result.loc[0, station+str(cur)+'A中心波长'] = central_wavelength
                                                    result.loc[0, station+str(cur)+'A质心波长'] = centroid_wavelength
                                                    result.loc[0, station+str(cur)+'A FWHM'] = FWHM
                                                    result.loc[0, station+str(cur)+'A PIB'] = PIB
                                                    final_test[str(cur)+'A最终测试中心波长'] = central_wavelength
                                                    final_test[str(cur)+'A最终测试质心波长'] = centroid_wavelength
                                                    final_test[str(cur)+'A最终测试FWHM'] = FWHM
                                                    final_test[str(cur)+'A最终测试PIB'] = PIB
                                    shift = self.spectra_shift_cal(wavelength_forecast_, intensity_forecast, \
                                        current_forecast, PIB_derange, 1)[0]
                                    result.loc[0, station+' SHIFT'] = shift
                                    final_test['最终测试SHIFT'] = shift
                        if sp_mtime:
                            if mtime:
                                sp_mtime_ = np.array(sp_mtime) - \
                                    dt.datetime.timestamp(mtime)
                                if method == '>=' or method == '>':
                                    state = sp_mtime_ < 0
                                    if state.all():
                                        index = None
                                    else:
                                        sp_mtime_[state] = sp_mtime_.max() + 1
                                        index = np.argmin(sp_mtime_)
                                if method == '<=' or method == '<':
                                    state = sp_mtime_ > 0
                                    if state.all():
                                        index = None
                                    else:
                                        sp_mtime_[state] = sp_mtime_.min() - 1
                                        index = np.argmax(sp_mtime_)
                            else:
                                sp_mtime_max = max(sp_mtime)
                                index = sp_mtime.index(sp_mtime_max)
                            if index != None:
                                path_require = sp_path[index]
                                try:
                                    data = pd.read_table(path_require, encoding='gbk', \
                                        header=None)
                                except:
                                    with open(path_require, mode='r') as f:
                                        data_ = f.readlines()
                                    data = pd.DataFrame([dat.split() for dat in data_])
                                wavelength = data.iloc[9:, 0].values.astype('float64')
                                intensity = data.iloc[9:, 1].values.astype('float64')
                                peak_wavelength = self.central_wavelength_cal(wavelength, intensity)
                                result.loc[0, station+'峰值波长'] = peak_wavelength
                                final_test['最终测试峰值波长'] = peak_wavelength
                except:
                    pass
                if current == [-1]:
                    result.loc[0, '最终测试功率'] = final_test['最终测试功率']
                    result.loc[0, '最终测试电压'] = final_test['最终测试电压']
                    result.loc[0, '最终测试电光效率'] = final_test['最终测试电光效率']
                    result.loc[0, '最终测试中心波长'] = final_test['最终测试中心波长']
                    result.loc[0, '最终测试质心波长'] = final_test['最终测试质心波长']
                    result.loc[0, '最终测试FWHM'] = final_test['最终测试FWHM']
                    result.loc[0, '最终测试PIB'] = final_test['最终测试PIB']
                else:
                    for cur in current:
                        result.loc[0, str(cur)+'A最终测试功率'] = final_test[str(cur)+'A最终测试功率']
                        result.loc[0, str(cur)+'A最终测试电压'] = final_test[str(cur)+'A最终测试电压']
                        result.loc[0, str(cur)+'A最终测试电光效率'] = final_test[str(cur)+'A最终测试电光效率']
                        result.loc[0, str(cur)+'A最终测试中心波长'] = final_test[str(cur)+'A最终测试中心波长']
                        result.loc[0, str(cur)+'A最终测试质心波长'] = final_test[str(cur)+'A最终测试质心波长']
                        result.loc[0, str(cur)+'A最终测试FWHM'] = final_test[str(cur)+'A最终测试FWHM']
                        result.loc[0, str(cur)+'A最终测试PIB'] = final_test[str(cur)+'A最终测试PIB']
                result.loc[0, '最终测试阈值'] = final_test['最终测试阈值']
                result.loc[0, '最终测试SHIFT'] = final_test['最终测试SHIFT']
                result.loc[0, '最终测试峰值波长'] = final_test['最终测试峰值波长']
            return result

        elif self.pull_state == 'moudle':
            if current == [-1]:
                columns1 = [str(station)+'功率' for station in stations]
                columns2 = [str(station)+'电压' for station in stations]
                columns3 = [str(station)+'电光效率' for station in stations]
                columns4 = [str(station)+'中心波长' for station in stations]
                columns5 = [str(station)+'质心波长' for station in stations]
                columns6 = [str(station)+' FWHM' for station in stations]
                columns7 = [str(station)+' PIB' for station in stations]
                columns10 = ['最终测试功率']
                columns11 = ['最终测试电压']
                columns12 = ['最终测试电光效率']
                columns13 = ['最终测试中心波长']
                columns14 = ['最终测试质心波长']
                columns15 = ['最终测试FWHM']
                columns16 = ['最终测试PIB']
            else:
                columns1 = [str(station)+str(cur)+'A功率' for station in stations for cur in current]
                columns2 = [str(station)+str(cur)+'A电压' for station in stations for cur in current]
                columns3 = [str(station)+str(cur)+'A电光效率' for station in stations for cur in current]
                columns4 = [str(station)+str(cur)+'A中心波长' for station in stations for cur in current]
                columns5 = [str(station)+str(cur)+'A质心波长' for station in stations for cur in current]
                columns6 = [str(station)+str(cur)+'A FWHM' for station in stations for cur in current]
                columns7 = [str(station)+str(cur)+'A PIB' for station in stations for cur in current]
                columns10 = [str(cur)+'A最终测试功率' for cur in current]
                columns11 = [str(cur)+'A最终测试电压' for cur in current]
                columns12 = [str(cur)+'A最终测试电光效率' for cur in current]
                columns13 = [str(cur)+'A最终测试中心波长' for cur in current]
                columns14 = [str(cur)+'A最终测试质心波长' for cur in current]
                columns15 = [str(cur)+'A最终测试FWHM' for cur in current]
                columns16 = [str(cur)+'A最终测试PIB' for cur in current]
            columns8 = [str(station)+ '阈值' for station in stations]
            columns9 = [str(station)+' SHIFT' for station in stations]
            columns17 = ['最终测试阈值']
            columns18 = ['最终测试SHIFT']
            result_columns = ['壳体号'] + columns1 + columns2 + columns3 + columns4 +columns5 + \
                columns6 + columns7 + columns8 + columns9 + columns10 + columns11 + columns12 + \
                columns13 + columns14 + columns15 + columns16 + columns17 + columns18
            result = pd.DataFrame(np.full([1, len(result_columns)], ''), \
                columns=result_columns)
            result.loc[0, '壳体号'] = number
            path = 'Z:\\Ldtd\\FCP'
            for num in str(number):
                path += f'\\{num}'
            if os.path.exists(path):
                walk = list(os.walk(path))
                final_test = {}
                if current == [-1]:
                    final_test['最终测试功率'] = np.nan
                    final_test['最终测试电压'] = np.nan
                    final_test['最终测试电光效率'] = np.nan
                    final_test['最终测试中心波长'] = np.nan
                    final_test['最终测试质心波长'] = np.nan
                    final_test['最终测试FWHM'] = np.nan
                    final_test['最终测试PIB'] = np.nan
                else:
                    for cur in current:
                        final_test[str(cur)+'A最终测试功率'] = np.nan
                        final_test[str(cur)+'A最终测试电压'] = np.nan
                        final_test[str(cur)+'A最终测试电光效率'] = np.nan
                        final_test[str(cur)+'A最终测试中心波长'] = np.nan
                        final_test[str(cur)+'A最终测试质心波长'] = np.nan
                        final_test[str(cur)+'A最终测试FWHM'] = np.nan
                        final_test[str(cur)+'A最终测试PIB'] = np.nan
                final_test['最终测试阈值'] = np.nan
                final_test['最终测试SHIFT'] = np.nan
                try:
                    for station in stations:
                        lvi_mtime = []
                        lvi_path = []
                        rth_mtime = []
                        rth_path = []
                        for content in walk:
                            basename = os.path.basename(content[0]).upper()
                            dirname3 = os.path.dirname(os.path.dirname(os.path.dirname(content[0]))).upper()
                            dirname2 = os.path.dirname(os.path.dirname(content[0])).upper()
                            dirname1 = os.path.dirname(content[0]).upper()
                            if basename == station and \
                                ((dirname3 == path.upper() and \
                                dirname2 == os.path.join(path, '-').upper()) or \
                                dirname1 == path.upper()):
                                for file in content[2]:
                                    if '=LVI' in file.upper():
                                        abspath = os.path.join(content[0], file)
                                        lvi_path.append(abspath)
                                        lvi_mtime.append(os.path.getmtime(abspath))
                                    if '=RTH' in file.upper():
                                        abspath = os.path.join(content[0], file)
                                        rth_path.append(abspath)
                                        rth_mtime.append(os.path.getmtime(abspath))
                        if lvi_mtime:
                            if mtime:
                                lvi_mtime_ = np.array(lvi_mtime) - \
                                    dt.datetime.timestamp(mtime)
                                if method == '>=' or method == '>':
                                    state = lvi_mtime_ < 0
                                    if state.all():
                                        index = None
                                    else:
                                        lvi_mtime_[state] = lvi_mtime_.max() + 1
                                        index = np.argmin(lvi_mtime_)
                                if method == '<=' or method == '<':
                                    state = lvi_mtime_ > 0
                                    if state.all():
                                        index = None
                                    else:
                                        lvi_mtime_[state] = lvi_mtime_.min() - 1
                                        index = np.argmax(lvi_mtime_)
                            else:
                                lvi_mtime_max = max(lvi_mtime)
                                index = lvi_mtime.index(lvi_mtime_max)
                            if index != None:
                                path_require = lvi_path[index]
                                try:
                                    data = pd.read_table(path_require, encoding='gbk', \
                                        header=None)
                                except:
                                    with open(path_require, mode='r') as f:
                                        data_ = f.readlines()
                                    data = pd.DataFrame([dat.split() for dat in data_])
                                data.iloc[:, 0] = data.iloc[:, 0].apply(self.str2float)
                                current_forecast = data.iloc[19:, 0].values.\
                                    astype(np.float64)
                                voltage_forecast = data.iloc[19:, 2].values.\
                                    astype(np.float64)
                                power_forecast = data.iloc[19:, 1].values.\
                                    astype(np.float64)
                                state = power_forecast > 1
                                current_forecast = current_forecast[state]
                                voltage_forecast = voltage_forecast[state]
                                power_forecast = power_forecast[state]
                                if current_forecast.size > 2:
                                    threshold = self.threshold_cal(current_forecast, power_forecast, 2)
                                    result.loc[0, station+'阈值'] = threshold
                                    final_test['最终测试阈值'] = threshold
                                if current == [-1]:
                                    power = self.str2float(data.iloc[-1, 1])
                                    voltage = self.str2float(data.iloc[-1, 2])
                                    eff = self.str2float(data.iloc[-1, 3])
                                    result.loc[0, station+'功率'] = power
                                    result.loc[0, station+'电压'] = voltage
                                    result.loc[0, station+'电光效率'] = eff
                                    final_test['最终测试功率'] = power
                                    final_test['最终测试电压'] = voltage
                                    final_test['最终测试电光效率'] = eff
                                else:
                                    for cur in current:
                                        data_require = data[data.iloc[:, 0].isin([cur])]
                                        if data_require.empty:
                                            if self.forecast_mode and len(current_forecast) > 2:
                                                voltage, power, eff = self.lvi_forecast(current_forecast, \
                                                    voltage_forecast, power_forecast, 2, cur)
                                                result.loc[0, station+str(cur)+'A功率'] = power
                                                result.loc[0, station+str(cur)+'A电压'] = voltage
                                                result.loc[0, station+str(cur)+'A电光效率'] = eff
                                                final_test[str(cur)+'A最终测试功率'] = power
                                                final_test[str(cur)+'A最终测试电压'] = voltage
                                                final_test[str(cur)+'A最终测试电光效率'] = eff
                                        else:
                                            power = self.str2float(data_require.iloc[-1, 1])
                                            voltage = self.str2float(data_require.iloc[-1, 2])
                                            result.loc[0, station+str(cur)+'A功率'] = power
                                            result.loc[0, station+str(cur)+'A电压'] = voltage
                                            if voltage and cur:
                                                result.loc[0, station+str(cur)+'A电光效率'] = power\
                                                    /voltage/cur
                                            final_test[str(cur)+'A最终测试功率'] = power
                                            final_test[str(cur)+'A最终测试电压'] = voltage
                                            if voltage and cur:
                                                final_test[str(cur)+'A最终测试电光效率'] = power\
                                                    /voltage/cur
                        if rth_mtime:
                            if mtime:
                                rth_mtime_ = np.array(rth_mtime) - \
                                    dt.datetime.timestamp(mtime)
                                if method == '>=' or method == '>':
                                    state = rth_mtime_ < 0
                                    if state.all():
                                        index = None
                                    else:
                                        rth_mtime_[state] = rth_mtime_.max() + 1
                                        index = np.argmin(rth_mtime_)
                                if method == '<=' or method == '<':
                                    state = rth_mtime_ > 0
                                    if state.all():
                                        index = None
                                    else:
                                        rth_mtime_[rth_mtime_>0] = rth_mtime_.min() - 1
                                        index = np.argmax(rth_mtime_)
                            else:
                                rth_mtime_max = max(rth_mtime)
                                index = rth_mtime.index(rth_mtime_max)
                            if index != None:
                                path_require = rth_path[index]
                                try:
                                    data = pd.read_table(path_require, encoding='gbk')
                                except:
                                    with open(path_require, mode='r') as f:
                                        data_ = f.readlines()
                                    data__ = [dat.split() for dat in data_]
                                    data = pd.DataFrame(data__[1:], header=data__[0])
                                data.dropna(axis=1, how='all', inplace=True)
                                data.dropna(axis=0, how='all', inplace=True)
                                spectra = data.iloc[8:, 3:]
                                if not spectra.empty:
                                    try:
                                        threshold_ = threshold
                                    except:
                                        threshold_ = None
                                    spectra_columns:list = spectra.columns.tolist()
                                    spectra_columns = [spec_col.strip('a') for spec_col in spectra_columns]
                                    spectra_columns = [spec_col.strip('A') for spec_col in spectra_columns]
                                    spectra_columns = list(map(self.str2float, spectra_columns))
                                    if threshold_:
                                        current_forecast = [spectra_columns[2*ii] for ii in \
                                            range(int(len(spectra_columns)/2))]
                                        compress = [True if cur>threshold else False for cur in current_forecast]
                                        current_forecast = list(it.compress(current_forecast, compress))
                                        wavelength_columns = [2*ii for ii in range(int(len(spectra_columns)/2))]
                                        intensity_columns = [2*ii+1 for ii in range(int(len(spectra_columns)/2))]
                                        wavelength_forecast_ = spectra.iloc[2:-1, wavelength_columns].values.\
                                            astype(np.float64)
                                        intensity_forecast = spectra.iloc[2:-1, intensity_columns].values.\
                                            astype(np.float64)
                                        wavelength_forecast_ = wavelength_forecast_[:, compress]
                                        intensity_forecast = intensity_forecast[:, compress]
                                    if current == [-1]:
                                        wavelength = spectra.iloc[:, -2].dropna().iloc[2:-1].values.astype(np.float64)
                                        intensity = spectra.iloc[:, -1].dropna().iloc[2:-1].values.astype(np.float64)
                                        central_wavelength = self.central_wavelength_cal(wavelength, intensity)
                                        FWHM = self.FWHM_cal(wavelength, intensity)
                                        centroid_wavelength = self.centroid_wavelength_cal(wavelength, intensity, \
                                            PIB_derange)
                                        PIB = self.PIB_cal(wavelength, intensity, PIB_morange, PIB_derange)
                                        result.loc[0, station+'中心波长'] = central_wavelength
                                        result.loc[0, station+'质心波长'] = centroid_wavelength
                                        result.loc[0, station+' FWHM'] = FWHM
                                        result.loc[0, station+' PIB'] = PIB
                                        final_test['最终测试中心波长'] = central_wavelength
                                        final_test['最终测试质心波长'] = centroid_wavelength
                                        final_test['最终测试FWHM'] = FWHM
                                        final_test['最终测试PIB'] = PIB
                                    else:
                                        for cur in current:
                                            if cur in spectra_columns:
                                                column_index = spectra_columns.index(cur)
                                                wavelength = spectra.iloc[:, column_index].dropna().\
                                                    iloc[2:-1].values.astype(np.float64)
                                                intensity = spectra.iloc[:, column_index+1].dropna().\
                                                    iloc[2:-1].values.astype(np.float64)
                                                central_wavelength = self.central_wavelength_cal(wavelength, intensity)
                                                FWHM = self.FWHM_cal(wavelength, intensity)
                                                centroid_wavelength = self.centroid_wavelength_cal(\
                                                    wavelength, intensity, PIB_derange)
                                                PIB = self.PIB_cal(wavelength, intensity, PIB_morange, PIB_derange)
                                                result.loc[0, station+str(cur)+'A中心波长'] = central_wavelength
                                                result.loc[0, station+str(cur)+'A质心波长'] = centroid_wavelength
                                                result.loc[0, station+str(cur)+'A FWHM'] = FWHM
                                                result.loc[0, station+str(cur)+'A PIB'] = PIB
                                                final_test[str(cur)+'A最终测试中心波长'] = central_wavelength
                                                final_test[str(cur)+'A最终测试质心波长'] = centroid_wavelength
                                                final_test[str(cur)+'A最终测试FWHM'] = FWHM
                                                final_test[str(cur)+'A最终测试PIB'] = PIB
                                            else:
                                                if self.forecast_mode and threshold_ and len(wavelength_forecast_) > 2:
                                                    central_wavelength, centroid_wavelength = self.wavelength_forecast(\
                                                        wavelength_forecast_, intensity_forecast, current_forecast, \
                                                        PIB_derange, 2, cur)
                                                    FWHM = self.FWHM_forecast(wavelength_forecast_, intensity_forecast, \
                                                        current_forecast, PIB_derange, cur)
                                                    PIB = self.PIB_forecast(wavelength_forecast_, intensity_forecast, \
                                                        current_forecast, PIB_morange, PIB_derange, cur)
                                                    result.loc[0, station+str(cur)+'A中心波长'] = central_wavelength
                                                    result.loc[0, station+str(cur)+'A质心波长'] = centroid_wavelength
                                                    result.loc[0, station+str(cur)+'A FWHM'] = FWHM
                                                    result.loc[0, station+str(cur)+'A PIB'] = PIB
                                                    final_test[str(cur)+'A最终测试中心波长'] = central_wavelength
                                                    final_test[str(cur)+'A最终测试质心波长'] = centroid_wavelength
                                                    final_test[str(cur)+'A最终测试FWHM'] = FWHM
                                                    final_test[str(cur)+'A最终测试PIB'] = PIB
                                    shift = self.spectra_shift_cal(wavelength_forecast_, intensity_forecast, \
                                        current_forecast, PIB_derange, 1)[0]
                                    result.loc[0, station+' SHIFT'] = shift
                                    final_test['最终测试SHIFT'] = shift
                except:
                    pass
                if current == [-1]:
                    result.loc[0, '最终测试功率'] = final_test['最终测试功率']
                    result.loc[0, '最终测试电压'] = final_test['最终测试电压']
                    result.loc[0, '最终测试电光效率'] = final_test['最终测试电光效率']
                    result.loc[0, '最终测试中心波长'] = final_test['最终测试中心波长']
                    result.loc[0, '最终测试质心波长'] = final_test['最终测试质心波长']
                    result.loc[0, '最终测试FWHM'] = final_test['最终测试FWHM']
                    result.loc[0, '最终测试PIB'] = final_test['最终测试PIB']
                else:
                    for cur in current:
                        result.loc[0, str(cur)+'A最终测试功率'] = final_test[str(cur)+'A最终测试功率']
                        result.loc[0, str(cur)+'A最终测试电压'] = final_test[str(cur)+'A最终测试电压']
                        result.loc[0, str(cur)+'A最终测试电光效率'] = final_test[str(cur)+'A最终测试电光效率']
                        result.loc[0, str(cur)+'A最终测试中心波长'] = final_test[str(cur)+'A最终测试中心波长']
                        result.loc[0, str(cur)+'A最终测试质心波长'] = final_test[str(cur)+'A最终测试质心波长']
                        result.loc[0, str(cur)+'A最终测试FWHM'] = final_test[str(cur)+'A最终测试FWHM']
                        result.loc[0, str(cur)+'A最终测试PIB'] = final_test[str(cur)+'A最终测试PIB']
                result.loc[0, '最终测试阈值'] = final_test['最终测试阈值']
                result.loc[0, '最终测试SHIFT'] = final_test['最终测试SHIFT']
            return result


if __name__ == '__main__':
    time_start = dt.datetime.now()

    pull_state = 'chip'      #chip or moudle
    # pull_state = 'moudle'      #chip or moudle
    data = pd.read_excel(r'C:\Users\jiawei.xu\Desktop\kth.xlsx')
    current = 2         #-1表示最大电流
    stations = ['pre', 'preQCW', 'post', 'postQCW']
    # stations = ['封盖测试']
    # stations = ['封盖测试', 'Post测试', '低温储存后测试', '耦合测试', '温循测试']
    PIB_morange = 0
    PIB_derange = 0
    method = '>'
    forecast_mode = True
    export_path = r'C:\Users\jiawei.xu\Desktop\ldtd_info.xlsx'

    data.dropna(axis=0, how='all', inplace=True)
    data.dropna(axis=1, how='all', inplace=True)
    number = data.iloc[:, 0]
    number = number.str.strip()
    if data.shape[1] >= 2:
        mtime = data.iloc[:, 1]
    else:
        mtime = pd.Series(np.full(number.shape[0], 0))

    if not isinstance(current, (list, tuple, np.ndarray)):
        current = [current]
    current = [cur for cur in current if (cur>=0 or cur==-1)]
    current_ = []
    for cur in current:
        if cur not in current_:
            current_.append(cur)
    current = current_
    if not isinstance(stations, (list, tuple, np.ndarray)):
        stations = [stations]
    stations = [ii.upper() for ii in stations]
    df = Data_Fetch(pull_state, forecast_mode)
    with fut.ThreadPoolExecutor() as p:
        result = [p.submit(df.data_pull, number.iloc[ii], current, stations, \
            PIB_morange, PIB_derange, mtime.iloc[ii], method) \
            for ii in range(number.shape[0])]
    # for ii in range(number.shape[0]):
    #     a = df.data_pull(number.iloc[ii], current, stations, \
    #     PIB_morange, PIB_derange, mtime.iloc[ii], method)
    if current == [-1]:
        columns1 = [str(station)+'功率' for station in stations]
        columns2 = [str(station)+'电压' for station in stations]
        columns3 = [str(station)+'电光效率' for station in stations]
        columns4 = [str(station)+'中心波长' for station in stations]
        columns5 = [str(station)+'质心波长' for station in stations]
        columns6 = [str(station)+' FWHM' for station in stations]
        columns7 = [str(station)+' PIB' for station in stations]
        columns11 = ['最终测试功率']
        columns12 = ['最终测试电压']
        columns13 = ['最终测试电光效率']
        columns14 = ['最终测试中心波长']
        columns15 = ['最终测试质心波长']
        columns16 = ['最终测试FWHM']
        columns17 = ['最终测试PIB']
    else:
        columns1 = [str(station)+str(cur)+'A功率' for station in stations for cur in current]
        columns2 = [str(station)+str(cur)+'A电压' for station in stations for cur in current]
        columns3 = [str(station)+str(cur)+'A电光效率' for station in stations for cur in current]
        columns4 = [str(station)+str(cur)+'A中心波长' for station in stations for cur in current]
        columns5 = [str(station)+str(cur)+'A质心波长' for station in stations for cur in current]
        columns6 = [str(station)+str(cur)+'A FWHM' for station in stations for cur in current]
        columns7 = [str(station)+str(cur)+'A PIB' for station in stations for cur in current]
        columns11 = [str(cur)+'A最终测试功率' for cur in current]
        columns12 = [str(cur)+'A最终测试电压' for cur in current]
        columns13 = [str(cur)+'A最终测试电光效率' for cur in current]
        columns14 = [str(cur)+'A最终测试中心波长' for cur in current]
        columns15 = [str(cur)+'A最终测试质心波长' for cur in current]
        columns16 = [str(cur)+'A最终测试FWHM' for cur in current]
        columns17 = [str(cur)+'A最终测试PIB' for cur in current]
    columns8 = [str(station)+ '阈值' for station in stations]
    columns9 = [str(station)+' SHIFT' for station in stations]
    columns10 = [str(station)+'峰值波长' for station in stations]
    columns18 = ['最终测试阈值']
    columns19 = ['最终测试SHIFT']
    columns20 = ['最终测试峰值波长']
    if pull_state == 'moudle':
        export_columns = ['壳体号'] + columns1 + columns2 + columns3 + columns4 +columns5 + \
            columns6 + columns7 + columns8 + columns9 + columns10 + columns11 + columns12 + \
            columns13 + columns14 + columns15 + columns16 + columns17 + columns18 + columns19 +\
            columns20
    elif pull_state == 'chip':
        export_columns = ['器件号'] + columns1 + columns2 + columns3 + columns4 +columns5 + \
            columns6 + columns7 + columns8 + columns9 + columns10 + columns11 + columns12 + \
            columns13 + columns14 + columns15 + columns16 + columns17 + columns18 + columns19 +\
            columns20
    export = pd.DataFrame(columns=export_columns)
    for res in result:
        export = pd.concat([export, res.result()], axis=0)
    export.to_excel(export_path, index=False)

    time_end = dt.datetime.now()
    print(f'运行时间:{time_end-time_start}')