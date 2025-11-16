"""
数据收集模块

该模块负责从Data_fetch、TestAnalysis和Progress页面收集指定壳体号的测试数据。
提供统一的数据收集接口，支持多数据源的数据提取和合并。
"""

from typing import Dict, List, Optional, Tuple
import pandas as pd
from datetime import datetime
from utils.error_handler import ErrorHandler, DataCollectionError


class DataCollector:
    """
    数据收集器，从各页面提取数据
    
    该类提供静态方法用于从不同的数据源（Data_fetch、TestAnalysis、Progress页面）
    收集测试数据，并将收集到的数据合并成统一的数据集结构。
    """
    
    @staticmethod
    def collect_from_data_fetch(
        df: pd.DataFrame, 
        shell_ids: List[str]
    ) -> Dict[str, Dict]:
        """
        从Data_fetch页面数据中提取指定壳体号的数据
        
        提取全电流范围的功率、电压、效率、中心波长和shift数据。
        
        Args:
            df: Data_fetch页面的DataFrame，包含测试数据
            shell_ids: 要收集的壳体号列表
            
        Returns:
            字典，键为壳体号，值为包含以下字段的字典:
            {
                'shell_id': {
                    'current': [电流值列表],
                    'power': [功率值列表],
                    'voltage': [电压值列表],
                    'efficiency': [效率值列表],
                    'wavelength': [中心波长列表],
                    'shift': [shift列表],
                    'test_type': 测试类型,
                    'collected_at': 收集时间戳,
                    'data_available': 数据是否可用
                }
            }
        """
        result = {}
        collected_at = datetime.now().isoformat()
        
        # 定义列名映射（与Data_fetch页面保持一致）
        shell_col = "壳体号"
        test_type_col = "测试类型"
        current_col = "电流(A)"
        power_col = "功率(W)"
        voltage_col = "电压(V)"
        efficiency_col = "电光效率(%)"
        wavelength_col = "波长lambda"
        shift_col = "波长shift"
        
        # 检查必需列是否存在
        if df is None or df.empty:
            # 返回所有壳体号的空数据标记
            for shell_id in shell_ids:
                result[shell_id] = {
                    'current': [],
                    'power': [],
                    'voltage': [],
                    'efficiency': [],
                    'wavelength': [],
                    'shift': [],
                    'test_type': None,
                    'collected_at': collected_at,
                    'data_available': False
                }
            return result
        
        # 遍历每个壳体号
        for shell_id in shell_ids:
            try:
                # 筛选该壳体号的数据
                if shell_col not in df.columns:
                    result[shell_id] = {
                        'current': [],
                        'power': [],
                        'voltage': [],
                        'efficiency': [],
                        'wavelength': [],
                        'shift': [],
                        'test_type': None,
                        'collected_at': collected_at,
                        'data_available': False
                    }
                    continue
                
                shell_data = df[df[shell_col].astype(str).str.strip() == str(shell_id).strip()].copy()
                
                if shell_data.empty:
                    # 该壳体号无数据
                    result[shell_id] = {
                        'current': [],
                        'power': [],
                        'voltage': [],
                        'efficiency': [],
                        'wavelength': [],
                        'shift': [],
                        'test_type': None,
                        'collected_at': collected_at,
                        'data_available': False
                    }
                    continue
                
                # 提取测试类型（取第一个非空值）
                test_type = None
                if test_type_col in shell_data.columns:
                    test_type_values = shell_data[test_type_col].dropna()
                    if not test_type_values.empty:
                        test_type = str(test_type_values.iloc[0])
                
                # 按电流排序
                if current_col in shell_data.columns:
                    shell_data[current_col] = pd.to_numeric(shell_data[current_col], errors='coerce')
                    shell_data = shell_data.sort_values(current_col)
                
                # 提取各列数据，处理缺失值
                def extract_column(col_name):
                    if col_name in shell_data.columns:
                        series = pd.to_numeric(shell_data[col_name], errors='coerce')
                        # 过滤掉NaN值，返回列表
                        return series.dropna().tolist()
                    return []
                
                current_list = extract_column(current_col)
                power_list = extract_column(power_col)
                voltage_list = extract_column(voltage_col)
                efficiency_list = extract_column(efficiency_col)
                wavelength_list = extract_column(wavelength_col)
                shift_list = extract_column(shift_col)
                
                # 判断数据是否可用（至少有电流和一个测量值）
                data_available = len(current_list) > 0 and (
                    len(power_list) > 0 or 
                    len(voltage_list) > 0 or 
                    len(efficiency_list) > 0 or
                    len(wavelength_list) > 0 or
                    len(shift_list) > 0
                )
                
                result[shell_id] = {
                    'current': current_list,
                    'power': power_list,
                    'voltage': voltage_list,
                    'efficiency': efficiency_list,
                    'wavelength': wavelength_list,
                    'shift': shift_list,
                    'test_type': test_type,
                    'collected_at': collected_at,
                    'data_available': data_available
                }
                
            except Exception as e:
                # 记录错误到日志
                ErrorHandler.log_error(
                    e,
                    f"从Data_fetch收集壳体'{shell_id}'的数据时出错"
                )
                # 处理异常情况，标记数据不可用
                result[shell_id] = {
                    'current': [],
                    'power': [],
                    'voltage': [],
                    'efficiency': [],
                    'wavelength': [],
                    'shift': [],
                    'test_type': None,
                    'collected_at': collected_at,
                    'data_available': False,
                    'error': str(e)
                }
        
        return result
    
    @staticmethod
    def collect_from_test_analysis(
        df: pd.DataFrame,
        shell_ids: List[str],
        target_current: float
    ) -> Dict[str, Dict]:
        """
        ?TestAnalysis?????????????NA????????????

        Args:
            df: TestAnalysis???DataFrame?????????
            shell_ids: ?????????
            target_current: ?????????

        Returns:
            ????????????????????:
            {
                'shell_id': {
                    'na': NA?,
                    'spectral_fwhm': ?????,
                    'thermal_resistance': ???,
                    'current': ????,
                    'test_type': ????,
                    'collected_at': ?????,
                    'data_available': ??????
                }
            }
        """
        result: Dict[str, Dict] = {}
        collected_at = datetime.now().isoformat()

        shell_col = "???"
        test_type_col = "????"
        current_col = "????"
        na_col = "NA"
        thermal_col = "??"
        fwhm_col = "?????"

        if df is None or df.empty:
            for shell_id in shell_ids:
                result[shell_id] = {
                    'na': None,
                    'spectral_fwhm': None,
                    'thermal_resistance': None,
                    'current': target_current,
                    'test_type': None,
                    'collected_at': collected_at,
                    'data_available': False,
                }
            return result

        for shell_id in shell_ids:
            try:
                if shell_col not in df.columns:
                    result[shell_id] = {
                        'na': None,
                        'spectral_fwhm': None,
                        'thermal_resistance': None,
                        'current': target_current,
                        'test_type': None,
                        'collected_at': collected_at,
                        'data_available': False,
                    }
                    continue

                shell_data = df[df[shell_col].astype(str).str.strip() == str(shell_id).strip()].copy()

                if shell_data.empty:
                    result[shell_id] = {
                        'na': None,
                        'spectral_fwhm': None,
                        'thermal_resistance': None,
                        'current': target_current,
                        'test_type': None,
                        'collected_at': collected_at,
                        'data_available': False,
                    }
                    continue

                test_type = None
                if test_type_col in shell_data.columns:
                    test_type_values = shell_data[test_type_col].dropna()
                    if not test_type_values.empty:
                        test_type = str(test_type_values.iloc[0])

                na_value = None
                spectral_fwhm_value = None
                thermal_value = None
                current_value = target_current

                if current_col in shell_data.columns:
                    shell_data[current_col] = pd.to_numeric(shell_data[current_col], errors="coerce")
                    valid_data = shell_data[shell_data[current_col].notna()].copy()

                    if not valid_data.empty:
                        valid_data["current_diff"] = abs(valid_data[current_col] - target_current)
                        closest_row = valid_data.loc[valid_data["current_diff"].idxmin()]

                        if pd.notna(closest_row.get(current_col)):
                            try:
                                current_value = float(closest_row[current_col])
                            except (TypeError, ValueError):
                                current_value = target_current

                        if pd.notna(closest_row.get(na_col)):
                            try:
                                na_value = float(closest_row[na_col])
                            except (TypeError, ValueError):
                                na_value = None

                        if pd.notna(closest_row.get(fwhm_col)):
                            try:
                                spectral_fwhm_value = float(closest_row[fwhm_col])
                            except (TypeError, ValueError):
                                spectral_fwhm_value = None

                        if pd.notna(closest_row.get(thermal_col)):
                            try:
                                thermal_value = float(closest_row[thermal_col])
                            except (TypeError, ValueError):
                                thermal_value = None
                else:
                    if fwhm_col in shell_data.columns:
                        fwhm_series = pd.to_numeric(shell_data[fwhm_col], errors="coerce").dropna()
                        if not fwhm_series.empty:
                            spectral_fwhm_value = float(fwhm_series.iloc[0])

                    if na_col in shell_data.columns:
                        na_series = pd.to_numeric(shell_data[na_col], errors="coerce").dropna()
                        if not na_series.empty:
                            na_value = float(na_series.iloc[0])

                    if thermal_col in shell_data.columns:
                        thermal_series = pd.to_numeric(shell_data[thermal_col], errors="coerce").dropna()
                        if not thermal_series.empty:
                            thermal_value = float(thermal_series.iloc[0])

                data_available = any(
                    value is not None
                    for value in (na_value, spectral_fwhm_value, thermal_value)
                )

                result[shell_id] = {
                    'na': na_value,
                    'spectral_fwhm': spectral_fwhm_value,
                    'thermal_resistance': thermal_value,
                    'current': current_value,
                    'test_type': test_type,
                    'collected_at': collected_at,
                    'data_available': data_available,
                }

            except Exception as exc:
                ErrorHandler.log_error(
                    exc,
                    f"?TestAnalysis????'{shell_id}'??????"
                )
                result[shell_id] = {
                    'na': None,
                    'spectral_fwhm': None,
                    'thermal_resistance': None,
                    'current': target_current,
                    'test_type': None,
                    'collected_at': collected_at,
                    'data_available': False,
                    'error': str(exc),
                }

        return result

    @staticmethod
    def collect_from_progress(
        df: pd.DataFrame,
        shell_ids: List[str]
    ) -> Dict[str, Dict]:
        """
        从Progress页面数据中提取指定壳体号的进度信息
        
        提取料号、生产订单、当前站点、已完成站别和各站别时间戳信息。
        
        Args:
            df: Progress页面的DataFrame，包含工艺流程进度数据
            shell_ids: 要收集的壳体号列表
            
        Returns:
            字典，键为壳体号，值为包含以下字段的字典:
            {
                'shell_id': {
                    'part_number': 料号,
                    'production_order': 生产订单,
                    'current_station': 当前站点,
                    'completed_stations': [已完成站别列表],
                    'station_times': {站别: 时间},
                    'collected_at': 收集时间戳,
                    'data_available': 数据是否可用
                }
            }
        """
        result = {}
        collected_at = datetime.now().isoformat()
        
        # 定义列名映射（与Progress页面保持一致）
        shell_col = "壳体号"
        part_number_col = "料号"
        production_order_col = "生产订单"
        current_station_col = "当前站点"
        
        # 检查数据是否为空
        if df is None or df.empty:
            for shell_id in shell_ids:
                result[shell_id] = {
                    'part_number': None,
                    'production_order': None,
                    'current_station': None,
                    'completed_stations': [],
                    'station_times': {},
                    'collected_at': collected_at,
                    'data_available': False
                }
            return result
        
        # 遍历每个壳体号
        for shell_id in shell_ids:
            try:
                # 筛选该壳体号的数据
                if shell_col not in df.columns:
                    result[shell_id] = {
                        'part_number': None,
                        'production_order': None,
                        'current_station': None,
                        'completed_stations': [],
                        'station_times': {},
                        'collected_at': collected_at,
                        'data_available': False
                    }
                    continue
                
                shell_data = df[df[shell_col].astype(str).str.strip() == str(shell_id).strip()]
                
                if shell_data.empty:
                    result[shell_id] = {
                        'part_number': None,
                        'production_order': None,
                        'current_station': None,
                        'completed_stations': [],
                        'station_times': {},
                        'collected_at': collected_at,
                        'data_available': False
                    }
                    continue
                
                # 取第一行数据（一个壳体号应该只有一条记录）
                row = shell_data.iloc[0]
                
                # 提取料号
                part_number = None
                if part_number_col in row.index and pd.notna(row[part_number_col]):
                    part_number = str(row[part_number_col]).strip()
                
                # 提取生产订单
                production_order = None
                if production_order_col in row.index and pd.notna(row[production_order_col]):
                    production_order = str(row[production_order_col]).strip()
                
                # 提取当前站点
                current_station = None
                if current_station_col in row.index and pd.notna(row[current_station_col]):
                    current_station = str(row[current_station_col]).strip()
                
                # 提取已完成站别和站别时间
                completed_stations = []
                station_times = {}
                
                # 定义站别映射（与Progress页面保持一致）
                station_mapping = {
                    "机械件打标": "打标",
                    "机械件清洗": "清洗",
                    "壳体组装": "壳体组装",
                    "光耦回流": "回流",
                    "FAC前备料": "fac前备料",
                    "打线": "打线",
                    "FAC": "fac",
                    "FAC补胶": "fac补胶",
                    "FAC补胶后烘烤": "fac补胶后烘烤",
                    "FAC测试": "fac测试",
                    "SAC组装": "sac组装",
                    "光纤组装": "光纤组装",
                    "光纤组装后烘烤": "光纤组装后烘烤",
                    "红光耦合": "红光耦合",
                    "装大反": "装大反",
                    "耦合后烘烤": "红光耦合后烘烤",
                    "红光耦合后烘烤": "红光耦合后烘烤",
                    "合束": "合束",
                    "合束后烘烤": "合束后烘烤",
                    "VBG": "VBG",
                    "VBG后烘烤": "VBG后烘烤",
                    "NA前镜检": "NA前镜检",
                    "NA前红光端面检": "NA前红光端面检",
                    "NA前红光端面检查": "NA前红光端面检",
                    "NA测试": "NA测试",
                    "耦合测试": "耦合测试",
                    "补胶": "补胶",
                    "温度循环": "温度循环",
                    "pre测试": "Pre测试",
                    "Pre测试": "Pre测试",
                    "低温存储": "低温存储",
                    "低温储存": "低温存储",
                    "低温存储后测试": "低温存储后测试",
                    "低温储存后测试": "低温存储后测试",
                    "高温存储": "高温存储",
                    "高温存储后测试": "高温存储后测试",
                    "老化": "老化前红光端面",
                    "老化前红光端面": "老化前红光端面",
                    "老化前红光端面检查": "老化前红光端面",
                    "post测试": "post测试",
                    "Post测试": "post测试",
                    "红光端面检查": "红光端面检查",
                    "镜检": "镜检",
                    "封盖": "封盖",
                    "封盖测试": "封盖测试",
                    "分级": "分级",
                    "入库检": "入库检",
                    "入库": "入库",
                    "RMA": "RMA"
                }
                
                # 遍历所有列，查找站别时间列
                for col in df.columns:
                    if col.endswith("时间"):
                        # 提取站别名称（去掉"时间"后缀）
                        station_name_raw = col[:-2]
                        # 标准化站别名称
                        station_name = station_mapping.get(station_name_raw, station_name_raw)
                        
                        # 检查该站别是否有时间值
                        if col in row.index and pd.notna(row[col]):
                            time_value = row[col]
                            # 转换为字符串格式
                            if isinstance(time_value, pd.Timestamp):
                                time_str = time_value.isoformat()
                            else:
                                time_str = str(time_value)
                            
                            completed_stations.append(station_name)
                            station_times[station_name] = time_str
                
                # 判断数据是否可用
                data_available = (
                    part_number is not None or 
                    production_order is not None or 
                    current_station is not None or 
                    len(completed_stations) > 0
                )
                
                result[shell_id] = {
                    'part_number': part_number,
                    'production_order': production_order,
                    'current_station': current_station,
                    'completed_stations': completed_stations,
                    'station_times': station_times,
                    'collected_at': collected_at,
                    'data_available': data_available
                }
                
            except Exception as e:
                # 记录错误到日志
                ErrorHandler.log_error(
                    e,
                    f"从Progress收集壳体'{shell_id}'的数据时出错"
                )
                result[shell_id] = {
                    'part_number': None,
                    'production_order': None,
                    'current_station': None,
                    'completed_stations': [],
                    'station_times': {},
                    'collected_at': collected_at,
                    'data_available': False,
                    'error': str(e)
                }
        
        return result
    
    @staticmethod
    def _closest_index_and_value(
        current_list: List[float],
        target_current: float
    ) -> Tuple[Optional[int], Optional[float]]:
        """
        ??????????????
        """
        if not current_list:
            return None, None

        best_index: Optional[int] = None
        best_diff = float("inf")
        best_value: Optional[float] = None

        for idx, raw_value in enumerate(current_list):
            try:
                numeric_value = float(raw_value)
            except (TypeError, ValueError):
                continue

            diff = abs(numeric_value - target_current)
            if diff < best_diff:
                best_diff = diff
                best_index = idx
                best_value = numeric_value

        return best_index, best_value

    @staticmethod
    def _value_from_sequence(
        sequence: List[float],
        index: Optional[int]
    ) -> Optional[float]:
        """
        ?????????????????
        """
        if sequence is None or index is None:
            return None
        if index < 0 or index >= len(sequence):
            return None

        value = sequence[index]
        if value is None:
            return None

        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def merge_collected_data(
        data_fetch_data: Dict,
        test_analysis_data: Dict,
        metadata: Dict,
        *,
        progress_data: Optional[Dict] = None
    ) -> Dict:
        """
        ???????????????? Data_fetch ? TestAnalysis ???????
        Progress ??????????????
        """
        _ = progress_data

        target_current = metadata.get('target_current', 15.0)
        records: List[Dict] = []

        if not data_fetch_data or not test_analysis_data:
            metadata_copy = {
                'version': metadata.get('version', '1.0'),
                'created_at': metadata.get('created_at', datetime.now().isoformat()),
                'created_by': metadata.get('created_by', 'unknown'),
                'description': metadata.get('description', ''),
                'target_current': target_current,
                'record_count': 0,
                'shell_count': 0,
                'source_pages': metadata.get('source_pages', []),
                'missing_pages': metadata.get('missing_pages', []),
            }
            return {
                'metadata': metadata_copy,
                'records': records,
            }

        for shell_id, fetch_entry in data_fetch_data.items():
            if not fetch_entry.get('data_available'):
                continue

            test_entry = test_analysis_data.get(shell_id)
            if not test_entry or not test_entry.get('data_available'):
                continue

            test_current = test_entry.get('current', target_current)

            try:
                target = float(test_current)
            except (TypeError, ValueError):
                try:
                    target = float(target_current)
                except (TypeError, ValueError):
                    target = None

            if target is None:
                continue

            closest_idx, matched_current = DataCollector._closest_index_and_value(
                fetch_entry.get('current', []),
                target
            )
            if closest_idx is None or matched_current is None:
                continue

            record = {
                'shell_id': shell_id,
                'current': matched_current,
                'power': DataCollector._value_from_sequence(fetch_entry.get('power', []), closest_idx),
                'efficiency': DataCollector._value_from_sequence(fetch_entry.get('efficiency', []), closest_idx),
                'wavelength': DataCollector._value_from_sequence(fetch_entry.get('wavelength', []), closest_idx),
                'shift': DataCollector._value_from_sequence(fetch_entry.get('shift', []), closest_idx),
                'na': test_entry.get('na'),
                'spectral_fwhm': test_entry.get('spectral_fwhm'),
                'thermal_resistance': test_entry.get('thermal_resistance'),
            }

            records.append(record)

        shell_count = len({record['shell_id'] for record in records})
        metadata_copy = {
            'version': metadata.get('version', '1.0'),
            'created_at': metadata.get('created_at', datetime.now().isoformat()),
            'created_by': metadata.get('created_by', 'unknown'),
            'description': metadata.get('description', ''),
            'target_current': target_current,
            'record_count': len(records),
            'shell_count': shell_count,
            'source_pages': metadata.get('source_pages', []),
            'missing_pages': metadata.get('missing_pages', []),
        }

        return {
            'metadata': metadata_copy,
            'records': records,
        }

