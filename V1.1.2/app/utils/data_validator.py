"""
数据验证模块

该模块负责验证数据的格式和完整性，确保数据集符合规范要求。
提供壳体号验证、数值数据验证、数据集结构验证和文件格式验证功能。
"""

import re
import os
from typing import Dict, List, Tuple, Optional, Any
from utils.error_handler import ErrorHandler, DataValidationError


class DataValidator:
    """
    数据验证器
    
    该类提供静态方法用于验证壳体号格式、数值数据、数据集结构和文件格式。
    所有验证方法返回元组，包含验证结果和错误消息。
    """
    
    # 文件大小限制（字节）：默认50MB
    MAX_FILE_SIZE = 50 * 1024 * 1024
    
    # 支持的文件扩展名
    SUPPORTED_EXTENSIONS = ['.json']
    
    @staticmethod
    def validate_shell_id(shell_id: str) -> Tuple[bool, str]:
        """
        验证壳体号格式
        
        壳体号应该是非空字符串，可以包含字母、数字、下划线和连字符。
        长度应在1-50个字符之间。
        
        Args:
            shell_id: 要验证的壳体号
            
        Returns:
            元组 (是否有效, 错误消息)
            - 如果有效，返回 (True, "")
            - 如果无效，返回 (False, "错误描述")
        """
        # 检查是否为None或空字符串
        if shell_id is None:
            return False, "壳体号不能为None"
        
        if not isinstance(shell_id, str):
            return False, f"壳体号必须是字符串类型，当前类型: {type(shell_id).__name__}"
        
        # 去除首尾空格
        shell_id = shell_id.strip()
        
        if not shell_id:
            return False, "壳体号不能为空"
        
        # 检查长度
        if len(shell_id) > 50:
            return False, f"壳体号长度不能超过50个字符，当前长度: {len(shell_id)}"
        
        # 检查格式：允许字母、数字、下划线、连字符和中文字符
        # 使用宽松的验证规则以支持各种命名格式
        if not re.match(r'^[\w\-\u4e00-\u9fa5]+$', shell_id):
            return False, f"壳体号格式不正确: '{shell_id}'，只能包含字母、数字、下划线、连字符和中文字符"
        
        return True, ""
    
    @staticmethod
    def validate_numeric_data(
        data: List[float],
        field_name: str,
        allow_empty: bool = True,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None
    ) -> Tuple[bool, str]:
        """
        验证数值型数据
        
        检查数据列表是否包含有效的数值，并可选地检查数值范围。
        
        Args:
            data: 要验证的数值列表
            field_name: 字段名称（用于错误消息）
            allow_empty: 是否允许空列表，默认True
            min_value: 最小值限制（可选）
            max_value: 最大值限制（可选）
            
        Returns:
            元组 (是否有效, 错误消息)
            - 如果有效，返回 (True, "")
            - 如果无效，返回 (False, "错误描述")
        """
        # 检查是否为None
        if data is None:
            return False, f"{field_name}不能为None"
        
        # 检查是否为列表
        if not isinstance(data, list):
            return False, f"{field_name}必须是列表类型，当前类型: {type(data).__name__}"
        
        # 检查是否为空
        if not data:
            if allow_empty:
                return True, ""
            else:
                return False, f"{field_name}不能为空列表"
        
        # 检查每个元素是否为数值
        for i, value in enumerate(data):
            if not isinstance(value, (int, float)):
                return False, f"{field_name}[{i}]必须是数值类型，当前类型: {type(value).__name__}"
            
            # 检查是否为NaN或无穷大
            if value != value:  # NaN检查
                return False, f"{field_name}[{i}]不能为NaN"
            
            if value == float('inf') or value == float('-inf'):
                return False, f"{field_name}[{i}]不能为无穷大"
            
            # 检查数值范围
            if min_value is not None and value < min_value:
                return False, f"{field_name}[{i}]={value}小于最小值{min_value}"
            
            if max_value is not None and value > max_value:
                return False, f"{field_name}[{i}]={value}大于最大值{max_value}"
        
        return True, ""
    
    @staticmethod
    def validate_dataset(dataset: Dict) -> Tuple[bool, List[str]]:
        """
        ?????????????????

        Args:
            dataset: ?????????

        Returns:
            (????, ?????????)
        """
        errors: List[str] = []
        warnings: List[str] = []

        if dataset is None:
            return False, ["??????None"]
        if not isinstance(dataset, dict):
            return False, [f"???????????????: {type(dataset).__name__}"]
        if not dataset:
            return False, ["???????"]

        metadata = dataset.get('metadata')
        if metadata is None:
            errors.append("??metadata??")
            metadata = {}
        elif not isinstance(metadata, dict):
            errors.append(f"metadata????????????: {type(metadata).__name__}")
            metadata = {}

        required_metadata_fields = ['version', 'created_at', 'target_current']
        for field in required_metadata_fields:
            if field not in metadata:
                warnings.append(f"metadata??????: {field}")

        target_current = metadata.get('target_current')
        if target_current is not None:
            if not isinstance(target_current, (int, float)):
                errors.append(f"metadata.target_current????????????: {type(target_current).__name__}")
            elif target_current <= 0:
                warnings.append(f"metadata.target_current?????????: {target_current}")

        records = dataset.get('records')
        if records is None:
            errors.append("??records??")
            return False, errors if errors else warnings
        if not isinstance(records, list):
            errors.append(f"records????????????: {type(records).__name__}")
            return False, errors

        if not records:
            warnings.append("records????????????")

        def _validate_optional_numeric(value, field_label: str, *, allow_negative: bool = False, upper_bound: Optional[float] = None) -> None:
            if value is None:
                return
            if not isinstance(value, (int, float)):
                errors.append(f"{field_label}????????????: {type(value).__name__}")
                return
            if value != value:
                errors.append(f"{field_label}???NaN")
                return
            if not allow_negative and value < 0:
                errors.append(f"{field_label}?????????: {value}")
                return
            if upper_bound is not None and value > upper_bound:
                warnings.append(f"{field_label}??????{upper_bound}????: {value}")

        unique_shell_ids = set()

        for idx, record in enumerate(records, start=1):
            location = f"??#{idx}"
            if not isinstance(record, dict):
                errors.append(f"{location}???????")
                continue

            shell_id = record.get('shell_id')
            if shell_id is None:
                errors.append(f"{location}??shell_id??")
                continue

            is_valid, message = DataValidator.validate_shell_id(str(shell_id))
            if not is_valid:
                errors.append(f"{location}??????: {message}")
                continue
            unique_shell_ids.add(str(shell_id).strip())

            current = record.get('current')
            if current is None:
                errors.append(f"{location}??current??")
            elif not isinstance(current, (int, float)):
                errors.append(f"{location}?current????????????: {type(current).__name__}")
            elif current < 0:
                errors.append(f"{location}?current?????????: {current}")

            _validate_optional_numeric(record.get('power'), f"{location}?power")
            _validate_optional_numeric(record.get('efficiency'), f"{location}?efficiency", upper_bound=100)
            _validate_optional_numeric(record.get('wavelength'), f"{location}?wavelength")
            _validate_optional_numeric(record.get('shift'), f"{location}?shift", allow_negative=True)

            na_value = record.get('na')
            if na_value is not None:
                if not isinstance(na_value, (int, float)):
                    errors.append(f"{location}?NA????????????: {type(na_value).__name__}")
                elif not (0 <= na_value <= 1):
                    warnings.append(f"{location}?NA={na_value}??????[0, 1]")

            _validate_optional_numeric(record.get('spectral_fwhm'), f"{location}?spectral_fwhm")
            _validate_optional_numeric(record.get('thermal_resistance'), f"{location}?thermal_resistance")

        record_count = len(records)
        metadata_record_count = metadata.get('record_count')
        if isinstance(metadata_record_count, int) and metadata_record_count != record_count:
            warnings.append(f"metadata.record_count({metadata_record_count})??????({record_count})???")

        shell_count = len(unique_shell_ids)
        metadata_shell_count = metadata.get('shell_count')
        if isinstance(metadata_shell_count, int) and metadata_shell_count != shell_count:
            warnings.append(f"metadata.shell_count({metadata_shell_count})??????({shell_count})???")

        if errors:
            return False, errors
        return True, warnings

    @staticmethod
    def validate_file_format(file_path: str) -> Tuple[bool, str]:
        """
        验证文件格式
        
        检查文件扩展名、文件是否存在以及文件大小是否在限制范围内。
        
        Args:
            file_path: 文件路径
            
        Returns:
            元组 (是否有效, 错误消息)
            - 如果有效，返回 (True, "")
            - 如果无效，返回 (False, "错误描述")
        """
        # 检查文件路径是否为None或空
        if not file_path:
            return False, "文件路径不能为空"
        
        if not isinstance(file_path, str):
            return False, f"文件路径必须是字符串类型，当前类型: {type(file_path).__name__}"
        
        # 检查文件扩展名
        _, ext = os.path.splitext(file_path)
        if ext.lower() not in DataValidator.SUPPORTED_EXTENSIONS:
            return False, f"不支持的文件格式: {ext}，支持的格式: {', '.join(DataValidator.SUPPORTED_EXTENSIONS)}"
        
        # 检查文件是否存在（仅在保存时不需要检查）
        if os.path.exists(file_path):
            # 检查是否为文件（而非目录）
            if not os.path.isfile(file_path):
                return False, f"路径'{file_path}'不是一个文件"
            
            # 检查文件大小
            try:
                file_size = os.path.getsize(file_path)
                if file_size > DataValidator.MAX_FILE_SIZE:
                    size_mb = file_size / (1024 * 1024)
                    max_mb = DataValidator.MAX_FILE_SIZE / (1024 * 1024)
                    return False, f"文件大小{size_mb:.2f}MB超过限制{max_mb:.2f}MB"
            except OSError as e:
                return False, f"无法获取文件大小: {e}"
        
        return True, ""
    
    @staticmethod
    def validate_save_path(file_path: str, allow_overwrite: bool = False) -> Tuple[bool, str]:
        """
        验证保存路径
        
        检查保存路径的有效性，包括目录是否存在、是否有写权限、文件是否已存在等。
        
        Args:
            file_path: 要保存的文件路径
            allow_overwrite: 是否允许覆盖已存在的文件，默认False
            
        Returns:
            元组 (是否有效, 错误消息)
            - 如果有效，返回 (True, "")
            - 如果无效，返回 (False, "错误描述")
        """
        # 先验证文件格式
        is_valid, error_msg = DataValidator.validate_file_format(file_path)
        if not is_valid:
            return False, error_msg
        
        # 获取目录路径
        dir_path = os.path.dirname(file_path)
        
        # 如果目录路径为空，使用当前目录
        if not dir_path:
            dir_path = '.'
        
        # 检查目录是否存在
        if not os.path.exists(dir_path):
            return False, f"目录'{dir_path}'不存在"
        
        # 检查是否为目录
        if not os.path.isdir(dir_path):
            return False, f"路径'{dir_path}'不是一个目录"
        
        # 检查目录是否有写权限
        if not os.access(dir_path, os.W_OK):
            return False, f"目录'{dir_path}'没有写权限"
        
        # 检查文件是否已存在
        if os.path.exists(file_path):
            if not allow_overwrite:
                return False, f"文件'{file_path}'已存在，如需覆盖请确认"
            
            # 检查文件是否有写权限
            if not os.access(file_path, os.W_OK):
                return False, f"文件'{file_path}'没有写权限"
        
        return True, ""
