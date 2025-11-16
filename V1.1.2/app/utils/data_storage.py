"""
数据存储模块

该模块负责数据集的保存和加载操作，支持JSON格式的文件存储。
提供默认路径管理、文件名生成、数据序列化和反序列化功能。
"""

from typing import Dict, Optional, Tuple, List
import json
from pathlib import Path
from datetime import datetime
from utils.error_handler import ErrorHandler, DataStorageError, DataLoadError


class DataStorage:
    """
    数据存储管理器
    
    提供数据集的保存和加载功能，管理文件路径和文件名生成。
    支持JSON格式的数据持久化。
    """
    
    @staticmethod
    def save_dataset(
        dataset: Dict,
        file_path: str,
        file_name: str
    ) -> Tuple[bool, str]:
        """
        保存数据集到文件
        
        将数据集序列化为JSON格式并保存到指定路径。如果路径不存在，将自动创建。
        如果文件已存在，将提示用户确认是否覆盖。
        
        Args:
            dataset: 数据集字典，包含metadata和records数据
            file_path: 保存路径（目录路径）
            file_name: 文件名（不含扩展名或含.json扩展名）
            
        Returns:
            元组 (是否成功, 消息)
            - 如果成功，返回 (True, "保存成功的消息")
            - 如果失败，返回 (False, "失败原因")
            
        异常处理:
            - FileNotFoundError: 路径不存在且无法创建
            - PermissionError: 没有写入权限
            - IOError: 其他IO错误
        """
        try:
            # 验证和处理路径
            save_dir = Path(file_path)
            
            # 如果路径不存在，尝试创建
            if not save_dir.exists():
                try:
                    save_dir.mkdir(parents=True, exist_ok=True)
                    ErrorHandler.get_logger().info(f"创建目录: {file_path}")
                except PermissionError as e:
                    ErrorHandler.log_error(e, f"创建目录'{file_path}'时权限不足")
                    return False, f"权限不足，无法创建目录: {file_path}"
                except Exception as e:
                    ErrorHandler.log_error(e, f"创建目录'{file_path}'失败")
                    return False, f"无法创建目录: {str(e)}"
            
            # 验证路径是否为目录
            if not save_dir.is_dir():
                error_msg = f"指定的路径不是目录: {file_path}"
                ErrorHandler.get_logger().error(error_msg)
                return False, error_msg
            
            # 处理文件名，确保有.json扩展名
            if not file_name.endswith('.json'):
                file_name = f"{file_name}.json"
            
            # 构建完整文件路径
            full_path = save_dir / file_name
            
            # 检查文件是否已存在
            file_exists = full_path.exists()
            
            # 记录保存操作
            ErrorHandler.get_logger().info(
                f"开始保存数据集: {full_path} (覆盖={file_exists})"
            )
            
            # 保存数据集为JSON格式
            try:
                with open(full_path, 'w', encoding='utf-8') as f:
                    json.dump(dataset, f, ensure_ascii=False, indent=2)
                
                # 记录成功
                ErrorHandler.get_logger().info(f"数据集保存成功: {full_path}")
                
            except PermissionError as e:
                ErrorHandler.log_error(e, f"写入文件'{full_path}'时权限不足")
                return False, f"权限不足，无法写入文件: {full_path}"
            except IOError as e:
                ErrorHandler.log_error(e, f"写入文件'{full_path}'失败")
                return False, f"文件写入失败: {str(e)}"
            except Exception as e:
                ErrorHandler.log_error(e, f"保存数据集到'{full_path}'时出错")
                return False, f"保存失败: {str(e)}"
            
            # 返回成功消息
            if file_exists:
                return True, f"数据集已成功保存（覆盖）: {full_path}"
            else:
                return True, f"数据集已成功保存: {full_path}"
                
        except Exception as e:
            ErrorHandler.log_error(e, "保存数据集时发生未预期的错误")
            return False, f"保存失败: {str(e)}"
    
    @staticmethod
    def load_dataset(file_path: str) -> Tuple[Optional[Dict], str]:
        """
        从文件加载数据集
        
        从指定路径读取JSON文件并反序列化为数据集字典。
        加载前会验证文件格式和数据结构。
        
        Args:
            file_path: 文件完整路径（包含文件名和扩展名）
            
        Returns:
            元组 (数据集字典或None, 消息)
            - 如果成功，返回 (数据集字典, "加载成功的消息")
            - 如果失败，返回 (None, "失败原因")
            
        异常处理:
            - FileNotFoundError: 文件不存在
            - json.JSONDecodeError: JSON格式错误
            - IOError: 其他IO错误
        """
        try:
            # 导入验证器
            from utils.data_validator import DataValidator
            
            # 记录加载操作
            ErrorHandler.get_logger().info(f"开始加载数据集: {file_path}")
            
            # 验证文件格式
            is_valid, error_msg = DataValidator.validate_file_format(file_path)
            if not is_valid:
                ErrorHandler.get_logger().error(f"文件格式验证失败: {error_msg}")
                return None, f"文件格式验证失败: {error_msg}"
            
            # 检查文件是否存在
            file_path_obj = Path(file_path)
            if not file_path_obj.exists():
                error_msg = f"文件不存在: {file_path}"
                ErrorHandler.get_logger().error(error_msg)
                return None, error_msg
            
            # 检查是否为文件
            if not file_path_obj.is_file():
                error_msg = f"指定的路径不是文件: {file_path}"
                ErrorHandler.get_logger().error(error_msg)
                return None, error_msg
            
            # 读取并解析JSON文件
            try:
                with open(file_path_obj, 'r', encoding='utf-8') as f:
                    dataset = json.load(f)
                ErrorHandler.get_logger().info(f"JSON文件解析成功: {file_path}")
            except json.JSONDecodeError as e:
                ErrorHandler.log_error(e, f"解析JSON文件'{file_path}'失败")
                return None, f"JSON格式错误: {str(e)}"
            except PermissionError as e:
                ErrorHandler.log_error(e, f"读取文件'{file_path}'时权限不足")
                return None, f"权限不足，无法读取文件: {file_path}"
            except IOError as e:
                ErrorHandler.log_error(e, f"读取文件'{file_path}'失败")
                return None, f"文件读取失败: {str(e)}"
            
            # 验证数据集结构
            is_valid, validation_messages = DataValidator.validate_dataset(dataset)
            
            # 构建返回消息
            if is_valid:
                records = dataset.get('records', []) or []
                record_count = len(records)
                shell_ids = {
                    str(record.get('shell_id')).strip()
                    for record in records
                    if record.get('shell_id') is not None
                }
                shell_count = len(shell_ids)
                
                success_msg = f"数据集加载成功: {file_path}\n"
                success_msg += f"记录数量: {record_count}\n"
                success_msg += f"壳体数量: {shell_count}"
                
                # 记录成功
                ErrorHandler.get_logger().info(
                    f"数据集加载成功: {file_path}, 记录数量: {record_count}, 壳体数量: {shell_count}"
                )
                
                # 如果有警告信息，添加到消息中
                if validation_messages:
                    success_msg += f"\n\n⚠️ 发现 {len(validation_messages)} 个警告:"
                    # 只显示前5个警告
                    for msg in validation_messages[:5]:
                        success_msg += f"\n  • {msg}"
                    if len(validation_messages) > 5:
                        success_msg += f"\n  • ... 还有 {len(validation_messages) - 5} 个警告"
                    
                    # 记录警告到日志
                    for msg in validation_messages:
                        ErrorHandler.get_logger().warning(f"数据验证警告: {msg}")
                
                return dataset, success_msg
            else:
                # 验证失败，但提供修复建议
                error_msg = f"❌ 数据集验证失败 ({len(validation_messages)} 个错误)\n\n"
                error_msg += "错误详情:\n"
                
                # 分类错误
                critical_errors = []
                field_errors = []
                data_errors = []
                
                for msg in validation_messages:
                    if '缺少' in msg or 'metadata' in msg or 'records' in msg:
                        critical_errors.append(msg)
                    elif '字段' in msg or '类型' in msg:
                        field_errors.append(msg)
                    else:
                        data_errors.append(msg)
                
                # 显示关键错误
                if critical_errors:
                    error_msg += "\n🔴 关键错误:\n"
                    for msg in critical_errors[:3]:
                        error_msg += f"  • {msg}\n"
                    if len(critical_errors) > 3:
                        error_msg += f"  • ... 还有 {len(critical_errors) - 3} 个\n"
                
                # 显示字段错误
                if field_errors:
                    error_msg += "\n🟡 字段错误:\n"
                    for msg in field_errors[:3]:
                        error_msg += f"  • {msg}\n"
                    if len(field_errors) > 3:
                        error_msg += f"  • ... 还有 {len(field_errors) - 3} 个\n"
                
                # 显示数据错误
                if data_errors:
                    error_msg += "\n🟠 数据错误:\n"
                    for msg in data_errors[:3]:
                        error_msg += f"  • {msg}\n"
                    if len(data_errors) > 3:
                        error_msg += f"  • ... 还有 {len(data_errors) - 3} 个\n"
                
                # 添加修复建议
                error_msg += "\n💡 修复建议:\n"
                if critical_errors:
                    error_msg += "  • 检查文件是否为有效的数据集格式\n"
                    error_msg += "  • 确保包含 metadata 和 records 字段\n"
                if field_errors:
                    error_msg += "  • 检查数据字段类型是否正确\n"
                if data_errors:
                    error_msg += "  • 检查数据值是否在合理范围内\n"
                
                # 记录验证错误
                ErrorHandler.get_logger().error(f"数据集验证失败: {file_path}")
                for msg in validation_messages:
                    ErrorHandler.get_logger().error(f"验证错误: {msg}")
                
                return None, error_msg
                
        except Exception as e:
            ErrorHandler.log_error(e, f"加载数据集'{file_path}'时发生未预期的错误")
            return None, f"加载失败: {str(e)}"
    
    @staticmethod
    def get_default_save_path() -> str:
        """
        获取默认保存路径
        
        从配置文件中读取默认的数据集保存路径。
        如果配置中未指定，返回系统默认路径。
        
        Returns:
            默认保存路径字符串
        """
        try:
            from config import get_dataset_save_path
            return str(get_dataset_save_path())
        except ImportError:
            return str(Path.cwd())
    
    @staticmethod
    def generate_default_filename() -> str:
        """
        生成默认文件名（包含时间戳）
        
        生成格式为 "dataset_YYYYMMDD_HHMMSS.json" 的文件名。
        
        Returns:
            默认文件名字符串，例如: "dataset_20251018_103000.json"
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"dataset_{timestamp}.json"
    
    @staticmethod
    def save_dataset_with_validation(
        dataset: Dict,
        file_path: str,
        file_name: str,
        allow_warnings: bool = True
    ) -> Tuple[bool, str, Optional[List[str]]]:
        """
        保存数据集并进行验证
        
        在保存前验证数据集，如果有错误则不保存，如果只有警告则根据参数决定是否保存。
        
        Args:
            dataset: 数据集字典
            file_path: 保存路径
            file_name: 文件名
            allow_warnings: 是否允许在有警告的情况下保存
            
        Returns:
            元组 (是否成功, 消息, 验证消息列表)
        """
        from utils.data_validator import DataValidator
        
        # 验证数据集
        is_valid, validation_messages = DataValidator.validate_dataset(dataset)
        
        if not is_valid:
            # 验证失败，不保存
            error_msg = f"❌ 数据集验证失败，无法保存\n\n"
            error_msg += f"发现 {len(validation_messages)} 个错误:\n"
            for msg in validation_messages[:5]:
                error_msg += f"  • {msg}\n"
            if len(validation_messages) > 5:
                error_msg += f"  • ... 还有 {len(validation_messages) - 5} 个错误\n"
            
            ErrorHandler.get_logger().error("数据集验证失败，取消保存")
            return False, error_msg, validation_messages
        
        # 验证通过或只有警告
        if validation_messages and not allow_warnings:
            # 有警告但不允许保存
            warning_msg = f"⚠️ 数据集有 {len(validation_messages)} 个警告，已取消保存\n\n"
            for msg in validation_messages[:5]:
                warning_msg += f"  • {msg}\n"
            if len(validation_messages) > 5:
                warning_msg += f"  • ... 还有 {len(validation_messages) - 5} 个警告\n"
            
            return False, warning_msg, validation_messages
        
        # 保存数据集
        success, save_msg = DataStorage.save_dataset(dataset, file_path, file_name)
        
        if success and validation_messages:
            # 保存成功但有警告
            save_msg += f"\n\n⚠️ 注意: 数据集有 {len(validation_messages)} 个警告"
        
        return success, save_msg, validation_messages if success else None
