"""
错误处理模块

该模块提供统一的错误处理和日志记录功能。
包括错误分类、错误消息格式化、日志记录和用户友好的错误提示。
"""

import logging
import traceback
from typing import Optional, Tuple, Callable, Any
from functools import wraps
from datetime import datetime
from pathlib import Path


class ErrorHandler:
    """
    错误处理器
    
    提供统一的错误处理、日志记录和用户友好的错误消息生成功能。
    """
    
    # 日志配置
    _logger = None
    _log_file = None
    
    @classmethod
    def initialize_logger(cls, log_dir: str = "logs") -> None:
        """
        初始化日志记录器
        
        Args:
            log_dir: 日志文件目录
        """
        if cls._logger is not None:
            return
        
        # 创建日志目录
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        
        # 生成日志文件名（按日期）
        log_filename = f"data_collection_{datetime.now().strftime('%Y%m%d')}.log"
        cls._log_file = log_path / log_filename
        
        # 配置日志记录器
        cls._logger = logging.getLogger('DataCollection')
        cls._logger.setLevel(logging.DEBUG)
        
        # 避免重复添加handler
        if not cls._logger.handlers:
            # 文件处理器
            file_handler = logging.FileHandler(cls._log_file, encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_formatter)
            cls._logger.addHandler(file_handler)
            
            # 控制台处理器（仅显示WARNING及以上级别）
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.WARNING)
            console_formatter = logging.Formatter('%(levelname)s - %(message)s')
            console_handler.setFormatter(console_formatter)
            cls._logger.addHandler(console_handler)
    
    @classmethod
    def get_logger(cls) -> logging.Logger:
        """
        获取日志记录器
        
        Returns:
            日志记录器实例
        """
        if cls._logger is None:
            cls.initialize_logger()
        return cls._logger
    
    @staticmethod
    def format_error_message(
        error: Exception,
        context: str = "",
        user_friendly: bool = True
    ) -> str:
        """
        格式化错误消息
        
        Args:
            error: 异常对象
            context: 错误上下文描述
            user_friendly: 是否返回用户友好的消息
            
        Returns:
            格式化后的错误消息
        """
        error_type = type(error).__name__
        error_msg = str(error)
        
        if user_friendly:
            # 用户友好的错误消息
            friendly_messages = {
                'FileNotFoundError': '文件或目录不存在',
                'PermissionError': '权限不足，无法访问文件或目录',
                'JSONDecodeError': 'JSON文件格式错误',
                'ValueError': '数据值不正确',
                'TypeError': '数据类型不正确',
                'KeyError': '缺少必需的数据字段',
                'IOError': '文件读写错误',
                'OSError': '系统操作错误',
            }
            
            base_msg = friendly_messages.get(error_type, '操作失败')
            
            if context:
                return f"{context}: {base_msg}"
            else:
                return base_msg
        else:
            # 详细的错误消息（用于日志）
            if context:
                return f"{context}: {error_type} - {error_msg}"
            else:
                return f"{error_type}: {error_msg}"
    
    @staticmethod
    def log_error(
        error: Exception,
        context: str = "",
        include_traceback: bool = True
    ) -> None:
        """
        记录错误到日志
        
        Args:
            error: 异常对象
            context: 错误上下文描述
            include_traceback: 是否包含堆栈跟踪
        """
        logger = ErrorHandler.get_logger()
        
        error_msg = ErrorHandler.format_error_message(
            error,
            context,
            user_friendly=False
        )
        
        if include_traceback:
            logger.error(error_msg, exc_info=True)
        else:
            logger.error(error_msg)
    
    @staticmethod
    def handle_error(
        error: Exception,
        context: str = "",
        log: bool = True
    ) -> Tuple[bool, str]:
        """
        处理错误并返回结果
        
        Args:
            error: 异常对象
            context: 错误上下文描述
            log: 是否记录到日志
            
        Returns:
            元组 (False, 用户友好的错误消息)
        """
        if log:
            ErrorHandler.log_error(error, context)
        
        user_msg = ErrorHandler.format_error_message(error, context, user_friendly=True)
        return False, user_msg
    
    @staticmethod
    def safe_execute(
        func: Callable,
        *args,
        context: str = "",
        default_return: Any = None,
        **kwargs
    ) -> Tuple[bool, Any, Optional[str]]:
        """
        安全执行函数，捕获并处理异常
        
        Args:
            func: 要执行的函数
            *args: 函数位置参数
            context: 错误上下文描述
            default_return: 发生错误时的默认返回值
            **kwargs: 函数关键字参数
            
        Returns:
            元组 (是否成功, 返回值或默认值, 错误消息或None)
        """
        try:
            result = func(*args, **kwargs)
            return True, result, None
        except Exception as e:
            ErrorHandler.log_error(e, context)
            error_msg = ErrorHandler.format_error_message(e, context, user_friendly=True)
            return False, default_return, error_msg


def handle_exceptions(context: str = "", log: bool = True):
    """
    装饰器：为函数添加异常处理
    
    Args:
        context: 错误上下文描述
        log: 是否记录到日志
        
    使用示例:
        @handle_exceptions(context="数据收集")
        def collect_data():
            # 函数实现
            pass
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log:
                    ErrorHandler.log_error(e, context or func.__name__)
                
                # 如果函数返回元组(bool, str)格式，返回错误结果
                error_msg = ErrorHandler.format_error_message(
                    e,
                    context or func.__name__,
                    user_friendly=True
                )
                return False, error_msg
        
        return wrapper
    return decorator


# 错误类型分类
class DataCollectionError(Exception):
    """数据收集错误基类"""
    pass


class DataValidationError(DataCollectionError):
    """数据验证错误"""
    pass


class DataStorageError(DataCollectionError):
    """数据存储错误"""
    pass


class DataLoadError(DataCollectionError):
    """数据加载错误"""
    pass


class ConfigurationError(DataCollectionError):
    """配置错误"""
    pass
