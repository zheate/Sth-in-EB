# Custom exceptions for local data storage operations.

class LocalStorageError(Exception):
    """本地存储基础异常"""
    pass


class StorageError(LocalStorageError):
    """存储操作错误"""
    pass


class DatasetNotFoundError(LocalStorageError):
    """数据集不存在"""
    pass


class CorruptedDataError(LocalStorageError):
    """数据损坏"""
    pass


class SerializationError(LocalStorageError):
    """序列化错误"""
    pass


class ExportError(LocalStorageError):
    """导出错误"""
    pass
