# Local data storage module for saving, loading, and managing datasets.

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import base64
import io
import json
import re
import uuid

import pandas as pd

from utils.exceptions import StorageError, DatasetNotFoundError, CorruptedDataError, SerializationError, ExportError


class DataCategory(Enum):
    """数据类别枚举"""
    EXTRACTION = "extraction"   # 数据提取
    PROGRESS = "progress"       # 进度追踪
    ANALYSIS = "analysis"       # 工程分析


@dataclass
class DatasetMetadata:
    """数据集元数据"""
    id: str                                 # 唯一标识 (UUID)
    name: str                               # 用户可读名称
    category: DataCategory                  # 数据类别
    created_at: str                         # 创建时间 ISO 格式
    row_count: int                          # 行数
    columns: List[str] = field(default_factory=list)  # 列名列表
    source_file: Optional[str] = None       # 原始数据来源
    note: Optional[str] = None              # 用户备注
    extra: Dict[str, Any] = field(default_factory=dict)  # 扩展元数据

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "id": self.id,
            "name": self.name,
            "category": self.category.value,
            "created_at": self.created_at,
            "row_count": self.row_count,
            "columns": self.columns,
            "source_file": self.source_file,
            "note": self.note,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetMetadata":
        """从字典创建元数据对象"""
        return cls(
            id=data["id"],
            name=data["name"],
            category=DataCategory(data["category"]),
            created_at=data["created_at"],
            row_count=data["row_count"],
            columns=data.get("columns", []),
            source_file=data.get("source_file"),
            note=data.get("note"),
            extra=data.get("extra", {}),
        )


class LocalDataStore:
    """本地数据存储管理器"""
    
    # 默认存储基础路径
    DEFAULT_BASE_PATH = Path(__file__).parent.parent / "data" / "saved"
    
    def __init__(self, base_path: Optional[Path] = None):
        """
        初始化存储管理器
        
        Args:
            base_path: 存储基础路径，默认为 app/data/saved/
        """
        self.base_path = Path(base_path) if base_path else self.DEFAULT_BASE_PATH
        self._ensure_directories()
    
    def _ensure_directories(self) -> None:
        """确保所有必要的存储目录存在"""
        for category in DataCategory:
            category_path = self.base_path / category.value
            category_path.mkdir(parents=True, exist_ok=True)
    
    def _get_category_path(self, category: DataCategory) -> Path:
        """获取指定类别的存储目录路径"""
        return self.base_path / category.value

    def _sanitize_filename(self, filename: str) -> str:
        """
        清理文件名，移除或替换不安全字符
        
        Args:
            filename: 原始文件名
            
        Returns:
            安全的文件名
        """
        # 移除或替换不安全字符，只保留字母、数字、中文、下划线、连字符
        sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
        # 移除前后空白
        sanitized = sanitized.strip()
        # 将多个连续的下划线或空格替换为单个下划线
        sanitized = re.sub(r'[\s_]+', '_', sanitized)
        # 移除首尾的下划线
        sanitized = sanitized.strip('_')
        # 如果结果为空，返回默认名称
        if not sanitized:
            sanitized = "unnamed"
        # 限制长度（保留扩展名空间）
        max_length = 200
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length]
        return sanitized

    def _generate_filename(
        self,
        df: pd.DataFrame,
        category: DataCategory,
        custom_filename: Optional[str] = None
    ) -> str:
        """
        生成文件名
        
        Args:
            df: DataFrame 数据
            category: 数据类别
            custom_filename: 用户自定义文件名（不含扩展名）
            
        Returns:
            生成的文件名（不含扩展名）
        """
        if custom_filename:
            return self._sanitize_filename(custom_filename)
        
        # 自动生成：时间戳 + 数据摘要
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        row_count = len(df)
        col_count = len(df.columns)
        summary = f"{row_count}rows_{col_count}cols"
        
        return f"{category.value}_{timestamp}_{summary}"

    def _generate_dataset_id(self) -> str:
        """生成唯一的数据集 ID"""
        return str(uuid.uuid4())

    def save(
        self,
        df: pd.DataFrame,
        category: DataCategory,
        name: Optional[str] = None,
        custom_filename: Optional[str] = None,
        note: Optional[str] = None,
        extra_data: Optional[Dict[str, Any]] = None,
        source_file: Optional[str] = None
    ) -> str:
        """
        保存数据集
        
        Args:
            df: 要保存的 DataFrame
            category: 数据类别
            name: 数据集显示名称（用于列表展示）
            custom_filename: 用户自定义文件名（不含扩展名），为空则自动生成
            note: 用户备注
            extra_data: 扩展数据（如绘图数据源）
            source_file: 原始数据来源
        
        Returns:
            str: 数据集 ID (UUID)
            
        Raises:
            StorageError: 存储操作失败时抛出
        """
        # 生成唯一 ID
        dataset_id = self._generate_dataset_id()
        
        # 生成文件名
        filename = self._generate_filename(df, category, custom_filename)
        
        # 获取存储目录
        category_path = self._get_category_path(category)
        
        # 文件路径
        parquet_path = category_path / f"{dataset_id}.parquet"
        meta_path = category_path / f"{dataset_id}.meta.json"
        extra_path = category_path / f"{dataset_id}.extra.json"
        
        try:
            # 保存 DataFrame 为 Parquet 格式
            df.to_parquet(parquet_path, index=False)
            
            # 生成显示名称
            display_name = name if name else filename
            
            # 创建元数据
            metadata = DatasetMetadata(
                id=dataset_id,
                name=display_name,
                category=category,
                created_at=datetime.now().isoformat(),
                row_count=len(df),
                columns=list(df.columns),
                source_file=source_file,
                note=note,
                extra={}
            )
            
            # 保存元数据 JSON
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(metadata.to_dict(), f, ensure_ascii=False, indent=2)
            
            # 如果有扩展数据，保存扩展数据 JSON
            if extra_data:
                with open(extra_path, 'w', encoding='utf-8') as f:
                    json.dump(extra_data, f, ensure_ascii=False, indent=2)
            
            return dataset_id
            
        except PermissionError as e:
            raise StorageError(f"无法写入文件，请检查权限: {e}")
        except OSError as e:
            # 检查是否是磁盘空间不足
            if "No space left" in str(e) or "disk full" in str(e).lower():
                raise StorageError("存储空间不足，请清理磁盘")
            raise StorageError(f"存储操作失败: {e}")

    def _find_dataset_files(self, dataset_id: str) -> Tuple[Path, DataCategory]:
        """
        查找数据集文件所在的路径和类别
        
        Args:
            dataset_id: 数据集 ID
            
        Returns:
            Tuple[Path, DataCategory]: (元数据文件路径, 数据类别)
            
        Raises:
            DatasetNotFoundError: 数据集不存在时抛出
        """
        for category in DataCategory:
            category_path = self._get_category_path(category)
            meta_path = category_path / f"{dataset_id}.meta.json"
            if meta_path.exists():
                return meta_path, category
        
        raise DatasetNotFoundError(f"未找到指定的数据集: {dataset_id}")

    def load(self, dataset_id: str) -> Tuple[pd.DataFrame, DatasetMetadata, Optional[Dict[str, Any]]]:
        """
        加载数据集
        
        Args:
            dataset_id: 数据集 ID
        
        Returns:
            Tuple[DataFrame, Metadata, ExtraData]: 
                - DataFrame: 还原的数据
                - DatasetMetadata: 元数据对象
                - Optional[Dict]: 扩展数据（如绘图数据源），如不存在则为 None
                
        Raises:
            DatasetNotFoundError: 数据集不存在时抛出
            CorruptedDataError: 数据文件损坏时抛出
        """
        # 查找数据集文件
        meta_path, category = self._find_dataset_files(dataset_id)
        category_path = self._get_category_path(category)
        
        parquet_path = category_path / f"{dataset_id}.parquet"
        extra_path = category_path / f"{dataset_id}.extra.json"
        
        # 检查 Parquet 文件是否存在
        if not parquet_path.exists():
            raise DatasetNotFoundError(f"数据文件不存在: {dataset_id}")
        
        try:
            # 读取元数据 JSON
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta_dict = json.load(f)
            metadata = DatasetMetadata.from_dict(meta_dict)
        except json.JSONDecodeError as e:
            raise CorruptedDataError(f"元数据文件损坏，无法解析 JSON: {e}")
        except KeyError as e:
            raise CorruptedDataError(f"元数据文件缺少必要字段: {e}")
        except Exception as e:
            raise CorruptedDataError(f"读取元数据失败: {e}")
        
        try:
            # 读取 Parquet 文件还原 DataFrame
            df = pd.read_parquet(parquet_path)
        except Exception as e:
            raise CorruptedDataError(f"数据文件损坏，无法读取 Parquet: {e}")
        
        # 读取扩展数据（如存在）
        extra_data = None
        if extra_path.exists():
            try:
                with open(extra_path, 'r', encoding='utf-8') as f:
                    extra_data = json.load(f)
            except json.JSONDecodeError as e:
                raise CorruptedDataError(f"扩展数据文件损坏，无法解析 JSON: {e}")
            except Exception as e:
                raise CorruptedDataError(f"读取扩展数据失败: {e}")
        
        return df, metadata, extra_data

    def list_datasets(
        self,
        category: Optional[DataCategory] = None
    ) -> List[DatasetMetadata]:
        """
        列出数据集
        
        Args:
            category: 可选的数据类别筛选，为 None 时返回所有类别
        
        Returns:
            List[DatasetMetadata]: 数据集元数据列表，按创建时间降序排列
        """
        datasets: List[DatasetMetadata] = []
        
        # 确定要扫描的类别
        categories_to_scan = [category] if category else list(DataCategory)
        
        for cat in categories_to_scan:
            category_path = self._get_category_path(cat)
            
            # 扫描目录中的所有 .meta.json 文件
            if not category_path.exists():
                continue
                
            for meta_file in category_path.glob("*.meta.json"):
                try:
                    with open(meta_file, 'r', encoding='utf-8') as f:
                        meta_dict = json.load(f)
                    metadata = DatasetMetadata.from_dict(meta_dict)
                    datasets.append(metadata)
                except (json.JSONDecodeError, KeyError, ValueError):
                    # 跳过损坏或无效的元数据文件
                    continue
        
        # 按创建时间降序排列（最新的在前）
        datasets.sort(key=lambda x: x.created_at, reverse=True)
        
        return datasets

    def delete(self, dataset_id: str) -> bool:
        """
        删除数据集
        
        删除数据集的所有关联文件（.parquet, .meta.json, .extra.json）
        
        Args:
            dataset_id: 数据集 ID
        
        Returns:
            bool: 删除成功返回 True
            
        Raises:
            DatasetNotFoundError: 数据集不存在时抛出
            StorageError: 删除操作失败时抛出
        """
        # 查找数据集文件所在位置
        meta_path, category = self._find_dataset_files(dataset_id)
        category_path = self._get_category_path(category)
        
        # 定义所有可能的关联文件
        files_to_delete = [
            category_path / f"{dataset_id}.parquet",
            category_path / f"{dataset_id}.meta.json",
            category_path / f"{dataset_id}.extra.json",
        ]
        
        try:
            # 删除所有存在的关联文件
            for file_path in files_to_delete:
                if file_path.exists():
                    file_path.unlink()
            
            return True
            
        except PermissionError as e:
            raise StorageError(f"无法删除文件，请检查权限: {e}")
        except OSError as e:
            raise StorageError(f"删除操作失败: {e}")

    def _generate_export_filename(
        self,
        category: Optional[DataCategory],
        export_type: str,
        extension: str
    ) -> str:
        """
        生成导出文件名
        
        Args:
            category: 数据类别（可选，用于批量导出时可能为 None）
            export_type: 导出类型描述
            extension: 文件扩展名（不含点）
            
        Returns:
            str: 生成的文件名（含扩展名）
        """
        date_str = datetime.now().strftime("%Y%m%d")
        category_str = category.value if category else "mixed"
        return f"{category_str}_{export_type}_{date_str}.{extension}"

    def _generate_summary_df(self, df: pd.DataFrame, metadata: DatasetMetadata) -> pd.DataFrame:
        """
        生成数据统计摘要 DataFrame
        
        Args:
            df: 原始数据 DataFrame
            metadata: 数据集元数据
            
        Returns:
            pd.DataFrame: 统计摘要 DataFrame
        """
        summary_data = []
        
        # 基本信息
        summary_data.append({"项目": "数据集名称", "值": metadata.name})
        summary_data.append({"项目": "数据类别", "值": metadata.category.value})
        summary_data.append({"项目": "创建时间", "值": metadata.created_at})
        summary_data.append({"项目": "行数", "值": str(metadata.row_count)})
        summary_data.append({"项目": "列数", "值": str(len(metadata.columns))})
        summary_data.append({"项目": "列名", "值": ", ".join(metadata.columns)})
        
        if metadata.source_file:
            summary_data.append({"项目": "数据来源", "值": metadata.source_file})
        if metadata.note:
            summary_data.append({"项目": "备注", "值": metadata.note})
        
        # 数值列统计
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        if numeric_cols:
            summary_data.append({"项目": "", "值": ""})  # 空行分隔
            summary_data.append({"项目": "=== 数值列统计 ===", "值": ""})
            
            for col in numeric_cols:
                col_stats = df[col].describe()
                summary_data.append({"项目": f"{col} - 均值", "值": f"{col_stats['mean']:.4f}"})
                summary_data.append({"项目": f"{col} - 标准差", "值": f"{col_stats['std']:.4f}"})
                summary_data.append({"项目": f"{col} - 最小值", "值": f"{col_stats['min']:.4f}"})
                summary_data.append({"项目": f"{col} - 最大值", "值": f"{col_stats['max']:.4f}"})
        
        return pd.DataFrame(summary_data)

    def export_to_excel(
        self,
        dataset_ids: List[str],
        output_path: Optional[Path] = None,
        include_summary: bool = True
    ) -> Path:
        """
        导出为 Excel 文件
        
        支持单个和批量数据集导出，支持多 Sheet（数据 + 统计摘要）。
        
        Args:
            dataset_ids: 要导出的数据集 ID 列表
            output_path: 输出文件路径（可选），为 None 时自动生成
            include_summary: 是否包含统计摘要 Sheet，默认 True
        
        Returns:
            Path: 导出文件的路径
            
        Raises:
            DatasetNotFoundError: 数据集不存在时抛出
            ExportError: 导出操作失败时抛出
        """
        if not dataset_ids:
            raise ExportError("未指定要导出的数据集")
        
        # 加载所有数据集
        datasets: List[Tuple[pd.DataFrame, DatasetMetadata]] = []
        for dataset_id in dataset_ids:
            df, metadata, _ = self.load(dataset_id)
            datasets.append((df, metadata))
        
        # 确定输出路径
        if output_path is None:
            # 自动生成文件名
            if len(datasets) == 1:
                category = datasets[0][1].category
            else:
                # 批量导出时检查是否所有数据集类别相同
                categories = set(meta.category for _, meta in datasets)
                category = list(categories)[0] if len(categories) == 1 else None
            
            filename = self._generate_export_filename(category, "export", "xlsx")
            output_path = self.base_path / filename
        else:
            output_path = Path(output_path)
        
        try:
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                if len(datasets) == 1:
                    # 单个数据集导出
                    df, metadata = datasets[0]
                    df.to_excel(writer, sheet_name='数据', index=False)
                    
                    if include_summary:
                        summary_df = self._generate_summary_df(df, metadata)
                        summary_df.to_excel(writer, sheet_name='统计摘要', index=False)
                else:
                    # 批量数据集导出 - 合并到一个 Sheet
                    merged_dfs = []
                    all_summaries = []
                    
                    for i, (df, metadata) in enumerate(datasets):
                        # 添加数据集标识列
                        df_copy = df.copy()
                        df_copy.insert(0, '_数据集', metadata.name)
                        df_copy.insert(1, '_数据集ID', metadata.id)
                        merged_dfs.append(df_copy)
                        
                        if include_summary:
                            summary_df = self._generate_summary_df(df, metadata)
                            summary_df.insert(0, '数据集', metadata.name)
                            all_summaries.append(summary_df)
                    
                    # 合并所有数据
                    merged_df = pd.concat(merged_dfs, ignore_index=True)
                    merged_df.to_excel(writer, sheet_name='数据', index=False)
                    
                    if include_summary and all_summaries:
                        merged_summary = pd.concat(all_summaries, ignore_index=True)
                        merged_summary.to_excel(writer, sheet_name='统计摘要', index=False)
            
            return output_path
            
        except PermissionError as e:
            raise ExportError(f"无法写入导出文件，请检查权限: {e}")
        except OSError as e:
            if "No space left" in str(e) or "disk full" in str(e).lower():
                raise ExportError("存储空间不足，请清理磁盘")
            raise ExportError(f"导出操作失败: {e}")
        except Exception as e:
            raise ExportError(f"导出 Excel 失败: {e}")

    def export_to_csv(
        self,
        dataset_id: str,
        output_path: Optional[Path] = None
    ) -> Path:
        """
        导出为 CSV 文件
        
        使用 UTF-8-BOM 编码以确保中文兼容性。
        
        Args:
            dataset_id: 要导出的数据集 ID
            output_path: 输出文件路径（可选），为 None 时自动生成
        
        Returns:
            Path: 导出文件的路径
            
        Raises:
            DatasetNotFoundError: 数据集不存在时抛出
            ExportError: 导出操作失败时抛出
        """
        # 加载数据集
        df, metadata, _ = self.load(dataset_id)
        
        # 确定输出路径
        if output_path is None:
            filename = self._generate_export_filename(metadata.category, "export", "csv")
            output_path = self.base_path / filename
        else:
            output_path = Path(output_path)
        
        try:
            # 使用 UTF-8-BOM 编码
            df.to_csv(output_path, index=False, encoding='utf-8-sig')
            return output_path
            
        except PermissionError as e:
            raise ExportError(f"无法写入导出文件，请检查权限: {e}")
        except OSError as e:
            if "No space left" in str(e) or "disk full" in str(e).lower():
                raise ExportError("存储空间不足，请清理磁盘")
            raise ExportError(f"导出操作失败: {e}")
        except Exception as e:
            raise ExportError(f"导出 CSV 失败: {e}")


# ============================================================================
# 序列化器函数 - 用于复杂对象（绘图数据源）的序列化和反序列化
# ============================================================================

def _dataframe_to_base64(df: pd.DataFrame) -> str:
    """
    将 DataFrame 转换为 Base64 编码的 Parquet 字符串
    
    Args:
        df: 要序列化的 DataFrame
        
    Returns:
        str: Base64 编码的字符串
        
    Raises:
        SerializationError: 序列化失败时抛出
    """
    try:
        buffer = io.BytesIO()
        df.to_parquet(buffer, index=False)
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode('utf-8')
    except Exception as e:
        raise SerializationError(f"DataFrame 序列化失败: {e}")


def _base64_to_dataframe(encoded: str) -> pd.DataFrame:
    """
    将 Base64 编码的 Parquet 字符串还原为 DataFrame
    
    Args:
        encoded: Base64 编码的字符串
        
    Returns:
        pd.DataFrame: 还原的 DataFrame
        
    Raises:
        SerializationError: 反序列化失败时抛出
    """
    try:
        buffer = io.BytesIO(base64.b64decode(encoded))
        return pd.read_parquet(buffer)
    except Exception as e:
        raise SerializationError(f"DataFrame 反序列化失败: {e}")


def serialize_plot_sources(
    lvi_sources: Dict[Tuple[str, str], Tuple[pd.DataFrame, Optional[pd.DataFrame]]],
    rth_sources: Dict[Tuple[str, str], pd.DataFrame]
) -> Dict[str, Any]:
    """
    序列化绘图数据源为可存储的 JSON 格式
    
    将 lvi_sources 和 rth_sources 字典序列化为 JSON 可存储格式，
    其中 DataFrame 转为 Base64 编码的 Parquet。
    
    Args:
        lvi_sources: LVI 绘图数据源字典
            键: (shell_id, test_type) 元组
            值: (full_df, selected_df) 元组，selected_df 可为 None
        rth_sources: Rth 绘图数据源字典
            键: (shell_id, test_type) 元组
            值: DataFrame
    
    Returns:
        Dict[str, Any]: 可序列化为 JSON 的字典，格式如下:
            {
                "lvi_sources": {
                    "shell_id|test_type": {
                        "full": "base64_encoded_parquet",
                        "selected": "base64_encoded_parquet_or_null"
                    }
                },
                "rth_sources": {
                    "shell_id|test_type": "base64_encoded_parquet"
                }
            }
    
    Raises:
        SerializationError: 序列化失败时抛出
    
    Example:
        >>> lvi = {("HHD550048", "Pre测试"): (df_full, df_selected)}
        >>> rth = {("HHD550048", "Pre测试"): df_rth}
        >>> data = serialize_plot_sources(lvi, rth)
    """
    try:
        result: Dict[str, Any] = {
            "lvi_sources": {},
            "rth_sources": {}
        }
        
        # 序列化 LVI 数据源
        for (shell_id, test_type), (full_df, selected_df) in lvi_sources.items():
            key = f"{shell_id}|{test_type}"
            result["lvi_sources"][key] = {
                "full": _dataframe_to_base64(full_df) if full_df is not None and not full_df.empty else None,
                "selected": _dataframe_to_base64(selected_df) if selected_df is not None and not selected_df.empty else None
            }
        
        # 序列化 Rth 数据源
        for (shell_id, test_type), df in rth_sources.items():
            key = f"{shell_id}|{test_type}"
            result["rth_sources"][key] = _dataframe_to_base64(df) if df is not None and not df.empty else None
        
        return result
        
    except SerializationError:
        raise
    except Exception as e:
        raise SerializationError(f"绘图数据源序列化失败: {e}")


def deserialize_plot_sources(
    data: Dict[str, Any]
) -> Tuple[Dict[Tuple[str, str], Tuple[pd.DataFrame, Optional[pd.DataFrame]]], 
           Dict[Tuple[str, str], pd.DataFrame]]:
    """
    反序列化绘图数据源，从 JSON 格式还原为原始字典结构
    
    Args:
        data: serialize_plot_sources 返回的字典格式数据
    
    Returns:
        Tuple[Dict, Dict]: (lvi_sources, rth_sources) 元组
            - lvi_sources: 键为 (shell_id, test_type)，值为 (full_df, selected_df)
            - rth_sources: 键为 (shell_id, test_type)，值为 DataFrame
    
    Raises:
        SerializationError: 反序列化失败时抛出
    
    Example:
        >>> lvi_sources, rth_sources = deserialize_plot_sources(data)
    """
    try:
        lvi_sources: Dict[Tuple[str, str], Tuple[pd.DataFrame, Optional[pd.DataFrame]]] = {}
        rth_sources: Dict[Tuple[str, str], pd.DataFrame] = {}
        
        # 反序列化 LVI 数据源
        lvi_data = data.get("lvi_sources", {})
        for key, value in lvi_data.items():
            # 解析键: "shell_id|test_type" -> (shell_id, test_type)
            parts = key.split("|", 1)
            if len(parts) != 2:
                continue  # 跳过格式不正确的键
            shell_id, test_type = parts
            
            full_df = None
            selected_df = None
            
            if value.get("full"):
                full_df = _base64_to_dataframe(value["full"])
            if value.get("selected"):
                selected_df = _base64_to_dataframe(value["selected"])
            
            # 只有当至少有一个 DataFrame 时才添加
            if full_df is not None:
                lvi_sources[(shell_id, test_type)] = (full_df, selected_df)
        
        # 反序列化 Rth 数据源
        rth_data = data.get("rth_sources", {})
        for key, encoded in rth_data.items():
            parts = key.split("|", 1)
            if len(parts) != 2:
                continue
            shell_id, test_type = parts
            
            if encoded:
                rth_sources[(shell_id, test_type)] = _base64_to_dataframe(encoded)
        
        return lvi_sources, rth_sources
        
    except SerializationError:
        raise
    except Exception as e:
        raise SerializationError(f"绘图数据源反序列化失败: {e}")


# ============================================================================
# 跨模块数据共享 - 数据格式转换
# ============================================================================

# 各模块的关键列定义
MODULE_COLUMN_SCHEMAS: Dict[DataCategory, Dict[str, List[str]]] = {
    DataCategory.EXTRACTION: {
        "required": ["壳体号"],
        "optional": ["站别", "电流(A)", "功率(W)", "电压(V)", "效率(%)", "波长(nm)", "波长Shift(nm)", "冷态波长(nm)"],
        "id_column": "壳体号",
    },
    DataCategory.PROGRESS: {
        "required": ["壳体号"],
        "optional": ["料号", "生产订单", "当前站点", "上一站", "完成站别", "站别时间", "是否工程分析"],
        "id_column": "壳体号",
    },
    DataCategory.ANALYSIS: {
        "required": [],
        "optional": ["SN", "不良站点", "不良现象", "原因分类", "分析时间", "生产线", "工单类型"],
        "id_column": "SN",
    },
}

# 列名映射（源列名 -> 目标列名）
COLUMN_MAPPINGS: Dict[str, Dict[str, str]] = {
    # 从 EXTRACTION 到其他模块的映射
    "extraction_to_progress": {
        "壳体号": "壳体号",
    },
    "extraction_to_analysis": {
        "壳体号": "SN",
    },
    # 从 PROGRESS 到其他模块的映射
    "progress_to_extraction": {
        "壳体号": "壳体号",
    },
    "progress_to_analysis": {
        "壳体号": "SN",
    },
    # 从 ANALYSIS 到其他模块的映射
    "analysis_to_extraction": {
        "SN": "壳体号",
    },
    "analysis_to_progress": {
        "SN": "壳体号",
    },
}


@dataclass
class ColumnCompatibilityResult:
    """列兼容性检查结果"""
    is_compatible: bool                     # 是否兼容
    matched_columns: List[str]              # 匹配的列
    missing_required: List[str]             # 缺少的必需列
    extra_columns: List[str]                # 额外的列（目标模块不需要）
    suggested_mappings: Dict[str, str]      # 建议的列映射
    warnings: List[str]                     # 警告信息
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "is_compatible": self.is_compatible,
            "matched_columns": self.matched_columns,
            "missing_required": self.missing_required,
            "extra_columns": self.extra_columns,
            "suggested_mappings": self.suggested_mappings,
            "warnings": self.warnings,
        }


def check_column_compatibility(
    df: pd.DataFrame,
    source_category: DataCategory,
    target_category: DataCategory
) -> ColumnCompatibilityResult:
    """
    检测源数据和目标模块的列兼容性
    
    Args:
        df: 源数据 DataFrame
        source_category: 源数据类别
        target_category: 目标模块类别
    
    Returns:
        ColumnCompatibilityResult: 兼容性检查结果
    
    Example:
        >>> result = check_column_compatibility(df, DataCategory.EXTRACTION, DataCategory.PROGRESS)
        >>> if result.is_compatible:
        ...     print("数据兼容")
    """
    source_columns = set(df.columns)
    target_schema = MODULE_COLUMN_SCHEMAS.get(target_category, {"required": [], "optional": []})
    
    target_required = set(target_schema.get("required", []))
    target_optional = set(target_schema.get("optional", []))
    target_all = target_required | target_optional
    
    # 获取列映射
    mapping_key = f"{source_category.value}_to_{target_category.value}"
    column_mapping = COLUMN_MAPPINGS.get(mapping_key, {})
    
    # 应用映射后的源列名
    mapped_source_columns = set()
    suggested_mappings: Dict[str, str] = {}
    
    for src_col in source_columns:
        if src_col in column_mapping:
            mapped_col = column_mapping[src_col]
            mapped_source_columns.add(mapped_col)
            suggested_mappings[src_col] = mapped_col
        else:
            mapped_source_columns.add(src_col)
    
    # 计算匹配和缺失
    matched_columns = list(mapped_source_columns & target_all)
    missing_required = list(target_required - mapped_source_columns)
    extra_columns = list(source_columns - target_all - set(column_mapping.keys()))
    
    # 生成警告
    warnings: List[str] = []
    
    if missing_required:
        warnings.append(f"目标模块需要以下列，但源数据中不存在: {', '.join(missing_required)}")
    
    if extra_columns:
        warnings.append(f"以下列在目标模块中不使用，将被保留: {', '.join(extra_columns[:5])}{'...' if len(extra_columns) > 5 else ''}")
    
    # 检查 ID 列是否存在
    source_id_col = MODULE_COLUMN_SCHEMAS.get(source_category, {}).get("id_column")
    target_id_col = MODULE_COLUMN_SCHEMAS.get(target_category, {}).get("id_column")
    
    if source_id_col and target_id_col:
        # 检查是否可以通过映射获得目标 ID 列
        has_id = (
            target_id_col in source_columns or
            source_id_col in source_columns and source_id_col in column_mapping
        )
        if not has_id:
            warnings.append(f"源数据缺少可映射到目标 ID 列 '{target_id_col}' 的列")
    
    # 判断是否兼容（没有缺少必需列即为兼容）
    is_compatible = len(missing_required) == 0
    
    return ColumnCompatibilityResult(
        is_compatible=is_compatible,
        matched_columns=matched_columns,
        missing_required=missing_required,
        extra_columns=extra_columns,
        suggested_mappings=suggested_mappings,
        warnings=warnings,
    )


def convert_dataframe_for_module(
    df: pd.DataFrame,
    source_category: DataCategory,
    target_category: DataCategory,
    apply_mappings: bool = True,
    preserve_extra_columns: bool = True
) -> Tuple[pd.DataFrame, ColumnCompatibilityResult]:
    """
    将 DataFrame 转换为目标模块的格式
    
    进行必要的列映射和转换，保留原始数据的关键列和元数据。
    
    Args:
        df: 源数据 DataFrame
        source_category: 源数据类别
        target_category: 目标模块类别
        apply_mappings: 是否应用列名映射（默认 True）
        preserve_extra_columns: 是否保留额外的列（默认 True）
    
    Returns:
        Tuple[pd.DataFrame, ColumnCompatibilityResult]: 
            - 转换后的 DataFrame
            - 兼容性检查结果
    
    Raises:
        ValueError: 数据不兼容且无法转换时抛出
    
    Example:
        >>> converted_df, result = convert_dataframe_for_module(
        ...     df, DataCategory.EXTRACTION, DataCategory.PROGRESS
        ... )
    """
    if df is None or df.empty:
        raise ValueError("源数据为空")
    
    # 检查兼容性
    compatibility = check_column_compatibility(df, source_category, target_category)
    
    # 复制数据以避免修改原始数据
    result_df = df.copy()
    
    if apply_mappings and compatibility.suggested_mappings:
        # 应用列名映射
        rename_map = {}
        for src_col, target_col in compatibility.suggested_mappings.items():
            if src_col in result_df.columns and src_col != target_col:
                rename_map[src_col] = target_col
        
        if rename_map:
            result_df = result_df.rename(columns=rename_map)
    
    # 如果不保留额外列，则删除它们
    if not preserve_extra_columns and compatibility.extra_columns:
        cols_to_drop = [c for c in compatibility.extra_columns if c in result_df.columns]
        result_df = result_df.drop(columns=cols_to_drop, errors='ignore')
    
    return result_df, compatibility


def get_sendable_modules(source_category: DataCategory) -> List[DataCategory]:
    """
    获取可以发送数据的目标模块列表
    
    Args:
        source_category: 源数据类别
    
    Returns:
        List[DataCategory]: 可发送的目标模块列表
    """
    # 所有模块都可以互相发送数据
    return [cat for cat in DataCategory if cat != source_category]


def get_module_display_name(category: DataCategory) -> str:
    """
    获取模块的显示名称
    
    Args:
        category: 数据类别
    
    Returns:
        str: 模块显示名称
    """
    names = {
        DataCategory.EXTRACTION: "数据提取",
        DataCategory.PROGRESS: "进度追踪",
        DataCategory.ANALYSIS: "工程分析",
    }
    return names.get(category, category.value)
