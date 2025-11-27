# Design Document: Local Data Storage

## Overview

本设计为 Streamlit 数据分析应用提供统一的本地数据存储解决方案。核心目标是让用户能够保存、加载、管理和导出来自三个模块（数据提取、进度追踪、工程分析）的数据，实现数据持久化和跨模块共享。

### 设计原则

1. **统一接口** - 所有模块使用相同的存储 API
2. **高性能** - 使用 Parquet 格式，支持大数据量
3. **可扩展** - 易于添加新的数据类别
4. **用户友好** - 最小化用户操作步骤

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Streamlit Pages                             │
├─────────────┬─────────────────┬─────────────────────────────────┤
│ Data_fetch  │    Progress     │      Engineering_Analysis       │
│   .py       │      .py        │            .py                  │
└──────┬──────┴────────┬────────┴──────────────┬──────────────────┘
       │               │                        │
       ▼               ▼                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                    UI Components Layer                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │ SaveButton  │  │ LoadSelect  │  │ DataManagerPage         │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LocalDataStore (Core)                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │   save()    │  │   load()    │  │   list() / delete()     │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
│  ┌─────────────┐  ┌─────────────┐                               │
│  │  export()   │  │ serialize() │                               │
│  └─────────────┘  └─────────────┘                               │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    File System                                   │
│  app/data/saved/                                                 │
│  ├── extraction/    (数据提取结果)                               │
│  ├── progress/      (进度追踪快照)                               │
│  └── analysis/      (工程分析数据)                               │
└─────────────────────────────────────────────────────────────────┘
```

## Components and Interfaces

### 1. LocalDataStore 类

核心存储管理器，提供所有数据操作的统一接口。

```python
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

class DataCategory(Enum):
    EXTRACTION = "extraction"   # 数据提取
    PROGRESS = "progress"       # 进度追踪
    ANALYSIS = "analysis"       # 工程分析

@dataclass
class DatasetMetadata:
    """数据集元数据"""
    id: str                     # 唯一标识 (UUID)
    name: str                   # 用户可读名称
    category: DataCategory      # 数据类别
    created_at: str             # 创建时间 ISO 格式
    row_count: int              # 行数
    columns: List[str]          # 列名列表
    source_file: Optional[str]  # 原始数据来源
    note: Optional[str]         # 用户备注
    extra: Dict[str, Any]       # 扩展元数据（如筛选条件）

class LocalDataStore:
    """本地数据存储管理器"""
    
    def __init__(self, base_path: Path = None):
        """初始化存储管理器"""
        
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
            str: 数据集 ID
        """
        
    def load(self, dataset_id: str) -> Tuple[pd.DataFrame, DatasetMetadata, Optional[Dict]]:
        """
        加载数据集
        
        Returns:
            Tuple[DataFrame, Metadata, ExtraData]
        """
        
    def list_datasets(
        self,
        category: Optional[DataCategory] = None
    ) -> List[DatasetMetadata]:
        """列出数据集"""
        
    def delete(self, dataset_id: str) -> bool:
        """删除数据集"""
        
    def export_to_excel(
        self,
        dataset_ids: List[str],
        output_path: Path,
        include_summary: bool = True
    ) -> Path:
        """导出为 Excel"""
        
    def export_to_csv(
        self,
        dataset_id: str,
        output_path: Path
    ) -> Path:
        """导出为 CSV"""
```

### 2. UI 组件接口

```python
# ui_components/storage_widgets.py

def render_save_button(
    df: pd.DataFrame,
    category: DataCategory,
    extra_data: Optional[Dict] = None,
    key: str = "save_btn"
) -> Optional[str]:
    """
    渲染保存按钮和对话框
    
    Returns:
        保存成功返回 dataset_id，否则返回 None
    """

def render_load_selector(
    category: Optional[DataCategory] = None,
    key: str = "load_select"
) -> Optional[str]:
    """
    渲染数据集加载选择器
    
    Returns:
        选中的 dataset_id 或 None
    """

def render_data_manager_page():
    """渲染数据管理页面"""
```

### 3. 序列化器接口

```python
# utils/serializers.py

def serialize_plot_sources(
    lvi_sources: Dict[Tuple[str, str], Tuple[pd.DataFrame, Optional[pd.DataFrame]]],
    rth_sources: Dict[Tuple[str, str], pd.DataFrame]
) -> Dict[str, Any]:
    """序列化绘图数据源为可存储格式"""

def deserialize_plot_sources(
    data: Dict[str, Any]
) -> Tuple[Dict, Dict]:
    """反序列化绘图数据源"""
```

## Data Models

### 文件存储结构

```
app/data/saved/
├── extraction/
│   ├── {uuid}.parquet          # DataFrame 数据
│   ├── {uuid}.meta.json        # 元数据
│   └── {uuid}.extra.json       # 扩展数据（绘图源等）
├── progress/
│   ├── {uuid}.parquet
│   ├── {uuid}.meta.json
│   └── {uuid}.extra.json
└── analysis/
    ├── {uuid}.parquet
    └── {uuid}.meta.json
```

### 元数据 JSON 结构

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "name": "HHD550048_20241126_extraction",
  "category": "extraction",
  "created_at": "2024-11-26T10:30:00+08:00",
  "row_count": 156,
  "columns": ["壳体号", "站别", "电流(A)", "功率(W)", "效率(%)"],
  "source_file": "HHD550048",
  "note": "Pre测试和Post测试数据",
  "extra": {
    "shell_ids": ["HHD550048"],
    "test_types": ["Pre测试", "Post测试"],
    "current_points": [12.0, 15.0]
  }
}
```

### 扩展数据 JSON 结构（数据提取专用）

```json
{
  "lvi_sources": {
    "HHD550048|Pre测试": {
      "full": "base64_encoded_parquet",
      "selected": "base64_encoded_parquet_or_null"
    }
  },
  "rth_sources": {
    "HHD550048|Pre测试": "base64_encoded_parquet"
  }
}
```

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system-essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Save-Load Round Trip

*For any* valid DataFrame and metadata, saving then loading should produce an equivalent DataFrame with identical data and preserved metadata.

**Validates: Requirements 1.1, 1.2, 1.3, 2.1**

### Property 2: Filename Generation and Custom Filename

*For any* save operation without custom_filename, the generated filename should contain a valid timestamp (ISO format date) and a sanitized version of the data summary. *For any* save operation with custom_filename, the actual filename should use the sanitized custom name.

**Validates: Requirements 1.4**

### Property 3: Save Returns Valid Identifier

*For any* successful save operation, the returned identifier should be a valid UUID that can be used to load the dataset.

**Validates: Requirements 1.5**

### Property 4: Load Non-existent File Returns Error

*For any* non-existent or invalid dataset ID, the load operation should raise a specific exception with a descriptive error message.

**Validates: Requirements 2.5**

### Property 5: List Datasets Contains Required Metadata Fields

*For any* dataset in the list, the metadata should contain all required fields: id, name, category, created_at, row_count.

**Validates: Requirements 3.2**

### Property 6: Delete Removes All Associated Files

*For any* delete operation on a valid dataset ID, all associated files (.parquet, .meta.json, .extra.json) should be removed.

**Validates: Requirements 3.3**

### Property 7: Category Filter Returns Only Matching Datasets

*For any* list operation with a category filter, all returned datasets should have the specified category.

**Validates: Requirements 3.4**

### Property 8: Complex Object Serialization Round Trip

*For any* complex object (like plot sources dictionary), serializing then deserializing should produce an equivalent object.

**Validates: Requirements 4.3**

### Property 9: Export Excel Contains Multiple Sheets

*For any* Excel export with include_summary=True, the output file should contain at least two sheets: data and summary.

**Validates: Requirements 6.2**

### Property 10: Export Filename Contains Date and Type

*For any* export operation, the generated filename should contain the current date and the data category.

**Validates: Requirements 6.4**

### Property 11: Batch Export Merges All Selected Datasets

*For any* batch export of N datasets, the output should contain data from all N datasets (verifiable by row count or unique identifiers).

**Validates: Requirements 6.5**

## Error Handling

| 错误场景 | 处理方式 | 用户提示 |
|---------|---------|---------|
| 存储目录不存在 | 自动创建目录 | 无（静默处理） |
| 磁盘空间不足 | 抛出 `StorageError` | "存储空间不足，请清理磁盘" |
| 文件损坏无法读取 | 抛出 `CorruptedDataError` | "数据文件损坏，无法加载" |
| 数据集 ID 不存在 | 抛出 `DatasetNotFoundError` | "未找到指定的数据集" |
| 序列化失败 | 抛出 `SerializationError` | "数据格式不支持保存" |
| 导出路径无写权限 | 抛出 `ExportError` | "无法写入导出文件，请检查权限" |

```python
# utils/exceptions.py

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
```

## Testing Strategy

### 单元测试

使用 `pytest` 框架，测试核心功能：

1. **LocalDataStore 类测试**
   - `test_save_creates_files` - 验证保存创建正确的文件
   - `test_load_returns_correct_data` - 验证加载返回正确数据
   - `test_list_returns_all_datasets` - 验证列表功能
   - `test_delete_removes_files` - 验证删除功能
   - `test_export_excel_format` - 验证 Excel 导出格式
   - `test_export_csv_encoding` - 验证 CSV 编码

2. **序列化器测试**
   - `test_serialize_plot_sources` - 验证绘图数据序列化
   - `test_deserialize_plot_sources` - 验证反序列化

### 属性测试

使用 `hypothesis` 库进行属性测试：

```python
# tests/test_local_storage_properties.py

from hypothesis import given, strategies as st
import pandas as pd

@given(st.data())
def test_save_load_round_trip(data):
    """
    **Feature: local-data-storage, Property 1: Save-Load Round Trip**
    **Validates: Requirements 1.1, 1.2, 1.3, 2.1**
    """
    # 生成随机 DataFrame
    df = generate_random_dataframe(data)
    store = LocalDataStore()
    
    # 保存
    dataset_id = store.save(df, DataCategory.EXTRACTION)
    
    # 加载
    loaded_df, metadata, _ = store.load(dataset_id)
    
    # 验证数据一致
    pd.testing.assert_frame_equal(df, loaded_df)
```

### 测试框架配置

- 测试框架: `pytest`
- 属性测试库: `hypothesis`
- 每个属性测试运行 100 次迭代
- 测试文件位置: `tests/test_local_storage.py`, `tests/test_local_storage_properties.py`
