"""
Data models for Data Manager (Zh's DataBase).

This module defines dataclasses for:
- ProductType: 产品类型
- ProductTypeSummary: 产品类型摘要（用于列表显示）
- ProductionOrder: 生产订单
- ShellProgress: 壳体进度
- Attachment: 附件
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import json
import uuid


def _generate_uuid() -> str:
    """Generate a new UUID string."""
    return str(uuid.uuid4())


def _now() -> datetime:
    """Get current datetime."""
    return datetime.now()


@dataclass
class ProductType:
    """
    产品类型数据模型。
    
    Attributes:
        id: UUID 唯一标识
        name: 产品类型名称，如 M20-AM-C
        created_at: 创建时间
        updated_at: 更新时间
        source_file: 来源文件名
        shell_count: 壳体数量
        order_count: 订单数量
        attachments: 附件 ID 列表
        threshold_config: 阈值配置
    """
    name: str
    id: str = field(default_factory=_generate_uuid)
    created_at: datetime = field(default_factory=_now)
    updated_at: datetime = field(default_factory=_now)
    source_file: Optional[str] = None
    shell_count: int = 0
    order_count: int = 0
    attachments: List[str] = field(default_factory=list)
    threshold_config: Optional[Dict[str, Tuple[Optional[float], Optional[float]]]] = None


    def validate(self) -> List[str]:
        """
        Validate the ProductType data.
        
        Returns:
            List of validation error messages. Empty list if valid.
        """
        errors = []
        
        if not self.name or not self.name.strip():
            errors.append("产品类型名称不能为空")
        
        if self.shell_count < 0:
            errors.append("壳体数量不能为负数")
        
        if self.order_count < 0:
            errors.append("订单数量不能为负数")
        
        if self.threshold_config:
            for col_name, (min_val, max_val) in self.threshold_config.items():
                if min_val is not None and max_val is not None and min_val > max_val:
                    errors.append(f"阈值配置错误: {col_name} 的最小值大于最大值")
        
        return errors

    def is_valid(self) -> bool:
        """Check if the ProductType is valid."""
        return len(self.validate()) == 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "source_file": self.source_file,
            "shell_count": self.shell_count,
            "order_count": self.order_count,
            "attachments": self.attachments,
            "threshold_config": self.threshold_config,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProductType":
        """Create ProductType from dictionary."""
        return cls(
            id=data.get("id", _generate_uuid()),
            name=data["name"],
            created_at=datetime.fromisoformat(data["created_at"]) if isinstance(data.get("created_at"), str) else data.get("created_at", _now()),
            updated_at=datetime.fromisoformat(data["updated_at"]) if isinstance(data.get("updated_at"), str) else data.get("updated_at", _now()),
            source_file=data.get("source_file"),
            shell_count=data.get("shell_count", 0),
            order_count=data.get("order_count", 0),
            attachments=data.get("attachments", []),
            threshold_config=data.get("threshold_config"),
        )


@dataclass
class ProductTypeSummary:
    """
    产品类型摘要，用于列表显示。
    
    Attributes:
        id: UUID 唯一标识
        name: 产品类型名称
        shell_count: 壳体数量
        order_count: 订单数量
        created_at: 创建时间
        has_attachments: 是否有附件
    """
    id: str
    name: str
    shell_count: int = 0
    order_count: int = 0
    created_at: datetime = field(default_factory=_now)
    has_attachments: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "shell_count": self.shell_count,
            "order_count": self.order_count,
            "created_at": self.created_at.isoformat(),
            "has_attachments": self.has_attachments,
        }

    @classmethod
    def from_product_type(cls, pt: ProductType) -> "ProductTypeSummary":
        """Create summary from ProductType."""
        return cls(
            id=pt.id,
            name=pt.name,
            shell_count=pt.shell_count,
            order_count=pt.order_count,
            created_at=pt.created_at,
            has_attachments=len(pt.attachments) > 0,
        )


@dataclass
class ProductionOrder:
    """
    生产订单数据模型。
    
    Attributes:
        id: 订单号，如 WO-MP-M20-HX-25090251
        product_type_id: 关联的产品类型 ID
        shell_count: 壳体数量
        latest_time: 最新时间
        earliest_time: 最早时间
    """
    id: str
    product_type_id: str
    shell_count: int = 0
    latest_time: Optional[datetime] = None
    earliest_time: Optional[datetime] = None

    def validate(self) -> List[str]:
        """Validate the ProductionOrder data."""
        errors = []
        
        if not self.id or not self.id.strip():
            errors.append("订单号不能为空")
        
        if not self.product_type_id or not self.product_type_id.strip():
            errors.append("产品类型 ID 不能为空")
        
        if self.shell_count < 0:
            errors.append("壳体数量不能为负数")
        
        if self.earliest_time and self.latest_time and self.earliest_time > self.latest_time:
            errors.append("最早时间不能晚于最新时间")
        
        return errors

    def is_valid(self) -> bool:
        """Check if the ProductionOrder is valid."""
        return len(self.validate()) == 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "product_type_id": self.product_type_id,
            "shell_count": self.shell_count,
            "latest_time": self.latest_time.isoformat() if self.latest_time else None,
            "earliest_time": self.earliest_time.isoformat() if self.earliest_time else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProductionOrder":
        """Create ProductionOrder from dictionary."""
        return cls(
            id=data["id"],
            product_type_id=data["product_type_id"],
            shell_count=data.get("shell_count", 0),
            latest_time=datetime.fromisoformat(data["latest_time"]) if data.get("latest_time") else None,
            earliest_time=datetime.fromisoformat(data["earliest_time"]) if data.get("earliest_time") else None,
        )



@dataclass
class ShellProgress:
    """
    壳体进度数据模型。
    
    Attributes:
        shell_id: 壳体号
        production_order: 生产订单
        current_station: 当前站别
        completed_stations: 已完成站别列表
        station_times: 站别时间映射
        is_engineering_analysis: 是否工程分析
        part_number: 料号
    """
    shell_id: str
    production_order: str
    current_station: str = ""
    completed_stations: List[str] = field(default_factory=list)
    station_times: Dict[str, datetime] = field(default_factory=dict)
    is_engineering_analysis: bool = False
    part_number: str = ""

    def validate(self) -> List[str]:
        """Validate the ShellProgress data."""
        errors = []
        
        if not self.shell_id or not self.shell_id.strip():
            errors.append("壳体号不能为空")
        
        if not self.production_order or not self.production_order.strip():
            errors.append("生产订单不能为空")
        
        return errors

    def is_valid(self) -> bool:
        """Check if the ShellProgress is valid."""
        return len(self.validate()) == 0

    def get_latest_station_time(self) -> Optional[datetime]:
        """Get the latest station time."""
        if not self.station_times:
            return None
        valid_times = [t for t in self.station_times.values() if t is not None]
        return max(valid_times) if valid_times else None

    def get_earliest_station_time(self) -> Optional[datetime]:
        """Get the earliest station time."""
        if not self.station_times:
            return None
        valid_times = [t for t in self.station_times.values() if t is not None]
        return min(valid_times) if valid_times else None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "shell_id": self.shell_id,
            "production_order": self.production_order,
            "current_station": self.current_station,
            "completed_stations": self.completed_stations,
            "station_times": {k: v.isoformat() if v else None for k, v in self.station_times.items()},
            "is_engineering_analysis": self.is_engineering_analysis,
            "part_number": self.part_number,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ShellProgress":
        """Create ShellProgress from dictionary."""
        station_times = {}
        for k, v in data.get("station_times", {}).items():
            if v:
                try:
                    station_times[k] = datetime.fromisoformat(v) if isinstance(v, str) else v
                except (ValueError, TypeError):
                    station_times[k] = None
            else:
                station_times[k] = None
        
        return cls(
            shell_id=data["shell_id"],
            production_order=data["production_order"],
            current_station=data.get("current_station", ""),
            completed_stations=data.get("completed_stations", []),
            station_times=station_times,
            is_engineering_analysis=data.get("is_engineering_analysis", False),
            part_number=data.get("part_number", ""),
        )


@dataclass
class Attachment:
    """
    附件数据模型。
    
    Attributes:
        id: UUID 唯一标识
        product_type_id: 关联的产品类型 ID
        original_name: 原始文件名
        stored_name: 存储文件名
        file_type: 文件类型 (pdf/excel)
        size: 文件大小（字节）
        uploaded_at: 上传时间
    """
    product_type_id: str
    original_name: str
    stored_name: str
    file_type: str
    size: int = 0
    id: str = field(default_factory=_generate_uuid)
    uploaded_at: datetime = field(default_factory=_now)

    ALLOWED_FILE_TYPES = {"pdf", "excel", "xlsx", "xls"}

    def validate(self) -> List[str]:
        """Validate the Attachment data."""
        errors = []
        
        if not self.product_type_id or not self.product_type_id.strip():
            errors.append("产品类型 ID 不能为空")
        
        if not self.original_name or not self.original_name.strip():
            errors.append("原始文件名不能为空")
        
        if not self.stored_name or not self.stored_name.strip():
            errors.append("存储文件名不能为空")
        
        if self.file_type not in self.ALLOWED_FILE_TYPES:
            errors.append(f"不支持的文件类型: {self.file_type}，允许的类型: {self.ALLOWED_FILE_TYPES}")
        
        if self.size < 0:
            errors.append("文件大小不能为负数")
        
        return errors

    def is_valid(self) -> bool:
        """Check if the Attachment is valid."""
        return len(self.validate()) == 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "product_type_id": self.product_type_id,
            "original_name": self.original_name,
            "stored_name": self.stored_name,
            "file_type": self.file_type,
            "size": self.size,
            "uploaded_at": self.uploaded_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Attachment":
        """Create Attachment from dictionary."""
        return cls(
            id=data.get("id", _generate_uuid()),
            product_type_id=data["product_type_id"],
            original_name=data["original_name"],
            stored_name=data["stored_name"],
            file_type=data["file_type"],
            size=data.get("size", 0),
            uploaded_at=datetime.fromisoformat(data["uploaded_at"]) if isinstance(data.get("uploaded_at"), str) else data.get("uploaded_at", _now()),
        )

    @classmethod
    def get_file_type_from_extension(cls, filename: str) -> str:
        """Determine file type from filename extension."""
        ext = filename.lower().split(".")[-1] if "." in filename else ""
        if ext == "pdf":
            return "pdf"
        elif ext in ("xlsx", "xls"):
            return "excel"
        else:
            return ext
