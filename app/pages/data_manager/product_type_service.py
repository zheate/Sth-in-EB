"""
ProductTypeService for Data Manager (Zh's DataBase).

This module provides CRUD operations for ProductType entities,
including attachment management and production order retrieval.

Requirements: 1.2, 1.3, 1.5, 2.2, 2.5, 3.3, 7.1, 7.2, 7.5
"""

import json
import logging
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

from .constants import (
    ATTACHMENTS_DIR,
    DATABASE_DIR,
    PRODUCT_TYPES_FILE,
    SHELLS_DIR,
    THRESHOLD_CONFIG_DIR,
    ensure_database_dirs,
)
from .models import (
    Attachment,
    ProductType,
    ProductTypeSummary,
    ProductionOrder,
)

logger = logging.getLogger(__name__)


class ProductTypeService:
    """
    产品类型服务类。
    
    负责产品类型的 CRUD 操作和附件管理。
    """

    def __init__(self, database_dir: Optional[Path] = None):
        """
        初始化 ProductTypeService。
        
        Args:
            database_dir: 数据库目录路径，默认使用 constants 中定义的路径
        """
        self.database_dir = database_dir or DATABASE_DIR
        self.metadata_file = self.database_dir / "product_types.json"
        self.shells_dir = self.database_dir / "shells"
        self.attachments_dir = self.database_dir / "attachments"
        self.threshold_config_dir = self.database_dir / "thresholds"
        
        # Ensure directories exist
        self._ensure_dirs()


    def _ensure_dirs(self) -> None:
        """Ensure all required directories exist."""
        self.database_dir.mkdir(parents=True, exist_ok=True)
        self.shells_dir.mkdir(parents=True, exist_ok=True)
        self.attachments_dir.mkdir(parents=True, exist_ok=True)
        self.threshold_config_dir.mkdir(parents=True, exist_ok=True)

    def _load_metadata(self) -> Dict[str, Any]:
        """
        Load product types metadata from JSON file.
        
        Returns:
            Dictionary containing product types metadata
        """
        if not self.metadata_file.exists():
            return {"product_types": {}}
        
        try:
            with open(self.metadata_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Failed to load metadata: {e}")
            return {"product_types": {}}

    def _save_metadata(self, metadata: Dict[str, Any]) -> bool:
        """
        Save product types metadata to JSON file.
        
        Args:
            metadata: Dictionary containing product types metadata
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(self.metadata_file, "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}", exc_info=True)
            return False

    # ========================================================================
    # CRUD Operations for ProductType
    # ========================================================================

    def save_product_type(
        self,
        name: str,
        shells_df: pd.DataFrame,
        production_orders: List[str],
        source_file: Optional[str] = None,
    ) -> str:
        """
        保存产品类型。
        
        Args:
            name: 产品类型名称 (如 M20-AM-C)
            shells_df: 壳体数据 DataFrame
            production_orders: 生产订单列表
            source_file: 来源文件名
            
        Returns:
            产品类型 ID
            
        Raises:
            ValueError: 如果名称为空或数据无效
        """
        if not name or not name.strip():
            raise ValueError("产品类型名称不能为空")
        
        # Generate new ID
        product_type_id = str(uuid.uuid4())
        now = datetime.now()
        
        # Create ProductType object
        product_type = ProductType(
            id=product_type_id,
            name=name.strip(),
            created_at=now,
            updated_at=now,
            source_file=source_file,
            shell_count=len(shells_df) if shells_df is not None else 0,
            order_count=len(production_orders) if production_orders else 0,
            attachments=[],
            threshold_config=None,
        )
        
        # Validate
        errors = product_type.validate()
        if errors:
            raise ValueError(f"数据验证失败: {', '.join(errors)}")
        
        # Save shell data to Parquet
        if shells_df is not None and not shells_df.empty:
            parquet_path = self.shells_dir / f"{product_type_id}.parquet"
            shells_df.to_parquet(parquet_path, index=False)
        
        # Update metadata
        metadata = self._load_metadata()
        metadata["product_types"][product_type_id] = product_type.to_dict()
        
        if not self._save_metadata(metadata):
            raise IOError("保存元数据失败")
        
        logger.info(f"Saved product type: {name} (ID: {product_type_id})")
        return product_type_id

    def upsert_product_type(
        self,
        name: str,
        shells_df: pd.DataFrame,
        production_orders: List[str],
        source_file: Optional[str] = None,
    ) -> str:
        """
        根据名称更新或创建产品类型（保持名称唯一，若存在则覆盖数据并更新时间）。
        """
        if not name or not name.strip():
            raise ValueError("产品类型名称不能为空")

        metadata = self._load_metadata()
        existing_id = None
        existing_data = None
        for pid, pdata in metadata.get("product_types", {}).items():
            if pdata.get("name") == name.strip():
                existing_id = pid
                existing_data = pdata
                break

        now = datetime.now()
        product_type_id = existing_id or str(uuid.uuid4())
        created_at = existing_data.get("created_at") if existing_data else None
        created_dt = (
            datetime.fromisoformat(created_at)
            if isinstance(created_at, str)
            else created_at
            if created_at
            else now
        )

        product_type = ProductType(
            id=product_type_id,
            name=name.strip(),
            created_at=created_dt,
            updated_at=now,
            source_file=source_file,
            shell_count=len(shells_df) if shells_df is not None else 0,
            order_count=len(production_orders) if production_orders else 0,
            attachments=existing_data.get("attachments", []) if existing_data else [],
            threshold_config=existing_data.get("threshold_config") if existing_data else None,
        )

        errors = product_type.validate()
        if errors:
            raise ValueError(f"数据验证失败: {', '.join(errors)}")

        if shells_df is not None and not shells_df.empty:
            parquet_path = self.shells_dir / f"{product_type_id}.parquet"
            shells_df.to_parquet(parquet_path, index=False)

        metadata["product_types"][product_type_id] = product_type.to_dict()
        if not self._save_metadata(metadata):
            raise IOError("保存元数据失败")

        logger.info(f"Upserted product type: {name} (ID: {product_type_id})")
        return product_type_id

    def get_product_type(self, product_type_id: str) -> Optional[ProductType]:
        """
        获取产品类型详情。
        
        Args:
            product_type_id: 产品类型 ID
            
        Returns:
            ProductType 对象，如果不存在返回 None
        """
        metadata = self._load_metadata()
        pt_data = metadata.get("product_types", {}).get(product_type_id)
        
        if pt_data is None:
            return None
        
        return ProductType.from_dict(pt_data)

    def list_product_types(self) -> List[ProductTypeSummary]:
        """
        列出所有产品类型。
        
        Returns:
            ProductTypeSummary 列表
        """
        metadata = self._load_metadata()
        summaries = []
        
        for pt_id, pt_data in metadata.get("product_types", {}).items():
            try:
                pt = ProductType.from_dict(pt_data)
                summary = ProductTypeSummary.from_product_type(pt)
                summaries.append(summary)
            except Exception as e:
                logger.warning(f"Failed to load product type {pt_id}: {e}")
                continue
        
        # Sort by created_at descending (newest first)
        summaries.sort(key=lambda x: x.created_at, reverse=True)
        return summaries


    def get_shells_dataframe(self, product_type_id: str) -> Optional[pd.DataFrame]:
        """
        获取产品类型的壳体数据 DataFrame。
        
        Args:
            product_type_id: 产品类型 ID
            
        Returns:
            壳体数据 DataFrame，如果不存在返回 None
        """
        parquet_path = self.shells_dir / f"{product_type_id}.parquet"
        
        if not parquet_path.exists():
            return None
        
        try:
            return pd.read_parquet(parquet_path)
        except Exception as e:
            logger.error(f"Failed to load shells data for {product_type_id}: {e}")
            return None

    # ========================================================================
    # Rename and Delete Operations (Task 2.3)
    # ========================================================================

    def rename_product_type(self, product_type_id: str, new_name: str) -> bool:
        """
        重命名产品类型。
        
        保持关联数据（壳体数据、附件）不变。
        
        Args:
            product_type_id: 产品类型 ID
            new_name: 新名称
            
        Returns:
            True if successful, False otherwise
            
        Raises:
            ValueError: 如果新名称为空
        """
        if not new_name or not new_name.strip():
            raise ValueError("新名称不能为空")
        
        metadata = self._load_metadata()
        pt_data = metadata.get("product_types", {}).get(product_type_id)
        
        if pt_data is None:
            logger.warning(f"Product type not found: {product_type_id}")
            return False
        
        # Update name and updated_at
        pt_data["name"] = new_name.strip()
        pt_data["updated_at"] = datetime.now().isoformat()
        
        metadata["product_types"][product_type_id] = pt_data
        
        if not self._save_metadata(metadata):
            return False
        
        logger.info(f"Renamed product type {product_type_id} to: {new_name}")
        return True

    def set_product_type_completed(self, product_type_id: str, is_completed: bool) -> bool:
        """
        设置产品类型的完成状态。
        
        Args:
            product_type_id: 产品类型 ID
            is_completed: 是否已完成
            
        Returns:
            True if successful, False otherwise
        """
        metadata = self._load_metadata()
        pt_data = metadata.get("product_types", {}).get(product_type_id)
        
        if pt_data is None:
            logger.warning(f"Product type not found: {product_type_id}")
            return False
        
        pt_data["is_completed"] = is_completed
        pt_data["updated_at"] = datetime.now().isoformat()
        
        metadata["product_types"][product_type_id] = pt_data
        
        try:
            if not self._save_metadata(metadata):
                logger.error(f"Failed to save metadata for product type {product_type_id}")
                return False
        except Exception as e:
            logger.error(f"Exception saving metadata: {e}", exc_info=True)
            return False
        
        logger.info(f"Set product type {product_type_id} completed: {is_completed}")
        return True

    def delete_product_type(self, product_type_id: str) -> bool:
        """
        删除产品类型（级联删除关联数据）。
        
        删除内容包括：
        - 产品类型元数据
        - 壳体数据文件 (Parquet)
        - 所有关联的附件文件
        - 阈值配置文件
        
        Args:
            product_type_id: 产品类型 ID
            
        Returns:
            True if successful, False otherwise
        """
        metadata = self._load_metadata()
        pt_data = metadata.get("product_types", {}).get(product_type_id)
        
        if pt_data is None:
            logger.warning(f"Product type not found: {product_type_id}")
            return False
        
        # Delete shell data file
        parquet_path = self.shells_dir / f"{product_type_id}.parquet"
        if parquet_path.exists():
            try:
                parquet_path.unlink()
                logger.info(f"Deleted shells file: {parquet_path}")
            except IOError as e:
                logger.error(f"Failed to delete shells file: {e}")
        
        # Delete all attachments
        attachments = pt_data.get("attachments", [])
        for attachment_id in attachments:
            self._delete_attachment_file(product_type_id, attachment_id)
        
        # Delete threshold config file
        threshold_path = self.threshold_config_dir / f"{product_type_id}.json"
        if threshold_path.exists():
            try:
                threshold_path.unlink()
                logger.info(f"Deleted threshold config: {threshold_path}")
            except IOError as e:
                logger.error(f"Failed to delete threshold config: {e}")
        
        # Remove from metadata
        del metadata["product_types"][product_type_id]
        
        if not self._save_metadata(metadata):
            return False
        
        logger.info(f"Deleted product type: {product_type_id}")
        return True

    # ========================================================================
    # Production Order Retrieval (Task 2.5)
    # ========================================================================

    def get_production_orders(self, product_type_id: str) -> List[ProductionOrder]:
        """
        获取产品类型下的生产订单列表。
        
        从壳体数据中提取生产订单信息，包括时间信息。
        
        Args:
            product_type_id: 产品类型 ID
            
        Returns:
            ProductionOrder 列表
        """
        shells_df = self.get_shells_dataframe(product_type_id)
        
        if shells_df is None or shells_df.empty:
            return []
        
        # Find production order column
        order_col = self._find_column(shells_df, [
            "生产订单", "ERP生产订单", "SAP生产订单", 
            "生产订单号", "订单号", "工单号"
        ])
        
        if order_col is None:
            logger.warning(f"No production order column found for {product_type_id}")
            return []
        
        # Find time column
        time_col = self._find_column(shells_df, [
            "时间", "日期", "更新时间", "创建时间", "Time", "Date"
        ])
        
        orders = []
        for order_id in shells_df[order_col].unique():
            if pd.isna(order_id) or str(order_id).strip() == "":
                continue
            
            order_df = shells_df[shells_df[order_col] == order_id]
            shell_count = len(order_df)
            
            # Extract time information
            latest_time = None
            earliest_time = None
            
            if time_col and time_col in order_df.columns:
                try:
                    times = pd.to_datetime(order_df[time_col], errors="coerce")
                    valid_times = times.dropna()
                    if not valid_times.empty:
                        latest_time = valid_times.max().to_pydatetime()
                        earliest_time = valid_times.min().to_pydatetime()
                except Exception as e:
                    logger.warning(f"Failed to parse time for order {order_id}: {e}")
            
            orders.append(ProductionOrder(
                id=str(order_id),
                product_type_id=product_type_id,
                shell_count=shell_count,
                latest_time=latest_time,
                earliest_time=earliest_time,
            ))
        
        # Sort by latest_time descending (newest first)
        orders.sort(key=lambda x: x.latest_time or datetime.min, reverse=True)
        return orders

    def _find_column(
        self, df: pd.DataFrame, candidates: List[str]
    ) -> Optional[str]:
        """
        Find a column in DataFrame by candidate names.
        
        Args:
            df: DataFrame to search
            candidates: List of candidate column names
            
        Returns:
            Found column name or None
        """
        df_cols_lower = {col.lower(): col for col in df.columns}
        
        for candidate in candidates:
            # Exact match
            if candidate in df.columns:
                return candidate
            # Case-insensitive match
            if candidate.lower() in df_cols_lower:
                return df_cols_lower[candidate.lower()]
        
        return None


    # ========================================================================
    # Attachment Management (Task 2.7)
    # ========================================================================

    def upload_attachment(
        self,
        product_type_id: str,
        file_content: bytes,
        original_name: str,
        file_type: Optional[str] = None,
        allow_overwrite: bool = False,
    ) -> str:
        """
        上传附件。
        
        Args:
            product_type_id: 产品类型 ID
            file_content: 文件内容（字节）
            original_name: 原始文件名
            file_type: 文件类型 (pdf/excel)，如果为 None 则从文件名推断
            
        Returns:
            附件 ID
            
        Raises:
            ValueError: 如果产品类型不存在或文件类型不支持
        """
        # Verify product type exists
        metadata = self._load_metadata()
        pt_data = metadata.get("product_types", {}).get(product_type_id)
        
        if pt_data is None:
            raise ValueError(f"产品类型不存在: {product_type_id}")

        # Prevent duplicate attachment (same original name)
        existing_ids = pt_data.get("attachments", [])
        attachments_data = metadata.get("attachments_data", {})
        for att_id in existing_ids:
            att_data = attachments_data.get(att_id)
            if att_data and att_data.get("original_name", "").lower() == original_name.lower():
                if not allow_overwrite:
                    raise ValueError(f"同名附件已存在: {original_name}，点击覆盖上传以替换")
                # 覆盖：先删除旧附件
                self.delete_attachment(product_type_id, att_id)
        
        # Determine file type
        if file_type is None:
            file_type = Attachment.get_file_type_from_extension(original_name)
        
        if file_type not in Attachment.ALLOWED_FILE_TYPES:
            raise ValueError(f"不支持的文件类型: {file_type}")
        
        # Generate attachment ID and stored name
        attachment_id = str(uuid.uuid4())
        ext = original_name.split(".")[-1] if "." in original_name else file_type
        stored_name = f"{attachment_id}.{ext}"
        
        # Save file
        file_path = self.attachments_dir / stored_name
        try:
            with open(file_path, "wb") as f:
                f.write(file_content)
        except IOError as e:
            raise IOError(f"保存附件失败: {e}")
        
        # Create Attachment object
        attachment = Attachment(
            id=attachment_id,
            product_type_id=product_type_id,
            original_name=original_name,
            stored_name=stored_name,
            file_type=file_type,
            size=len(file_content),
            uploaded_at=datetime.now(),
        )
        
        # Update metadata
        if "attachments_data" not in metadata:
            metadata["attachments_data"] = {}
        metadata["attachments_data"][attachment_id] = attachment.to_dict()
        
        # Add attachment ID to product type
        if "attachments" not in pt_data:
            pt_data["attachments"] = []
        pt_data["attachments"].append(attachment_id)
        pt_data["updated_at"] = datetime.now().isoformat()
        
        metadata["product_types"][product_type_id] = pt_data
        
        if not self._save_metadata(metadata):
            # Rollback: delete the file
            file_path.unlink(missing_ok=True)
            raise IOError("保存元数据失败")
        
        logger.info(f"Uploaded attachment: {original_name} (ID: {attachment_id})")
        return attachment_id

    def list_attachments(self, product_type_id: str) -> List[Attachment]:
        """
        列出产品类型的附件。
        
        Args:
            product_type_id: 产品类型 ID
            
        Returns:
            Attachment 列表
        """
        metadata = self._load_metadata()
        pt_data = metadata.get("product_types", {}).get(product_type_id)
        
        if pt_data is None:
            return []
        
        attachment_ids = pt_data.get("attachments", [])
        attachments_data = metadata.get("attachments_data", {})
        
        attachments = []
        for att_id in attachment_ids:
            att_data = attachments_data.get(att_id)
            if att_data:
                try:
                    attachments.append(Attachment.from_dict(att_data))
                except Exception as e:
                    logger.warning(f"Failed to load attachment {att_id}: {e}")
        
        # Sort by uploaded_at descending (newest first)
        attachments.sort(key=lambda x: x.uploaded_at, reverse=True)
        return attachments

    def delete_attachment(self, product_type_id: str, attachment_id: str) -> bool:
        """
        删除附件。
        
        Args:
            product_type_id: 产品类型 ID
            attachment_id: 附件 ID
            
        Returns:
            True if successful, False otherwise
        """
        metadata = self._load_metadata()
        pt_data = metadata.get("product_types", {}).get(product_type_id)
        
        if pt_data is None:
            logger.warning(f"Product type not found: {product_type_id}")
            return False
        
        # Get attachment data
        attachments_data = metadata.get("attachments_data", {})
        att_data = attachments_data.get(attachment_id)
        
        if att_data is None:
            logger.warning(f"Attachment not found: {attachment_id}")
            return False
        
        # Delete file
        stored_name = att_data.get("stored_name", "")
        if stored_name:
            file_path = self.attachments_dir / stored_name
            if file_path.exists():
                try:
                    file_path.unlink()
                    logger.info(f"Deleted attachment file: {file_path}")
                except IOError as e:
                    logger.error(f"Failed to delete attachment file: {e}")
        
        # Remove from metadata
        if attachment_id in attachments_data:
            del attachments_data[attachment_id]
        
        if attachment_id in pt_data.get("attachments", []):
            pt_data["attachments"].remove(attachment_id)
            pt_data["updated_at"] = datetime.now().isoformat()
        
        metadata["attachments_data"] = attachments_data
        metadata["product_types"][product_type_id] = pt_data
        
        if not self._save_metadata(metadata):
            return False
        
        logger.info(f"Deleted attachment: {attachment_id}")
        return True

    def _delete_attachment_file(
        self, product_type_id: str, attachment_id: str
    ) -> bool:
        """
        Delete attachment file only (used during cascading delete).
        
        Args:
            product_type_id: 产品类型 ID
            attachment_id: 附件 ID
            
        Returns:
            True if successful, False otherwise
        """
        metadata = self._load_metadata()
        attachments_data = metadata.get("attachments_data", {})
        att_data = attachments_data.get(attachment_id)
        
        if att_data is None:
            return False
        
        stored_name = att_data.get("stored_name", "")
        if stored_name:
            file_path = self.attachments_dir / stored_name
            if file_path.exists():
                try:
                    file_path.unlink()
                    logger.info(f"Deleted attachment file: {file_path}")
                    return True
                except IOError as e:
                    logger.error(f"Failed to delete attachment file: {e}")
                    return False
        
        return False

    def get_attachment_path(self, attachment_id: str) -> Optional[Path]:
        """
        获取附件文件路径。
        
        Args:
            attachment_id: 附件 ID
            
        Returns:
            文件路径，如果不存在返回 None
        """
        metadata = self._load_metadata()
        attachments_data = metadata.get("attachments_data", {})
        att_data = attachments_data.get(attachment_id)
        
        if att_data is None:
            return None
        
        stored_name = att_data.get("stored_name", "")
        if stored_name:
            file_path = self.attachments_dir / stored_name
            if file_path.exists():
                return file_path
        
        return None
