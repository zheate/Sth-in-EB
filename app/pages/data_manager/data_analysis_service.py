"""
DataAnalysisService for Data Manager (Zh's DataBase).

This module provides data analysis functionality including:
- Fetching test data by integrating with Data_fetch logic
- Applying threshold filtering to partition data into pass/fail groups
- Persisting and loading threshold configurations
- Caching analysis data locally to avoid repeated fetches

Requirements: 5.1, 6.2, 6.3, 6.4, 6.5
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

from .constants import (
    DATABASE_DIR,
    THRESHOLD_CONFIG_DIR,
    COMMON_METRIC_COLUMNS,
    ensure_database_dirs,
)

# 分析数据缓存目录
ANALYSIS_CACHE_DIR = DATABASE_DIR / "analysis_cache"

logger = logging.getLogger(__name__)


# Type alias for threshold configuration
# {column_name: (min_value, max_value)} where None means no limit
ThresholdConfig = Dict[str, Tuple[Optional[float], Optional[float]]]


class DataAnalysisService:
    """
    数据分析服务类。
    
    负责调用 Data_fetch 逻辑进行数据分析，以及阈值筛选功能。
    """

    def __init__(self, database_dir: Optional[Path] = None):
        """
        初始化 DataAnalysisService。
        
        Args:
            database_dir: 数据库目录路径，默认使用 constants 中定义的路径
        """
        self.database_dir = database_dir or DATABASE_DIR
        self.threshold_config_dir = self.database_dir / "thresholds"
        self.analysis_cache_dir = self.database_dir / "analysis_cache"
        
        # Ensure directories exist
        self._ensure_dirs()

    def _ensure_dirs(self) -> None:
        """Ensure all required directories exist."""
        self.database_dir.mkdir(parents=True, exist_ok=True)
        self.threshold_config_dir.mkdir(parents=True, exist_ok=True)
        self.analysis_cache_dir.mkdir(parents=True, exist_ok=True)

    # ========================================================================
    # Test Data Fetching (Task 5.1)
    # ========================================================================

    def fetch_test_data(
        self,
        shell_ids: List[str],
        test_types: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        获取壳体测试数据（调用 Data_fetch 逻辑）。
        
        This method integrates with the existing Data_fetch module to retrieve
        test data for the specified shell IDs.
        
        Args:
            shell_ids: 壳体号列表
            test_types: 测试类型列表（如 ["LVI", "Rth"]），None 表示全部
            
        Returns:
            测试数据 DataFrame，如果无数据则返回空 DataFrame
        """
        if not shell_ids:
            logger.warning("No shell IDs provided for fetch_test_data")
            return pd.DataFrame()
        
        try:
            # Import Data_fetch modules dynamically to avoid circular imports
            from ..data_fetch.data_extraction import (
                extract_lvi_data,
                extract_rth_data,
                extract_generic_excel,
                align_output_columns,
                merge_measurement_rows,
            )
            from ..data_fetch.file_utils import (
                interpret_folder_input,
                resolve_test_folder,
                find_measurement_file,
                build_module_measurement_index_cached,
            )
            from ..data_fetch.constants import (
                OUTPUT_COLUMNS,
                SHELL_COLUMN,
                TEST_TYPE_COLUMN,
                MEASUREMENT_OPTIONS,
                TEST_CATEGORY_OPTIONS,
            )
        except ImportError as e:
            logger.error(f"Failed to import Data_fetch modules: {e}")
            return pd.DataFrame()
        
        # Default test types if not specified
        if test_types is None:
            test_types = list(MEASUREMENT_OPTIONS.keys())
        
        combined_frames: List[pd.DataFrame] = []
        errors: List[str] = []
        
        for shell_id in shell_ids:
            try:
                # Interpret shell ID to path
                base_path = interpret_folder_input(shell_id)
                
                # Process each test category
                for test_category in TEST_CATEGORY_OPTIONS:
                    try:
                        # Resolve test folder
                        test_folder = resolve_test_folder(base_path, test_category)
                        
                        # Build file index
                        folder_mtime = test_folder.stat().st_mtime
                        index = build_module_measurement_index_cached(
                            str(test_folder), folder_mtime
                        )
                        
                        # Extract data for each measurement type
                        for measurement_label in test_types:
                            if measurement_label not in MEASUREMENT_OPTIONS:
                                continue
                            
                            token = MEASUREMENT_OPTIONS[measurement_label]
                            
                            try:
                                file_path, _, file_mtime = find_measurement_file(
                                    test_folder, token, index=index
                                )
                                
                                # Extract data based on measurement type
                                if measurement_label == "LVI":
                                    extracted, _, _ = extract_lvi_data(
                                        file_path=file_path,
                                        current_points=None,  # Get all current points
                                        mtime=file_mtime
                                    )
                                elif measurement_label == "Rth":
                                    extracted, _, _ = extract_rth_data(
                                        file_path=file_path,
                                        current_points=None,
                                        mtime=file_mtime
                                    )
                                else:
                                    extracted = extract_generic_excel(
                                        file_path, mtime=file_mtime
                                    )
                                
                                if extracted is not None and not extracted.empty:
                                    # Tag with shell ID and test type
                                    tagged = extracted.copy()
                                    tagged.insert(0, TEST_TYPE_COLUMN, test_category)
                                    tagged.insert(0, SHELL_COLUMN, shell_id)
                                    
                                    aligned = align_output_columns(
                                        tagged, columns=OUTPUT_COLUMNS
                                    )
                                    combined_frames.append(aligned)
                                    
                            except FileNotFoundError:
                                # File not found is expected for some combinations
                                continue
                            except Exception as e:
                                errors.append(
                                    f"{shell_id}/{test_category}/{measurement_label}: {e}"
                                )
                                
                    except FileNotFoundError:
                        # Test folder not found is expected
                        continue
                    except Exception as e:
                        errors.append(f"{shell_id}/{test_category}: {e}")
                        
            except ValueError as e:
                errors.append(f"{shell_id}: {e}")
            except Exception as e:
                errors.append(f"{shell_id}: {e}")
        
        if errors:
            logger.warning(f"Errors during fetch_test_data: {errors[:5]}")
        
        if not combined_frames:
            return pd.DataFrame()
        
        # Combine all frames
        try:
            result_df = pd.concat(combined_frames, ignore_index=True)
            
            # Merge duplicate rows
            result_df = merge_measurement_rows(result_df, columns=OUTPUT_COLUMNS)
            
            return result_df
            
        except Exception as e:
            logger.error(f"Failed to combine data frames: {e}")
            return pd.DataFrame()

    # ========================================================================
    # Threshold Filtering (Task 5.2)
    # ========================================================================

    def apply_thresholds(
        self,
        df: pd.DataFrame,
        thresholds: ThresholdConfig,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        """
        应用阈值筛选。
        
        将数据分为符合标准和不符合标准两组，并计算统计信息。
        
        Args:
            df: 数据 DataFrame
            thresholds: 阈值字典 {列名: (最小值, 最大值)}
                       None 表示该方向无限制
            
        Returns:
            Tuple of (合格数据, 不合格数据, 统计信息)
            
            统计信息包含:
            - total_count: 总数据量
            - pass_count: 合格数量
            - fail_count: 不合格数量
            - pass_rate: 合格率 (0-100)
            - failure_reasons: 不合格原因分析 {列名: 不合格数量}
        """
        if df is None or df.empty:
            return pd.DataFrame(), pd.DataFrame(), {
                "total_count": 0,
                "pass_count": 0,
                "fail_count": 0,
                "pass_rate": 0.0,
                "failure_reasons": {},
            }
        
        if not thresholds:
            # No thresholds, all data passes
            return df.copy(), pd.DataFrame(), {
                "total_count": len(df),
                "pass_count": len(df),
                "fail_count": 0,
                "pass_rate": 100.0,
                "failure_reasons": {},
            }
        
        # Initialize pass mask (all True initially)
        pass_mask = pd.Series(True, index=df.index)
        
        # Track failure reasons
        failure_reasons: Dict[str, int] = {}
        column_fail_masks: Dict[str, pd.Series] = {}
        
        for column, (min_val, max_val) in thresholds.items():
            if column not in df.columns:
                logger.warning(f"Column '{column}' not found in DataFrame")
                continue
            
            # Convert column to numeric
            col_values = pd.to_numeric(df[column], errors="coerce")
            
            # Create fail mask for this column
            col_fail_mask = pd.Series(False, index=df.index)
            
            if min_val is not None:
                col_fail_mask |= (col_values < min_val)
            
            if max_val is not None:
                col_fail_mask |= (col_values > max_val)
            
            # Count failures for this column (excluding NaN values)
            valid_mask = col_values.notna()
            fail_count = (col_fail_mask & valid_mask).sum()
            
            if fail_count > 0:
                failure_reasons[column] = int(fail_count)
            
            column_fail_masks[column] = col_fail_mask
            
            # Update overall pass mask
            pass_mask &= ~col_fail_mask
        
        # Split data
        pass_df = df[pass_mask].copy()
        fail_df = df[~pass_mask].copy()
        
        # Calculate statistics
        total_count = len(df)
        pass_count = len(pass_df)
        fail_count = len(fail_df)
        pass_rate = (pass_count / total_count * 100) if total_count > 0 else 0.0
        
        stats = {
            "total_count": total_count,
            "pass_count": pass_count,
            "fail_count": fail_count,
            "pass_rate": round(pass_rate, 2),
            "failure_reasons": failure_reasons,
        }
        
        return pass_df, fail_df, stats

    def get_numeric_columns(self, df: pd.DataFrame) -> List[str]:
        """
        获取 DataFrame 中的数值列。
        
        Args:
            df: 数据 DataFrame
            
        Returns:
            数值列名列表
        """
        if df is None or df.empty:
            return []

        numeric_cols = []
        for col in df.columns:
            # Try to convert to numeric
            try:
                numeric_values = pd.to_numeric(df[col], errors="coerce")
                # If more than 50% of values are numeric, consider it a numeric column
                if numeric_values.notna().sum() > len(df) * 0.5:
                    numeric_cols.append(col)
            except Exception:
                continue

        if not numeric_cols:
            return []

        # 仅保留指定的指标列，顺序按常量列表呈现
        filtered: List[str] = []
        for allowed in COMMON_METRIC_COLUMNS:
            for col in numeric_cols:
                if col.lower() == allowed.lower():
                    if col not in filtered:
                        filtered.append(col)
                    break

        return filtered

    def suggest_thresholds(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        percentile_low: float = 5.0,
        percentile_high: float = 95.0,
    ) -> ThresholdConfig:
        """
        基于数据分布建议阈值。
        
        Args:
            df: 数据 DataFrame
            columns: 要分析的列，None 表示所有数值列
            percentile_low: 下限百分位数
            percentile_high: 上限百分位数
            
        Returns:
            建议的阈值配置
        """
        if df is None or df.empty:
            return {}
        
        if columns is None:
            columns = self.get_numeric_columns(df)
        
        suggested: ThresholdConfig = {}
        
        for col in columns:
            if col not in df.columns:
                continue
            
            numeric_values = pd.to_numeric(df[col], errors="coerce").dropna()
            
            if numeric_values.empty:
                continue
            
            min_val = float(numeric_values.quantile(percentile_low / 100))
            max_val = float(numeric_values.quantile(percentile_high / 100))
            
            suggested[col] = (round(min_val, 3), round(max_val, 3))
        
        return suggested

    # ========================================================================
    # Threshold Config Persistence (Task 5.4)
    # ========================================================================

    def save_threshold_config(
        self,
        product_type_id: str,
        thresholds: ThresholdConfig,
    ) -> bool:
        """
        保存阈值配置。
        
        Args:
            product_type_id: 产品类型 ID
            thresholds: 阈值配置
            
        Returns:
            True if successful, False otherwise
        """
        if not product_type_id:
            logger.error("Product type ID is required")
            return False
        
        config_path = self.threshold_config_dir / f"{product_type_id}.json"
        
        try:
            # Convert thresholds to JSON-serializable format
            config_data = {
                "product_type_id": product_type_id,
                "updated_at": datetime.now().isoformat(),
                "thresholds": {
                    col: {
                        "min": min_val,
                        "max": max_val,
                    }
                    for col, (min_val, max_val) in thresholds.items()
                },
            }
            
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Saved threshold config for {product_type_id}")
            return True
            
        except IOError as e:
            logger.error(f"Failed to save threshold config: {e}")
            return False

    def load_threshold_config(
        self,
        product_type_id: str,
    ) -> Optional[ThresholdConfig]:
        """
        加载阈值配置。
        
        Args:
            product_type_id: 产品类型 ID
            
        Returns:
            阈值配置，如果不存在返回 None
        """
        if not product_type_id:
            return None
        
        config_path = self.threshold_config_dir / f"{product_type_id}.json"
        
        if not config_path.exists():
            return None
        
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config_data = json.load(f)
            
            # Convert from JSON format to ThresholdConfig
            thresholds: ThresholdConfig = {}
            
            for col, bounds in config_data.get("thresholds", {}).items():
                min_val = bounds.get("min")
                max_val = bounds.get("max")
                thresholds[col] = (min_val, max_val)
            
            return thresholds
            
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Failed to load threshold config: {e}")
            return None

    def delete_threshold_config(self, product_type_id: str) -> bool:
        """
        删除阈值配置。
        
        Args:
            product_type_id: 产品类型 ID
            
        Returns:
            True if successful, False otherwise
        """
        if not product_type_id:
            return False
        
        config_path = self.threshold_config_dir / f"{product_type_id}.json"
        
        if not config_path.exists():
            return True  # Already deleted
        
        try:
            config_path.unlink()
            logger.info(f"Deleted threshold config for {product_type_id}")
            return True
        except IOError as e:
            logger.error(f"Failed to delete threshold config: {e}")
            return False

    def list_threshold_configs(self) -> List[str]:
        """
        列出所有已保存的阈值配置。
        
        Returns:
            产品类型 ID 列表
        """
        if not self.threshold_config_dir.exists():
            return []
        
        configs = []
        for config_file in self.threshold_config_dir.glob("*.json"):
            product_type_id = config_file.stem
            configs.append(product_type_id)
        
        return configs

    # ========================================================================
    # Analysis Data Cache (本地缓存分析数据)
    # ========================================================================

    def _get_cache_key(
        self,
        product_type_id: str,
        order_ids: List[str],
        stations: Optional[List[str]] = None,
    ) -> str:
        """
        生成缓存键。
        
        使用哈希来生成安全的文件名，避免特殊字符和过长文件名问题。
        
        Args:
            product_type_id: 产品类型 ID
            order_ids: 订单 ID 列表
            stations: 站别列表
            
        Returns:
            缓存键字符串
        """
        import hashlib
        
        # 组合所有参数生成唯一标识
        orders_str = ",".join(sorted(order_ids)) if order_ids else "all"
        stations_str = ",".join(sorted(stations)) if stations else "all"
        combined = f"{product_type_id}|{orders_str}|{stations_str}"
        
        # 使用 MD5 哈希生成短且安全的文件名
        hash_str = hashlib.md5(combined.encode("utf-8")).hexdigest()[:16]
        
        # 保留产品类型 ID 前缀便于识别
        return f"{product_type_id[:8]}_{hash_str}"

    def save_analysis_cache(
        self,
        product_type_id: str,
        order_ids: List[str],
        df: pd.DataFrame,
        stations: Optional[List[str]] = None,
        note: Optional[str] = None,
    ) -> bool:
        """
        保存分析数据到本地缓存。
        
        Args:
            product_type_id: 产品类型 ID
            order_ids: 订单 ID 列表
            df: 分析数据 DataFrame
            stations: 站别列表
            note: 备注
            
        Returns:
            True if successful, False otherwise
        """
        if not product_type_id:
            logger.warning("save_analysis_cache: product_type_id is empty")
            return False
        if df is None or df.empty:
            logger.warning("save_analysis_cache: DataFrame is None or empty")
            return False
        if not order_ids:
            logger.warning("save_analysis_cache: order_ids is empty")
            return False
        
        # 确保缓存目录存在
        self.analysis_cache_dir.mkdir(parents=True, exist_ok=True)
        
        cache_key = self._get_cache_key(product_type_id, order_ids, stations)
        parquet_path = self.analysis_cache_dir / f"{cache_key}.parquet"
        meta_path = self.analysis_cache_dir / f"{cache_key}.meta.json"
        
        try:
            # 保存数据
            df.to_parquet(parquet_path, index=False)
            logger.info(f"Saved parquet file: {parquet_path}")
            
            # 保存元数据
            meta = {
                "cache_key": cache_key,
                "product_type_id": product_type_id,
                "order_ids": order_ids,
                "stations": stations,
                "row_count": len(df),
                "columns": list(df.columns),
                "created_at": datetime.now().isoformat(),
                "note": note,
            }
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Saved analysis cache: {cache_key}, rows: {len(df)}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save analysis cache: {e}", exc_info=True)
            return False

    def load_analysis_cache(
        self,
        product_type_id: str,
        order_ids: List[str],
        stations: Optional[List[str]] = None,
    ) -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, Any]]]:
        """
        加载本地缓存的分析数据。
        
        Args:
            product_type_id: 产品类型 ID
            order_ids: 订单 ID 列表
            stations: 站别列表
            
        Returns:
            (DataFrame, 元数据字典) 元组，如果不存在返回 (None, None)
        """
        cache_key = self._get_cache_key(product_type_id, order_ids, stations)
        parquet_path = self.analysis_cache_dir / f"{cache_key}.parquet"
        meta_path = self.analysis_cache_dir / f"{cache_key}.meta.json"
        
        if not parquet_path.exists():
            return None, None
        
        try:
            df = pd.read_parquet(parquet_path)
            
            meta = None
            if meta_path.exists():
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            
            return df, meta
            
        except Exception as e:
            logger.error(f"Failed to load analysis cache: {e}")
            return None, None

    def has_analysis_cache(
        self,
        product_type_id: str,
        order_ids: List[str],
        stations: Optional[List[str]] = None,
    ) -> bool:
        """
        检查是否存在缓存的分析数据。
        
        Args:
            product_type_id: 产品类型 ID
            order_ids: 订单 ID 列表
            stations: 站别列表
            
        Returns:
            True if cache exists, False otherwise
        """
        cache_key = self._get_cache_key(product_type_id, order_ids, stations)
        parquet_path = self.analysis_cache_dir / f"{cache_key}.parquet"
        return parquet_path.exists()

    def get_analysis_cache_info(
        self,
        product_type_id: str,
        order_ids: List[str],
        stations: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        获取缓存的元数据信息。
        
        Args:
            product_type_id: 产品类型 ID
            order_ids: 订单 ID 列表
            stations: 站别列表
            
        Returns:
            元数据字典，如果不存在返回 None
        """
        cache_key = self._get_cache_key(product_type_id, order_ids, stations)
        meta_path = self.analysis_cache_dir / f"{cache_key}.meta.json"
        
        if not meta_path.exists():
            return None
        
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to get analysis cache info: {e}")
            return None

    def delete_analysis_cache(
        self,
        product_type_id: str,
        order_ids: List[str],
        stations: Optional[List[str]] = None,
    ) -> bool:
        """
        删除缓存的分析数据。
        
        Args:
            product_type_id: 产品类型 ID
            order_ids: 订单 ID 列表
            stations: 站别列表
            
        Returns:
            True if successful, False otherwise
        """
        cache_key = self._get_cache_key(product_type_id, order_ids, stations)
        parquet_path = self.analysis_cache_dir / f"{cache_key}.parquet"
        meta_path = self.analysis_cache_dir / f"{cache_key}.meta.json"
        
        try:
            if parquet_path.exists():
                parquet_path.unlink()
            if meta_path.exists():
                meta_path.unlink()
            logger.info(f"Deleted analysis cache: {cache_key}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete analysis cache: {e}")
            return False

    def list_analysis_caches(
        self,
        product_type_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        列出所有缓存的分析数据。
        
        Args:
            product_type_id: 可选的产品类型 ID 筛选
            
        Returns:
            缓存元数据列表
        """
        if not self.analysis_cache_dir.exists():
            return []
        
        caches = []
        for meta_file in self.analysis_cache_dir.glob("*.meta.json"):
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                
                # 筛选产品类型
                if product_type_id and meta.get("product_type_id") != product_type_id:
                    continue
                
                caches.append(meta)
            except Exception as e:
                logger.warning(f"Failed to load cache meta {meta_file}: {e}")
                continue
        
        # 按创建时间降序排列
        caches.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return caches

    def clear_all_analysis_caches(self, product_type_id: Optional[str] = None) -> int:
        """
        清除所有缓存的分析数据。
        
        Args:
            product_type_id: 可选的产品类型 ID，如果指定则只清除该产品类型的缓存
            
        Returns:
            删除的缓存数量
        """
        if not self.analysis_cache_dir.exists():
            return 0
        
        deleted_count = 0
        
        for parquet_file in list(self.analysis_cache_dir.glob("*.parquet")):
            cache_key = parquet_file.stem
            meta_path = self.analysis_cache_dir / f"{cache_key}.meta.json"
            
            # 如果指定了产品类型，检查是否匹配
            if product_type_id and meta_path.exists():
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    if meta.get("product_type_id") != product_type_id:
                        continue
                except Exception:
                    continue
            
            try:
                parquet_file.unlink()
                if meta_path.exists():
                    meta_path.unlink()
                deleted_count += 1
            except Exception as e:
                logger.warning(f"Failed to delete cache {cache_key}: {e}")
        
        logger.info(f"Cleared {deleted_count} analysis caches")
        return deleted_count
