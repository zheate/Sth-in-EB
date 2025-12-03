"""
ShellProgressService for Data Manager (Zh's DataBase).

This module provides shell progress query and Gantt chart data generation.

Requirements: 3.4, 3.5, 4.1, 4.3
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .constants import (
    BASE_STATIONS,
    DATABASE_DIR,
    SHELLS_DIR,
    STATION_MAPPING,
    get_stations_for_part,
    get_station_index,
)
from .models import ShellProgress

logger = logging.getLogger(__name__)


class ShellProgressService:
    """
    壳体进度服务类。
    
    负责壳体进度查询和甘特图数据生成。
    """

    def __init__(self, database_dir: Optional[Path] = None):
        """
        初始化 ShellProgressService。
        
        Args:
            database_dir: 数据库目录路径，默认使用 constants 中定义的路径
        """
        self.database_dir = database_dir or DATABASE_DIR
        self.shells_dir = self.database_dir / "shells"

    def _load_shells_dataframe(self, product_type_id: str) -> Optional[pd.DataFrame]:
        """
        加载产品类型的壳体数据 DataFrame。
        
        Args:
            product_type_id: 产品类型 ID
            
        Returns:
            壳体数据 DataFrame，如果不存在返回 None
        """
        parquet_path = self.shells_dir / f"{product_type_id}.parquet"
        
        if not parquet_path.exists():
            logger.warning(f"Shells data file not found: {parquet_path}")
            return None
        
        try:
            return pd.read_parquet(parquet_path)
        except Exception as e:
            logger.error(f"Failed to load shells data for {product_type_id}: {e}")
            return None

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

    def get_shells_by_orders(
        self,
        product_type_id: str,
        order_ids: List[str],
        time_filter: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        根据生产订单获取壳体数据。
        
        支持多选生产订单，合并显示所有选中订单的壳体数据。
        默认选择最新的时间记录。
        
        Args:
            product_type_id: 产品类型 ID
            order_ids: 生产订单 ID 列表（支持多选）
            time_filter: 时间筛选（默认最新）
            
        Returns:
            壳体数据 DataFrame，包含壳体号、最新站别、站别时间等
            如果没有数据返回空 DataFrame
        """
        shells_df = self._load_shells_dataframe(product_type_id)
        
        if shells_df is None or shells_df.empty:
            return pd.DataFrame()
        
        if not order_ids:
            return pd.DataFrame()
        
        # Find production order column
        order_col = self._find_column(shells_df, [
            "生产订单", "ERP生产订单", "SAP生产订单",
            "生产订单号", "订单号", "工单号"
        ])
        
        if order_col is None:
            logger.warning(f"No production order column found for {product_type_id}")
            return pd.DataFrame()
        
        # Filter by order IDs (multi-select support)
        # Normalize order IDs for comparison
        order_ids_set = {str(oid).strip() for oid in order_ids}
        shells_df[order_col] = shells_df[order_col].fillna("").astype(str).str.strip()
        
        filtered_df = shells_df[shells_df[order_col].isin(order_ids_set)].copy()
        
        if filtered_df.empty:
            return pd.DataFrame()
        
        # Find time column for filtering
        time_col = self._find_column(filtered_df, [
            "时间", "日期", "更新时间", "创建时间", "Time", "Date"
        ])
        
        # Apply time filter or default to latest
        if time_col and time_col in filtered_df.columns:
            try:
                filtered_df[time_col] = pd.to_datetime(filtered_df[time_col], errors="coerce")
                
                if time_filter is not None:
                    # Filter by specific time
                    filtered_df = filtered_df[filtered_df[time_col] <= time_filter]
                else:
                    # Default: select latest time record for each shell
                    # Group by shell ID and keep the latest record
                    shell_col = self._find_column(filtered_df, [
                        "壳体号", "壳体编号", "Shell ID", "ShellID", "SN", "序列号"
                    ])
                    
                    if shell_col:
                        # Sort by time descending and drop duplicates keeping first (latest)
                        filtered_df = filtered_df.sort_values(
                            by=time_col, ascending=False, na_position="last"
                        )
                        filtered_df = filtered_df.drop_duplicates(
                            subset=[shell_col], keep="first"
                        )
            except Exception as e:
                logger.warning(f"Failed to apply time filter: {e}")
        
        # Remove duplicates based on shell ID (multi-order aggregation)
        shell_col = self._find_column(filtered_df, [
            "壳体号", "壳体编号", "Shell ID", "ShellID", "SN", "序列号"
        ])
        
        if shell_col:
            # Keep unique shells (no duplicates across orders)
            filtered_df = filtered_df.drop_duplicates(subset=[shell_col], keep="first")
        
        return filtered_df.reset_index(drop=True)

    def get_shell_progress_list(
        self,
        product_type_id: str,
        order_ids: List[str],
        time_filter: Optional[datetime] = None,
    ) -> List[ShellProgress]:
        """
        获取壳体进度列表。
        
        Args:
            product_type_id: 产品类型 ID
            order_ids: 生产订单 ID 列表
            time_filter: 时间筛选
            
        Returns:
            ShellProgress 对象列表
        """
        shells_df = self.get_shells_by_orders(product_type_id, order_ids, time_filter)
        
        if shells_df.empty:
            return []
        
        # Find relevant columns
        shell_col = self._find_column(shells_df, [
            "壳体号", "壳体编号", "Shell ID", "ShellID", "SN", "序列号"
        ])
        order_col = self._find_column(shells_df, [
            "生产订单", "ERP生产订单", "SAP生产订单",
            "生产订单号", "订单号", "工单号"
        ])
        station_col = self._find_column(shells_df, [
            "当前站点", "当前站别", "站别", "Station"
        ])
        part_col = self._find_column(shells_df, [
            "料号", "物料号", "Part Number", "PartNumber", "PN", "型号"
        ])
        
        progress_list = []
        
        for _, row in shells_df.iterrows():
            shell_id = str(row.get(shell_col, "")).strip() if shell_col else ""
            if not shell_id:
                continue
            
            production_order = str(row.get(order_col, "")).strip() if order_col else ""
            current_station = str(row.get(station_col, "")).strip() if station_col else ""
            part_number = str(row.get(part_col, "")).strip() if part_col else ""
            
            # Normalize station name
            current_station = self._normalize_station_name(current_station)
            
            # Extract completed stations and station times from time columns
            completed_stations, station_times = self._extract_station_times(row, shells_df.columns)
            
            # Check if engineering analysis
            is_engineering = current_station == "工程分析"
            
            progress = ShellProgress(
                shell_id=shell_id,
                production_order=production_order,
                current_station=current_station,
                completed_stations=completed_stations,
                station_times=station_times,
                is_engineering_analysis=is_engineering,
                part_number=part_number,
            )
            
            progress_list.append(progress)
        
        return progress_list

    def _normalize_station_name(self, station_name: str) -> str:
        """
        将站别名称标准化。
        
        Args:
            station_name: 原始站别名称
            
        Returns:
            标准化后的站别名称
        """
        if not station_name or station_name.strip() == "":
            return ""
        
        station_name = station_name.strip()
        station_name_lower = station_name.lower()
        
        # RMA related
        if "rma" in station_name_lower:
            return "RMA"
        
        # Direct mapping (case-insensitive)
        station_mapping_lower = {k.lower(): v for k, v in STATION_MAPPING.items()}
        if station_name_lower in station_mapping_lower:
            return station_mapping_lower[station_name_lower]
        
        # Check if in BASE_STATIONS
        base_stations_lower = {s.lower(): s for s in BASE_STATIONS}
        if station_name_lower in base_stations_lower:
            return base_stations_lower[station_name_lower]
        
        return station_name

    def _extract_station_times(
        self, row: pd.Series, columns: pd.Index
    ) -> Tuple[List[str], Dict[str, datetime]]:
        """
        从行数据中提取站别时间信息。
        
        Args:
            row: DataFrame 行
            columns: DataFrame 列名
            
        Returns:
            (已完成站别列表, 站别时间映射)
        """
        completed_stations = []
        station_times = {}
        
        # Build lowercase lookup maps
        station_mapping_lower = {k.lower(): v for k, v in STATION_MAPPING.items()}
        base_stations_lower = {s.lower(): s for s in BASE_STATIONS}
        
        # Look for time columns (format: "{station}时间")
        for col in columns:
            col_str = str(col)
            if col_str.endswith("时间"):
                station_name = col_str[:-2]  # Remove "时间" suffix
                station_name_lower = station_name.lower()
                
                # Get standard station name
                standard_station = None
                
                # 1. Try direct mapping
                if station_name in STATION_MAPPING:
                    standard_station = STATION_MAPPING[station_name]
                # 2. Try case-insensitive mapping
                elif station_name_lower in station_mapping_lower:
                    standard_station = station_mapping_lower[station_name_lower]
                # 3. Try direct match in BASE_STATIONS
                elif station_name in BASE_STATIONS:
                    standard_station = station_name
                # 4. Try case-insensitive match in BASE_STATIONS
                elif station_name_lower in base_stations_lower:
                    standard_station = base_stations_lower[station_name_lower]
                
                if standard_station:
                    time_val = row.get(col)
                    if pd.notna(time_val) and str(time_val).strip():
                        completed_stations.append(standard_station)
                        try:
                            if isinstance(time_val, datetime):
                                station_times[standard_station] = time_val
                            else:
                                station_times[standard_station] = pd.to_datetime(time_val)
                        except Exception:
                            pass
        
        return completed_stations, station_times

    def generate_gantt_data(
        self,
        shells_df: pd.DataFrame,
        stations: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        生成甘特图数据。
        
        按站别顺序显示每个壳体的完成情况。
        
        Args:
            shells_df: 壳体数据 DataFrame
            stations: 站别顺序列表，如果为 None 则使用 BASE_STATIONS
            
        Returns:
            甘特图数据列表，每个元素包含壳体号、站别、开始时间、结束时间、状态
        """
        if shells_df is None or shells_df.empty:
            return []
        
        # Use BASE_STATIONS if not provided
        if stations is None:
            stations = BASE_STATIONS.copy()
            stations.append("已完成")
        
        gantt_data = []
        
        # Find relevant columns
        shell_col = self._find_column(shells_df, [
            "壳体号", "壳体编号", "Shell ID", "ShellID", "SN", "序列号"
        ])
        station_col = self._find_column(shells_df, [
            "当前站点", "当前站别", "站别", "Station"
        ])
        part_col = self._find_column(shells_df, [
            "料号", "物料号", "Part Number", "PartNumber", "PN", "型号"
        ])
        
        if not shell_col:
            logger.warning("No shell ID column found")
            return []
        
        for _, row in shells_df.iterrows():
            shell_id = str(row.get(shell_col, "")).strip()
            if not shell_id:
                continue
            
            current_station = str(row.get(station_col, "")).strip() if station_col else ""
            current_station = self._normalize_station_name(current_station)
            
            part_number = str(row.get(part_col, "")).strip() if part_col else ""
            
            # Get stations for this part
            shell_stations = get_stations_for_part(part_number)
            
            # Extract station times
            completed_stations, station_times = self._extract_station_times(row, shells_df.columns)
            
            # Determine current station index
            current_idx = -1
            if current_station in shell_stations:
                current_idx = shell_stations.index(current_station)
            elif current_station == "工程分析":
                # Engineering analysis - use last completed station
                for i, s in enumerate(shell_stations):
                    if s in completed_stations:
                        current_idx = i
            
            # Generate Gantt entries for each station
            for station_idx, station in enumerate(shell_stations):
                # Determine status
                if station in completed_stations:
                    status = "completed"
                elif station_idx == current_idx:
                    status = "current"
                elif station_idx < current_idx:
                    status = "completed"
                else:
                    status = "pending"
                
                # Get time info
                start_time = station_times.get(station)
                end_time = None
                
                # If we have the next station's time, use it as end time
                if station_idx + 1 < len(shell_stations):
                    next_station = shell_stations[station_idx + 1]
                    end_time = station_times.get(next_station)
                
                gantt_entry = {
                    "shell_id": shell_id,
                    "station": station,
                    "station_index": station_idx,
                    "status": status,
                    "start_time": start_time.isoformat() if start_time else None,
                    "end_time": end_time.isoformat() if end_time else None,
                    "part_number": part_number,
                }
                
                gantt_data.append(gantt_entry)
        
        # Sort by station index to ensure proper ordering
        gantt_data.sort(key=lambda x: (x["shell_id"], x["station_index"]))
        
        return gantt_data

    def generate_gantt_dataframe(
        self,
        shells_df: pd.DataFrame,
        stations: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        生成甘特图 DataFrame（便于可视化）。
        
        Args:
            shells_df: 壳体数据 DataFrame
            stations: 站别顺序列表
            
        Returns:
            甘特图 DataFrame
        """
        gantt_data = self.generate_gantt_data(shells_df, stations)
        
        if not gantt_data:
            return pd.DataFrame()
        
        return pd.DataFrame(gantt_data)

    def get_shell_detail(
        self,
        product_type_id: str,
        shell_id: str,
    ) -> Optional[ShellProgress]:
        """
        获取壳体详情。
        
        Args:
            product_type_id: 产品类型 ID
            shell_id: 壳体号
            
        Returns:
            ShellProgress 对象，如果不存在返回 None
        """
        shells_df = self._load_shells_dataframe(product_type_id)
        
        if shells_df is None or shells_df.empty:
            return None
        
        # Find shell ID column
        shell_col = self._find_column(shells_df, [
            "壳体号", "壳体编号", "Shell ID", "ShellID", "SN", "序列号"
        ])
        
        if not shell_col:
            return None
        
        # Filter by shell ID
        shells_df[shell_col] = shells_df[shell_col].fillna("").astype(str).str.strip()
        shell_row = shells_df[shells_df[shell_col] == shell_id.strip()]
        
        if shell_row.empty:
            return None
        
        # Get the first matching row
        row = shell_row.iloc[0]
        
        # Find relevant columns
        order_col = self._find_column(shells_df, [
            "生产订单", "ERP生产订单", "SAP生产订单",
            "生产订单号", "订单号", "工单号"
        ])
        station_col = self._find_column(shells_df, [
            "当前站点", "当前站别", "站别", "Station"
        ])
        part_col = self._find_column(shells_df, [
            "料号", "物料号", "Part Number", "PartNumber", "PN", "型号"
        ])
        
        production_order = str(row.get(order_col, "")).strip() if order_col else ""
        current_station = str(row.get(station_col, "")).strip() if station_col else ""
        part_number = str(row.get(part_col, "")).strip() if part_col else ""
        
        # Normalize station name
        current_station = self._normalize_station_name(current_station)
        
        # Extract completed stations and station times
        completed_stations, station_times = self._extract_station_times(row, shells_df.columns)
        
        # Check if engineering analysis
        is_engineering = current_station == "工程分析"
        
        return ShellProgress(
            shell_id=shell_id,
            production_order=production_order,
            current_station=current_station,
            completed_stations=completed_stations,
            station_times=station_times,
            is_engineering_analysis=is_engineering,
            part_number=part_number,
        )
