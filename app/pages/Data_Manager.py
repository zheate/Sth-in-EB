import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import json
import io
import inspect

import pandas as pd
import streamlit as st
import altair as alt

# 路径设置
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# 导入服务层
from pages.data_manager.product_type_service import ProductTypeService
from pages.data_manager.shell_progress_service import ShellProgressService
from pages.data_manager.data_analysis_service import DataAnalysisService
from pages.data_manager.models import ProductType, ProductTypeSummary, ProductionOrder, Attachment
from pages.data_manager.constants import (
    DATABASE_DIR,
    ATTACHMENTS_DIR,
    ensure_database_dirs,
    get_stations_for_part,
    STATION_MAPPING,
    BASE_STATIONS,
    SHELL_ID_CANDIDATES,
    PRODUCTION_ORDER_CANDIDATES,
)
from pages.data_fetch.constants import (
    SHELL_COLUMN,
    TEST_TYPE_COLUMN,
    CURRENT_COLUMN,
    CURRENT_TOLERANCE,
    TEST_CATEGORY_OPTIONS,
    OUTPUT_COLUMNS,
)
from pages.data_fetch.data_extraction import align_output_columns
from pages.data_fetch.ui_components import parse_current_points

logger = logging.getLogger(__name__)

# 页面配置
st.set_page_config(
    page_title="数据管理",
    page_icon="🗄️",
    layout="wide",
)

# 确保目录存在
ensure_database_dirs()

# 初始化服务
@st.cache_resource
def get_product_type_service():
    """获取 ProductTypeService 单例"""
    return ProductTypeService()

@st.cache_resource
def get_shell_progress_service():
    """获取 ShellProgressService 单例"""
    return ShellProgressService()


def _ensure_service_has_method():
    """确保服务有新方法，否则清除缓存"""
    service = get_product_type_service()
    if not hasattr(service, "set_product_type_completed"):
        get_product_type_service.clear()


_ensure_service_has_method()

def get_data_analysis_service():
    """获取 DataAnalysisService 实例"""
    if "dm_data_analysis_service" not in st.session_state:
        st.session_state.dm_data_analysis_service = DataAnalysisService()
    return st.session_state.dm_data_analysis_service


# ============================================================================
# Auto Update Helpers (for Data_fetch/TestAnalysis integration)
# ============================================================================

def _find_shell_record(shell_id: str) -> Optional[Dict[str, Any]]:
    """
    Locate a shell inside Zh's DataBase and return minimal context.
    """
    normalized = str(shell_id).strip() if shell_id is not None else ""
    if not normalized:
        return None

    try:
        pt_service = get_product_type_service()
        shell_service = get_shell_progress_service()
        product_types = pt_service.list_product_types()
    except Exception as exc:
        logger.warning("Failed to load product types while locating shell %s: %s", normalized, exc)
        return None

    for pt in product_types:
        try:
            shells_df = pt_service.get_shells_dataframe(pt.id)
        except Exception as exc:
            logger.debug("Failed to load shells for %s: %s", pt.id, exc)
            continue

        if shells_df is None or shells_df.empty:
            continue

        shell_col = shell_service._find_column(shells_df, SHELL_ID_CANDIDATES)
        if not shell_col:
            continue

        try:
            normalized_df = shells_df.copy()
            normalized_df[shell_col] = normalized_df[shell_col].fillna("").astype(str).str.strip()
        except Exception as exc:
            logger.debug("Failed to normalize shell column for %s: %s", pt.id, exc)
            continue

        matches = normalized_df[normalized_df[shell_col] == normalized]
        if matches.empty:
            continue

        row = matches.iloc[0]
        order_col = shell_service._find_column(normalized_df, PRODUCTION_ORDER_CANDIDATES)
        order_id = str(row.get(order_col, "")).strip() if order_col else ""

        return {
            "product_type_id": pt.id,
            "product_type_name": pt.name,
            "order_id": order_id or "__unknown__",
        }

    return None


def check_shell_in_database(shell_id: str) -> bool:
    """
    Check whether a shell exists in Zh's DataBase.
    """
    return _find_shell_record(shell_id) is not None


def update_shell_test_data(
    shell_id: str,
    test_data: Dict[str, Any],
    current_station: Optional[str] = None,
    test_time: Optional[Any] = None,
    source: str = "auto_update",
) -> bool:
    """
    Persist test data for a shell into the analysis cache so it can be reused by Data Manager.
    """
    shell_info = _find_shell_record(shell_id)
    if shell_info is None:
        logger.debug("Shell %s not found in database; skip auto update", shell_id)
        return False
    if not test_data:
        return False

    normalized_shell_id = str(shell_id).strip()
    record: Dict[str, Any] = {SHELL_COLUMN: normalized_shell_id}
    if current_station is not None:
        record[TEST_TYPE_COLUMN] = str(current_station).strip()
    for key, value in test_data.items():
        record[str(key)] = value

    parsed_time = None
    if test_time is not None:
        try:
            parsed_time = pd.to_datetime(test_time, errors="coerce")
        except Exception:
            parsed_time = None
    if parsed_time is None or pd.isna(parsed_time):
        parsed_time = datetime.now()
    record["测试时间"] = parsed_time

    df = pd.DataFrame([record])
    base_columns = OUTPUT_COLUMNS.copy()
    extra_cols = [col for col in df.columns if col not in base_columns]
    df = align_output_columns(df, columns=base_columns + extra_cols)

    try:
        service = get_data_analysis_service()
        order_ids = [shell_info["order_id"]] if shell_info.get("order_id") else ["__unknown__"]
        existing_df, _ = service.load_analysis_cache(
            shell_info["product_type_id"],
            order_ids,
            stations=None,
        )

        if existing_df is not None and not existing_df.empty:
            combined = pd.concat([existing_df, df], ignore_index=True, sort=False)
        else:
            combined = df

        key_cols = [SHELL_COLUMN]
        if TEST_TYPE_COLUMN in combined.columns:
            key_cols.append(TEST_TYPE_COLUMN)
        if "测试时间" in combined.columns:
            combined = combined.sort_values(by=["测试时间"], ascending=True)
        combined = combined.drop_duplicates(subset=key_cols, keep="last")

        note = f"auto_update:{source}" if source else "auto_update"

        # 同步更新 Data Manager 中已存在的壳体进度（仅更新已存在壳体，不新增）
        try:
            pt_service = get_product_type_service()
            shell_service = get_shell_progress_service()
            shells_df = pt_service.get_shells_dataframe(shell_info["product_type_id"])
            if shells_df is not None and not shells_df.empty:
                shell_col = shell_service._find_column(shells_df, SHELL_ID_CANDIDATES)
                station_col = shell_service._find_column(shells_df, ["当前站点", "当前站别", "站别", "Station"])

                if shell_col:
                    df_norm = shells_df.copy()
                    df_norm[shell_col] = df_norm[shell_col].fillna("").astype(str).str.strip()
                    mask = df_norm[shell_col] == normalized_shell_id
                    if mask.any():
                        if current_station is not None and station_col:
                            normalized_station = shell_service._normalize_station_name(str(current_station).strip())
                            df_norm.loc[mask, station_col] = normalized_station

                        order_col = shell_service._find_column(df_norm, PRODUCTION_ORDER_CANDIDATES)
                        orders = (
                            df_norm[order_col]
                            .dropna()
                            .astype(str)
                            .str.strip()
                            .loc[lambda s: s != ""]
                            .unique()
                            .tolist()
                            if order_col
                            else []
                        )

                        pt_service.upsert_product_type(
                            name=shell_info["product_type_name"],
                            shells_df=df_norm,
                            production_orders=orders,
                        )
        except Exception as exc_update:
            logger.debug("Failed to sync shell progress for %s: %s", normalized_shell_id, exc_update)

        return bool(
            service.save_analysis_cache(
                shell_info["product_type_id"],
                order_ids,
                combined,
                stations=None,
                note=note,
            )
        )
    except Exception as exc:
        logger.warning("Failed to update test data for shell %s: %s", normalized_shell_id, exc, exc_info=True)
        return False


# ============================================================================
# Session State 初始化
# ============================================================================

def init_session_state():
    """初始化 session state"""
    defaults = {
        # Layer 1: Product Type Management
        "dm_selected_product_type_id": None,
        "dm_selected_product_type_name": None,
        "dm_show_rename_dialog": False,
        "dm_show_delete_confirm": False,
        "dm_delete_target_ids": [],
        "dm_attachment_preview_expanded": True,
        
        # Layer 1: Production Order Selection
        "dm_selected_orders": [],
        "dm_order_select_mode": "all",  # "single", "multi" or "all"
        "dm_selected_time": None,
        
        # Layer 2: Shell Progress
        "dm_shells_df": None,
        "dm_shell_progress_list": None,
        "dm_shell_cache_key": None,
        "dm_gantt_data": None,
        "dm_shell_list_page": 0,  # Pagination for shell list
        "dm_gantt_page": 0,       # Pagination for Gantt chart
        
        # Layer 3: Data Analysis
        "dm_analysis_df": None,
        "dm_thresholds": {},
        "dm_selected_stations": [],
        "dm_current_input": "",
        "dm_current_points": None,
        "dm_selected_product_type_ids": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ============================================================================
# Layer 1: Product Type Management UI
# ============================================================================


def _build_product_type_display(pt: ProductTypeSummary) -> str:
    """Format product type display text."""
    text = f"{pt.name} ({pt.shell_count} 壳体, {pt.order_count} 订单)"
    if getattr(pt, "has_attachments", False):
        text += " 📎"
    return text


def _apply_product_type_selection(
    selected_ids: List[str],
    product_types: Optional[List[ProductTypeSummary]] = None,
) -> None:
    """Sync selection state and reset downstream caches."""
    primary_id = selected_ids[0] if selected_ids else None
    st.session_state.dm_selected_product_type_ids = selected_ids
    st.session_state.dm_selected_product_type_id = primary_id
    st.session_state.dm_selected_product_type_name = None

    target = None
    targets = []
    if product_types is None:
        service = get_product_type_service()
        product_types = service.list_product_types()
    
    for sid in selected_ids:
        for pt in product_types:
            if pt.id == sid:
                targets.append(pt)
                if sid == primary_id:
                    target = pt
                break
    
    if target:
        st.session_state.dm_selected_product_type_name = target.name
    
    # 标记需要更新 multiselect（在下次渲染前应用）
    if targets:
        display_texts = [_build_product_type_display(pt) for pt in targets]
        st.session_state._dm_pending_product_type_select = display_texts

    # Reset dependent state
    st.session_state.dm_selected_orders = []
    st.session_state.dm_shells_df = None
    st.session_state.dm_gantt_data = None
    st.session_state.dm_analysis_df = None
    st.session_state.dm_thresholds = {}
    for key in list(st.session_state.keys()):
        if str(key).startswith("dm_loaded_config_"):
            del st.session_state[key]


def render_product_type_selector():
    """
    渲染产品类型选择器。
    
    Requirements: 1.3 - 显示产品类型列表，包含壳体数量和订单数量
    """
    st.markdown(
        """
        <style>
        div[data-baseweb="tag"] {
            max-width: none !important;
            width: auto !important;
            flex-shrink: 0 !important;
        }
        div[data-baseweb="tag"] span {
            max-width: none !important;
            overflow: visible !important;
            text-overflow: unset !important;
            white-space: nowrap !important;
        }
        /* 玻璃拟态按钮样式 */
        button[kind="secondary"], button[kind="primary"] {
            background: rgba(255, 255, 255, 0.15) !important;
            backdrop-filter: blur(8px) !important;
            -webkit-backdrop-filter: blur(8px) !important;
            border-radius: 12px !important;
            border: 1px solid rgba(255, 255, 255, 0.3) !important;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08),
                        inset 0 1px 0 rgba(255, 255, 255, 0.4) !important;
            transition: all 0.3s ease !important;
        }
        button[kind="secondary"]:hover, button[kind="primary"]:hover {
            background: rgba(255, 255, 255, 0.25) !important;
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.12),
                        inset 0 1px 0 rgba(255, 255, 255, 0.5) !important;
            transform: translateY(-1px) !important;
        }
        button[kind="primary"] {
            background: rgba(99, 102, 241, 0.7) !important;
            border: 1px solid rgba(129, 140, 248, 0.5) !important;
        }
        button[kind="primary"]:hover {
            background: rgba(99, 102, 241, 0.85) !important;
        }
        /* 按钮文字不换行 */
        button[kind="secondary"] p, button[kind="primary"] p {
            white-space: nowrap !important;
        }
        /* 玻璃拟态容器样式 - 应用于带边框的容器 */
        [data-testid="stVerticalBlockBorderWrapper"] > div {
            background: rgba(255, 255, 255, 0.06) !important;
            backdrop-filter: blur(10px) !important;
            -webkit-backdrop-filter: blur(10px) !important;
            border-radius: 16px !important;
            border: 1px solid rgba(255, 255, 255, 0.12) !important;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.08),
                        inset 0 1px 0 rgba(255, 255, 255, 0.15) !important;
        }
        /* DataFrame 表格美化 */
        [data-testid="stDataFrame"] {
            border-radius: 12px !important;
            overflow: hidden !important;
        }
        [data-testid="stDataFrame"] > div {
            border-radius: 12px !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    service = get_product_type_service()
    product_types = service.list_product_types()
    
    if not product_types:
        st.info("📭 暂无产品类型数据，请从 Progress 页面保存数据")
        return None
    
    # 创建选项列表，显示名称、壳体数量和订单数量
    options = []
    id_map = {}
    for pt in product_types:
        display_text = _build_product_type_display(pt)
        options.append(display_text)
        id_map[display_text] = pt.id
    selected_ids = st.session_state.get("dm_selected_product_type_ids", [])
    
    # 当前选择（供按钮使用）- 优先使用已存储的 ID 列表
    selected_ids_for_action = st.session_state.get("dm_selected_product_type_ids") or []
    if not selected_ids_for_action and st.session_state.get("dm_selected_product_type_id"):
        selected_ids_for_action = [st.session_state.dm_selected_product_type_id]

    # 第一行：标题 + 按钮（水平对齐）
    title_col, rename_col, complete_col, delete_col = st.columns([3, 2, 2, 2], gap="small", vertical_alignment="center")
    with title_col:
        st.markdown("**选择产品类型**")
    with rename_col:
        if st.button("✏️ 重命名", key="dm_rename_btn", use_container_width=True):
            st.session_state.dm_show_rename_dialog = True
    with complete_col:
        if st.button("✅ 已完成", key="dm_complete_btn", use_container_width=True, help="将选中产品标记为已完成"):
            _mark_selected_product_types_completed(selected_ids_for_action)
    with delete_col:
        if st.button("🗑️ 删除", key="dm_delete_btn", use_container_width=True, help="删除已选产品类型及关联数据"):
            st.session_state.dm_show_delete_confirm = True
            st.session_state.dm_delete_target_ids = selected_ids_for_action

    # 准备 multiselect 默认值
    pending = st.session_state.pop("_dm_pending_product_type_select", None)
    has_widget_value = "dm_product_type_select" in st.session_state

    if pending:
        st.session_state.dm_product_type_select = pending
        has_widget_value = True

    default_values = None
    if not has_widget_value:
        default_values = []
        if st.session_state.get("dm_selected_product_type_ids"):
            for opt in options:
                if id_map[opt] in st.session_state.dm_selected_product_type_ids:
                    default_values.append(opt)
        elif st.session_state.dm_selected_product_type_id:
            for opt in options:
                if id_map[opt] == st.session_state.dm_selected_product_type_id:
                    default_values.append(opt)
                    break
        if not default_values and options:
            default_values = [options[0]]
        default_values = [d for d in default_values if d in options]

    # 第二行：选择器（全宽）
    if default_values is not None:
        selected_displays = st.multiselect(
            "选择产品类型",
            options=options,
            default=default_values,
            key="dm_product_type_select",
            label_visibility="collapsed",
            help="选择要查看的产品类型（可多选，首个为当前）"
        )
    else:
        selected_displays = st.multiselect(
            "选择产品类型",
            options=options,
            key="dm_product_type_select",
            label_visibility="collapsed",
            help="选择要查看的产品类型（可多选，首个为当前）"
        )

    if selected_displays:
        selected_ids = [id_map[d] for d in selected_displays]
        primary_id = selected_ids[0]
        if selected_ids != st.session_state.get("dm_selected_product_type_ids", []) or primary_id != st.session_state.dm_selected_product_type_id:
            _apply_product_type_selection(selected_ids, product_types)
            st.rerun()
    else:
        selected_ids = []
        st.session_state.dm_selected_product_type_ids = []
        st.session_state.dm_selected_product_type_id = None
        st.session_state.dm_selected_product_type_name = None
    
    return st.session_state.dm_selected_product_type_id


def _is_shell_completed(shell: Any) -> bool:
    """Check whether a shell is treated as completed."""
    stations_for_part = get_stations_for_part(shell.part_number)
    final_station = stations_for_part[-1] if stations_for_part else "已完成"
    return (
        shell.current_station in {"已完成", "出货检", final_station}
        or final_station in getattr(shell, "completed_stations", [])
    )


def _load_product_type_board_data() -> List[Dict[str, Any]]:
    service = get_product_type_service()
    shell_service = get_shell_progress_service()
    product_types = service.list_product_types()
    board_items: List[Dict[str, Any]] = []

    for pt in product_types:
        orders = service.get_production_orders(pt.id)
        order_ids = [o.id for o in orders] if orders else []
        progress_list = shell_service.get_shell_progress_list(pt.id, order_ids) if order_ids else []

        completed_shells = 0
        for shell in progress_list:
            if _is_shell_completed(shell):
                completed_shells += 1

        total_shells = len(progress_list) or pt.shell_count
        # 优先使用手动标记的 is_completed 字段（兼容旧数据）
        is_completed = getattr(pt, "is_completed", False)
        status = "completed" if is_completed else "wip"

        board_items.append({
            "id": pt.id,
            "name": pt.name,
            "order_count": pt.order_count,
            "shell_count": pt.shell_count,
            "completed_shells": completed_shells,
            "total_shells": total_shells,
            "has_attachments": pt.has_attachments,
            "created_at": pt.created_at,
            "status": status,
            "is_completed": is_completed,
        })

    return board_items


def _render_product_type_board_column(
    container,
    title: str,
    items: List[Dict[str, Any]],
    product_type_map: Dict[str, ProductTypeSummary],
    show_title: bool = True,
) -> None:
    with container:
        if show_title:
            st.markdown(f"**{title} ({len(items)})**")
        if not items:
            st.caption("暂无数据")
            return

        # 超过6个产品时添加滚动容器
        if len(items) > 6:
            scroll_container = st.container(height=320)
        else:
            scroll_container = st.container()

        with scroll_container:
            for item in items:
                attachment_flag = " 📎" if item.get("has_attachments") else ""
                if st.button(
                    f"{item['name']}{attachment_flag}",
                    key=f"dm_pt_board_select_{item['id']}",
                    use_container_width=True
                ):
                    _apply_product_type_selection([item["id"]], list(product_type_map.values()))
                    st.session_state.dm_focus_progress_tab = True
                    st.rerun()


def render_product_type_kanban():
    """Render product type Kanban grouped by WIP/completed."""
    # 添加样式让看板按钮文字靠左且加粗
    st.markdown(
        """
        <style>
        [data-testid="stExpander"] button[kind="secondary"] {
            text-align: left !important;
            justify-content: flex-start !important;
        }
        [data-testid="stExpander"] button[kind="secondary"] p {
            text-align: left !important;
            font-weight: 600 !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    service = get_product_type_service()
    product_types = service.list_product_types()
    if not product_types:
        st.info("📋 暂无产品类型数据")
        return

    board_data = _load_product_type_board_data()
    product_type_map = {pt.id: pt for pt in product_types}

    header_col, action_col = st.columns([5, 1])
    with header_col:
        st.markdown("#### 📋 产品类型看板")
        st.caption("按进度快速分组，点击卡片右上角即可跳转到该产品类型。")
    with action_col:
        if st.button("🔄 刷新", key="dm_pt_board_refresh", use_container_width=True):
            st.rerun()

    wip_items = [item for item in board_data if item.get("status") == "wip"]
    done_items = [item for item in board_data if item.get("status") == "completed"]

    col_wip, col_done = st.columns(2)
    # 折叠区，始终展开（有滚动容器控制高度）
    wip_expanded = True
    with col_wip:
        exp_wip = st.expander(f"🛠 WIP ({len(wip_items)})", expanded=wip_expanded)
        _render_product_type_board_column(exp_wip, "🛠 WIP", wip_items, product_type_map, show_title=False)
        # 手动添加产品
        with exp_wip:
            # 应用 pending 清空状态
            if st.session_state.pop("_dm_clear_new_product_name", False):
                st.session_state.dm_new_product_name = ""
            
            add_col1, add_col2 = st.columns([3, 1])
            with add_col1:
                new_product_name = st.text_input(
                    "新产品名称",
                    key="dm_new_product_name",
                    placeholder="输入产品名称",
                    label_visibility="collapsed"
                )
            with add_col2:
                if st.button("➕ 添加", key="dm_add_product_btn", use_container_width=True):
                    if new_product_name and new_product_name.strip():
                        try:
                            service.upsert_product_type(
                                name=new_product_name.strip(),
                                shells_df=None,
                                production_orders=[]
                            )
                            st.session_state._dm_clear_new_product_name = True  # 标记清空
                            st.toast(f"✅ 已添加: {new_product_name.strip()}")
                            st.rerun()
                        except Exception as e:
                            st.toast(f"❌ 添加失败: {str(e)}", icon="❌")
                    else:
                        st.toast("请输入产品名称", icon="⚠️")
    with col_done:
        exp_done = st.expander(f"✅ 已完成 ({len(done_items)})", expanded=False)
        _render_product_type_board_column(exp_done, "✅ 已完成", done_items, product_type_map, show_title=False)


def render_rename_dialog():
    """
    渲染重命名对话框。
    
    Requirements: 1.5 - 重命名产品类型
    """
    if not st.session_state.dm_show_rename_dialog:
        return
    
    service = get_product_type_service()
    current_name = st.session_state.dm_selected_product_type_name or ""
    
    with st.container(border=True):
        st.markdown("### ✏️ 重命名产品类型")
        
        new_name = st.text_input(
            "新名称",
            value=current_name,
            key="dm_rename_input",
            placeholder="输入新的产品类型名称"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ 确认", key="dm_rename_confirm", use_container_width=True, type="primary"):
                if new_name and new_name.strip():
                    try:
                        success = service.rename_product_type(
                            st.session_state.dm_selected_product_type_id,
                            new_name.strip()
                        )
                        if success:
                            st.session_state.dm_selected_product_type_name = new_name.strip()
                            st.session_state.dm_show_rename_dialog = False
                            st.toast(f"✅ 已重命名为: {new_name.strip()}")
                            st.rerun()
                        else:
                            st.toast("❌ 重命名失败", icon="❌")
                    except ValueError as e:
                        st.toast(f"❌ {str(e)}", icon="❌")
                else:
                    st.toast("❌ 名称不能为空", icon="❌")
        
        with col2:
            if st.button("❌ 取消", key="dm_rename_cancel", use_container_width=True):
                st.session_state.dm_show_rename_dialog = False
                st.rerun()


def render_delete_confirm_dialog():
    """渲染删除确认对话框。"""
    if not st.session_state.get("dm_show_delete_confirm"):
        return

    service = get_product_type_service()
    target_ids = st.session_state.get("dm_delete_target_ids", [])

    # 获取要删除的产品名称
    names = []
    for pid in target_ids:
        pt = service.get_product_type(pid)
        if pt:
            names.append(pt.name)

    if not names:
        st.session_state.dm_show_delete_confirm = False
        return

    with st.container(border=True):
        st.markdown("### ⚠️ 确认删除")
        st.warning(f"确定要删除以下产品类型吗？此操作不可恢复！\n\n**{', '.join(names)}**")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ 确认删除", key="dm_delete_confirm_yes", use_container_width=True, type="primary"):
                _delete_selected_product_types(target_ids)
        with col2:
            if st.button("❌ 取消", key="dm_delete_confirm_no", use_container_width=True):
                st.session_state.dm_show_delete_confirm = False
                st.session_state.dm_delete_target_ids = []
                st.rerun()


def _mark_selected_product_types_completed(selected_ids: Optional[List[str]] = None) -> None:
    """
    将选中的产品类型标记为已完成。
    """
    service = get_product_type_service()
    ids = selected_ids or st.session_state.get("dm_selected_product_type_ids") or []
    if not ids and st.session_state.get("dm_selected_product_type_id"):
        ids = [st.session_state.dm_selected_product_type_id]

    logger.info(f"Marking product types as completed: {ids}")

    if not ids:
        st.warning("请选择要标记的产品类型")
        return

    errors: List[str] = []
    completed = 0
    for pid in ids:
        pt = service.get_product_type(pid)
        pt_name = pt.name if pt else pid
        logger.info(f"Processing product type: {pid} ({pt_name})")
        try:
            result = service.set_product_type_completed(pid, True)
            logger.info(f"set_product_type_completed result: {result}")
            if result:
                completed += 1
            else:
                errors.append(pt_name)
        except Exception as e:
            logger.error(f"Failed to mark product type {pid} as completed: {e}", exc_info=True)
            errors.append(pt_name)

    if completed:
        st.session_state.dm_show_balloons = True
        st.toast(f"✅ 已将 {completed} 个产品标记为已完成")
        st.rerun()
    elif errors:
        st.toast(f"❌ 标记失败: {', '.join(errors)}", icon="❌")


def _delete_selected_product_types(selected_ids: Optional[List[str]] = None) -> None:
    """
    Delete selected product types immediately (cascade delete related data).
    
    Requirements: 7.5 - 删除产品类型时级联删除关联数据
    """
    service = get_product_type_service()
    ids = selected_ids or st.session_state.get("dm_selected_product_type_ids") or []
    if not ids and st.session_state.dm_selected_product_type_id:
        ids = [st.session_state.dm_selected_product_type_id]

    product_types = []
    for pid in ids:
        pt = service.get_product_type(pid)
        if pt:
            product_types.append(pt)

    if not product_types:
        st.session_state.dm_show_delete_confirm = False
        st.warning("请选择要删除的产品类型")
        return

    errors: List[str] = []
    deleted = 0
    with st.spinner("正在删除所选产品类型..."):
        for pt in product_types:
            try:
                if service.delete_product_type(pt.id):
                    deleted += 1
                else:
                    errors.append(pt.name)
            except Exception:
                errors.append(pt.name)

    _apply_product_type_selection([], product_types)
    st.session_state.dm_show_delete_confirm = False
    if deleted:
        st.toast(f"✅ 已删除 {deleted} 个产品类型")
    if errors:
        st.toast(f"❌ 未能删除: {', '.join(errors)}", icon="❌")
    st.rerun()


def render_attachment_upload():
    """
    渲染附件上传UI。
    
    Requirements: 2.1, 2.2 - 上传 PDF 或 Excel 附件
    """
    if not st.session_state.dm_selected_product_type_id:
        return
    
    service = get_product_type_service()
    existing_attachments = service.list_attachments(st.session_state.dm_selected_product_type_id)
    existing_names = {att.original_name.lower() for att in existing_attachments}
    
    # 上传区域
    uploaded_file = st.file_uploader(
        "上传附件",
        type=["pdf", "xlsx", "xls"],
        key="dm_attachment_uploader",
        help="支持 PDF 和 Excel 文件"
    )
    
    if uploaded_file:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.caption(f"📄 {uploaded_file.name} ({uploaded_file.size / 1024:.1f} KB)")
        with col2:
            duplicate = uploaded_file.name.lower() in existing_names
            btn_label = "📤 覆盖上传" if duplicate else "📤 上传"
            btn_help = "同名附件已存在，点击覆盖上传" if duplicate else "上传附件"
            if duplicate:
                st.warning(f"同名附件已存在：{uploaded_file.name}，点击“覆盖上传”确认覆盖。")
            if st.button(btn_label, key="dm_upload_btn", use_container_width=True, type="primary", help=btn_help):
                try:
                    file_content = uploaded_file.read()
                    sig = inspect.signature(service.upload_attachment)
                    if "allow_overwrite" in sig.parameters:
                        attachment_id = service.upload_attachment(
                            product_type_id=st.session_state.dm_selected_product_type_id,
                            file_content=file_content,
                            original_name=uploaded_file.name,
                            allow_overwrite=duplicate,
                        )
                    else:
                        # 兼容旧版本：若需覆盖则先删除再上传
                        if duplicate:
                            for att in existing_attachments:
                                if att.original_name.lower() == uploaded_file.name.lower():
                                    try:
                                        service.delete_attachment(st.session_state.dm_selected_product_type_id, att.id)
                                    except Exception:
                                        pass
                                    break
                        attachment_id = service.upload_attachment(
                            product_type_id=st.session_state.dm_selected_product_type_id,
                            file_content=file_content,
                            original_name=uploaded_file.name,
                        )
                    st.toast(f"✅ 附件上传成功: {uploaded_file.name}")
                    st.rerun()
                except ValueError as e:
                    st.toast(f"上传失败: {str(e)}", icon="❌")
                except IOError as e:
                    st.toast(f"文件保存失败: {str(e)}", icon="❌")


def render_attachment_preview():
    """
    渲染附件预览UI。
    
    Requirements: 2.3, 2.4 - 默认折叠，可展开预览
    """
    if not st.session_state.dm_selected_product_type_id:
        return
    
    service = get_product_type_service()
    attachments = service.list_attachments(st.session_state.dm_selected_product_type_id)
    
    if not attachments:
        st.caption("暂无附件")
        return
    
    # 默认折叠的附件列表
    with st.expander(f"📎 附件列表 ({len(attachments)})", expanded=st.session_state.dm_attachment_preview_expanded):
        for att in attachments:
            col1, col2, col3, col4 = st.columns([5, 1, 1, 1])
            
            with col1:
                icon = "📄" if att.file_type == "pdf" else "📊"
                st.markdown(f"{icon} **{att.original_name}** <span style='color:grey; font-size:0.8em'>({att.size / 1024:.1f} KB | {att.uploaded_at.strftime('%Y-%m-%d %H:%M')})</span>", unsafe_allow_html=True)
            
            with col2:
                # 预览按钮
                if st.button("👁️", key=f"preview_{att.id}", help="预览"):
                    st.session_state[f"dm_preview_{att.id}"] = not st.session_state.get(f"dm_preview_{att.id}", False)
            
            with col3:
                # 下载按钮
                file_path = service.get_attachment_path(att.id)
                if file_path and file_path.exists():
                    with open(file_path, "rb") as f:
                        st.download_button(
                            "📥",
                            data=f.read(),
                            file_name=att.original_name,
                            key=f"download_{att.id}",
                            help="下载"
                        )
            
            with col4:
                # 删除按钮
                if st.button("🗑️", key=f"delete_att_{att.id}", help="删除"):
                    try:
                        success = service.delete_attachment(
                            st.session_state.dm_selected_product_type_id,
                            att.id
                        )
                        if success:
                            st.toast(f"✅ 已删除: {att.original_name}")
                            st.rerun()
                    except Exception as e:
                        st.toast(f"❌ 删除失败: {str(e)}", icon="❌")
            
            # 预览内容
            if st.session_state.get(f"dm_preview_{att.id}", False):
                file_path = service.get_attachment_path(att.id)
                if file_path and file_path.exists():
                    if att.file_type == "pdf":
                        # 使用 base64 + iframe 预览 PDF 文件
                        try:
                            import base64
                            with open(file_path, "rb") as pdf_file:
                                pdf_bytes = pdf_file.read()
                            base64_pdf = base64.b64encode(pdf_bytes).decode("utf-8")
                            pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
                            st.markdown(pdf_display, unsafe_allow_html=True)
                        except Exception as e:
                            st.toast(f"PDF 预览失败: {str(e)}", icon="❌")
                    elif att.file_type in ("excel", "xlsx", "xls"):
                        # 用系统默认程序打开 Excel 文件
                        try:
                            import os
                            import subprocess
                            import platform
                            
                            file_str = str(file_path)
                            if platform.system() == "Windows":
                                os.startfile(file_str)
                            elif platform.system() == "Darwin":  # macOS
                                subprocess.run(["open", file_str])
                            else:  # Linux
                                subprocess.run(["xdg-open", file_str])
                            st.toast(f"✅ 已用系统默认程序打开: {att.original_name}")
                            # 关闭预览状态
                            st.session_state[f"dm_preview_{att.id}"] = False
                        except Exception as e:
                            st.toast(f"打开文件失败: {str(e)}", icon="❌")


def render_production_order_selector():
    if not st.session_state.dm_selected_product_type_id:
        return
    
    service = get_product_type_service()
    orders = service.get_production_orders(st.session_state.dm_selected_product_type_id)
    
    if not orders:
        st.info("📭 该产品类型下暂无生产订单数据")
        return
    if not st.session_state.dm_selected_orders:
        if st.session_state.dm_order_select_mode == "all":
            st.session_state.dm_selected_orders = [order.id for order in orders]
        else:
            st.session_state.dm_selected_orders = [orders[0].id]
      
    # 选择模式切换
    col1, col2 = st.columns([3, 1])
    with col2:
        mode_map = {"single": 0, "multi": 1, "all": 2}
        current_index = mode_map.get(st.session_state.dm_order_select_mode, 0)
        
        select_mode = st.radio(
            "选择模式",
            options=["单选", "多选", "全选"],
            index=current_index,
            key="dm_order_mode_radio",
            horizontal=False,
            label_visibility="collapsed"
        )
        
        new_mode = {"单选": "single", "多选": "multi", "全选": "all"}[select_mode]
        if new_mode != st.session_state.dm_order_select_mode:
            st.session_state.dm_order_select_mode = new_mode
            if new_mode == "all":
                st.session_state.dm_selected_orders = [order.id for order in orders]
            st.session_state.dm_shell_list_page = 0
            st.session_state.dm_gantt_page = 0
            st.rerun()
    
    # 创建订单选项，显示时间信息
    order_options = []
    order_id_map = {}
    for order in orders:
        time_info = ""
        if order.latest_time:
            time_info = f" | {order.latest_time.strftime('%Y-%m-%d')}"
        display_text = f"{order.id} ({order.shell_count} 壳体{time_info})"
        order_options.append(display_text)
        order_id_map[display_text] = order.id
    
    with col1:
        if st.session_state.dm_order_select_mode == "single":
            # 单选模式
            default_index = 0
            if st.session_state.dm_selected_orders:
                for i, opt in enumerate(order_options):
                    if order_id_map[opt] in st.session_state.dm_selected_orders:
                        default_index = i
                        break
            
            selected_display = st.selectbox(
                "选择生产订单",
                options=order_options,
                index=default_index,
                key="dm_order_select_single",
                help="选择要查看的生产订单"
            )
            
            if selected_display:
                selected_id = order_id_map[selected_display]
                if [selected_id] != st.session_state.dm_selected_orders:
                    st.session_state.dm_selected_orders = [selected_id]
                    st.session_state.dm_shell_list_page = 0
                    st.session_state.dm_gantt_page = 0
        elif st.session_state.dm_order_select_mode == "multi":
            # 多选模式
            default_values = []
            for opt in order_options:
                if order_id_map[opt] in st.session_state.dm_selected_orders:
                    default_values.append(opt)
            
            if not default_values and order_options:
                default_values = [order_options[0]]
            
            selected_displays = st.multiselect(
                "选择生产订单（可多选）",
                options=order_options,
                default=default_values,
                key="dm_order_select_multi",
                help="选择要查看的生产订单，支持多选"
            )
            
            if selected_displays:
                selected_ids = [order_id_map[d] for d in selected_displays]
                if selected_ids != st.session_state.dm_selected_orders:
                    st.session_state.dm_selected_orders = selected_ids
                    st.session_state.dm_shell_list_page = 0
                    st.session_state.dm_gantt_page = 0
        else:
            # 全选模式 - 显示已选中所有订单
            st.multiselect(
                "选择生产订单（已全选）",
                options=order_options,
                default=order_options,
                key="dm_order_select_all",
                disabled=True,
                help="已选择所有生产订单"
            )
    
    # 显示选中订单的统计信息
    if st.session_state.dm_selected_orders:
        total_shells = sum(
            order.shell_count for order in orders 
            if order.id in st.session_state.dm_selected_orders
        )
        st.caption(f"已选择 {len(st.session_state.dm_selected_orders)} 个订单，共 {total_shells} 个壳体")

# Pagination constants
SHELLS_PER_PAGE = 20
GANTT_MAX_SHELLS = 50
ANALYSIS_ROWS_PER_PAGE = 50


def render_shell_progress_section():
    st.markdown('<a id="shell-progress"></a>', unsafe_allow_html=True)
    if not st.session_state.dm_selected_orders:
        st.info("📭 请先选择生产订单以查看壳体进度")
        return
    
    if not st.session_state.dm_selected_product_type_id:
        st.info("📭 请先选择产品类型")
        return
    
    # 使用缓存键来避免重复加载
    cache_key = f"{st.session_state.dm_selected_product_type_id}_{','.join(sorted(st.session_state.dm_selected_orders))}"
    
    # 检查是否需要重新加载数据
    if (st.session_state.get("dm_shell_cache_key") != cache_key or 
        st.session_state.get("dm_shells_df") is None):
        
        shell_service = get_shell_progress_service()
        shells_df = shell_service.get_shells_by_orders(
            product_type_id=st.session_state.dm_selected_product_type_id,
            order_ids=st.session_state.dm_selected_orders,
        )
        
        if shells_df.empty:
            st.warning("⚠️ 所选订单下没有壳体数据")
            return
        
        shell_progress_list = shell_service.get_shell_progress_list(
            product_type_id=st.session_state.dm_selected_product_type_id,
            order_ids=st.session_state.dm_selected_orders,
        )
        
        # 缓存数据
        st.session_state.dm_shells_df = shells_df
        st.session_state.dm_shell_progress_list = shell_progress_list
        st.session_state.dm_shell_cache_key = cache_key
    else:
        shells_df = st.session_state.dm_shells_df
        shell_progress_list = st.session_state.dm_shell_progress_list
    
    if shells_df.empty:
        st.warning("⚠️ 所选订单下没有壳体数据")
        return
    
    total_shells = len(shell_progress_list)
    
    # 站别当前数量（参考进度追踪逻辑）
    counts_df = calculate_shell_station_counts(shell_progress_list)
    if not counts_df.empty:
        # 玻璃拟态容器样式
        st.markdown(
            """
            <style>
            .glass-container {
                background: rgba(255, 255, 255, 0.1);
                backdrop-filter: blur(10px);
                -webkit-backdrop-filter: blur(10px);
                border-radius: 16px;
                border: 1px solid rgba(255, 255, 255, 0.2);
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.08);
                padding: 1rem;
                margin: 0.5rem 0;
            }
            .stDataFrame, [data-testid="stDataFrame"] {
                background: rgba(255, 255, 255, 0.05) !important;
                border-radius: 12px !important;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("### 各站别当前数量")
        with st.container(border=True):
            table_col, chart_col = st.columns([2, 3])

            with table_col:
                counts_style = counts_df.style.format({"占比": "{:.1%}"})
                # 高度自适应：每行36px + 表头60px，最小100px
                table_height = max(100, min(320, 36 * len(counts_df) + 60))
                st.dataframe(counts_style, use_container_width=True, height=table_height)

            with chart_col:
                station_order = counts_df["站别"].tolist()
                chart_height = max(160, min(360, 28 * len(counts_df)))
                max_val = counts_df["数量"].max() if not counts_df.empty else 0
                color_scale = alt.Scale(
                    scheme="blues",
                    domain=[0, max(max_val, 1)],
                )

                chart = (
                    alt.Chart(counts_df)
                    .mark_bar(cornerRadius=12, opacity=0.85)
                    .encode(
                        x=alt.X("数量:Q", title="完成数量", axis=alt.Axis(grid=True, gridOpacity=0.15, tickMinStep=1)),
                        y=alt.Y("站别:N", sort=station_order, title="站别", axis=alt.Axis(labelFontSize=12, labelFontWeight="bold")),
                        color=alt.Color("数量:Q", scale=color_scale, legend=None),
                        tooltip=["站别", "数量", alt.Tooltip("占比:Q", title="占比", format=".1%")],
                    )
                ).properties(height=chart_height).configure_view(strokeWidth=0).configure_axis(titleFontSize=13, titleFontWeight="bold")

                st.altair_chart(chart, use_container_width=True, theme="streamlit")

    # 工程分析站别分布
    eng_counts_df = calculate_engineering_station_counts(shell_progress_list)
    if not eng_counts_df.empty:
        st.markdown("---")
        st.markdown("### 🔍 工程分析站别分布")

        with st.container(border=True):
            table_col, pie_col = st.columns([2, 3])

            with table_col:
                st.caption(f"工程分析总数: {int(eng_counts_df['数量'].sum())} 个")
                eng_style = eng_counts_df.style.format({"占比": "{:.1%}"})
                st.dataframe(eng_style, use_container_width=True, hide_index=True)

            with pie_col:
                st.caption("工程分析站别占比")
                # 悬停高亮效果
                hover = alt.selection_point(fields=["站别"], on="pointerover", empty=False)
                pie_chart = (
                    alt.Chart(eng_counts_df)
                    .mark_arc(innerRadius=25, outerRadius=75, opacity=0.85)
                    .encode(
                        theta=alt.Theta("数量:Q", stack=True),
                        color=alt.Color("站别:N", legend=alt.Legend(title="站别", orient="right"), scale=alt.Scale(scheme="category20")),
                        tooltip=[
                            alt.Tooltip("站别:N", title="站别"),
                            alt.Tooltip("数量:Q", title="数量"),
                            alt.Tooltip("占比:Q", title="占比", format=".1%"),
                        ],
                        opacity=alt.condition(hover, alt.value(1), alt.value(0.7)),
                        stroke=alt.condition(hover, alt.value("#333"), alt.value(None)),
                        strokeWidth=alt.condition(hover, alt.value(2), alt.value(0)),
                    )
                    .add_params(hover)
                    .properties(height=180)
                )
                st.altair_chart(pie_chart, use_container_width=True)
    
    # Render shell list
    st.markdown("---")
    st.markdown("### 📋 壳体列表")
    render_shell_list(shell_progress_list, total_shells)


def render_shell_list(shell_progress_list: List, total_shells: int):

    if not shell_progress_list:
        st.info("暂无壳体数据")
        return

    display_data: List[Dict[str, Any]] = []
    for idx, shell in enumerate(shell_progress_list, start=1):
        stations_for_part = get_stations_for_part(shell.part_number)
        final_station = stations_for_part[-1] if stations_for_part else "已完成"
        is_completed = (
            shell.current_station in {"已完成", "出货检", final_station}
            or final_station in shell.completed_stations
        )
        
        if shell.is_engineering_analysis:
            status_icon = "🔬"
            status_text = "工程分析"
        elif shell.current_station == "报废":
            status_icon = "❌"
            status_text = "报废"
        elif is_completed:
            status_icon = "✅"
            status_text = "已完成"
        elif shell.current_station:
            status_icon = "🔄"
            status_text = "进行中"
        else:
            status_icon = "⏳"
            status_text = "未开始"

        latest_time = shell.get_latest_station_time()
        time_str = latest_time.strftime("%Y-%m-%d %H:%M") if latest_time else "-"

        display_data.append({
            "序号": idx,
            "壳体号": shell.shell_id,
            "当前站别": shell.current_station or "-",
            "状态": f"{status_icon} {status_text}",
            "已完成站数": len(shell.completed_stations),
            "最新时间": time_str,
            "生产订单": shell.production_order,
        })

    display_df = pd.DataFrame(display_data)
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "序号": st.column_config.NumberColumn("序号", width="small"),
            "壳体号": st.column_config.TextColumn("壳体号", width="medium"),
            "当前站别": st.column_config.TextColumn("当前站别", width="medium"),
            "状态": st.column_config.TextColumn("状态", width="small"),
            "已完成站数": st.column_config.NumberColumn("已完成站数", width="small"),
            "最新时间": st.column_config.TextColumn("最新时间", width="medium"),
            "生产订单": st.column_config.TextColumn("生产订单", width="large"),
        },
    )

    st.caption(f"共 {total_shells} 个壳体")


def render_data_analysis_section():
    """
    渲染数据分析区域（第三层）。
    
    Requirements: 5.1, 5.2, 5.3, 5.4, 6.1, 6.2, 6.3, 6.4, 6.5
    """
    if not st.session_state.dm_selected_orders:
        st.info("📭 请先选择生产订单以进行数据分析")
        return
    
    if not st.session_state.dm_selected_product_type_id:
        st.info("📭 请先选择产品类型")
        return
    
    st.markdown("### 📈 数据分析")
    
    # Initialize analysis-specific session state
    if "dm_analysis_df" not in st.session_state:
        st.session_state.dm_analysis_df = None
    if "dm_thresholds" not in st.session_state:
        st.session_state.dm_thresholds = {}
    if "dm_analysis_page" not in st.session_state:
        st.session_state.dm_analysis_page = 0
    if "dm_show_threshold_editor" not in st.session_state:
        st.session_state.dm_show_threshold_editor = False
    if "dm_filter_columns" not in st.session_state:
        st.session_state.dm_filter_columns = []
    
    # Auto-load threshold config when product type is selected
    _auto_load_threshold_config()
    
    # Render test data fetch UI (Task 10.1)
    render_test_data_fetch_ui()
    
    # Only show analysis UI if data is loaded
    if st.session_state.dm_analysis_df is not None and not st.session_state.dm_analysis_df.empty:
        st.divider()
        
        # Render threshold setting UI (Task 10.3)
        render_threshold_setting_ui()
        
        st.divider()
        
        # Render analysis results with filtering (Task 10.2, 10.4)
        render_analysis_results_table()


def _auto_load_threshold_config():
    """
    自动加载产品类型的指标配置。
    
    Requirements: 6.5 - Auto-load saved config when selecting product type
    """
    if not st.session_state.dm_selected_product_type_id:
        return
    
    # Check if we need to load config (only on product type change)
    config_key = f"dm_loaded_config_{st.session_state.dm_selected_product_type_id}"
    if st.session_state.get(config_key):
        return
    
    service = get_data_analysis_service()
    saved_config = service.load_threshold_config(st.session_state.dm_selected_product_type_id)
    
    if saved_config:
        st.session_state.dm_thresholds = saved_config
        st.session_state[config_key] = True


def render_test_data_fetch_ui():
    """
    渲染测试数据获取UI。
    
    Requirements: 5.1 - Add button to fetch test data for selected shells
                   Show loading indicator during fetch
    """
    # Get shell IDs from selected orders
    shell_service = get_shell_progress_service()
    shell_progress_list = shell_service.get_shell_progress_list(
        product_type_id=st.session_state.dm_selected_product_type_id,
        order_ids=st.session_state.dm_selected_orders,
    )
    
    if not shell_progress_list:
        st.warning("⚠️ 所选订单下没有壳体数据")
        return
    
    shell_ids = [sp.shell_id for sp in shell_progress_list]
    analysis_service = get_data_analysis_service()
    
    # 检查是否有缓存数据（不按站别区分，统一缓存）
    cache_info = analysis_service.get_analysis_cache_info(
        product_type_id=st.session_state.dm_selected_product_type_id,
        order_ids=st.session_state.dm_selected_orders,
        stations=None,  # 不按站别区分缓存
    )
    
    # 过滤条件：站别 + 电流点
    filt_col1, filt_col2 = st.columns([2, 1])
    with filt_col1:
        default_stations = st.session_state.dm_selected_stations or TEST_CATEGORY_OPTIONS
        selected_stations = st.multiselect(
            "指定站别",
            options=TEST_CATEGORY_OPTIONS,
            default=default_stations,
            key="dm_station_select",
            help="选择要显示的测试站别；加载缓存后会按此筛选"
        )
        st.session_state.dm_selected_stations = selected_stations
    with filt_col2:
        current_input = st.text_input(
            "指定电流点",
            value=st.session_state.dm_current_input or "",
            placeholder="例: 4 或 2,5 或 12~19，a 表示全部",
            key="dm_current_input",
            help="加载缓存后会按此电流点筛选；留空取最高电流点；a 表示全部"
        )
    
    col1, col2, col3, col4 = st.columns([3, 1.5, 1.5, 1])
    
    with col1:
        st.caption(f"📋 已选择 {len(shell_ids)} 个壳体")
        if len(shell_ids) > 5:
            st.caption(f"壳体号: {', '.join(shell_ids[:5])}...")
        else:
            st.caption(f"壳体号: {', '.join(shell_ids)}")
    
    with col2:
        # 加载缓存按钮（如果有缓存）
        if cache_info:
            cache_time = cache_info.get("created_at", "")[:16].replace("T", " ")
            cache_rows = cache_info.get("row_count", 0)
            if st.button(
                f"📂 加载缓存",
                key="dm_load_cache_btn",
                use_container_width=True,
                help=f"缓存: {cache_rows}条 | {cache_time}\n加载后按当前站别和电流点筛选"
            ):
                # 解析电流点
                try:
                    if current_input.strip():
                        current_points = parse_current_points(current_input)
                    else:
                        current_points = []
                except ValueError:
                    current_points = []
                _load_cached_analysis_data(selected_stations, current_points)
        else:
            st.button("📂 加载缓存", key="dm_load_cache_btn_disabled", use_container_width=True, disabled=True)
    
    with col3:
        # Fetch data button
        fetch_clicked = st.button(
            "🔄 获取数据",
            key="dm_fetch_test_data_btn",
            use_container_width=True,
            type="primary",
            help="从测试系统获取全部数据并更新缓存"
        )
    
    with col4:
        # Clear data button
        if st.session_state.dm_analysis_df is not None:
            if st.button(
                "🗑️ 清除",
                key="dm_clear_analysis_btn",
                use_container_width=True,
                help="清除当前分析数据"
            ):
                st.session_state.dm_analysis_df = None
                st.session_state.dm_analysis_page = 0
                st.rerun()
        else:
            st.button("🗑️ 清除", key="dm_clear_analysis_btn_disabled", use_container_width=True, disabled=True)
    
    # Handle fetch button click - 获取全部数据，保存到缓存，然后按当前筛选条件显示
    if fetch_clicked:
        # 解析电流点
        try:
            if current_input.strip():
                current_points = parse_current_points(current_input)
            else:
                current_points = []
        except ValueError:
            current_points = []
        _fetch_test_data(shell_ids, selected_stations, current_points)
    
    # Show data status
    if st.session_state.dm_analysis_df is not None:
        df = st.session_state.dm_analysis_df
        if df.empty:
            st.toast("📭 未找到测试数据", icon="ℹ️")


def _load_cached_analysis_data(selected_stations: List[str], current_points: Optional[List[float]]):
    """加载缓存的分析数据，并按站别和电流点筛选"""
    service = get_data_analysis_service()
    
    # 加载缓存（不按站别区分）
    df, meta = service.load_analysis_cache(
        product_type_id=st.session_state.dm_selected_product_type_id,
        order_ids=st.session_state.dm_selected_orders,
        stations=None,  # 加载全部缓存数据
    )
    
    if df is not None and not df.empty:
        original_count = len(df)
        
        # 按站别筛选
        if selected_stations and TEST_TYPE_COLUMN in df.columns:
            df = df[df[TEST_TYPE_COLUMN].isin(selected_stations)]
        
        # 按电流点筛选
        df = _filter_by_current_points(df, current_points)
        
        st.session_state.dm_analysis_df = df
        st.session_state.dm_analysis_page = 0
        cache_time = meta.get("created_at", "")[:16].replace("T", " ") if meta else ""
        
        if len(df) < original_count:
            st.toast(f"从缓存筛选出 {len(df)}/{original_count} 条记录", icon="✅")
        else:
            st.toast(f"已从缓存加载 {len(df)} 条记录 (缓存时间: {cache_time})", icon="✅")
        st.rerun()
    else:
        st.toast("缓存数据加载失败或为空", icon="⚠️")


def _save_analysis_data_to_cache(df: pd.DataFrame, selected_stations: List[str]):
    """保存分析数据到本地缓存"""
    # 检查必要参数
    if not st.session_state.dm_selected_product_type_id:
        st.error("❌ 请先选择产品类型")
        return
    if not st.session_state.dm_selected_orders:
        st.error("❌ 请先选择生产订单")
        return
    if df is None or df.empty:
        st.error("❌ 没有数据可保存")
        return
    
    try:
        service = get_data_analysis_service()
        
        # 直接尝试保存，捕获详细错误
        import hashlib
        from pathlib import Path
        
        # 生成缓存路径
        product_type_id = st.session_state.dm_selected_product_type_id
        order_ids = st.session_state.dm_selected_orders
        
        orders_str = ",".join(sorted(order_ids)) if order_ids else "all"
        stations_str = ",".join(sorted(selected_stations)) if selected_stations else "all"
        combined = f"{product_type_id}|{orders_str}|{stations_str}"
        hash_str = hashlib.md5(combined.encode("utf-8")).hexdigest()[:16]
        cache_key = f"{product_type_id[:8]}_{hash_str}"
        
        cache_dir = service.analysis_cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        parquet_path = cache_dir / f"{cache_key}.parquet"
        meta_path = cache_dir / f"{cache_key}.meta.json"
        
        # 保存 parquet
        df.to_parquet(parquet_path, index=False)
        
        # 保存元数据
        from datetime import datetime
        meta = {
            "cache_key": cache_key,
            "product_type_id": product_type_id,
            "order_ids": order_ids,
            "stations": selected_stations,
            "row_count": len(df),
            "columns": list(df.columns),
            "created_at": datetime.now().isoformat(),
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        
        st.toast(f"已保存 {len(df)} 条记录到本地缓存", icon="✅")
        st.rerun()
        
    except Exception as e:
        st.error(f"❌ 保存失败: {str(e)}")


def _fetch_test_data(shell_ids: List[str], selected_stations: List[str], current_points: Optional[List[float]]):
    """
    获取测试数据的内部函数。
    
    Args:
        shell_ids: 壳体号列表
        selected_stations: 需要的测试站别（空列表表示全部）
        current_points: 电流点列表；None 表示全部，不为空列表表示按输入点过滤，空列表时取最高电流点
    """
    service = get_data_analysis_service()
    total = len(shell_ids)
    
    # 使用进度条显示提取进度
    progress_bar = st.progress(0, text="正在获取测试数据...")
    status_text = st.empty()
    
    try:
        combined_frames: List[pd.DataFrame] = []
        errors: List[str] = []
        
        for idx, shell_id in enumerate(shell_ids):
            # 更新进度
            progress = (idx + 1) / total
            progress_bar.progress(progress, text=f"正在提取: {shell_id} ({idx + 1}/{total})")
            
            try:
                # 单个壳体提取
                shell_df = service.fetch_test_data([shell_id])
                if shell_df is not None and not shell_df.empty:
                    combined_frames.append(shell_df)
            except Exception as e:
                errors.append(f"{shell_id}: {str(e)[:50]}")
        
        progress_bar.progress(1.0, text="数据提取完成！")
        
        # 合并所有数据
        if combined_frames:
            df = pd.concat(combined_frames, ignore_index=True)
        else:
            df = pd.DataFrame()
        
        if df.empty:
            st.session_state.dm_analysis_df = df
            status_text.empty()
            st.toast("⚠️ 未找到测试数据，请确认壳体号是否正确", icon="⚠️")
        else:
            # 保存全部数据到缓存
            _auto_save_to_cache(df)
            
            # 然后按当前选择的站别和电流点筛选显示
            display_df = df.copy()
            if selected_stations and TEST_TYPE_COLUMN in display_df.columns:
                display_df = display_df[display_df[TEST_TYPE_COLUMN].isin(selected_stations)]
            display_df = _filter_by_current_points(display_df, current_points)
            
            st.session_state.dm_analysis_df = display_df
            st.session_state.dm_analysis_page = 0
            
            if errors:
                status_text.warning(f"获取 {len(df)} 条数据，{len(errors)} 个壳体失败")
            else:
                status_text.empty()
                st.toast(f"获取 {len(df)} 条数据，筛选后 {len(display_df)} 条", icon="✅")
            st.rerun()
            
    except Exception as e:
        progress_bar.empty()
        st.error(f"❌ 获取数据失败: {str(e)}")
        st.session_state.dm_analysis_df = pd.DataFrame()


def _auto_save_to_cache(df: pd.DataFrame):
    """自动保存全部数据到缓存（不按站别区分）"""
    if df is None or df.empty:
        return
    if not st.session_state.dm_selected_product_type_id:
        return
    if not st.session_state.dm_selected_orders:
        return
    
    try:
        import hashlib
        from datetime import datetime
        
        service = get_data_analysis_service()
        product_type_id = st.session_state.dm_selected_product_type_id
        order_ids = st.session_state.dm_selected_orders
        
        # 不按站别区分，统一缓存
        orders_str = ",".join(sorted(order_ids)) if order_ids else "all"
        combined = f"{product_type_id}|{orders_str}|all"
        hash_str = hashlib.md5(combined.encode("utf-8")).hexdigest()[:16]
        cache_key = f"{product_type_id[:8]}_{hash_str}"
        
        cache_dir = service.analysis_cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        parquet_path = cache_dir / f"{cache_key}.parquet"
        meta_path = cache_dir / f"{cache_key}.meta.json"
        
        df.to_parquet(parquet_path, index=False)
        
        meta = {
            "cache_key": cache_key,
            "product_type_id": product_type_id,
            "order_ids": order_ids,
            "stations": None,  # 全部站别
            "row_count": len(df),
            "columns": list(df.columns),
            "created_at": datetime.now().isoformat(),
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
    except Exception:
        pass  # 静默失败，不影响主流程


def _filter_by_current_points(df: pd.DataFrame, currents: Optional[List[float]]) -> pd.DataFrame:
    """按电流点过滤数据，逻辑与数据提取保持一致"""
    if df is None or df.empty or CURRENT_COLUMN not in df.columns:
        return df
    if currents is None:
        return df  # 'a' 表示全部电流点

    numeric = df.copy()
    numeric[CURRENT_COLUMN] = pd.to_numeric(numeric[CURRENT_COLUMN], errors="coerce")
    numeric = numeric.dropna(subset=[CURRENT_COLUMN])
    if numeric.empty:
        return numeric

    if currents:
        mask = pd.Series(False, index=numeric.index)
        for c in currents:
            mask |= (numeric[CURRENT_COLUMN] - c).abs() <= CURRENT_TOLERANCE
        filtered = numeric.loc[mask]
        if not filtered.empty:
            return filtered

    # 若指定电流未匹配，回退到最高电流点
    max_c = numeric[CURRENT_COLUMN].max()
    return numeric.loc[(numeric[CURRENT_COLUMN] - max_c).abs() <= CURRENT_TOLERANCE] if pd.notna(max_c) else numeric


def _normalize_station_name_dm(name: str) -> str:
    """站别归一化，兼容大小写/映射，空值返回 未开工"""
    if name is None:
        return "未开工"
    text = str(name).strip()
    if not text:
        return "未开工"
    if text in STATION_MAPPING:
        return STATION_MAPPING[text]
    lower_map = {k.lower(): v for k, v in STATION_MAPPING.items()}
    return lower_map.get(text.lower(), text)


def calculate_shell_station_counts(shell_progress_list: List) -> pd.DataFrame:
    """统计壳体当前站别数量与占比，参考进度追踪逻辑"""
    if not shell_progress_list:
        return pd.DataFrame(columns=["站别", "数量", "占比"])

    unknown_label = "未识别"
    stations = []
    for shell in shell_progress_list:
        current = "工程分析" if shell.is_engineering_analysis else (shell.current_station or "")
        normalized = _normalize_station_name_dm(current)
        if normalized in ("", None):
            normalized = "未开工"
        stations.append(normalized)

    counts = pd.Series(stations).value_counts(dropna=False).reset_index()
    counts.columns = ["站别", "数量"]
    counts["占比"] = counts["数量"] / len(stations)

    ordered_labels = BASE_STATIONS + ["工程分析", "已完成", "未开工", unknown_label]
    order_map = {label: idx for idx, label in enumerate(ordered_labels)}
    counts["排序"] = counts["站别"].map(order_map)
    fallback_order = len(ordered_labels) + counts.index.to_series()
    counts["排序"] = counts["排序"].fillna(fallback_order)

    counts = counts.sort_values(["排序", "站别"]).drop(columns="排序").reset_index(drop=True)
    return counts


def _get_engineering_station(shell: Any) -> Optional[str]:
    """获取工程分析的上一站，用于工程分析分布统计"""
    # 如果 completed_stations 有记录，取最后一个
    if getattr(shell, "completed_stations", None):
        return shell.completed_stations[-1]
    return None


def calculate_engineering_station_counts(shell_progress_list: List) -> pd.DataFrame:
    """统计工程分析壳体的上一站分布"""
    if not shell_progress_list:
        return pd.DataFrame(columns=["站别", "数量", "占比"])

    stations = []
    for shell in shell_progress_list:
        if not getattr(shell, "is_engineering_analysis", False):
            continue
        prev_station = _get_engineering_station(shell)
        normalized = _normalize_station_name_dm(prev_station) if prev_station else "未识别"
        stations.append(normalized)

    if not stations:
        return pd.DataFrame(columns=["站别", "数量", "占比"])

    counts = pd.Series(stations).value_counts(dropna=False).reset_index()
    counts.columns = ["站别", "数量"]
    counts["占比"] = counts["数量"] / len(stations)
    counts = counts.reset_index(drop=True)
    return counts


def render_threshold_setting_ui():
    df = st.session_state.dm_analysis_df
    if df is None or df.empty:
        return
    
    service = get_data_analysis_service()
    
    # Get numeric columns
    numeric_cols = service.get_numeric_columns(df)
    
    if not numeric_cols:
        st.info("📭 数据中没有可用于指标筛选的数值列")
        return
    
    # Header with toggle and save buttons
    col1, col2, col3 = st.columns([3, 1, 1])
    
    with col1:
        st.markdown("#### ⚙️ 指标设置")
    
    with col2:
        # Toggle threshold editor
        if st.button(
            "📝 编辑指标" if not st.session_state.dm_show_threshold_editor else "✅ 完成编辑",
            key="dm_toggle_threshold_editor",
            use_container_width=True,
        ):
            st.session_state.dm_show_threshold_editor = not st.session_state.dm_show_threshold_editor
            st.rerun()
    
    with col3:
        # Save threshold config (Task 10.5)
        if st.button(
            "💾 保存配置",
            key="dm_save_threshold_btn",
            use_container_width=True,
            help="保存当前指标配置到产品类型"
        ):
            _save_threshold_config()
    
    # Show current thresholds summary
    if st.session_state.dm_thresholds:
        active_thresholds = [
            col for col, (min_v, max_v) in st.session_state.dm_thresholds.items()
            if min_v is not None or max_v is not None
        ]
        if active_thresholds:
            st.caption(f"📊 已设置 {len(active_thresholds)} 个指标: {', '.join(active_thresholds[:5])}{'...' if len(active_thresholds) > 5 else ''}")
    
    # Threshold editor (expandable)
    if st.session_state.dm_show_threshold_editor:
        _render_threshold_editor(numeric_cols, df)


def _render_threshold_editor(numeric_cols: List[str], df: pd.DataFrame):
    """
    渲染指标编辑器。
    
    Args:
        numeric_cols: 数值列列表
        df: 数据 DataFrame
    """
    with st.container(border=True):
        st.markdown("##### 设置指标")
        st.caption("设置每个指标的最小值和最大值，留空表示不限制")
        
        # Create columns for threshold inputs
        for i in range(0, len(numeric_cols), 2):
            cols = st.columns(2)
            
            for j, col_idx in enumerate([i, i + 1]):
                if col_idx >= len(numeric_cols):
                    break
                
                col_name = numeric_cols[col_idx]
                
                with cols[j]:
                    # Get current threshold values
                    current_min, current_max = st.session_state.dm_thresholds.get(col_name, (None, None))
                    
                    # Get column statistics for reference
                    col_values = pd.to_numeric(df[col_name], errors="coerce").dropna()
                    if not col_values.empty:
                        data_min = float(col_values.min())
                        data_max = float(col_values.max())
                        data_mean = float(col_values.mean())
                        stats_text = f"范围: {data_min:.3f} ~ {data_max:.3f}, 均值: {data_mean:.3f}"
                    else:
                        stats_text = "无有效数据"
                    
                    st.markdown(f"**{col_name}**")
                    st.caption(stats_text)
                    
                    sub_col1, sub_col2 = st.columns(2)
                    
                    with sub_col1:
                        new_min = st.number_input(
                            "最小值",
                            value=current_min,
                            key=f"dm_threshold_min_{col_name}",
                            format="%.3f",
                            label_visibility="collapsed",
                            placeholder="最小值",
                        )
                    
                    with sub_col2:
                        new_max = st.number_input(
                            "最大值",
                            value=current_max,
                            key=f"dm_threshold_max_{col_name}",
                            format="%.3f",
                            label_visibility="collapsed",
                            placeholder="最大值",
                        )
                    
                    # Update threshold in session state
                    if new_min is not None or new_max is not None:
                        st.session_state.dm_thresholds[col_name] = (new_min, new_max)
                    elif col_name in st.session_state.dm_thresholds:
                        # Remove if both are None
                        if new_min is None and new_max is None:
                            del st.session_state.dm_thresholds[col_name]
        
        # Clear all thresholds button
        if st.button("🗑️ 清除所有指标", key="dm_clear_thresholds_btn"):
            st.session_state.dm_thresholds = {}
            st.rerun()


def _save_threshold_config():
    """
    保存指标配置。
    
    Requirements: 6.5 - Add save button for current threshold config
    """
    if not st.session_state.dm_selected_product_type_id:
        st.toast("❌ 请先选择产品类型", icon="❌")
        return
    
    service = get_data_analysis_service()
    
    success = service.save_threshold_config(
        st.session_state.dm_selected_product_type_id,
        st.session_state.dm_thresholds
    )
    
    if success:
        st.toast("指标配置已保存", icon="✅")
    else:
        st.toast("保存失败", icon="❌")


def render_analysis_results_table():
    """
    渲染分析结果表格。
    
    Requirements: 5.2 - Display multi-station analysis results
                  5.3 - Support column filtering
                  5.4 - Highlight out-of-threshold values
                  6.3 - Show pass/fail statistics
                  6.4 - Display failure reason analysis
    """
    df = st.session_state.dm_analysis_df
    if df is None or df.empty:
        return
    
    service = get_data_analysis_service()
    
    # Apply thresholds and get statistics
    pass_df, fail_df, stats = service.apply_thresholds(df, st.session_state.dm_thresholds)
    
    # Render statistics (Task 10.4)
    _render_filtering_statistics(stats)
    
    st.divider()
    
    # Column filter (Task 10.2)
    col_title, col_filter = st.columns([1, 4])
    with col_title:
        st.markdown("#### 📊 分析结果")
    with col_filter:
        # View mode selector - 水平排列
        view_mode = st.radio(
            "显示",
            options=["全部", "合格", "不合格"],
            index=0,
            key="dm_view_mode",
            horizontal=True,
            label_visibility="collapsed"
        )
    
    # Column filter multiselect
    all_columns = df.columns.tolist()
    selected_columns = st.multiselect(
        "选择显示的列",
        options=all_columns,
        default=st.session_state.dm_filter_columns if st.session_state.dm_filter_columns else all_columns[:10],
        key="dm_column_filter",
        help="选择要在表格中显示的列",
        label_visibility="collapsed"
    )
    st.session_state.dm_filter_columns = selected_columns
    
    # Select data based on view mode
    if view_mode == "合格":
        display_df = pass_df
    elif view_mode == "不合格":
        display_df = fail_df
    else:
        display_df = df
    
    if display_df.empty:
        st.info(f"📭 没有{view_mode}的数据")
        return
    
    # Filter columns
    if selected_columns:
        display_cols = [c for c in selected_columns if c in display_df.columns]
        if display_cols:
            display_df = display_df[display_cols]
    
    # Apply highlighting for out-of-threshold values (Task 10.4)
    styled_df = _apply_threshold_highlighting(display_df, st.session_state.dm_thresholds)
    
    # Display the table - 高度自适应数据行数
    # 每行约35px，表头约40px，最小150px，最大600px
    table_height = min(600, max(150, len(display_df) * 35 + 40))
    
    # 玻璃拟态表格容器
    with st.container(border=True):
        st.caption(f"共 {len(display_df)} 条数据")
        st.dataframe(
            styled_df,
            use_container_width=True,
            hide_index=True,
            height=table_height
        )


def _render_filtering_statistics(stats: Dict[str, Any]):
    """
    渲染筛选统计信息。
    
    Requirements: 6.3 - Show pass/fail statistics
                  6.4 - Display failure reason analysis
    
    Args:
        stats: 统计信息字典
    """
    st.markdown("#### 📈 筛选统计")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("总数据量", stats["total_count"])
    
    with col2:
        st.metric("合格数量", stats["pass_count"], delta=None)
    
    with col3:
        st.metric("不合格数量", stats["fail_count"], delta=None)
    
    with col4:
        pass_rate = stats["pass_rate"]
        # Color code the pass rate
        if pass_rate >= 95:
            st.metric("合格率", f"{pass_rate:.1f}%", delta="优秀")
        elif pass_rate >= 80:
            st.metric("合格率", f"{pass_rate:.1f}%", delta="良好")
        else:
            st.metric("合格率", f"{pass_rate:.1f}%", delta="需改进", delta_color="inverse")
    
    # Failure reason analysis (Task 10.4)
    failure_reasons = stats.get("failure_reasons", {})
    if failure_reasons:
        with st.expander("📋 不合格原因分析", expanded=True):
            # Sort by failure count descending
            sorted_reasons = sorted(failure_reasons.items(), key=lambda x: x[1], reverse=True)
            
            # Create a simple bar chart using columns
            for col_name, fail_count in sorted_reasons[:10]:  # Show top 10
                col1, col2 = st.columns([3, 1])
                with col1:
                    # Calculate percentage
                    pct = (fail_count / stats["total_count"] * 100) if stats["total_count"] > 0 else 0
                    st.progress(min(pct / 100, 1.0), text=f"{col_name}")
                with col2:
                    st.caption(f"{fail_count} 条 ({pct:.1f}%)")


def _apply_threshold_highlighting(df: pd.DataFrame, thresholds: Dict) -> pd.DataFrame:
    """
    应用指标高亮显示。
    
    Requirements: 5.4 - Highlight out-of-threshold values
    
    Args:
        df: 数据 DataFrame
        thresholds: 指标配置
        
    Returns:
        带样式的 DataFrame
    """
    if not thresholds or df.empty:
        return df
    
    def highlight_out_of_threshold(val, col_name):
        """高亮超出指标的值"""
        if col_name not in thresholds:
            return ""
        
        min_val, max_val = thresholds[col_name]
        
        try:
            numeric_val = float(val)
        except (ValueError, TypeError):
            return ""
        
        if pd.isna(numeric_val):
            return ""
        
        if min_val is not None and numeric_val < min_val:
            return "background-color: #ffcccc"  # Light red for below min
        if max_val is not None and numeric_val > max_val:
            return "background-color: #ffcccc"  # Light red for above max
        
        return "background-color: #ccffcc"  # Light green for within range
    
    # Apply styling
    styled = df.style
    
    for col in df.columns:
        if col in thresholds:
            styled = styled.applymap(
                lambda val, c=col: highlight_out_of_threshold(val, c),
                subset=[col]
            )
    
    return styled


# ============================================================================
# Sidebar
# ============================================================================

def render_sidebar():
    """渲染侧边栏"""
    with st.sidebar:
        st.header("🗄️ Zh's DataBase")
        st.caption("产品数据管理与分析系统")
        
        st.divider()
        
        # 显示当前选中的产品类型信息
        if st.session_state.dm_selected_product_type_id:
            service = get_product_type_service()
            product_type = service.get_product_type(st.session_state.dm_selected_product_type_id)
            
            if product_type:
                st.markdown("### 📦 当前产品类型")
                st.metric("名称", product_type.name)
                
                col1, col2 = st.columns(2)
                col1.metric("壳体数", product_type.shell_count)
                col2.metric("订单数", product_type.order_count)
                
                if product_type.attachments:
                    st.caption(f"📎 {len(product_type.attachments)} 个附件")
                
                if product_type.source_file:
                    st.caption(f"📄 来源: {product_type.source_file}")
                
                st.caption(f"🕐 创建: {product_type.created_at.strftime('%Y-%m-%d %H:%M')}")
        else:
            st.info("请选择产品类型")
        
        st.divider()
        
        # 数据库统计
        st.markdown("### 📊 数据库统计")
        service = get_product_type_service()
        product_types = service.list_product_types()
        
        total_shells = sum(pt.shell_count for pt in product_types)
        total_orders = sum(pt.order_count for pt in product_types)
        
        col1, col2 = st.columns(2)
        col1.metric("产品类型", len(product_types))
        col2.metric("总壳体数", total_shells)
        
        st.divider()
        


# ============================================================================
# Main Page
# ============================================================================

def main():
    """主函数"""
    init_session_state()
    
    # 显示气球效果（标记完成后）
    if st.session_state.pop("dm_show_balloons", False):
        st.balloons()
    
    # 渲染侧边栏
    render_sidebar()
    
    # 主标题
    st.title("🏠 :rainbow[ZH's MiaoMiao House]")
    # 渲染对话框
    render_rename_dialog()
    render_delete_confirm_dialog()
    
    # 主内容区域 - 使用 tabs 组织三层结构
    tab1, tab2, tab3 = st.tabs([
        "📦 产品管理", 
        "📊 进度分析", 
        "📈 数据分析"
    ])
    
    with tab1:
        st.markdown("### 产品类型管理")
        
        render_product_type_kanban()
        st.divider()

        # 产品类型选择 + 生产订单并排
        col_pt, col_order = st.columns(2, vertical_alignment="top")
        with col_pt:
            st.markdown("#### 选择产品类型")
            render_product_type_selector()

        with col_order:
            if st.session_state.dm_selected_product_type_id:
                st.markdown("#### 生产订单")
                render_production_order_selector()
            else:
                st.info("请选择产品类型后再选择生产订单")

        if st.session_state.dm_selected_product_type_id:
            st.divider()
            st.markdown("#### 📎 附件管理")
            render_attachment_upload()
            render_attachment_preview()
    
    focus_progress = st.session_state.pop("dm_focus_progress_tab", False)
    
    with tab2:
        render_shell_progress_section()
    
    with tab3:
        render_data_analysis_section()

    # If focus flag set, switch to progress tab via JavaScript
    if focus_progress:
        from streamlit.components.v1 import html as st_html
        import time
        st_html(
            f"""
            <script>
            // 等待 DOM 加载完成后点击第二个 tab - {time.time()}
            setTimeout(function() {{
                const tabs = window.parent.document.querySelectorAll('[data-baseweb="tab"]');
                if (tabs && tabs.length > 1) {{
                    tabs[1].click();
                }}
            }}, 50);
            </script>
            """,
            height=0
        )


if __name__ == "__main__":
    main()
