# Storage UI widgets for Streamlit pages
"""
提供保存、加载和数据管理的 UI 组件。

这些组件可以在各个 Streamlit 页面中复用，提供统一的数据存储交互体验。
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

from utils.local_storage import (
    LocalDataStore,
    DataCategory,
    DatasetMetadata,
    serialize_plot_sources,
    deserialize_plot_sources,
    check_column_compatibility,
    convert_dataframe_for_module,
    get_sendable_modules,
    get_module_display_name,
    ColumnCompatibilityResult,
)
from utils.exceptions import (
    LocalStorageError,
    DatasetNotFoundError,
    ExportError,
)


def _get_store() -> LocalDataStore:
    """获取或创建 LocalDataStore 实例"""
    if "local_data_store" not in st.session_state:
        st.session_state.local_data_store = LocalDataStore()
    return st.session_state.local_data_store


def _format_datetime(iso_str: str) -> str:
    """格式化 ISO 日期时间字符串为可读格式"""
    try:
        dt = datetime.fromisoformat(iso_str)
        return dt.strftime("%Y-%m-%d %H:%M")
    except (ValueError, TypeError):
        return iso_str


def _get_category_label(category: DataCategory) -> str:
    """获取数据类别的中文标签"""
    labels = {
        DataCategory.EXTRACTION: "数据提取",
        DataCategory.PROGRESS: "进度追踪",
        DataCategory.ANALYSIS: "工程分析",
    }
    return labels.get(category, category.value)


def render_save_button(
    df: pd.DataFrame,
    category: DataCategory,
    extra_data: Optional[Dict[str, Any]] = None,
    source_file: Optional[str] = None,
    key: str = "save_btn",
    show_expander: bool = True,
) -> Optional[str]:
    """
    渲染保存按钮和对话框
    
    在侧边栏或页面中显示保存按钮，点击后展开输入框允许用户
    输入自定义文件名和备注信息。
    
    Args:
        df: 要保存的 DataFrame
        category: 数据类别
        extra_data: 扩展数据（如绘图数据源字典）
        source_file: 原始数据来源描述
        key: Streamlit 组件的唯一键
        show_expander: 是否使用 expander 包装（默认 True）
    
    Returns:
        保存成功返回 dataset_id，否则返回 None
    
    Example:
        >>> dataset_id = render_save_button(
        ...     df=result_df,
        ...     category=DataCategory.EXTRACTION,
        ...     extra_data={"lvi_sources": ..., "rth_sources": ...},
        ...     key="extraction_save"
        ... )
    """
    if df is None or df.empty:
        st.warning("没有可保存的数据")
        return None
    
    store = _get_store()
    saved_id = None
    
    # 初始化 session state
    state_key = f"{key}_expanded"
    if state_key not in st.session_state:
        st.session_state[state_key] = False
    
    def _render_save_form():
        nonlocal saved_id
        
        st.markdown(f"**数据概览**")
        col1, col2 = st.columns(2)
        col1.metric("行数", len(df))
        col2.metric("列数", len(df.columns))
        
        # 初始化输入框的默认值
        custom_name_key = f"{key}_custom_name"
        note_key = f"{key}_note"
        
        # 自定义文件名输入
        custom_name = st.text_input(
            "自定义文件名（可选）",
            placeholder="留空则自动生成",
            key=custom_name_key,
            help="输入自定义文件名，不需要扩展名。留空将自动生成包含时间戳的文件名。"
        )
        
        # 备注输入
        note = st.text_area(
            "备注（可选）",
            placeholder="添加备注信息...",
            key=note_key,
            height=80,
            help="可以添加备注信息，方便后续查找和识别数据集。"
        )
        
        # 保存按钮
        if st.button("💾 确认保存", key=f"{key}_confirm", use_container_width=True):
            try:
                with st.spinner("正在保存..."):
                    dataset_id = store.save(
                        df=df,
                        category=category,
                        name=custom_name if custom_name.strip() else None,
                        custom_filename=custom_name if custom_name.strip() else None,
                        note=note if note.strip() else None,
                        extra_data=extra_data,
                        source_file=source_file,
                    )
                st.success(f"✅ 保存成功！")
                st.caption(f"数据集 ID: {dataset_id[:8]}...")
                saved_id = dataset_id
            except LocalStorageError as e:
                st.error(f"保存失败: {e}")
            except Exception as e:
                st.error(f"保存时发生错误: {e}")
    
    if show_expander:
        with st.expander("💾 保存数据", expanded=st.session_state[state_key]):
            _render_save_form()
    else:
        _render_save_form()
    
    return saved_id



def render_load_selector(
    category: Optional[DataCategory] = None,
    key: str = "load_select",
    show_details: bool = True,
    on_load_callback: Optional[callable] = None,
) -> Optional[Tuple[pd.DataFrame, DatasetMetadata, Optional[Dict[str, Any]]]]:
    """
    渲染数据集加载选择器
    
    显示数据集列表，允许用户选择并加载已保存的数据集。
    
    Args:
        category: 可选的数据类别筛选，为 None 时显示所有类别
        key: Streamlit 组件的唯一键
        show_details: 是否显示数据集详细信息
        on_load_callback: 加载成功后的回调函数，接收 (df, metadata, extra_data) 参数
    
    Returns:
        加载成功返回 (DataFrame, Metadata, ExtraData) 元组，否则返回 None
    
    Example:
        >>> result = render_load_selector(
        ...     category=DataCategory.EXTRACTION,
        ...     key="extraction_load"
        ... )
        >>> if result:
        ...     df, metadata, extra_data = result
    """
    store = _get_store()
    
    # 获取数据集列表
    try:
        datasets = store.list_datasets(category=category)
    except Exception as e:
        st.error(f"获取数据集列表失败: {e}")
        return None
    
    if not datasets:
        category_label = _get_category_label(category) if category else "任何"
        st.info(f"暂无{category_label}类型的已保存数据集")
        return None
    
    # 构建选择项
    options = ["-- 选择数据集 --"]
    option_map = {}  # 显示文本 -> dataset_id
    
    for meta in datasets:
        # 格式化显示文本
        time_str = _format_datetime(meta.created_at)
        category_label = _get_category_label(meta.category)
        display_text = f"{meta.name} | {category_label} | {meta.row_count}行 | {time_str}"
        options.append(display_text)
        option_map[display_text] = meta.id
    
    # 选择框
    selected = st.selectbox(
        "选择要加载的数据集",
        options,
        key=f"{key}_select",
        help="选择一个已保存的数据集进行加载"
    )
    
    if selected == "-- 选择数据集 --":
        return None
    
    dataset_id = option_map.get(selected)
    if not dataset_id:
        return None
    
    # 显示详细信息
    if show_details:
        # 找到对应的元数据
        meta = next((m for m in datasets if m.id == dataset_id), None)
        if meta:
            with st.container():
                st.markdown("**数据集详情**")
                col1, col2, col3 = st.columns(3)
                col1.caption(f"📁 类别: {_get_category_label(meta.category)}")
                col2.caption(f"📊 行数: {meta.row_count}")
                col3.caption(f"📅 创建: {_format_datetime(meta.created_at)}")
                
                if meta.columns:
                    st.caption(f"📋 列: {', '.join(meta.columns[:5])}{'...' if len(meta.columns) > 5 else ''}")
                
                if meta.note:
                    st.caption(f"📝 备注: {meta.note}")
                
                if meta.source_file:
                    st.caption(f"📄 来源: {meta.source_file}")
    
    # 加载按钮
    if st.button("📂 加载数据", key=f"{key}_load_btn", use_container_width=True):
        try:
            with st.spinner("正在加载..."):
                df, metadata, extra_data = store.load(dataset_id)
            
            st.success(f"✅ 加载成功！共 {len(df)} 行数据")
            
            # 调用回调函数
            if on_load_callback:
                on_load_callback(df, metadata, extra_data)
            
            return df, metadata, extra_data
            
        except DatasetNotFoundError as e:
            st.error(f"数据集不存在: {e}")
        except LocalStorageError as e:
            st.error(f"加载失败: {e}")
        except Exception as e:
            st.error(f"加载时发生错误: {e}")
    
    return None


def render_dataset_list(
    category: Optional[DataCategory] = None,
    key: str = "dataset_list",
    selectable: bool = False,
) -> List[str]:
    """
    渲染数据集列表（用于数据管理页面）
    
    显示数据集的详细列表，支持多选。
    
    Args:
        category: 可选的数据类别筛选
        key: Streamlit 组件的唯一键
        selectable: 是否支持多选
    
    Returns:
        如果 selectable=True，返回选中的 dataset_id 列表；否则返回空列表
    """
    store = _get_store()
    
    try:
        datasets = store.list_datasets(category=category)
    except Exception as e:
        st.error(f"获取数据集列表失败: {e}")
        return []
    
    if not datasets:
        st.info("暂无已保存的数据集")
        return []
    
    selected_ids = []
    
    # 显示统计信息
    st.markdown(f"**共 {len(datasets)} 个数据集**")
    
    # 按类别分组显示
    by_category: Dict[DataCategory, List[DatasetMetadata]] = {}
    for meta in datasets:
        if meta.category not in by_category:
            by_category[meta.category] = []
        by_category[meta.category].append(meta)
    
    for cat in DataCategory:
        if cat not in by_category:
            continue
        
        cat_datasets = by_category[cat]
        cat_label = _get_category_label(cat)
        
        with st.expander(f"📁 {cat_label} ({len(cat_datasets)})", expanded=True):
            for i, meta in enumerate(cat_datasets):
                col1, col2, col3, col4 = st.columns([0.5, 3, 1.5, 1])
                
                # 选择框（如果启用多选）
                if selectable:
                    is_selected = col1.checkbox(
                        "",
                        key=f"{key}_{meta.id}_select",
                        label_visibility="collapsed"
                    )
                    if is_selected:
                        selected_ids.append(meta.id)
                
                # 名称和信息
                with col2:
                    st.markdown(f"**{meta.name}**")
                    info_parts = [f"{meta.row_count} 行"]
                    if meta.note:
                        info_parts.append(meta.note[:30] + "..." if len(meta.note) > 30 else meta.note)
                    st.caption(" | ".join(info_parts))
                
                # 时间
                col3.caption(_format_datetime(meta.created_at))
                
                # 操作按钮（删除）
                if col4.button("🗑️", key=f"{key}_{meta.id}_delete", help="删除此数据集"):
                    try:
                        store.delete(meta.id)
                        st.success(f"已删除: {meta.name}")
                        st.rerun()
                    except LocalStorageError as e:
                        st.error(f"删除失败: {e}")
    
    return selected_ids



def render_export_section(
    selected_ids: List[str],
    key: str = "export",
) -> Optional[Path]:
    """
    渲染导出功能区域
    
    Args:
        selected_ids: 选中的数据集 ID 列表
        key: Streamlit 组件的唯一键
    
    Returns:
        导出成功返回文件路径，否则返回 None
    """
    if not selected_ids:
        st.info("请先选择要导出的数据集")
        return None
    
    store = _get_store()
    
    st.markdown(f"**已选择 {len(selected_ids)} 个数据集**")
    
    col1, col2 = st.columns(2)
    
    # Excel 导出
    with col1:
        include_summary = st.checkbox(
            "包含统计摘要",
            value=True,
            key=f"{key}_include_summary",
            help="导出时包含数据统计摘要 Sheet"
        )
        
        if st.button("📊 导出为 Excel", key=f"{key}_excel", use_container_width=True):
            try:
                with st.spinner("正在导出..."):
                    output_path = store.export_to_excel(
                        dataset_ids=selected_ids,
                        include_summary=include_summary
                    )
                st.success(f"✅ 导出成功！")
                st.caption(f"文件: {output_path.name}")
                
                # 提供下载链接
                with open(output_path, "rb") as f:
                    st.download_button(
                        label="📥 下载 Excel 文件",
                        data=f.read(),
                        file_name=output_path.name,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key=f"{key}_download_excel"
                    )
                return output_path
            except ExportError as e:
                st.error(f"导出失败: {e}")
            except Exception as e:
                st.error(f"导出时发生错误: {e}")
    
    # CSV 导出（仅支持单个数据集）
    with col2:
        if len(selected_ids) == 1:
            if st.button("📄 导出为 CSV", key=f"{key}_csv", use_container_width=True):
                try:
                    with st.spinner("正在导出..."):
                        output_path = store.export_to_csv(dataset_id=selected_ids[0])
                    st.success(f"✅ 导出成功！")
                    st.caption(f"文件: {output_path.name}")
                    
                    # 提供下载链接
                    with open(output_path, "rb") as f:
                        st.download_button(
                            label="📥 下载 CSV 文件",
                            data=f.read(),
                            file_name=output_path.name,
                            mime="text/csv",
                            key=f"{key}_download_csv"
                        )
                    return output_path
                except ExportError as e:
                    st.error(f"导出失败: {e}")
                except Exception as e:
                    st.error(f"导出时发生错误: {e}")
        else:
            st.caption("CSV 导出仅支持单个数据集")
    
    return None


def render_send_to_section(
    selected_ids: List[str],
    key: str = "send_to",
) -> Optional[Dict[str, Any]]:
    """
    渲染"发送到"功能区域
    
    允许将选中的数据集发送到指定模块，进行必要的格式转换。
    
    Args:
        selected_ids: 选中的数据集 ID 列表
        key: Streamlit 组件的唯一键
    
    Returns:
        发送成功返回包含目标模块和转换结果的字典，否则返回 None
    """
    if not selected_ids:
        st.info("请先选择要发送的数据集")
        return None
    
    if len(selected_ids) > 1:
        st.warning("⚠️ 发送到功能目前仅支持单个数据集，请只选择一个数据集")
        return None
    
    store = _get_store()
    dataset_id = selected_ids[0]
    
    # 加载数据集
    try:
        df, metadata, extra_data = store.load(dataset_id)
    except LocalStorageError as e:
        st.error(f"加载数据集失败: {e}")
        return None
    
    source_category = metadata.category
    
    # 获取可发送的目标模块
    target_modules = get_sendable_modules(source_category)
    
    if not target_modules:
        st.info("没有可发送的目标模块")
        return None
    
    st.markdown(f"**当前数据集:** {metadata.name}")
    st.caption(f"类别: {_get_category_label(source_category)} | 行数: {metadata.row_count}")
    
    # 目标模块选择
    target_options = {get_module_display_name(cat): cat for cat in target_modules}
    selected_target_label = st.selectbox(
        "选择目标模块",
        list(target_options.keys()),
        key=f"{key}_target_select",
        help="选择要将数据发送到的目标模块"
    )
    
    target_category = target_options.get(selected_target_label)
    
    if target_category:
        # 检查兼容性
        compatibility = check_column_compatibility(df, source_category, target_category)
        
        # 显示兼容性信息
        with st.expander("📋 兼容性检查", expanded=True):
            if compatibility.is_compatible:
                st.success("✅ 数据格式兼容")
            else:
                st.warning("⚠️ 数据格式部分不兼容")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**匹配的列:**")
                if compatibility.matched_columns:
                    st.caption(", ".join(compatibility.matched_columns[:10]))
                    if len(compatibility.matched_columns) > 10:
                        st.caption(f"...等 {len(compatibility.matched_columns)} 列")
                else:
                    st.caption("无")
            
            with col2:
                if compatibility.suggested_mappings:
                    st.markdown("**列名映射:**")
                    for src, tgt in compatibility.suggested_mappings.items():
                        st.caption(f"{src} → {tgt}")
            
            if compatibility.warnings:
                st.markdown("**注意事项:**")
                for warning in compatibility.warnings:
                    st.caption(f"⚠️ {warning}")
        
        # 发送选项
        col1, col2 = st.columns(2)
        with col1:
            apply_mappings = st.checkbox(
                "应用列名映射",
                value=True,
                key=f"{key}_apply_mappings",
                help="将源列名转换为目标模块的列名"
            )
        with col2:
            preserve_extra = st.checkbox(
                "保留额外列",
                value=True,
                key=f"{key}_preserve_extra",
                help="保留目标模块不需要的列"
            )
        
        # 发送按钮
        if st.button("📤 发送到模块", key=f"{key}_send_btn", use_container_width=True, type="primary"):
            try:
                # 转换数据
                converted_df, result = convert_dataframe_for_module(
                    df=df,
                    source_category=source_category,
                    target_category=target_category,
                    apply_mappings=apply_mappings,
                    preserve_extra_columns=preserve_extra
                )
                
                # 存储到 session_state 供目标模块使用
                session_key = f"shared_data_to_{target_category.value}"
                st.session_state[session_key] = {
                    "df": converted_df,
                    "source_metadata": metadata,
                    "source_category": source_category,
                    "target_category": target_category,
                    "compatibility": result.to_dict(),
                    "timestamp": datetime.now().isoformat(),
                }
                
                st.success(f"✅ 数据已准备发送到 {get_module_display_name(target_category)}")
                st.info(f"💡 请打开 {get_module_display_name(target_category)} 页面，在加载历史数据中选择「从其他模块接收」")
                
                return {
                    "target_category": target_category,
                    "converted_df": converted_df,
                    "compatibility": result,
                }
                
            except ValueError as e:
                st.error(f"数据转换失败: {e}")
            except Exception as e:
                st.error(f"发送失败: {e}")
    
    return None


def render_receive_shared_data(
    target_category: DataCategory,
    key: str = "receive_shared",
    on_receive_callback: Optional[callable] = None,
) -> Optional[Tuple[pd.DataFrame, Dict[str, Any]]]:
    """
    渲染接收共享数据的组件
    
    检查是否有其他模块发送的数据，并提供接收选项。
    
    Args:
        target_category: 当前模块的类别
        key: Streamlit 组件的唯一键
        on_receive_callback: 接收数据后的回调函数
    
    Returns:
        接收成功返回 (DataFrame, 元数据字典)，否则返回 None
    """
    session_key = f"shared_data_to_{target_category.value}"
    
    if session_key not in st.session_state:
        return None
    
    shared_data = st.session_state[session_key]
    
    if not shared_data:
        return None
    
    source_category = shared_data.get("source_category")
    source_metadata = shared_data.get("source_metadata")
    df = shared_data.get("df")
    
    if df is None or source_metadata is None:
        return None
    
    st.markdown("---")
    st.markdown("### 📥 接收共享数据")
    
    with st.container(border=True):
        st.markdown(f"**来自:** {get_module_display_name(source_category)}")
        st.markdown(f"**数据集:** {source_metadata.name}")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("行数", len(df))
        col2.metric("列数", len(df.columns))
        col3.caption(f"发送时间: {_format_datetime(shared_data.get('timestamp', ''))}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("✅ 接收数据", key=f"{key}_accept", use_container_width=True, type="primary"):
                # 调用回调函数
                if on_receive_callback:
                    on_receive_callback(df, source_metadata, shared_data.get("compatibility", {}))
                
                # 清除共享数据
                del st.session_state[session_key]
                
                st.success("✅ 数据接收成功！")
                return df, shared_data
        
        with col2:
            if st.button("❌ 忽略", key=f"{key}_ignore", use_container_width=True):
                del st.session_state[session_key]
                st.info("已忽略共享数据")
                st.rerun()
    
    return None


def render_data_manager_page() -> None:
    """
    渲染数据管理页面
    
    提供完整的数据集管理界面，包括：
    - 数据集列表（按类别分组）
    - 批量选择和删除
    - 导出功能（Excel/CSV）
    - 发送到其他模块功能
    - 数据预览
    
    Example:
        在 Streamlit 页面中调用:
        >>> render_data_manager_page()
    """
    st.title("📁 数据管理")
    st.markdown("管理已保存的数据集，支持查看、删除、导出和跨模块共享操作。")
    
    store = _get_store()
    
    # 类别筛选
    st.markdown("---")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        category_options = ["全部"] + [_get_category_label(cat) for cat in DataCategory]
        selected_category_label = st.selectbox(
            "筛选类别",
            category_options,
            key="dm_category_filter"
        )
    
    # 确定筛选的类别
    selected_category = None
    if selected_category_label != "全部":
        for cat in DataCategory:
            if _get_category_label(cat) == selected_category_label:
                selected_category = cat
                break
    
    with col2:
        if st.button("🔄 刷新列表", key="dm_refresh", use_container_width=True):
            st.rerun()
    
    st.markdown("---")
    
    # 数据集列表（支持多选）
    selected_ids = render_dataset_list(
        category=selected_category,
        key="dm_list",
        selectable=True
    )
    
    # 操作区域（导出和发送到）
    if selected_ids:
        st.markdown("---")
        
        # 使用标签页组织导出和发送功能
        tab_export, tab_send = st.tabs(["📤 导出", "📨 发送到其他模块"])
        
        with tab_export:
            render_export_section(selected_ids, key="dm_export")
        
        with tab_send:
            render_send_to_section(selected_ids, key="dm_send_to")
    
    # 数据预览区域
    st.markdown("---")
    st.subheader("👁️ 数据预览")
    
    # 获取所有数据集用于预览选择
    try:
        all_datasets = store.list_datasets(category=selected_category)
    except Exception:
        all_datasets = []
    
    if all_datasets:
        preview_options = ["-- 选择数据集预览 --"] + [
            f"{meta.name} ({meta.row_count}行)" for meta in all_datasets
        ]
        preview_map = {
            f"{meta.name} ({meta.row_count}行)": meta.id for meta in all_datasets
        }
        
        preview_selected = st.selectbox(
            "选择要预览的数据集",
            preview_options,
            key="dm_preview_select"
        )
        
        if preview_selected != "-- 选择数据集预览 --":
            dataset_id = preview_map.get(preview_selected)
            if dataset_id:
                try:
                    df, metadata, _ = store.load(dataset_id)
                    
                    # 显示元数据
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("行数", metadata.row_count)
                    col2.metric("列数", len(metadata.columns))
                    col3.metric("类别", _get_category_label(metadata.category))
                    col4.metric("创建时间", _format_datetime(metadata.created_at))
                    
                    if metadata.note:
                        st.info(f"📝 备注: {metadata.note}")
                    
                    # 显示数据预览
                    st.dataframe(df, use_container_width=True, height=400)
                    
                except LocalStorageError as e:
                    st.error(f"加载预览失败: {e}")
    else:
        st.info("暂无数据集可预览")
