# UI 组件模块
"""
包含 Streamlit UI 渲染相关的辅助函数
"""

import io
from typing import Iterable, List, Optional

import pandas as pd
import streamlit as st

from .constants import SHELL_COLUMN, TEST_TYPE_COLUMN
from .file_utils import ensure_xlsx_suffix


def show_toast(message: str, icon: str = "ℹ️", duration: int = 2000) -> None:
    """
    显示 toast 消息。
    
    Args:
        message: 消息内容
        icon: 图标
        duration: 持续时间（毫秒）
    """
    st.toast(message, icon=icon)


def trigger_scroll_if_needed(anchor_id: str) -> None:
    """
    将页面滚动到指定锚点。
    
    Args:
        anchor_id: 锚点 ID
    """
    pending = st.session_state.get("pending_scroll_target")
    if pending != anchor_id:
        return

    st.markdown(
        f"""
        <script>
        const anchor = document.getElementById("{anchor_id}");
        if (anchor) {{
            anchor.scrollIntoView({{ behavior: "smooth", block: "start" }});
        }}
        </script>
        """,
        unsafe_allow_html=True,
    )
    st.session_state.pending_scroll_target = None


def render_extraction_results_section(
    container,
    result_df: Optional[pd.DataFrame],
    error_messages: Optional[Iterable[str]],
    info_messages: Optional[Iterable[str]],
    *,
    entity_label: str = "壳体",
) -> None:
    """
    渲染数据提取结果展示区段。
    
    Args:
        container: Streamlit 容器
        result_df: 结果 DataFrame
        error_messages: 错误消息列表
        info_messages: 信息消息列表
        entity_label: 实体标签（壳体/芯片）
    """
    if result_df is None:
        return

    errors = list(error_messages or [])
    infos = list(info_messages or [])

    with container:
        st.markdown('<div id="results"></div>', unsafe_allow_html=True)
        st.markdown("---")
        st.subheader("📊 抽取结果概览")

        # 概览指标
        overview_cols = st.columns(3)
        shell_series = (
            result_df[SHELL_COLUMN]
            if SHELL_COLUMN in result_df.columns
            else pd.Series(dtype=str)
        )
        test_series = (
            result_df[TEST_TYPE_COLUMN]
            if TEST_TYPE_COLUMN in result_df.columns
            else pd.Series(dtype=str)
        )
        
        with overview_cols[0]:
            st.metric("记录数", len(result_df))
        with overview_cols[1]:
            st.metric(f"{entity_label}数量", int(shell_series.nunique()))
        with overview_cols[2]:
            st.metric("站别数量", int(test_series.nunique()))

        # 结果明细
        with st.expander("查看抽取结果明细", expanded=True):
            row_count = len(result_df)
            table_height = max(140, min(600, row_count * 34 + 60))
            st.dataframe(
                result_df,
                use_container_width=True,
                hide_index=False,
                height=table_height,
            )

        st.markdown("---")
        st.subheader("💾 导出数据")

        col_name, col_btn = st.columns([3, 1])
        with col_name:
            download_name_input = st.text_input(
                "文件名称",
                value="combined_subset",
                help="输入文件名（无需扩展名，自动添加.xlsx)",
                key="download_name_input",
            )
        with col_btn:
            st.markdown("<div style='margin-top: 32px;'></div>", unsafe_allow_html=True)
            download_requested = st.button("💾 生成下载文件", key="download_btn")

        if download_requested:
            _handle_download_request(result_df, download_name_input)

        # 显示下载按钮
        download_payload = st.session_state.get("download_payload")
        download_counter = st.session_state.get("download_request_counter", 0)
        if download_payload and download_counter:
            st.download_button(
                "📥 点击下载保存文件",
                data=download_payload,
                file_name=st.session_state.get("download_filename", "combined_subset.xlsx"),
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key=f"download_button_{download_counter}",
                use_container_width=True,
            )

        # 错误和信息提示
        if errors or infos:
            col1, col2 = st.columns(2)

            if errors:
                with col1:
                    with st.expander(f"展开查看失败详情（{len(errors)} 条）", expanded=False):
                        for message in errors:
                            st.markdown(f"- {message}")

            if infos:
                with col2:
                    with st.expander(f"处理提示（{len(infos)} 条）", expanded=False):
                        for message in infos:
                            st.markdown(f"- {message}")


def _handle_download_request(result_df: pd.DataFrame, download_name_input: str) -> None:
    """处理下载请求"""
    default_download_name = "combined_subset.xlsx"
    requested_name = (download_name_input or "").strip()
    
    try:
        download_filename = ensure_xlsx_suffix(requested_name or default_download_name)
    except ValueError:
        show_toast("请输入有效的文件名", icon="⚠️")
        return
    
    buffer = io.BytesIO()
    try:
        with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
            result_df.to_excel(writer, index=False, sheet_name="Sheet1")
    except ImportError:
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            result_df.to_excel(writer, index=False, sheet_name="Sheet1")
    buffer.seek(0)

    st.session_state.download_payload = buffer.getvalue()
    st.session_state.download_filename = download_filename
    st.session_state.download_request_counter = (
        st.session_state.get("download_request_counter", 0) + 1
    )
    show_toast(f"数据已准备，请点击下方按钮下载：{download_filename}", icon="📁")


def parse_folder_entries(raw_folders: str) -> List[str]:
    """
    解析文件夹输入。
    
    Args:
        raw_folders: 原始输入字符串
        
    Returns:
        解析后的条目列表
    """
    entries: List[str] = []
    for line in raw_folders.replace("，", "\n").splitlines():
        entry = line.strip()
        if entry:
            entries.append(entry)
    return entries


def parse_current_points(raw_points: str) -> Optional[List[float]]:
    """
    解析电流点输入。
    
    Args:
        raw_points: 原始输入字符串
        
    Returns:
        电流点列表，输入 'a' 或 'A' 时返回 None 表示全部
        
    Raises:
        ValueError: 解析失败时
    """
    text = raw_points.strip()
    if text.lower() == "a":
        return None

    currents: List[float] = []
    cleaned = text.replace("，", ",").replace("～", "~")

    for line in cleaned.splitlines():
        for piece in line.split(","):
            piece = piece.strip()
            if not piece:
                continue

            normalized = piece.replace("～", "~")

            # 空格分隔的多个值
            if "~" not in normalized and "-" not in normalized[1:]:
                space_tokens = [token for token in normalized.split() if token]
                if len(space_tokens) > 1:
                    try:
                        currents.extend(float(token) for token in space_tokens)
                    except ValueError as exc:
                        raise ValueError(f"无法解析电流值: {piece}") from exc
                    continue

            # 范围表示
            range_tokens: Optional[List[str]] = None
            if "~" in normalized:
                range_tokens = normalized.split("~", 1)
            else:
                hyphen_index = normalized.find("-", 1)
                if hyphen_index != -1:
                    range_tokens = [normalized[:hyphen_index], normalized[hyphen_index + 1:]]

            if range_tokens:
                start_str, end_str = [token.strip() for token in range_tokens]
                try:
                    start = float(start_str)
                    end = float(end_str)
                except ValueError as exc:
                    raise ValueError(f"无法解析电流范围: {piece}") from exc

                if start.is_integer() and end.is_integer():
                    start_int = int(start)
                    end_int = int(end)
                    step = 1 if end_int >= start_int else -1
                    for value in range(start_int, end_int + step, step):
                        currents.append(float(value))
                else:
                    currents.extend([start, end])
                continue

            # 单个值
            try:
                currents.append(float(normalized))
            except ValueError as exc:
                raise ValueError(f"无法解析电流值: {piece}") from exc

    return currents


def init_session_state() -> None:
    """初始化 session state 默认值"""
    defaults = {
        'pending_scroll_target': None,
        'show_multi_station': False,
        'show_boxplot': False,
        'show_single_analysis': False,
        'show_multi_power': False,
        'download_payload': None,
        'download_filename': "combined_subset.xlsx",
        'download_request_counter': 0,
        'lvi_plot_sources': {},
        'rth_plot_sources': {},
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value
