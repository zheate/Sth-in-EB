# UI 组件模块
"""包含 Streamlit UI 渲染相关的辅助函数"""

import io
from typing import Iterable, List, Optional
import pandas as pd, streamlit as st
from .constants import SHELL_COLUMN, TEST_TYPE_COLUMN
from .file_utils import ensure_xlsx_suffix


def show_toast(message: str, icon: str = "ℹ️", duration: int = 2000) -> None:
    """显示 toast 消息"""
    st.toast(message, icon=icon)


def trigger_scroll_if_needed(anchor_id: str) -> None:
    """将页面滚动到指定锚点"""
    if st.session_state.get("pending_scroll_target") != anchor_id: return
    st.markdown(f'<script>document.getElementById("{anchor_id}")?.scrollIntoView({{behavior:"smooth",block:"start"}})</script>', unsafe_allow_html=True)
    st.session_state.pending_scroll_target = None


def render_extraction_results_section(container, result_df: Optional[pd.DataFrame], error_messages: Optional[Iterable[str]],
                                      info_messages: Optional[Iterable[str]], *, entity_label: str = "壳体") -> None:
    """渲染数据提取结果展示区段"""
    if result_df is None: return
    errors, infos = list(error_messages or []), list(info_messages or [])

    with container:
        st.markdown('<div id="results"></div>', unsafe_allow_html=True)
        st.markdown("---")
        st.subheader("📊 抽取结果概览")

        c1, c2, c3 = st.columns(3)
        shell_s = result_df.get(SHELL_COLUMN, pd.Series(dtype=str))
        test_s = result_df.get(TEST_TYPE_COLUMN, pd.Series(dtype=str))
        c1.metric("记录数", len(result_df))
        c2.metric(f"{entity_label}数量", int(shell_s.nunique()))
        c3.metric("站别数量", int(test_s.nunique()))

        with st.expander("查看抽取结果明细", expanded=True):
            st.dataframe(result_df, use_container_width=True, hide_index=False, height=max(140, min(600, len(result_df) * 34 + 60)))

        st.markdown("---")
        st.subheader("💾 导出数据")
        cn, cb = st.columns([3, 1])
        name_input = cn.text_input("文件名称", value="combined_subset", help="输入文件名（无需扩展名，自动添加.xlsx)", key="download_name_input")
        cb.markdown("<div style='margin-top:32px'></div>", unsafe_allow_html=True)
        if cb.button("💾 生成下载文件", key="download_btn"): _handle_download_request(result_df, name_input)

        payload, counter = st.session_state.get("download_payload"), st.session_state.get("download_request_counter", 0)
        if payload and counter:
            st.download_button("📥 点击下载保存文件", data=payload, file_name=st.session_state.get("download_filename", "combined_subset.xlsx"),
                              mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key=f"download_button_{counter}", use_container_width=True)

        if errors or infos:
            c1, c2 = st.columns(2)
            if errors:
                with c1, st.expander(f"展开查看失败详情（{len(errors)} 条）", expanded=False):
                    for m in errors: st.markdown(f"- {m}")
            if infos:
                with c2, st.expander(f"处理提示（{len(infos)} 条）", expanded=False):
                    for m in infos: st.markdown(f"- {m}")


def _handle_download_request(result_df: pd.DataFrame, name_input: str) -> None:
    """处理下载请求"""
    try: filename = ensure_xlsx_suffix((name_input or "").strip() or "combined_subset.xlsx")
    except ValueError: show_toast("请输入有效的文件名", icon="⚠️"); return
    
    buf = io.BytesIO()
    try:
        with pd.ExcelWriter(buf, engine="xlsxwriter") as w: result_df.to_excel(w, index=False, sheet_name="Sheet1")
    except ImportError:
        with pd.ExcelWriter(buf, engine="openpyxl") as w: result_df.to_excel(w, index=False, sheet_name="Sheet1")
    buf.seek(0)
    st.session_state.download_payload = buf.getvalue()
    st.session_state.download_filename = filename
    st.session_state.download_request_counter = st.session_state.get("download_request_counter", 0) + 1
    show_toast(f"数据已准备，请点击下方按钮下载：{filename}", icon="📁")


def parse_folder_entries(raw: str) -> List[str]:
    """解析文件夹输入"""
    return [e.strip() for line in raw.replace("，", "\n").splitlines() if (e := line.strip())]


def parse_current_points(raw: str) -> Optional[List[float]]:
    """解析电流点输入，'a'/'A' 返回 None 表示全部"""
    text = raw.strip()
    if text.lower() == "a": return None

    currents: List[float] = []
    for line in text.replace("，", ",").replace("～", "~").splitlines():
        for piece in line.split(","):
            p = piece.strip().replace("～", "~")
            if not p: continue

            # 空格分隔
            if "~" not in p and "-" not in p[1:]:
                tokens = p.split()
                if len(tokens) > 1:
                    try: currents.extend(float(t) for t in tokens)
                    except ValueError: raise ValueError(f"无法解析电流值: {piece}")
                    continue

            # 范围
            rt = p.split("~", 1) if "~" in p else ([p[:i], p[i+1:]] if (i := p.find("-", 1)) != -1 else None)
            if rt:
                try: s, e = float(rt[0].strip()), float(rt[1].strip())
                except ValueError: raise ValueError(f"无法解析电流范围: {piece}")
                if s.is_integer() and e.is_integer():
                    step = 1 if int(e) >= int(s) else -1
                    currents.extend(float(v) for v in range(int(s), int(e) + step, step))
                else: currents.extend([s, e])
                continue

            try: currents.append(float(p))
            except ValueError: raise ValueError(f"无法解析电流值: {piece}")
    return currents


def init_session_state() -> None:
    """初始化 session state 默认值"""
    defaults = {'pending_scroll_target': None, 'show_multi_station': False, 'show_boxplot': False,
                'show_single_analysis': False, 'show_multi_power': False, 'download_payload': None,
                'download_filename': "combined_subset.xlsx", 'download_request_counter': 0,
                'lvi_plot_sources': {}, 'rth_plot_sources': {}}
    for k, v in defaults.items():
        if k not in st.session_state: st.session_state[k] = v
