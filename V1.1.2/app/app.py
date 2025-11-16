import os
import platform
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd
import streamlit as st

from config import APP_ICON, APP_TITLE, DEFAULT_DATA_FOLDER, PAGE_LAYOUT, SIDEBAR_STATE

REPORT_PREFIX = "常用测试数据报表"
ALLOWED_REPORT_EXTENSIONS = {".xlsx", ".xls"}
DATA_FILE_EXTENSIONS = ALLOWED_REPORT_EXTENSIONS | {".csv"}
HOME_REPORT_SESSION_KEY = "test_analysis_home_reports"
APP_ROOT = Path(__file__).resolve().parent
PAGES_ROOT = APP_ROOT / "pages"


def _ensure_session_defaults() -> None:
    if "recent_folders" not in st.session_state:
        st.session_state["recent_folders"] = []
    if "folder_path_input" not in st.session_state:
        st.session_state["folder_path_input"] = DEFAULT_DATA_FOLDER
    if HOME_REPORT_SESSION_KEY not in st.session_state:
        st.session_state[HOME_REPORT_SESSION_KEY] = []


def _use_recent_folder() -> None:
    selected = st.session_state.get("recent_folder_select")
    if selected:
        st.session_state.folder_path_input = selected


def _clear_recent_folders() -> None:
    st.session_state.recent_folders = []
    st.session_state.pop("recent_folder_select", None)


def _find_data_files(directory: Path, recursive: bool = False) -> List[Path]:
    matched: List[Path] = []
    if recursive:
        try:
            for root, _, filenames in os.walk(directory):
                for name in filenames:
                    if os.path.splitext(name)[1].lower() in DATA_FILE_EXTENSIONS:
                        matched.append(Path(root) / name)
        except OSError:
            return matched
    else:
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    try:
                        if not entry.is_file():
                            continue
                    except OSError:
                        continue
                    if os.path.splitext(entry.name)[1].lower() in DATA_FILE_EXTENSIONS:
                        matched.append(Path(entry.path))
        except OSError:
            return matched
    return matched


def _format_file_table(files: Iterable[Path]) -> tuple[pd.DataFrame, List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    report_candidates: List[Dict[str, Any]] = []

    file_stats: List[tuple[Path, os.stat_result]] = []
    for file in files:
        try:
            file_stat = file.stat()
        except OSError:
            continue
        file_stats.append((file, file_stat))

    for file, file_stat in sorted(file_stats, key=lambda item: item[1].st_mtime, reverse=True):
        rows.append(
            {
                "文件名": file.name,
                "类型": file.suffix.upper(),
                "大小": f"{file_stat.st_size / 1024:.1f} KB",
                "修改时间": file_stat.st_mtime,
            }
        )
        if file.suffix.lower() in ALLOWED_REPORT_EXTENSIONS and file.name.startswith(REPORT_PREFIX):
            try:
                resolved_file = str(file.resolve())
            except OSError:
                resolved_file = str(file)
            report_candidates.append(
                {
                    "name": file.name,
                    "path": resolved_file,
                    "modified": file_stat.st_mtime,
                }
            )

    df = pd.DataFrame(rows)
    if not df.empty:
        df["修改时间"] = pd.to_datetime(df["修改时间"], unit="s").dt.strftime("%Y-%m-%d %H:%M:%S")
    return df, report_candidates

def render_home_page() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon=APP_ICON,
        layout=PAGE_LAYOUT,
        initial_sidebar_state=SIDEBAR_STATE,
    )

    # 注入 polyfill 以支持旧版浏览器
    st.components.v1.html(
        """
        <script>
        if (typeof structuredClone === 'undefined') {
            window.structuredClone = function(obj) {
                return JSON.parse(JSON.stringify(obj));
            };
        }
        </script>
        """,
        height=0,
    )

    _ensure_session_defaults()

    st.title("🔬 光耦数据分析系统")
    st.markdown("### 🚀 快捷入口")

    shortcut_cols = st.columns(3)
    with shortcut_cols[0]:
        if st.button("📊 测试数据分析", use_container_width=True, type="primary"):
            st.switch_page("pages/TestAnalysis.py")
    with shortcut_cols[1]:
        if st.button("📥 数据提取", use_container_width=True, type="primary"):
            st.switch_page("pages/Data_fetch.py")
    with shortcut_cols[2]:
        if st.button("📈 进度追踪", use_container_width=True, type="primary"):
            st.switch_page("pages/Progress.py")

    st.markdown("---")
    st.markdown("### 🛠 系统状态")

    available_pages = [path for path in PAGES_ROOT.glob("*.py")]
    status_cols = st.columns(4)
    with status_cols[0]:
        st.metric("Python 版本", platform.python_version())
    with status_cols[1]:
        st.metric("Streamlit 版本", st.__version__)
    with status_cols[2]:
        st.metric("Pandas 版本", pd.__version__)
    with status_cols[3]:
        st.metric("可用模块数", len(available_pages))

    with st.expander("运行环境详情"):
        data_dir = APP_ROOT / "data"
        package_df = pd.DataFrame(
            [
                {"组件": "Python", "版本": platform.python_version()},
                {"组件": "Streamlit", "版本": st.__version__},
                {"组件": "Pandas", "版本": pd.__version__},
            ]
        )
        st.markdown(f"- 应用目录：`{APP_ROOT}`")
        st.markdown(
            f"- 数据目录：`{data_dir}`"
            f"{' ✅' if data_dir.exists() else ' （未创建）'}"
        )
        st.markdown(f"- 页面脚本数量：`{len(available_pages)}`")
        st.dataframe(package_df, hide_index=True, width="stretch", height=150)

    st.markdown("---")
    st.markdown("### 📁 数据文件浏览")

    # 自动加载 ./data 目录
    data_dir = APP_ROOT / "data"
    if "auto_loaded_data_dir" not in st.session_state:
        st.session_state["auto_loaded_data_dir"] = False
        if data_dir.exists() and data_dir.is_dir():
            try:
                matched_files = _find_data_files(data_dir)
                if matched_files:
                    st.info(f"📂 自动加载 `./data` 目录，找到 {len(matched_files)} 个数据文件")
                    df_files, home_report_candidates = _format_file_table(matched_files)
                    if not df_files.empty:
                        st.dataframe(
                            df_files,
                            width="stretch",
                            hide_index=True,
                            height=min(400, len(df_files) * 35 + 38),
                        )
                    if home_report_candidates:
                        st.session_state[HOME_REPORT_SESSION_KEY] = home_report_candidates
                        st.caption(
                            f"📂 已识别 {len(home_report_candidates)} 个以「{REPORT_PREFIX}」开头的 Excel 报表"
                        )
                st.session_state["auto_loaded_data_dir"] = True
            except Exception as error:
                st.warning(f"自动加载 ./data 目录时出错: {error}")

    recent_folders = st.session_state.get("recent_folders", [])
    if recent_folders:
        st.markdown("#### 🕘 最近使用的路径")
        st.selectbox(
            "选择一个路径快速填充输入框",
            options=recent_folders,
            key="recent_folder_select",
        )
        action_cols = st.columns([1, 1, 4])
        with action_cols[0]:
            st.button(
                "使用该路径",
                key="use_recent_folder",
                width="stretch",
                on_click=_use_recent_folder,
            )
        with action_cols[1]:
            st.button(
                "清空记录",
                key="clear_recent_folders",
                width="stretch",
                on_click=_clear_recent_folders,
            )

    col_path, col_btn = st.columns([3, 1])
    with col_path:
        folder_path = st.text_input(
            "浏览器下载文件夹路径",
            key="folder_path_input",
            placeholder=f"默认: {DEFAULT_DATA_FOLDER}",
            help="浏览器按 Ctrl+J，复制下载文件所在的文件夹路径到此处",
        )
    with col_btn:
        st.markdown("<div style='margin-top: 32px;'></div>", unsafe_allow_html=True)
        search_btn = st.button("🔍 查找文件", width="stretch")

    if folder_path and (search_btn or st.session_state.get("recent_folder_select") == folder_path):
        try:
            search_path = Path(folder_path).expanduser()
            if not search_path.exists():
                st.error(f"路径不存在：{folder_path}")
            elif not search_path.is_dir():
                st.error(f"路径不是文件夹：{folder_path}")
            else:
                resolved_path = search_path.resolve()
                normalized = str(resolved_path)
                existing = st.session_state.get("recent_folders", [])
                filtered_existing = [path for path in existing if path != normalized]
                st.session_state.recent_folders = [normalized] + filtered_existing[:4]

                matched_files = _find_data_files(resolved_path)
                st.session_state[HOME_REPORT_SESSION_KEY] = []

                if matched_files:
                    st.success(f"在 `{normalized}` 中找到 {len(matched_files)} 个数据文件")

                    df_files, home_report_candidates = _format_file_table(matched_files)
                    if not df_files.empty:
                        st.dataframe(
                            df_files,
                            width="stretch",
                            hide_index=True,
                            height=min(400, len(df_files) * 35 + 38),
                        )
                    if home_report_candidates:
                        st.session_state[HOME_REPORT_SESSION_KEY] = home_report_candidates
                        st.caption(
                            f"📂 已识别 {len(home_report_candidates)} 个以“{REPORT_PREFIX}”开头的 Excel 报表，可在测试数据分析页面直接加载"
                        )

                    if st.checkbox("🔎 递归搜索子文件夹", key="recursive_search_toggle"):
                        all_files_recursive = _find_data_files(resolved_path, recursive=True)

                        if len(all_files_recursive) > len(matched_files):
                            st.info(f"递归搜索找到 {len(all_files_recursive)} 个文件（包含子文件夹）")

                            file_stats_recursive: List[tuple[Path, os.stat_result]] = []
                            for file in all_files_recursive:
                                try:
                                    file_stat = file.stat()
                                except OSError:
                                    continue
                                file_stats_recursive.append((file, file_stat))

                            top_entries = sorted(
                                file_stats_recursive,
                                key=lambda item: item[1].st_mtime,
                                reverse=True,
                            )[:50]
                            file_data_recursive = []
                            report_candidates_recursive: List[Dict[str, Any]] = []
                            for file, file_stat in top_entries:
                                try:
                                    relative_path = file.relative_to(resolved_path)
                                except ValueError:
                                    relative_path = file
                                file_data_recursive.append(
                                    {
                                        "相对路径": str(relative_path),
                                        "类型": file.suffix.upper(),
                                        "大小": f"{file_stat.st_size / 1024:.1f} KB",
                                        "修改时间": file_stat.st_mtime,
                                    }
                                )
                                if (
                                    file.suffix.lower() in ALLOWED_REPORT_EXTENSIONS
                                    and file.name.startswith(REPORT_PREFIX)
                                ):
                                    try:
                                        resolved_file = str(file.resolve())
                                    except OSError:
                                        resolved_file = str(file)
                                    report_candidates_recursive.append(
                                        {
                                            "name": file.name,
                                            "path": resolved_file,
                                            "modified": file_stat.st_mtime,
                                        }
                                    )

                            if file_data_recursive:
                                df_files_recursive = pd.DataFrame(file_data_recursive)
                                df_files_recursive["修改时间"] = pd.to_datetime(
                                    df_files_recursive["修改时间"], unit="s"
                                ).dt.strftime("%Y-%m-%d %H:%M:%S")

                                st.dataframe(
                                    df_files_recursive,
                                    width="stretch",
                                    hide_index=True,
                                    height=400,
                                )
                            if report_candidates_recursive:
                                st.session_state[HOME_REPORT_SESSION_KEY] = report_candidates_recursive
                                st.caption(
                                    f"📂 已更新主页可用报表：{len(report_candidates_recursive)} 个文件"
                                )
                else:
                    st.warning(f"在 `{folder_path}` 中未找到数据文件（支持 .xlsx、.xls、.csv）")
                    st.session_state[HOME_REPORT_SESSION_KEY] = []

        except Exception as error:
            st.error(f"读取文件夹时出错: {error}")


def main() -> None:
    pages = [
        st.Page(render_home_page, title=APP_TITLE, icon=APP_ICON, default=True),
        st.Page(PAGES_ROOT / "Data_fetch.py", title="数据提取", icon="📥"),
        st.Page(PAGES_ROOT / "Progress.py", title="进度追踪", icon="📈"),
        st.Page(PAGES_ROOT / "TestAnalysis.py", title="测试数据分析", icon="📊"),
        st.Page(PAGES_ROOT / "DataAnalysis.py", title="数据集分析", icon="📁"),
        st.Page(PAGES_ROOT / "BFD_Calculator.py", title="后焦距计算器", icon="🔧"),
        st.Page(PAGES_ROOT / "NA_Calculator.py", title="数值孔径计算器", icon="🔧")
    ]

    page = st.navigation(pages, position="sidebar", expanded=True)
    page.run()


if __name__ == "__main__":
    main()
