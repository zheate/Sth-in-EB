import platform
from pathlib import Path

import pandas as pd
import streamlit as st

from config import APP_ICON, APP_TITLE, PAGE_LAYOUT, SIDEBAR_STATE

APP_ROOT = Path(__file__).resolve().parent
PAGES_ROOT = APP_ROOT / "pages"




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

    st.title("🔬 光耦数据分析系统")

    st.markdown("### 🛠 系统状态")

    available_pages = [path for path in PAGES_ROOT.glob("*.py")]
    status_cols = st.columns(5)
    with status_cols[0]:
        st.metric("Python 版本", platform.python_version())
    with status_cols[1]:
        st.metric("Streamlit 版本", st.__version__)
    with status_cols[2]:
        st.metric("Pandas 版本", pd.__version__)
    with status_cols[3]:
        st.metric("可用模块数", len(available_pages))
    with status_cols[4]:
        st.metric("Chrome 最低版本", "118+")

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
        st.markdown("- 浏览器要求：Chrome 118 及以上版本（推荐使用最新稳定版）")
        st.markdown(f"- 页面脚本数量：`{len(available_pages)}`")
        st.dataframe(package_df, hide_index=True, width='stretch', height=150)

    st.markdown("---")
    st.markdown("### ✅ 启动检查")
    data_dir = APP_ROOT / "data"
    excel_files = []
    csv_files = []
    if data_dir.exists() and data_dir.is_dir():
        excel_files = [p for pattern in ("*.xlsx", "*.xls") for p in data_dir.glob(pattern)]
        csv_files = list(data_dir.glob("*.csv"))

    check_cols = st.columns(3)
    with check_cols[0]:
        st.metric("数据目录状态", "已就绪" if data_dir.exists() and data_dir.is_dir() else "未找到")
    with check_cols[1]:
        st.metric("Excel 文件数", len(excel_files))
    with check_cols[2]:
        st.metric("CSV 文件数", len(csv_files))

    st.markdown("---")
    st.markdown("### 📌 使用提示")
    st.markdown(
        "- 使用左侧导航进入各功能页面；\n"
        "- `测试数据分析` 支持从文件夹选择常用测试报表或直接上传；\n"
        "- `数据提取` 可批量汇总多站别数据并在页面底部导出；\n"
        "- `进度追踪` 页面提供壳体进度的甘特图与表格。"
    )

    st.markdown("### 🧭 页面说明")
    page_overview = pd.DataFrame(
        [
            {"页面": "测试数据分析", "用途": "筛选并分析常用测试数据报表"},
            {"页面": "数据提取", "用途": "多站别数据合并与趋势分析"},
            {"页面": "进度追踪", "用途": "查看壳体在各工序的实时进度"},
            {"页面": "BFD/NA 计算器", "用途": "辅助计算光学参数"},
        ]
    )
    st.dataframe(page_overview, hide_index=True, width='stretch', height=200)

def main() -> None:
    pages = [
        st.Page(render_home_page, title=APP_TITLE, icon=APP_ICON, default=True),
        st.Page(PAGES_ROOT / "Data_fetch.py", title="数据提取", icon="📥"),
        st.Page(PAGES_ROOT / "Progress.py", title="进度追踪", icon="📈"),
        st.Page(PAGES_ROOT / "TestAnalysis.py", title="测试数据分析", icon="📊"),
        st.Page(PAGES_ROOT / "BFD_Calculator.py", title="后焦距计算器", icon="🔧"),
        st.Page(PAGES_ROOT / "NA_Calculator.py", title="数值孔径计算器", icon="🔧")
    ]

    page = st.navigation(pages, position="sidebar", expanded=True)
    page.run()


if __name__ == "__main__":
    main()
