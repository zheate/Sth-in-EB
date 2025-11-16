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

    # 注入自定义CSS - 强制所有元素使用Times New Roman字体
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@400;500;700&display=swap');
        
        /* 最高优先级全局字体设置 */
        * {
            font-family: "Times New Roman", "Noto Sans SC", "Microsoft YaHei", sans-serif !important;
        }
        
        html, body {
            font-family: "Times New Roman", "Noto Sans SC", "Microsoft YaHei", sans-serif !important;
        }
        
        /* Streamlit所有元素 */
        .stApp, .stApp *, 
        .main, .main *,
        .block-container, .block-container *,
        section, section *,
        div, div *, span, span *, p, p *,
        label, label *, input, input *,
        button, button *, select, select *,
        table, table *, thead, thead *, tbody, tbody *,
        tr, tr *, th, th *, td, td * {
            font-family: "Times New Roman", "Noto Sans SC", "Microsoft YaHei", sans-serif !important;
        }
        
        /* 特别针对表格数据 */
        [data-testid="stDataFrame"] *,
        [data-testid="stTable"] *,
        .dataframe *,
        .stDataFrame * {
            font-family: "Times New Roman", "Noto Sans SC", "Microsoft YaHei", sans-serif !important;
        }
        
        /* 指标组件 */
        [data-testid="stMetric"] *,
        [data-testid="stMetricLabel"] *,
        [data-testid="stMetricValue"] *,
        [data-testid="stMetricDelta"] * {
            font-family: "Times New Roman", "Noto Sans SC", "Microsoft YaHei", sans-serif !important;
        }
        
        /* 输入框 */
        [data-baseweb="input"] *,
        [data-baseweb="select"] *,
        [data-baseweb="base-input"] * {
            font-family: "Times New Roman", "Noto Sans SC", "Microsoft YaHei", sans-serif !important;
        }
        
        /* 标题 */
        h1, h2, h3, h4, h5, h6,
        h1 *, h2 *, h3 *, h4 *, h5 *, h6 * {
            font-family: "Times New Roman", "Noto Sans SC", "Microsoft YaHei", sans-serif !important;
        }
        
        /* 代码块保持等宽字体 */
        code, code *, pre, pre *,
        .stCode, .stCode * {
            font-family: "Courier New", "Consolas", monospace !important;
        }
        </style>
        """,
        unsafe_allow_html=True
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

    st.title("🔬 ZH’s 妙妙屋")

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
            {"页面": "测试数据分析", 
             "用途": "筛选并分析常用测试数据报表"},
            {"页面": "数据提取", 
             "用途": "多站别数据合并与趋势分析"},
            {"页面": "进度追踪", 
             "用途": "查看壳体在各工序的实时进度"},
            {"页面": "COS筛选", 
             "用途": "按波长和仓库筛选批次实例数据"},
            {"页面": "光学计算器", 
             "用途": "NA 和后焦距计算器"},
        ]
    )
    st.dataframe(page_overview, hide_index=True, use_container_width=True, height=220)

def main() -> None:
    pages = [
        st.Page(render_home_page, title=APP_TITLE, icon=APP_ICON, default=True),
        st.Page(PAGES_ROOT / "Data_fetch.py", title="数据提取", icon="📥"),
        st.Page(PAGES_ROOT / "Progress.py", title="进度追踪", icon="📈"),
        st.Page(PAGES_ROOT / "TestAnalysis.py", title="测试数据分析", icon="📊"),
        st.Page(PAGES_ROOT / "COS_Filter.py", title="COS筛选", icon="🔍"),
        st.Page(PAGES_ROOT / "Optical_Calculators.py", title="光学计算器", icon="🔬"),
    ]

    page = st.navigation(pages, position="sidebar", expanded=True)
    page.run()


if __name__ == "__main__":
    main()
