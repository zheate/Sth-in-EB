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

    st.title("🔬 ZH's 妙妙屋")

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
    st.info(
        "💡 **快速开始**\n\n"
        "- 使用左侧导航栏进入各功能模块\n"
        "- 数据文件放置在 `app/data/` 目录下\n"
        "- 支持 Excel (.xlsx, .xls) 和 CSV 格式\n"
        "- 大部分功能支持数据导出"
    )

    st.markdown("### 🧭 功能模块")
    
    # 数据分析模块
    st.markdown("#### 📊 数据分析")
    analysis_pages = pd.DataFrame([
        {"模块": "📥 数据提取", 
         "功能": "多站别数据合并、趋势分析、拟合预测、箱线图分析"},
        {"模块": "📈 进度追踪", 
         "功能": "WIP进度监控、生产进度可视化、甘特图展示"},
        {"模块": "📊 测试数据分析", 
         "功能": "测试报表筛选与统计分析"},
        {"模块": "🔍 COS筛选", 
         "功能": "按波长和仓库筛选批次实例数据"},
        {"模块": "📉 工程分析", 
         "功能": "不良分析、帕累托分析、交叉分析、趋势分析"},
    ])
    st.dataframe(analysis_pages, hide_index=True, use_container_width=True, height=200)
    
    # 工具模块
    st.markdown("#### 🔧 计算工具")
    tool_pages = pd.DataFrame([
        {"工具": "🎯 NA计算器", 
         "功能": "数值孔径计算、端帽光阑计算、材料库管理"},
        {"工具": "🔧 BFD计算器", 
         "功能": "后焦距计算、光学系统参数优化"},
        {"工具": "🔍 折射率查询", 
         "功能": "查询数千种光学材料折射率、可视化展示"},
    ])
    st.dataframe(tool_pages, hide_index=True, use_container_width=True, height=140)
    
    st.markdown("---")
    st.markdown("### 🎯 核心特性")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**📈 数据分析**")
        st.markdown("- 多维度对比分析\n- 数据拟合预测\n- 帕累托分析\n- 交叉分析热力图")
    with col2:
        st.markdown("**🔬 光学计算**")
        st.markdown("- NA/BFD计算\n- 折射率查询\n- 材料数据库\n- 实时计算反馈")
    with col3:
        st.markdown("**💾 数据处理**")
        st.markdown("- 智能缓存\n- 批量处理\n- 多格式导出\n- 数据清洗")

def main() -> None:
    pages = {
        "主页": [
            st.Page(render_home_page, title=APP_TITLE, icon=APP_ICON, default=True),
        ],
        "数据分析": [
            st.Page(PAGES_ROOT / "Data_fetch.py", title="数据提取", icon="📥"),
            st.Page(PAGES_ROOT / "Progress.py", title="进度追踪", icon="📈"),
            st.Page(PAGES_ROOT / "TestAnalysis.py", title="测试数据分析", icon="📊"),
            st.Page(PAGES_ROOT / "COS_Filter.py", title="COS筛选", icon="🔍"),
            st.Page(PAGES_ROOT / "Engineering_Analysis.py", title="工程分析", icon="📉"),
        ],
        "工具": [
            st.Page(PAGES_ROOT / "NA_Calculator.py", title="NA计算器", icon="🎯"),
            st.Page(PAGES_ROOT / "BFD_Calculator.py", title="BFD计算器", icon="🔧"),
            st.Page(PAGES_ROOT / "Refractive_Index.py", title="Refractive_Index", icon="🔍"),
        ],
    }

    page = st.navigation(pages, position="sidebar")
    page.run()


if __name__ == "__main__":
    main()
