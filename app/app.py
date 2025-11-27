import platform
from pathlib import Path

import pandas as pd
import streamlit as st

from config import APP_ICON, APP_TITLE, PAGE_LAYOUT, SIDEBAR_STATE

APP_ROOT = Path(__file__).resolve().parent
PAGES_ROOT = APP_ROOT / "pages"

DATA_PAGES = [
    {
        "path": PAGES_ROOT / "Data_fetch.py",
        "title": "数据提取",
        "icon": "📥",
        "desc": "多站别数据合并、趋势分析、拟合预测、箱线图分析",
    },
    {
        "path": PAGES_ROOT / "Progress.py",
        "title": "进度追踪",
        "icon": "📈",
        "desc": "WIP进度监控、生产进度可视化、甘特图展示",
    },
    {
        "path": PAGES_ROOT / "TestAnalysis.py",
        "title": "测试数据分析",
        "icon": "📊",
        "desc": "测试报表筛选与统计分析",
    },
    {
        "path": PAGES_ROOT / "COS_Filter.py",
        "title": "COS筛选",
        "icon": "🔍",
        "desc": "按波长和仓库筛选批次实例数据",
    },
    {
        "path": PAGES_ROOT / "Engineering_Analysis.py",
        "title": "工程分析",
        "icon": "📉",
        "desc": "不良分析、帕累托分析、交叉分析、趋势分析",
    },
    {
        "path": PAGES_ROOT / "Data_Manager.py",
        "title": "数据管理",
        "icon": "📁",
        "desc": "管理已保存的数据集，支持查看、删除和导出",
    },
]

TOOL_PAGES = [
    {
        "path": PAGES_ROOT / "NA_Calculator.py",
        "title": "NA计算器",
        "icon": "🎯",
        "desc": "数值孔径计算、端帽光阑计算、材料库管理",
    },
    {
        "path": PAGES_ROOT / "BFD_Calculator.py",
        "title": "BFD计算器",
        "icon": "🔧",
        "desc": "后焦距计算、光学系统参数优化",
    },
    {
        "path": PAGES_ROOT / "Refractive_Index.py",
        "title": "Refractive_Index",
        "icon": "🔍",
        "desc": "查询数千种光学材料折射率、可视化展示",
    },
]


def render_home_page(enable_data: bool = True, enable_tools: bool = True) -> None:
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

    available_pages = []
    if enable_data:
        available_pages.extend(DATA_PAGES)
    if enable_tools:
        available_pages.extend(TOOL_PAGES)
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
    if enable_data:
        st.markdown("#### 📊 数据分析")
        analysis_pages = pd.DataFrame(
            [{"模块": f"{page['icon']} {page['title']}", "功能": page["desc"]} for page in DATA_PAGES]
        )
        st.dataframe(analysis_pages, hide_index=True, use_container_width=True, height=200)
    
    if enable_tools:
        st.markdown("#### 🔧 计算工具")
        tool_pages = pd.DataFrame(
            [{"工具": f"{page['icon']} {page['title']}", "功能": page["desc"]} for page in TOOL_PAGES]
        )
        st.dataframe(tool_pages, hide_index=True, use_container_width=True, height=140)
    
    st.markdown("---")
    st.markdown("### 🎯 核心特性")
    
    feature_sections = []
    if enable_data:
        feature_sections.append(("📈 数据分析", "- 多维度对比分析\n- 数据拟合预测\n- 帕累托分析\n- 交叉分析热力图"))
    if enable_tools:
        feature_sections.append(("🔬 光学计算", "- NA/BFD计算\n- 折射率查询\n- 材料数据库\n- 实时计算反馈"))
    feature_sections.append(("💾 数据处理", "- 智能缓存\n- 批量处理\n- 多格式导出\n- 数据清洗"))
    
    cols = st.columns(len(feature_sections))
    for col, (title, body) in zip(cols, feature_sections):
        col.markdown(f"**{title}**")
        col.markdown(body)


def build_pages(enable_data: bool = True, enable_tools: bool = True) -> dict:
    def _home():
        render_home_page(enable_data=enable_data, enable_tools=enable_tools)

    pages = {
        "主页": [
            st.Page(_home, title=APP_TITLE, icon=APP_ICON, default=True),
        ],
    }

    if enable_data:
        pages["数据分析"] = [
            st.Page(page_cfg["path"], title=page_cfg["title"], icon=page_cfg["icon"])
            for page_cfg in DATA_PAGES
        ]

    if enable_tools:
        pages["工具"] = [
            st.Page(page_cfg["path"], title=page_cfg["title"], icon=page_cfg["icon"])
            for page_cfg in TOOL_PAGES
        ]

    return pages


def run_app(enable_data: bool = True, enable_tools: bool = True) -> None:
    pages = build_pages(enable_data=enable_data, enable_tools=enable_tools)
    page = st.navigation(pages, position="sidebar")
    page.run()


def main() -> None:
    run_app()


if __name__ == "__main__":
    main()
