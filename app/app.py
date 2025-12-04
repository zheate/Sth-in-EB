from pathlib import Path

import streamlit as st

from config import APP_ICON, APP_TITLE
from auth import enforce_login, render_logout_button

APP_ROOT = Path(__file__).resolve().parent
PAGES_ROOT = APP_ROOT / "pages"

DATA_MANAGER_PAGE = {
    "path": PAGES_ROOT / "Data_Manager.py",
    "title": "数据管理",
    "desc": "管理已保存的数据集，支持查看、删除和导出",
}

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


def build_pages(enable_data: bool = True, enable_tools: bool = True) -> dict:
    # Data_Manager 作为主页
    home_pages = [
        st.Page(
            DATA_MANAGER_PAGE["path"],
            title=DATA_MANAGER_PAGE["title"],
            default=True,
        )
    ]

    pages = {"主页": home_pages}

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
    enforce_login()
    with st.sidebar:
        render_logout_button()
    pages = build_pages(enable_data=enable_data, enable_tools=enable_tools)
    page = st.navigation(pages, position="sidebar")
    page.run()


def main() -> None:
    run_app()


if __name__ == "__main__":
    main()
