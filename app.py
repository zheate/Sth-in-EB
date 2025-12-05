from pathlib import Path

import streamlit as st

from config import APP_ICON, APP_TITLE, ROLE_PERMISSIONS
from auth import enforce_login, render_logout_button, get_current_user

APP_ROOT = Path(__file__).resolve().parent
PAGES_ROOT = APP_ROOT / "pages"

DATA_MANAGER_PAGE = {
    "path": PAGES_ROOT / "Data_Manager.py",
    "title": "项目管理",
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
        "title": "折射率查询",
        "icon": "🔍",
        "desc": "查询数千种光学材料折射率、可视化展示",
    },
    {
        "path": PAGES_ROOT / "LD_Calculator.py",
        "title": "LD光纤耦合",
        "icon": "💡",
        "desc": "光纤耦合模块设计、远近场分析、光线追迹",
    },
]


def _filter_pages_by_role(page_list: list, user_role: str) -> list:
    """根据用户角色过滤页面列表"""
    perms = ROLE_PERMISSIONS.get(user_role, {"all": True})
    
    # 如果有 all 权限，返回全部页面
    if perms.get("all"):
        return page_list
    
    # 否则只返回指定的页面
    allowed_pages = perms.get("pages", [])
    return [p for p in page_list if p["path"].name in allowed_pages]


def build_pages(user_role: str = "user") -> dict:
    """根据用户角色构建页面导航"""
    perms = ROLE_PERMISSIONS.get(user_role, {"all": True})
    
    # 如果是受限用户，只显示允许的页面作为主页
    if not perms.get("all"):
        allowed_pages = perms.get("pages", [])
        filtered_tools = [p for p in TOOL_PAGES if p["path"].name in allowed_pages]
        filtered_data = [p for p in DATA_PAGES if p["path"].name in allowed_pages]
        
        all_allowed = filtered_tools + filtered_data
        if all_allowed:
            # 第一个允许的页面作为主页
            first_page = all_allowed[0]
            return {
                "主页": [
                    st.Page(first_page["path"], title=first_page["title"], icon=first_page.get("icon", "🏠"), default=True)
                ]
            }
        else:
            # 没有任何权限，返回空
            return {}
    
    # 完整权限用户
    home_pages = [
        st.Page(
            DATA_MANAGER_PAGE["path"],
            title=DATA_MANAGER_PAGE["title"],
            default=True,
        )
    ]

    pages = {"主页": home_pages}

    pages["数据分析"] = [
        st.Page(page_cfg["path"], title=page_cfg["title"], icon=page_cfg["icon"])
        for page_cfg in DATA_PAGES
    ]

    pages["工具"] = [
        st.Page(page_cfg["path"], title=page_cfg["title"], icon=page_cfg["icon"])
        for page_cfg in TOOL_PAGES
    ]

    return pages


def run_app() -> None:
    enforce_login()
    
    # 获取当前用户角色
    current_user = get_current_user()
    user_role = current_user.get("role", "user") if current_user else "user"
    
    with st.sidebar:
        render_logout_button()
    
    pages = build_pages(user_role=user_role)
    
    if not pages:
        st.error("您没有访问任何页面的权限，请联系管理员。")
        return
    
    page = st.navigation(pages, position="sidebar")
    page.run()


def main() -> None:
    run_app()


if __name__ == "__main__":
    main()
