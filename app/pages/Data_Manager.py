# title: 数据管理
"""数据管理页面 - 管理已保存的数据集"""

import sys
from pathlib import Path

# 路径设置
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import streamlit as st

# 页面配置
st.set_page_config(
    page_title="数据管理",
    page_icon="📁",
    layout="wide",
)

# 导入并渲染数据管理页面
from utils.storage_widgets import render_data_manager_page

render_data_manager_page()
