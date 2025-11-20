# title: 光学计算器

import streamlit as st
import sys
from pathlib import Path

# 配置页面
st.set_page_config(page_title="光学计算器", page_icon="🔬", layout="wide")

# 添加 pages 目录到路径，方便导入同级工具
pages_dir = Path(__file__).parent
if str(pages_dir) not in sys.path:
    sys.path.insert(0, str(pages_dir))

# 导入两个计算器模块
import NA_Calculator
import BFD_Calculator

st.title("🔬 光学计算器")

# 创建标签页
tab1, tab2 = st.tabs(["🎯 NA 计算器（数值孔径）", "🔧 BFD 计算器（后焦距）"])

with tab1:
    NA_Calculator.main()

with tab2:
    BFD_Calculator.main()
