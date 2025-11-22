import streamlit as st
import pandas as pd
import altair as alt
import os
import glob
from datetime import datetime, timedelta

st.set_page_config(page_title="工程分析", layout="wide", page_icon="📊")

# 自定义样式
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
        border-bottom: 3px solid #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">📊 工程分析明细报表</div>', unsafe_allow_html=True)

@st.cache_data
def load_data():
    # Define the directory and pattern
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    pattern = os.path.join(data_dir, "工程分析明细报表*.xlsx")
    
    # Find all matching files
    files = glob.glob(pattern)
    
    if not files:
        return None, "未找到匹配的文件 '工程分析明细报表*.xlsx'"
    
    # Get the latest file based on modification time
    latest_file = max(files, key=os.path.getmtime)
    
    try:
        df = pd.read_excel(latest_file)
        return df, latest_file
    except Exception as e:
        return None, str(e)

df, msg = load_data()

if df is None:
    st.error(f"❌ 加载数据失败: {msg}")
    st.info("💡 请确保 data 目录下存在 '工程分析明细报表*.xlsx' 文件")
else:
    # File Info
    with st.container(border=True):
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.markdown(f"**✅ 已加载数据:** `{os.path.basename(msg)}`")
        with col2:
            file_time = datetime.fromtimestamp(os.path.getmtime(msg))
            st.markdown(f"**📅 更新时间:** `{file_time.strftime('%Y-%m-%d %H:%M')}`")
        with col3:
            if st.button("🔄 刷新数据", use_container_width=True):
                st.cache_data.clear()
                st.rerun()
    
    # Sidebar Filters
    st.sidebar.markdown("### 🔍 筛选条件")
    st.sidebar.markdown("---")
    
    # Date Filter
    if '分析时间' in df.columns:
        df['分析时间'] = pd.to_datetime(df['分析时间'])
        min_date = df['分析时间'].min().date()
        max_date = df['分析时间'].max().date()
        
        # 快速日期选择
        st.sidebar.markdown("#### 📅 日期范围")
        date_preset = st.sidebar.radio(
            "快速选择",
            ["自定义", "今天", "最近7天", "最近30天", "全部"],
            horizontal=True
        )
        
        if date_preset == "今天":
            start_date = end_date = datetime.now().date()
        elif date_preset == "最近7天":
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=7)
        elif date_preset == "最近30天":
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=30)
        elif date_preset == "全部":
            start_date, end_date = min_date, max_date
        else:
            date_range = st.sidebar.date_input(
                "选择日期范围",
                [min_date, max_date],
                min_value=min_date,
                max_value=max_date
            )
            if len(date_range) == 2:
                start_date, end_date = date_range
            else:
                start_date = end_date = date_range[0]
        
        df = df[(df['分析时间'].dt.date >= start_date) & (df['分析时间'].dt.date <= end_date)]

    # Production Line Filter
    st.sidebar.markdown("#### 🏭 生产线")
    if '生产线' in df.columns:
        lines = sorted(df['生产线'].unique().tolist())
        select_all_lines = st.sidebar.checkbox("全选生产线", value=True, key="all_lines")
        if select_all_lines:
            selected_lines = lines
        else:
            selected_lines = st.sidebar.multiselect("选择生产线", lines, default=lines)
        if selected_lines:
            df = df[df['生产线'].isin(selected_lines)]

    # Work Order Type Filter
    st.sidebar.markdown("#### 📋 工单类型")
    if '工单类型' in df.columns:
        types = sorted(df['工单类型'].unique().tolist())
        select_all_types = st.sidebar.checkbox("全选工单类型", value=True, key="all_types")
        if select_all_types:
            selected_types = types
        else:
            selected_types = st.sidebar.multiselect("选择工单类型", types, default=types)
        if selected_types:
            df = df[df['工单类型'].isin(selected_types)]
    
    # 搜索功能
    st.sidebar.markdown("#### 🔎 搜索")
    search_term = st.sidebar.text_input("搜索 SN 或关键词", "")
    if search_term:
        mask = df.astype(str).apply(lambda x: x.str.contains(search_term, case=False, na=False)).any(axis=1)
        df = df[mask]
    
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**当前筛选结果: {len(df)} 条记录**")

    # Overview Metrics
    with st.container(border=True):
        st.markdown("### 📈 数据概览")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📊 总不良数", f"{len(df):,}", help="筛选后的不良记录总数")
        
        with col2:
            if 'SN' in df.columns:
                unique_sn = df['SN'].nunique()
                st.metric("🔢 唯一SN数", f"{unique_sn:,}", help="不同的产品序列号数量")
        
        with col3:
            if '不良站点' in df.columns:
                unique_sites = df['不良站点'].nunique()
                st.metric("🏭 涉及站点", f"{unique_sites}", help="出现不良的站点数量")
        
        with col4:
            if '不良现象' in df.columns:
                unique_phenomena = df['不良现象'].nunique()
                st.metric("⚠️ 不良类型", f"{unique_phenomena}", help="不同的不良现象类型")

    # Trend Analysis (Moved Up)
    if '分析时间' in df.columns:
        with st.container(border=True):
            st.markdown("### � 趋势分析")
            
            # 趋势统计放在上方
            if len(df) > 1:
                col1, col2, col3, col4, col5 = st.columns(5)
                
                # 先计算趋势数据用于统计
                df_temp = df.set_index('分析时间').resample('D').size().reset_index(name='数量')
                
                with col1:
                    avg_defects = df_temp['数量'].mean()
                    st.metric("📊 日均不良", f"{avg_defects:.1f}")
                with col2:
                    max_defects = df_temp['数量'].max()
                    st.metric("📈 峰值", f"{max_defects}")
                with col3:
                    min_defects = df_temp['数量'].min()
                    st.metric("📉 最低", f"{min_defects}")
                with col4:
                    if len(df_temp) >= 2:
                        recent_trend = df_temp['数量'].iloc[-1] - df_temp['数量'].iloc[-2]
                        st.metric("🔄 最近变化", f"{df_temp['数量'].iloc[-1]}", delta=f"{recent_trend:+.0f}")
                with col5:
                    trend_granularity = st.selectbox(
                        "时间粒度",
                        ["按日", "按周", "按月"],
                        label_visibility="collapsed"
                    )
            else:
                trend_granularity = "按日"
            
            # 图表
            if trend_granularity == "按日":
                df_trend = df.set_index('分析时间').resample('D').size().reset_index(name='数量')
                x_title = '日期'
            elif trend_granularity == "按周":
                df_trend = df.set_index('分析时间').resample('W').size().reset_index(name='数量')
                x_title = '周'
            else:
                df_trend = df.set_index('分析时间').resample('M').size().reset_index(name='数量')
                x_title = '月'
            
            # 添加移动平均线
            if len(df_trend) > 3:
                df_trend['移动平均'] = df_trend['数量'].rolling(window=3, min_periods=1).mean()
            
            # 主趋势线
            line_chart = alt.Chart(df_trend).mark_line(
                point=alt.OverlayMarkDef(filled=True, size=80),
                color='#1f77b4',
                strokeWidth=3
            ).encode(
                x=alt.X('分析时间:T', title=x_title),
                y=alt.Y('数量:Q', title='不良数量'),
                tooltip=[
                    alt.Tooltip('分析时间:T', title='时间', format='%Y-%m-%d'),
                    alt.Tooltip('数量:Q', title='数量')
                ]
            )
            
            # 移动平均线
            if '移动平均' in df_trend.columns:
                ma_line = alt.Chart(df_trend).mark_line(
                    strokeDash=[5, 5],
                    color='#ff7f0e',
                    strokeWidth=2
                ).encode(
                    x=alt.X('分析时间:T'),
                    y=alt.Y('移动平均:Q'),
                    tooltip=[
                        alt.Tooltip('分析时间:T', title='时间', format='%Y-%m-%d'),
                        alt.Tooltip('移动平均:Q', title='移动平均', format='.1f')
                    ]
                )
                chart_trend = (line_chart + ma_line).properties(
                    height=350,
                    title="不良趋势（蓝线：实际值，橙线：移动平均）"
                )
            else:
                chart_trend = line_chart.properties(height=350, title="不良趋势分析")
            
            st.altair_chart(chart_trend, use_container_width=True)

    # Visualizations
    with st.container(border=True):
        st.markdown("### 📊 不良分析")
        
        # 创建标签页
        tab1, tab2, tab3 = st.tabs(["📍 站点分析", "⚠️ 现象分析", "🔍 原因分析"])
        
        with tab1:
            if '不良站点' in df.columns:
                site_counts = df['不良站点'].value_counts().reset_index()
                site_counts.columns = ['站点', '数量']
                site_counts['占比'] = (site_counts['数量'] / site_counts['数量'].sum() * 100).round(2)
                
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    chart_site = alt.Chart(site_counts).mark_bar(color='#1f77b4').encode(
                        x=alt.X('站点:N', sort='-y', title='站点', axis=alt.Axis(labelAngle=-45)),
                        y=alt.Y('数量:Q', title='数量'),
                        tooltip=[
                            alt.Tooltip('站点:N', title='站点'),
                            alt.Tooltip('数量:Q', title='数量'),
                            alt.Tooltip('占比:Q', title='占比(%)', format='.2f')
                        ]
                    ).properties(height=350, title="各站点不良数量分布")
                    
                    st.altair_chart(chart_site, use_container_width=True)
                
                with col2:
                    st.markdown("#### 📋 Top 5")
                    for idx, row in site_counts.head(5).iterrows():
                        with st.container():
                            st.markdown(f"**{idx+1}. {row['站点']}**")
                            st.progress(row['数量'] / site_counts['数量'].max())
                            st.caption(f"{row['数量']} 次 ({row['占比']:.1f}%)")
                            st.markdown("")
        
        with tab2:
            if '不良现象' in df.columns:
                phenomena_counts = df['不良现象'].value_counts().reset_index()
                phenomena_counts.columns = ['现象', '数量']
                phenomena_counts['占比'] = (phenomena_counts['数量'] / phenomena_counts['数量'].sum() * 100).round(2)
                
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    chart_phenomena = alt.Chart(phenomena_counts).mark_bar(color='#ff7f0e').encode(
                        x=alt.X('现象:N', sort='-y', title='现象', axis=alt.Axis(labelAngle=-45)),
                        y=alt.Y('数量:Q', title='数量'),
                        tooltip=[
                            alt.Tooltip('现象:N', title='现象'),
                            alt.Tooltip('数量:Q', title='数量'),
                            alt.Tooltip('占比:Q', title='占比(%)', format='.2f')
                        ]
                    ).properties(height=350, title="各现象不良数量分布")
                    
                    st.altair_chart(chart_phenomena, use_container_width=True)
                
                with col2:
                    st.markdown("#### � Top 5")
                    for idx, row in phenomena_counts.head(5).iterrows():
                        with st.container():
                            st.markdown(f"**{idx+1}. {row['现象']}**")
                            st.progress(row['数量'] / phenomena_counts['数量'].max())
                            st.caption(f"{row['数量']} 次 ({row['占比']:.1f}%)")
                            st.markdown("")
        
        with tab3:
            if '原因分类' in df.columns:
                cause_counts = df['原因分类'].value_counts().reset_index()
                cause_counts.columns = ['原因', '数量']
                cause_counts['占比'] = (cause_counts['数量'] / cause_counts['数量'].sum() * 100).round(2)
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    chart_cause = alt.Chart(cause_counts).mark_arc(innerRadius=60).encode(
                        theta=alt.Theta("数量:Q", stack=True),
                        color=alt.Color("原因:N", legend=alt.Legend(title="原因分类", orient="bottom")),
                        tooltip=[
                            alt.Tooltip('原因:N', title='原因'),
                            alt.Tooltip('数量:Q', title='数量'),
                            alt.Tooltip('占比:Q', title='占比(%)', format='.2f')
                        ]
                    ).properties(height=350, title="不良原因分布")
                    
                    st.altair_chart(chart_cause, use_container_width=True)
                
                with col2:
                    st.markdown("#### 📊 原因统计")
                    st.dataframe(
                        cause_counts.style.format({'占比': '{:.2f}%'}),
                        use_container_width=True,
                        hide_index=True,
                        height=350
                    )

    # 交叉分析和帕累托分析并排
    col_left, col_right = st.columns(2)
    
    with col_left:
        with st.container(border=True):
            st.markdown("### 🔄 交叉分析")
            
            if '不良站点' in df.columns and '不良现象' in df.columns:
                cross_tab = pd.crosstab(df['不良站点'], df['不良现象'], margins=True, margins_name='总计')
                
                # 热力图数据准备
                cross_tab_no_margin = pd.crosstab(df['不良站点'], df['不良现象'])
                cross_tab_melted = cross_tab_no_margin.reset_index().melt(
                    id_vars='不良站点',
                    var_name='不良现象',
                    value_name='数量'
                )
                
                # 热力图
                heatmap = alt.Chart(cross_tab_melted).mark_rect().encode(
                    x=alt.X('不良现象:N', title='不良现象', axis=alt.Axis(labelAngle=-45)),
                    y=alt.Y('不良站点:N', title='不良站点'),
                    color=alt.Color('数量:Q', scale=alt.Scale(scheme='blues'), title='数量'),
                    tooltip=[
                        alt.Tooltip('不良站点:N', title='站点'),
                        alt.Tooltip('不良现象:N', title='现象'),
                        alt.Tooltip('数量:Q', title='数量')
                    ]
                ).properties(height=400, title="站点-现象交叉热力图")
                
                st.altair_chart(heatmap, use_container_width=True)
                
                with st.expander("📊 查看交叉统计表"):
                    st.dataframe(cross_tab, use_container_width=True)
    
    with col_right:
        with st.container(border=True):
            st.markdown("### 📊 帕累托分析")
            
            pareto_col = st.selectbox(
                "选择分析维度",
                ['不良站点', '不良现象', '原因分类'] if all(col in df.columns for col in ['不良站点', '不良现象', '原因分类']) 
                else [col for col in ['不良站点', '不良现象', '原因分类'] if col in df.columns],
                label_visibility="collapsed"
            )
            
            if pareto_col:
                pareto_data = df[pareto_col].value_counts().reset_index()
                pareto_data.columns = ['类别', '数量']
                pareto_data['累计数量'] = pareto_data['数量'].cumsum()
                pareto_data['累计占比'] = (pareto_data['累计数量'] / pareto_data['数量'].sum() * 100).round(2)
                pareto_data['占比'] = (pareto_data['数量'] / pareto_data['数量'].sum() * 100).round(2)
                
                # 找出累计占比达到80%的项
                pareto_80 = pareto_data[pareto_data['累计占比'] <= 80]
                
                # 柱状图
                bars = alt.Chart(pareto_data.head(10)).mark_bar(color='#1f77b4').encode(
                    x=alt.X('类别:N', sort='-y', title='类别', axis=alt.Axis(labelAngle=-45)),
                    y=alt.Y('数量:Q', title='数量'),
                    tooltip=[
                        alt.Tooltip('类别:N', title='类别'),
                        alt.Tooltip('数量:Q', title='数量'),
                        alt.Tooltip('占比:Q', title='占比(%)', format='.2f')
                    ]
                )
                
                # 累计占比线
                line = alt.Chart(pareto_data.head(10)).mark_line(
                    color='#ff7f0e',
                    strokeWidth=3,
                    point=alt.OverlayMarkDef(filled=True, size=80)
                ).encode(
                    x=alt.X('类别:N', sort='-y'),
                    y=alt.Y('累计占比:Q', title='累计占比(%)', axis=alt.Axis(titleColor='#ff7f0e')),
                    tooltip=[
                        alt.Tooltip('类别:N', title='类别'),
                        alt.Tooltip('累计占比:Q', title='累计占比(%)', format='.2f')
                    ]
                )
                
                # 80%参考线
                rule = alt.Chart(pd.DataFrame({'y': [80]})).mark_rule(
                    strokeDash=[5, 5],
                    color='red',
                    strokeWidth=2
                ).encode(y='y:Q')
                
                pareto_chart = alt.layer(
                    bars,
                    line,
                    rule
                ).resolve_scale(
                    y='independent'
                ).properties(height=350, title=f"{pareto_col} 帕累托图（Top 10）")
                
                st.altair_chart(pareto_chart, use_container_width=True)
                
                # 关键项展示
                st.info(f"🎯 前 **{len(pareto_80)}** 项占总数的 **80%**")
                
                with st.expander("📋 查看关键项详情"):
                    for idx, row in pareto_80.iterrows():
                        st.markdown(f"**{idx+1}. {row['类别']}** - {row['数量']} 次 ({row['占比']:.1f}%)")

    # Detailed Data
    with st.container(border=True):
        st.markdown("### 📋 详细数据")
        
        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
        with col1:
            st.markdown(f"**共 {len(df):,} 条记录**")
        with col2:
            show_rows = st.selectbox("显示行数", [10, 50, 100, 500, "全部"], index=1, label_visibility="collapsed")
        with col3:
            # 导出CSV
            csv = df.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="📥 导出CSV",
                data=csv,
                file_name=f"工程分析_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        with col4:
            # 导出Excel
            from io import BytesIO
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='工程分析')
            excel_data = output.getvalue()
            st.download_button(
                label="📥 导出Excel",
                data=excel_data,
                file_name=f"工程分析_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        
        if show_rows == "全部":
            st.dataframe(df, use_container_width=True, height=500)
        else:
            st.dataframe(df.head(show_rows), use_container_width=True, height=500)
