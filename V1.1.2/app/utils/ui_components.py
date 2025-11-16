"""
UI组件模块

该模块提供可复用的Streamlit UI组件，用于数据收集和分析界面。
包括数据收集对话框、数据集概览、图表组件和数据表格等。
"""

from typing import Dict, List, Tuple, Optional
import streamlit as st
import pandas as pd

# Altair将在需要时导入（延迟导入以提高性能）


def render_data_collection_dialog(
    shell_ids: List[str],
    default_path: str,
    default_filename: str
) -> Tuple[bool, str, str, float]:
    """
    渲染数据收集配置对话框
    
    显示一个对话框，允许用户配置数据收集参数，包括保存路径、文件名、
    指定电流值和数据来源选择。
    
    Args:
        shell_ids: 待收集的壳体号列表
        default_path: 默认保存路径
        default_filename: 默认文件名
        
    Returns:
        元组 (是否确认, 保存路径, 文件名, 指定电流)
        - 是否确认: 用户是否点击了"开始收集"按钮
        - 保存路径: 用户指定的保存路径
        - 文件名: 用户指定的文件名
        - 指定电流: 用户指定的电流值（用于TestAnalysis数据收集）
        
    UI元素:
        - 壳体号列表显示
        - 保存路径输入框和浏览按钮
        - 文件名输入框
        - 指定电流输入框
        - 数据来源复选框
        - 取消和开始收集按钮
    """
    st.subheader("📦 数据收集配置")
    
    # 显示待收集的壳体号列表
    st.write("**将收集以下壳体的数据:**")
    if len(shell_ids) <= 10:
        # 如果壳体号较少，显示为标签
        cols = st.columns(min(5, len(shell_ids)))
        for idx, shell_id in enumerate(shell_ids):
            with cols[idx % len(cols)]:
                st.info(f"🔹 {shell_id}")
    else:
        # 如果壳体号较多，显示为可展开的列表
        with st.expander(f"查看全部 {len(shell_ids)} 个壳体号"):
            # 分列显示
            num_cols = 5
            cols = st.columns(num_cols)
            for idx, shell_id in enumerate(shell_ids):
                with cols[idx % num_cols]:
                    st.write(f"• {shell_id}")
    
    st.divider()
    
    # 指定电流输入
    target_current = st.number_input(
        "指定电流 (A)",
        min_value=0.0,
        max_value=100.0,
        value=15.0,
        step=0.5,
        help="用于从TestAnalysis页面提取NA和热阻数据的电流值"
    )
    
    st.divider()
    
    # 保存路径和文件名配置
    col1, col2 = st.columns([3, 1])
    with col1:
        save_path = st.text_input(
            "保存路径",
            value=default_path,
            help="数据集文件的保存目录"
        )
    with col2:
        st.write("")  # 占位，对齐
        st.write("")  # 占位，对齐
        if st.button("📁 浏览", use_container_width=True):
            st.info("请在文本框中直接输入路径")
    
    filename = st.text_input(
        "文件名",
        value=default_filename,
        help="数据集文件名（自动添加.json扩展名）"
    )
    
    st.divider()
    
    # 数据来源选项（默认全选）
    st.write("**数据来源:**")
    col1, col2 = st.columns(2)
    
    with col1:
        collect_data_fetch = st.checkbox(
            "Data_fetch",
            value=True,
            help="功率、电流、效率、波长、Shift 数据"
        )
    with col2:
        collect_test_analysis = st.checkbox(
            "TestAnalysis",
            value=True,
            help="NA、热阻、光谱全高宽数据"
        )
    
    # 存储数据来源选择到session_state
    if 'data_sources' not in st.session_state:
        st.session_state.data_sources = {}
    
    st.session_state.data_sources = {
        'data_fetch': collect_data_fetch,
        'test_analysis': collect_test_analysis
    }
    
    st.divider()
    
    # 操作按钮
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        confirm = st.button("✅ 开始收集", type="primary", use_container_width=True)
    with col2:
        cancel = st.button("❌ 取消", use_container_width=True)
    
    # 处理取消操作
    if cancel:
        return False, "", "", 0.0
    
    # 返回配置结果
    return confirm, save_path, filename, target_current


def render_dataset_summary(dataset: Dict) -> None:
    """
    ????????
    """
    if not dataset or 'metadata' not in dataset or 'records' not in dataset:
        st.error("????????")
        return

    metadata = dataset.get('metadata', {}) or {}
    records = dataset.get('records', []) or []

    record_count = metadata.get('record_count', len(records))
    shell_count = metadata.get('shell_count')
    if shell_count is None:
        shell_count = len({rec.get('shell_id') for rec in records if rec.get('shell_id')})

    st.subheader("?? ?????")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("????", record_count)

    with col2:
        st.metric("????", shell_count)

    with col3:
        created_at = metadata.get('created_at')
        if created_at:
            try:
                from datetime import datetime
                created_dt = datetime.fromisoformat(created_at)
                created_str = created_dt.strftime("%Y-%m-%d %H:%M")
            except Exception:
                created_str = created_at
        else:
            created_str = "N/A"
        st.metric("????", created_str)

    with col4:
        target_current = metadata.get('target_current')
        if isinstance(target_current, (int, float)):
            st.metric("????", f"{target_current} A")
        else:
            st.metric("????", "N/A")

    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**????:** {metadata.get('created_by', 'unknown')}")
    with col2:
        description = metadata.get('description')
        if description:
            st.write(f"**??:** {description}")

    source_labels = {
        'Data_fetch': '????',
        'TestAnalysis': '????????',
    }

    def _format_sources(keys):
        if not keys:
            return ""
        return "?".join(source_labels.get(key, key) for key in keys)

    source_pages = metadata.get('source_pages', [])
    missing_pages = metadata.get('missing_pages', [])

    sources_text = _format_sources(source_pages)
    st.write(f"**????:** {sources_text or '???'}")

    if missing_pages:
        missing_text = _format_sources(missing_pages)
        st.warning(f"?? ????????????{missing_text}")

    st.divider()

    if not records:
        st.info("?????????????")

def render_shell_comparison_chart(
    dataset: Dict,
    metric: str,
    shell_ids: List[str]
) -> None:
    """
    渲染壳体对比图表
    
    根据指定的指标类型，绘制多个壳体的数据对比图表。
    支持功率-电流曲线、效率-电流曲线、波长分布、NA和热阻对比等。
    
    Args:
        dataset: 数据集字典
        metric: 指标类型，可选值:
            - 'power': 功率-电流曲线
            - 'efficiency': 效率-电流曲线
            - 'wavelength': 波长分布
            - 'na_thermal': NA和热阻对比
        shell_ids: 要对比的壳体号列表
        
    图表类型:
        - 折线图: 用于功率、效率随电流变化
        - 散点图: 用于波长分布
        - 柱状图: 用于NA和热阻对比
    """
    if not dataset or 'shells' not in dataset:
        st.error("数据集格式不正确")
        return
    
    if not shell_ids:
        st.warning("请选择至少一个壳体号")
        return
    
    shells = dataset['shells']
    
    # 根据指标类型渲染不同的图表
    if metric == 'power':
        _render_power_current_chart(shells, shell_ids)
    elif metric == 'efficiency':
        _render_efficiency_current_chart(shells, shell_ids)
    elif metric == 'wavelength':
        _render_wavelength_chart(shells, shell_ids)
    elif metric == 'voltage':
        _render_voltage_current_chart(shells, shell_ids)
    elif metric == 'shift':
        _render_shift_chart(shells, shell_ids)
    elif metric == 'na_thermal':
        _render_na_thermal_chart(shells, shell_ids)
    else:
        st.error(f"不支持的指标类型: {metric}")


def _render_power_current_chart(shells: Dict, shell_ids: List[str]) -> None:
    """渲染功率-电流曲线图"""
    import altair as alt
    
    # 准备数据
    chart_data = []
    for shell_id in shell_ids:
        if shell_id not in shells:
            continue
        
        data_fetch = shells[shell_id].get('data_fetch', {})
        if not data_fetch.get('data_available', False):
            continue
        
        current = data_fetch.get('current', [])
        power = data_fetch.get('power', [])
        
        if current and power and len(current) == len(power):
            for c, p in zip(current, power):
                chart_data.append({
                    '壳体号': shell_id,
                    '电流 (A)': c,
                    '功率 (W)': p
                })
    
    if not chart_data:
        st.warning("所选壳体没有可用的功率数据")
        return
    
    df = pd.DataFrame(chart_data)
    
    # 创建图表
    chart = alt.Chart(df).mark_line(point=True).encode(
        x=alt.X('电流 (A):Q', title='电流 (A)'),
        y=alt.Y('功率 (W):Q', title='功率 (W)'),
        color=alt.Color('壳体号:N', legend=alt.Legend(title='壳体号')),
        tooltip=['壳体号:N', alt.Tooltip('电流 (A):Q', format='.2f'), alt.Tooltip('功率 (W):Q', format='.2f')]
    ).properties(
        title='功率-电流曲线对比',
        height=500
    ).interactive()
    
    st.altair_chart(chart, use_container_width=True)


def _render_efficiency_current_chart(shells: Dict, shell_ids: List[str]) -> None:
    """渲染效率-电流曲线图"""
    import altair as alt
    
    # 准备数据
    chart_data = []
    for shell_id in shell_ids:
        if shell_id not in shells:
            continue
        
        data_fetch = shells[shell_id].get('data_fetch', {})
        if not data_fetch.get('data_available', False):
            continue
        
        current = data_fetch.get('current', [])
        efficiency = data_fetch.get('efficiency', [])
        
        if current and efficiency and len(current) == len(efficiency):
            for c, e in zip(current, efficiency):
                chart_data.append({
                    '壳体号': shell_id,
                    '电流 (A)': c,
                    '效率 (%)': e
                })
    
    if not chart_data:
        st.warning("所选壳体没有可用的效率数据")
        return
    
    df = pd.DataFrame(chart_data)
    
    # 创建图表
    chart = alt.Chart(df).mark_line(point=True).encode(
        x=alt.X('电流 (A):Q', title='电流 (A)'),
        y=alt.Y('效率 (%):Q', title='效率 (%)'),
        color=alt.Color('壳体号:N', legend=alt.Legend(title='壳体号')),
        tooltip=['壳体号:N', alt.Tooltip('电流 (A):Q', format='.2f'), alt.Tooltip('效率 (%):Q', format='.2f')]
    ).properties(
        title='效率-电流曲线对比',
        height=500
    ).interactive()
    
    st.altair_chart(chart, use_container_width=True)


def _render_voltage_current_chart(shells: Dict, shell_ids: List[str]) -> None:
    """渲染电压-电流曲线图"""
    import altair as alt
    
    # 准备数据
    chart_data = []
    for shell_id in shell_ids:
        if shell_id not in shells:
            continue
        
        data_fetch = shells[shell_id].get('data_fetch', {})
        if not data_fetch.get('data_available', False):
            continue
        
        current = data_fetch.get('current', [])
        voltage = data_fetch.get('voltage', [])
        
        if current and voltage and len(current) == len(voltage):
            for c, v in zip(current, voltage):
                chart_data.append({
                    '壳体号': shell_id,
                    '电流 (A)': c,
                    '电压 (V)': v
                })
    
    if not chart_data:
        st.warning("所选壳体没有可用的电压数据")
        return
    
    df = pd.DataFrame(chart_data)
    
    # 创建图表
    chart = alt.Chart(df).mark_line(point=True).encode(
        x=alt.X('电流 (A):Q', title='电流 (A)'),
        y=alt.Y('电压 (V):Q', title='电压 (V)'),
        color=alt.Color('壳体号:N', legend=alt.Legend(title='壳体号')),
        tooltip=['壳体号:N', alt.Tooltip('电流 (A):Q', format='.2f'), alt.Tooltip('电压 (V):Q', format='.2f')]
    ).properties(
        title='电压-电流曲线对比',
        height=500
    ).interactive()
    
    st.altair_chart(chart, use_container_width=True)


def _render_wavelength_chart(shells: Dict, shell_ids: List[str]) -> None:
    """渲染波长分布图"""
    import altair as alt
    
    # 准备数据
    chart_data = []
    for shell_id in shell_ids:
        if shell_id not in shells:
            continue
        
        data_fetch = shells[shell_id].get('data_fetch', {})
        if not data_fetch.get('data_available', False):
            continue
        
        current = data_fetch.get('current', [])
        wavelength = data_fetch.get('wavelength', [])
        
        if current and wavelength and len(current) == len(wavelength):
            for c, w in zip(current, wavelength):
                chart_data.append({
                    '壳体号': shell_id,
                    '电流 (A)': c,
                    '波长 (nm)': w
                })
    
    if not chart_data:
        st.warning("所选壳体没有可用的波长数据")
        return
    
    df = pd.DataFrame(chart_data)
    
    # 创建图表
    chart = alt.Chart(df).mark_line(point=True).encode(
        x=alt.X('电流 (A):Q', title='电流 (A)'),
        y=alt.Y('波长 (nm):Q', title='波长 (nm)'),
        color=alt.Color('壳体号:N', legend=alt.Legend(title='壳体号')),
        tooltip=['壳体号:N', alt.Tooltip('电流 (A):Q', format='.2f'), alt.Tooltip('波长 (nm):Q', format='.2f')]
    ).properties(
        title='波长-电流曲线对比',
        height=500
    ).interactive()
    
    st.altair_chart(chart, use_container_width=True)


def _render_shift_chart(shells: Dict, shell_ids: List[str]) -> None:
    """渲染shift分布图"""
    import altair as alt
    
    # 准备数据
    chart_data = []
    for shell_id in shell_ids:
        if shell_id not in shells:
            continue
        
        data_fetch = shells[shell_id].get('data_fetch', {})
        if not data_fetch.get('data_available', False):
            continue
        
        current = data_fetch.get('current', [])
        shift = data_fetch.get('shift', [])
        
        if current and shift and len(current) == len(shift):
            for c, s in zip(current, shift):
                chart_data.append({
                    '壳体号': shell_id,
                    '电流 (A)': c,
                    'Shift': s
                })
    
    if not chart_data:
        st.warning("所选壳体没有可用的Shift数据")
        return
    
    df = pd.DataFrame(chart_data)
    
    # 创建图表
    chart = alt.Chart(df).mark_line(point=True).encode(
        x=alt.X('电流 (A):Q', title='电流 (A)'),
        y=alt.Y('Shift:Q', title='Shift'),
        color=alt.Color('壳体号:N', legend=alt.Legend(title='壳体号')),
        tooltip=['壳体号:N', alt.Tooltip('电流 (A):Q', format='.2f'), alt.Tooltip('Shift:Q', format='.2f')]
    ).properties(
        title='Shift-电流曲线对比',
        height=500
    ).interactive()
    
    st.altair_chart(chart, use_container_width=True)


def _render_na_thermal_chart(shells: Dict, shell_ids: List[str]) -> None:
    """渲染NA和热阻对比柱状图"""
    import altair as alt
    
    # 准备数据
    na_data = []
    thermal_data = []
    
    for shell_id in shell_ids:
        if shell_id not in shells:
            continue
        
        test_analysis = shells[shell_id].get('test_analysis', {})
        if not test_analysis.get('data_available', False):
            continue
        
        na = test_analysis.get('na')
        thermal = test_analysis.get('thermal_resistance')
        
        if na is not None:
            na_data.append({'壳体号': shell_id, 'NA': na})
        if thermal is not None:
            thermal_data.append({'壳体号': shell_id, '热阻 (K/W)': thermal})
    
    if not na_data and not thermal_data:
        st.warning("所选壳体没有可用的NA或热阻数据")
        return
    
    col1, col2 = st.columns(2)
    
    # NA柱状图
    with col1:
        if na_data:
            df_na = pd.DataFrame(na_data)
            chart_na = alt.Chart(df_na).mark_bar(color='lightblue').encode(
                x=alt.X('壳体号:N', title='壳体号'),
                y=alt.Y('NA:Q', title='NA'),
                tooltip=['壳体号:N', alt.Tooltip('NA:Q', format='.4f')]
            ).properties(
                title='NA值对比',
                height=400
            )
            st.altair_chart(chart_na, use_container_width=True)
        else:
            st.info("没有可用的NA数据")
    
    # 热阻柱状图
    with col2:
        if thermal_data:
            df_thermal = pd.DataFrame(thermal_data)
            chart_thermal = alt.Chart(df_thermal).mark_bar(color='lightcoral').encode(
                x=alt.X('壳体号:N', title='壳体号'),
                y=alt.Y('热阻 (K/W):Q', title='热阻 (K/W)'),
                tooltip=['壳体号:N', alt.Tooltip('热阻 (K/W):Q', format='.2f')]
            ).properties(
                title='热阻对比',
                height=400
            )
            st.altair_chart(chart_thermal, use_container_width=True)
        else:
            st.info("没有可用的热阻数据")


def render_data_table(
    dataset: Dict,
    columns: List[str]
) -> None:
    """
    ?????????????
    """
    if not dataset or 'records' not in dataset:
        st.error("????????")
        return

    records = dataset.get('records', []) or []
    metadata = dataset.get('metadata', {}) or {}

    if not records:
        st.warning("????????")
        return

    df = pd.DataFrame(records)
    if df.empty:
        st.warning("????????")
        return

    base_columns = ['shell_id', 'current']
    requested = columns or []
    selected = base_columns + [col for col in requested if col not in base_columns and col in df.columns]
    selected = [col for col in selected if col in df.columns]

    if not selected:
        st.warning("???????")
        return

    display_df = df[selected].copy()

    rounding_rules = {
        'current': 3,
        'power': 3,
        'efficiency': 3,
        'wavelength': 3,
        'shift': 3,
        'spectral_fwhm': 3,
        'thermal_resistance': 3,
        'na': 4,
    }

    for column, decimals in rounding_rules.items():
        if column in display_df.columns:
            display_df[column] = pd.to_numeric(display_df[column], errors='coerce').round(decimals)

    column_labels = {
        'shell_id': '???',
        'current': '?? (A)',
        'power': '?? (W)',
        'efficiency': '?? (%)',
        'wavelength': '?? (nm)',
        'shift': '?? shift',
        'na': 'NA',
        'spectral_fwhm': '?????',
        'thermal_resistance': '?? (K/W)',
    }

    display_df.rename(columns={key: column_labels.get(key, key) for key in display_df.columns}, inplace=True)

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        height=400
    )

    target_current = metadata.get('target_current')
    if isinstance(target_current, (int, float)):
        st.caption(f"? {len(display_df)} ??? | ????: {target_current} A")
    else:
        st.caption(f"? {len(display_df)} ???")

def _get_value_at_current(
    current_list: List[float],
    value_list: List[float],
    target_current: float
) -> Optional[float]:
    """
    获取指定电流下的值（使用最近值匹配）
    
    Args:
        current_list: 电流值列表
        value_list: 对应的测量值列表
        target_current: 目标电流
        
    Returns:
        最接近目标电流的测量值，如果没有数据则返回None
    """
    if not current_list or not value_list or len(current_list) != len(value_list):
        return None
    
    # 找到最接近目标电流的索引
    min_diff = float('inf')
    closest_idx = 0
    
    for idx, current in enumerate(current_list):
        diff = abs(current - target_current)
        if diff < min_diff:
            min_diff = diff
            closest_idx = idx
    
    return value_list[closest_idx]
