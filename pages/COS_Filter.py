
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import streamlit as st

# 添加父目录到路径以导入config
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

st.set_page_config(
    page_title="COS筛选",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 COS筛选")
st.markdown("---")

# 数据目录
DATA_DIR = Path(__file__).parent.parent / "data"

@st.cache_data(ttl=300)  # 缓存5分钟
def find_all_batch_files():
    """查找data文件夹中所有批次实例Excel文件"""
    if not DATA_DIR.exists():
        return []
    
    # 查找包含"批次实例"的Excel文件
    batch_files = list(DATA_DIR.glob("*批次实例*.xlsx")) + list(DATA_DIR.glob("*批次实例*.xls"))
    
    if not batch_files:
        return []
    
    # 按修改时间排序（最新的在前）
    batch_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return [str(f) for f in batch_files]

def find_latest_batch_file():
    """查找data文件夹中最新的批次实例Excel文件"""
    all_files = find_all_batch_files()
    return all_files[0] if all_files else None

@st.cache_data(ttl=300)  # 缓存5分钟
def load_batch_data(file_path, load_all_columns=False):
    """加载批次实例数据（带缓存，优化速度）
    
    Args:
        file_path: Excel文件路径
        load_all_columns: 是否加载所有列（用于导出）
    """
    import pickle
    import os
    
    try:
        # 生成pickle缓存文件路径
        excel_path = Path(file_path)
        cache_suffix = "_all" if load_all_columns else ""
        cache_path = excel_path.parent / f".cache_{excel_path.stem}{cache_suffix}.pkl"
        
        # 检查缓存是否存在且比Excel文件新
        use_cache = False
        if cache_path.exists():
            excel_mtime = os.path.getmtime(file_path)
            cache_mtime = os.path.getmtime(cache_path)
            if cache_mtime > excel_mtime:
                use_cache = True
        
        # 尝试从缓存加载
        if use_cache:
            try:
                with open(cache_path, 'rb') as f:
                    df = pickle.load(f)
                return df
            except Exception:
                # 缓存损坏，删除并重新加载
                cache_path.unlink(missing_ok=True)
        
        # 从Excel加载
        if load_all_columns:
            # 加载所有列
            df = pd.read_excel(file_path, engine='openpyxl')
        else:
            # 只加载必需列
            usecols = ['LOT | SN', '仓库', '2A波长', '是否隔离', '料件名称', 
                       '料件规格', '入库时间', '库存状态', 'ItemNum']
            
            df = pd.read_excel(
                file_path, 
                usecols=usecols,
                engine='openpyxl',
                dtype={
                    'LOT | SN': str,
                    '仓库': str,
                    '是否隔离': str,
                    '料件名称': str,
                    '料件规格': str,
                    '库存状态': str,
                    'ItemNum': str
                }
            )
        
        # 预处理：转换2A波长为数值
        if '2A波长' in df.columns:
            df['2A波长_numeric'] = pd.to_numeric(df['2A波长'], errors='coerce')
        
        # 保存到缓存（异步，不阻塞）
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(df, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:
            pass  # 缓存失败不影响主流程
        
        return df
    except Exception as e:
        st.error(f"读取文件失败: {e}")
        return None

def get_available_itemnums(df):
    """从筛选后的数据中获取可用的ItemNum"""
    if df is None or df.empty or 'ItemNum' not in df.columns:
        return []
    unique_items = sorted(df['ItemNum'].dropna().unique().tolist())
    return unique_items

def filter_cos_data_step1(df, wavelength_min, wavelength_max, progress_callback=None):
    """
    第一步筛选：基础条件筛选（不限制数量）
    条件：
    - 隔离为否
    - 仓库为良品仓或研发工程仓
    - 2A波长在用户输入范围内
    """
    if df is None or df.empty:
        return None
    
    # 使用布尔索引一次性筛选
    mask = pd.Series(True, index=df.index)
    
    if progress_callback:
        progress_callback(50, "🔍 应用基础筛选条件...")
    
    # 1. 筛选隔离为否
    if '是否隔离' in df.columns:
        mask &= (df['是否隔离'] == '否')
    
    # 2. 筛选仓库（良品仓或研发工程仓）
    if '仓库' in df.columns:
        mask &= df['仓库'].isin(['良品仓', '研发工程仓'])
    
    # 3. 筛选2A波长范围（使用预处理的数值列）
    if '2A波长_numeric' in df.columns:
        mask &= (df['2A波长_numeric'] >= wavelength_min) & (df['2A波长_numeric'] <= wavelength_max)
    
    # 应用筛选
    df_filtered = df[mask].copy()
    
    if progress_callback:
        progress_callback(60, f"✅ 第一步筛选完成，找到 {len(df_filtered)} 条记录")
    
    return df_filtered

def filter_cos_data_step2(df, wavelength_min, wavelength_max, required_count, itemnum_filter=None, progress_callback=None):
    """
    第二步筛选：在第一步结果基础上进行ItemNum筛选和排序
    """
    if df is None or df.empty:
        return None
    
    # 计算目标波长（范围中间值）
    target_wavelength = (wavelength_min + wavelength_max) / 2
    
    df_filtered = df.copy()
    
    # 如果指定了ItemNum，进行第二次筛选
    if itemnum_filter and 'ItemNum' in df_filtered.columns:
        if progress_callback:
            progress_callback(70, "🔢 应用ItemNum筛选...")
        df_filtered = df_filtered[df_filtered['ItemNum'].isin(itemnum_filter)]
    
    if df_filtered.empty:
        return df_filtered
    
    if progress_callback:
        progress_callback(80, f"✅ 找到 {len(df_filtered)} 条符合条件的记录")
    
    # 计算每条记录与目标波长的距离
    if '2A波长_numeric' in df_filtered.columns:
        df_filtered['_wavelength_distance'] = abs(df_filtered['2A波长_numeric'] - target_wavelength)
    
    if progress_callback:
        progress_callback(90, "📊 按优先级排序...")
    
    # 按仓库优先级和波长距离排序
    # 良品仓优先，然后按波长距离从小到大排序
    if '仓库' in df_filtered.columns:
        df_filtered['_warehouse_priority'] = df_filtered['仓库'].map({'良品仓': 1, '研发工程仓': 2})
        df_filtered = df_filtered.sort_values(['_warehouse_priority', '_wavelength_distance'])
    else:
        df_filtered = df_filtered.sort_values('_wavelength_distance')
    
    # 取前N条记录，并删除临时列
    result = df_filtered.head(required_count)
    result = result.drop(columns=[col for col in ['_wavelength_distance', '_warehouse_priority'] if col in result.columns])
    
    return result

# 主界面
st.markdown("### 文件选择")

# 查找所有批次文件
all_batch_files = find_all_batch_files()

if not all_batch_files:
    st.warning("⚠️ 未找到批次实例Excel文件，请确保data文件夹中存在包含'批次实例'的Excel文件。")
    st.stop()

# 准备文件选项（显示文件名和修改时间）
from datetime import datetime
file_options = {}
for file_path in all_batch_files:
    file = Path(file_path)
    mtime = datetime.fromtimestamp(file.stat().st_mtime)
    size_mb = file.stat().st_size / (1024 * 1024)
    display_name = f"{file.name} ({mtime.strftime('%Y-%m-%d %H:%M')} | {size_mb:.1f}MB)"
    file_options[display_name] = file_path

# 文件选择下拉框和刷新按钮
col_select, col_refresh = st.columns([5, 1])

with col_select:
    selected_display_name = st.selectbox(
        "选择批次实例文件",
        options=list(file_options.keys()),
        index=0,
        help="默认选择最新的文件，可以切换到其他文件"
    )

with col_refresh:
    st.markdown("<br>", unsafe_allow_html=True)  # 对齐按钮
    if st.button("🔄", help="刷新文件列表"):
        st.cache_data.clear()
        st.rerun()

selected_file_path = file_options[selected_display_name]
selected_file = Path(selected_file_path)

# 检查是否有缓存
cache_path = selected_file.parent / f".cache_{selected_file.stem}.pkl"
has_cache = cache_path.exists()

# 获取Excel文件大小
excel_size_mb = selected_file.stat().st_size / (1024 * 1024)

# 显示文件信息和缓存管理
col_info, col_cache = st.columns([3, 1])

with col_info:
    if has_cache:
        cache_size = cache_path.stat().st_size / (1024 * 1024)  # MB
        st.success(f"⚡ 已缓存 ({cache_size:.1f}MB)，加载速度快（约1-3秒）")
    else:
        # 根据文件大小估算加载时间（粗略估算：每MB约0.5-1秒）
        estimated_time = int(excel_size_mb * 0.5)
        st.warning(f"⏳ Excel文件 {excel_size_mb:.1f}MB，首次加载预计 {estimated_time}-{estimated_time*2} 秒")
        st.info("💡 首次加载后会自动生成缓存文件，后续加载只需1-3秒")

with col_cache:
    if has_cache:
        if st.button("🗑️ 清除缓存", help="删除pickle缓存文件，下次加载会重新从Excel读取"):
            try:
                cache_path.unlink()
                st.success("✅ 缓存已清除")
                st.rerun()
            except Exception as e:
                st.error(f"清除失败: {e}")

st.markdown("---")
st.markdown("### 📋 第一步：基础筛选条件")

col1, col2, col3 = st.columns(3)

with col1:
    wavelength_min = st.number_input(
        "2A波长最小值 (nm)",
        min_value=0.0,
        max_value=10000.0,
        value=900.0,
        step=1.0,
        format="%.2f"
    )

with col2:
    wavelength_max = st.number_input(
        "2A波长最大值 (nm)",
        min_value=0.0,
        max_value=10000.0,
        value=1000.0,
        step=1.0,
        format="%.2f"
    )

with col3:
    required_count = st.number_input(
        "需要数量",
        min_value=1,
        max_value=10000,
        value=10,
        step=1
    )

# 初始化session state
if 'step1_result' not in st.session_state:
    st.session_state.step1_result = None
if 'step1_params' not in st.session_state:
    st.session_state.step1_params = None
if 'selected_file' not in st.session_state:
    st.session_state.selected_file = None

# 检查文件是否改变，如果改变则清空筛选结果
if st.session_state.selected_file != selected_file_path:
    st.session_state.selected_file = selected_file_path
    st.session_state.step1_result = None
    st.session_state.step1_params = None

# 第一步筛选按钮
if st.button("🔍 第一步筛选", type="primary", use_container_width=True):
    # 创建进度条
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # 定义进度回调函数
        def update_progress(percent, message):
            progress_bar.progress(percent)
            status_text.text(message)
        
        # 加载数据
        if has_cache:
            update_progress(10, "📂 正在从缓存加载数据...")
        else:
            update_progress(10, f"📂 正在读取Excel文件 ({excel_size_mb:.1f}MB)，请耐心等待...")
        
        df = load_batch_data(selected_file_path)
        
        if df is not None:
            update_progress(40, f"✅ 数据加载完成，共 {len(df):,} 条记录")
            
            # 第一步筛选
            df_step1 = filter_cos_data_step1(
                df, 
                wavelength_min, 
                wavelength_max,
                progress_callback=update_progress
            )
            
            if df_step1 is not None and not df_step1.empty:
                # 保存第一步结果到session state
                st.session_state.step1_result = df_step1
                st.session_state.step1_params = {
                    'wavelength_min': wavelength_min,
                    'wavelength_max': wavelength_max,
                    'required_count': required_count
                }
                
                update_progress(100, f"✅ 第一步筛选完成！找到 {len(df_step1)} 条符合条件的记录")
                
                # 清除进度条
                import time
                time.sleep(0.8)
                progress_bar.empty()
                status_text.empty()
            else:
                progress_bar.progress(100)
                status_text.empty()
                progress_bar.empty()
                st.warning("⚠️ 未找到符合条件的数据，请调整筛选条件。")
                st.session_state.step1_result = None
        else:
            progress_bar.empty()
            status_text.empty()
            st.error("❌ 数据加载失败")
    
    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f"❌ 处理过程中出错: {str(e)}")

# 显示第一步筛选结果和ItemNum选择
if st.session_state.step1_result is not None:
    st.markdown("---")
    st.markdown("### 📋 第二步：ItemNum筛选")
    
    df_step1 = st.session_state.step1_result
    params = st.session_state.step1_params
    
    # 获取第一步结果中的ItemNum
    available_itemnums = get_available_itemnums(df_step1)
    
    if available_itemnums:
        # 显示ItemNum统计
        itemnum_counts = df_step1['ItemNum'].value_counts()
        
        # ItemNum选择
        selected_itemnums = st.multiselect(
            "选择ItemNum（留空表示不筛选，直接使用第一步结果）",
            options=available_itemnums,
            default=None,
            help="可以选择一个或多个ItemNum进行进一步筛选"
        )
        
        # 显示每个ItemNum的数量
        with st.expander("📈 查看各ItemNum的数量分布"):
            itemnum_df = pd.DataFrame({
                'ItemNum': itemnum_counts.index,
                '数量': itemnum_counts.values
            })
            st.dataframe(itemnum_df, use_container_width=True, hide_index=True)
    else:
        selected_itemnums = []
        st.info("ℹ️ 第一步筛选结果中未找到ItemNum数据")
    
    # 完成筛选按钮
    if st.button("✅ 完成筛选", type="primary", use_container_width=True):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            def update_progress(percent, message):
                progress_bar.progress(percent)
                status_text.text(message)
            
            # 第二步筛选
            itemnum_filter = selected_itemnums if selected_itemnums else None
            df_filtered = filter_cos_data_step2(
                df_step1,
                params['wavelength_min'],
                params['wavelength_max'],
                params['required_count'],
                itemnum_filter,
                progress_callback=update_progress
            )
            
            if df_filtered is not None and not df_filtered.empty:
                update_progress(100, f"✅ 筛选完成！")
                
                import time
                time.sleep(0.5)
                progress_bar.empty()
                status_text.empty()
                
                st.markdown("---")
                st.markdown("### 📊 最终筛选结果")
                
                # 显示统计信息
                stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
                
                good_count = (df_filtered['仓库'] == '良品仓').sum() if '仓库' in df_filtered.columns else 0
                rd_count = (df_filtered['仓库'] == '研发工程仓').sum() if '仓库' in df_filtered.columns else 0
                avg_wl = df_filtered['2A波长_numeric'].mean() if '2A波长_numeric' in df_filtered.columns else 0
                
                target_wl = (params['wavelength_min'] + params['wavelength_max']) / 2
                
                stat_col1.metric("筛选结果数量", len(df_filtered))
                stat_col2.metric("良品仓数量", good_count)
                stat_col3.metric("研发工程仓数量", rd_count)
                stat_col4.metric("平均2A波长", f"{avg_wl:.2f} nm" if avg_wl > 0 else "N/A")
                
                # 显示筛选条件信息
                filter_info = f"🎯 目标波长（范围中间值）: {target_wl:.2f} nm | 筛选范围: {params['wavelength_min']:.2f} - {params['wavelength_max']:.2f} nm"
                if itemnum_filter:
                    filter_info += f" | ItemNum: {', '.join(map(str, itemnum_filter))}"
                st.info(filter_info)
                
                # 显示LOT | SN号码列表
                st.markdown("#### 📝 LOT | SN 号码列表")
                if 'LOT | SN' in df_filtered.columns:
                    lot_sn_text = "\n".join(df_filtered['LOT | SN'].astype(str))
                    st.text_area(
                        "LOT | SN 号码（每行一个）",
                        value=lot_sn_text,
                        height=200
                    )
                
                # 显示详细数据表
                st.markdown("#### 📋 详细数据")
                
                # 选择要显示的关键列
                display_columns = ['LOT | SN', '仓库', '2A波长', 'ItemNum', '料件名称', '料件规格', 
                                 '入库时间', '是否隔离', '库存状态']
                available_columns = [col for col in display_columns if col in df_filtered.columns]
                
                st.dataframe(
                    df_filtered[available_columns],
                    use_container_width=True,
                    hide_index=True,
                    height=400
                )
                
            else:
                progress_bar.progress(100)
                status_text.empty()
                progress_bar.empty()
                st.warning("⚠️ 未找到符合条件的数据，请调整筛选条件。")
        
        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"❌ 处理过程中出错: {str(e)}")

# 缓存管理工具
with st.expander("缓存管理工具"):
    st.markdown("### 批量缓存管理")
    
    # 查找所有缓存文件
    all_cache_files = list(DATA_DIR.glob(".cache_*.pkl"))
    
    if all_cache_files:
        total_size = sum(f.stat().st_size for f in all_cache_files) / (1024 * 1024)
        st.info(f"📊 当前共有 {len(all_cache_files)} 个缓存文件，总大小: {total_size:.1f} MB")
        
        # 显示缓存文件列表
        cache_data = []
        for cache_file in all_cache_files:
            size_mb = cache_file.stat().st_size / (1024 * 1024)
            from datetime import datetime
            mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
            cache_data.append({
                "文件名": cache_file.name,
                "大小(MB)": f"{size_mb:.2f}",
                "修改时间": mtime.strftime("%Y-%m-%d %H:%M:%S")
            })
        
        if cache_data:
            st.dataframe(pd.DataFrame(cache_data), use_container_width=True, hide_index=True)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ 清除所有缓存", type="secondary", use_container_width=True):
                try:
                    deleted_count = 0
                    for cache_file in all_cache_files:
                        cache_file.unlink()
                        deleted_count += 1
                    st.success(f"✅ 已清除 {deleted_count} 个缓存文件，释放 {total_size:.1f} MB 空间")
                    st.rerun()
                except Exception as e:
                    st.error(f"清除失败: {e}")
        
        with col2:
            st.markdown("**说明**: 清除缓存后，下次加载会重新从Excel读取")
    else:
        st.info("ℹ️ 当前没有缓存文件")

