import json
import streamlit as st
from pathlib import Path

# 配置页面
st.set_page_config(page_title="后焦距计算器", page_icon="🔧", layout="wide")

# 文件路径
MATERIAL_FILE = Path("material.json")
INPUT_FILE = Path("BFD_Calculator_input.json")


def load_json(filename, default_data):
    """从JSON文件加载数据"""
    try:
        with open(filename, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default_data


def save_json(data, filename):
    """保存数据到JSON文件"""
    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        return True
    except Exception as e:
        st.error(f"保存失败: {str(e)}")
        return False


def validate_float(text, condition, error_msg, field_name=""):
    """验证浮点数输入"""
    if not text:
        return False, f"{field_name}不能为空"
    try:
        value = float(text)
        if not condition(value):
            return False, f"{field_name}{error_msg}"
        return True, ""
    except ValueError:
        return False, f"{field_name}必须是数字"


def init_session_state():
    """初始化 session state"""
    if "materials" not in st.session_state:
        st.session_state.materials = load_json(MATERIAL_FILE, {"ZF52-976": "1.8145"})
    
    if "inputs" not in st.session_state:
        saved_inputs = load_json(INPUT_FILE, {})
        st.session_state.inputs = {
            "material_fast": saved_inputs.get("material_fast", "ZF52-976"),
            "re_index_fast": saved_inputs.get("re_index_fast", "1.8145"),
            "curvature_fast": saved_inputs.get("foc_curvature", ""),
            "efl_fast": saved_inputs.get("foc_efl", ""),
            "thickness_fast": saved_inputs.get("foc_thickness", ""),
            "material_slow": saved_inputs.get("material_slow", "ZF52-976"),
            "re_index_slow": saved_inputs.get("re_index_slow", "1.8145"),
            "curvature_slow": saved_inputs.get("soc_curvature", ""),
            "efl_slow": saved_inputs.get("soc_efl", ""),
            "thickness_slow": saved_inputs.get("soc_thickness", ""),
            "precision": saved_inputs.get("precision", 3),
        }
    
    if "show_material_manager" not in st.session_state:
        st.session_state.show_material_manager = False


def calculate_related_param(n, r, efl, source_type):
    """计算相关参数 (R 或 EFL)"""
    if n is None or n <= 1:
        return None
    
    n_minus_1 = n - 1
    if n_minus_1 == 0:
        return None
    
    if source_type == "r" and r is not None:
        return r / n_minus_1  # 计算 EFL
    elif source_type == "efl" and efl is not None:
        return efl * n_minus_1  # 计算 R
    return None


def material_manager():
    """材料管理界面"""
    st.subheader("📦 材料管理")
    
    materials = st.session_state.materials
    
    # 搜索框
    search = st.text_input("🔍 搜索材料", key="material_search")
    
    # 过滤材料
    filtered_materials = {
        k: v for k, v in materials.items()
        if search.lower() in k.lower() or search.lower() in str(v)
    }
    
    # 显示材料列表
    if filtered_materials:
        selected_material = st.selectbox(
            "选择材料进行编辑",
            options=[""] + list(filtered_materials.keys()),
            key="selected_material_edit"
        )
    else:
        selected_material = ""
        st.warning("未找到匹配的材料")
    
    st.markdown("---")
    
    # 编辑区域
    col1, col2 = st.columns(2)
    
    with col1:
        if selected_material:
            material_name = st.text_input("材料名称", value=selected_material, key="edit_material_name")
            re_index = st.text_input("折射率", value=materials.get(selected_material, "1.5"), key="edit_re_index")
        else:
            material_name = st.text_input("材料名称", key="new_material_name")
            re_index = st.text_input("折射率", value="1.5", key="new_re_index")
    
    with col2:
        st.write("")  # 占位
        st.write("")  # 占位
        
        btn_col1, btn_col2, btn_col3 = st.columns(3)
        
        with btn_col1:
            if st.button("💾 保存", use_container_width=True):
                valid, msg = validate_float(re_index, lambda x: x > 1, "必须大于1", "折射率")
                if not material_name.strip():
                    st.error("材料名称不能为空")
                elif not valid:
                    st.error(msg)
                elif material_name == "Custom":
                    st.error("不能将材质命名为 'Custom'")
                else:
                    materials[material_name] = re_index
                    save_json(materials, MATERIAL_FILE)
                    st.success(f"材料 '{material_name}' 已保存")
                    st.rerun()
        
        with btn_col2:
            if st.button("🗑️ 删除", use_container_width=True, disabled=not selected_material):
                if selected_material in materials:
                    del materials[selected_material]
                    save_json(materials, MATERIAL_FILE)
                    st.success(f"材料 '{selected_material}' 已删除")
                    st.rerun()
        
        with btn_col3:
            if st.button("❌ 关闭", use_container_width=True):
                st.session_state.show_material_manager = False
                st.rerun()
    
    # 显示所有材料
    st.markdown("---")
    st.markdown("### 📋 材料列表")
    if materials:
        material_data = [{"材料名称": k, "折射率": v} for k, v in sorted(materials.items())]
        st.dataframe(material_data, use_container_width=True, hide_index=True)
    else:
        st.info("暂无材料数据")


def main():
    init_session_state()
    
    st.title("🔧 后焦距计算器")
    
    # 顶部按钮
    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        if st.button("📦 管理材料", use_container_width=True):
            st.session_state.show_material_manager = not st.session_state.show_material_manager
            st.rerun()
    
    with col2:
        if st.button("ℹ️ 公式说明", use_container_width=True):
            st.session_state.show_formula = not st.session_state.get("show_formula", False)
    
    # 显示材料管理器
    if st.session_state.show_material_manager:
        material_manager()
        return
    
    # 显示公式说明
    if st.session_state.get("show_formula", False):
        with st.expander("📐 计算公式说明", expanded=True):
            st.markdown("### 后焦距 (BFD) 计算公式")
            
            st.markdown("#### 🟡 快轴后焦距 (FOC BFD):")
            st.latex(r"""
            BFD_{FOC} = EFL_{FOC} - \frac{T_{FOC}}{n_{FOC}} + \frac{T_{SOC} \times (n_{SOC} - 1)}{n_{SOC}}
            """)
            
            st.markdown("#### 🔵 慢轴后焦距 (SOC BFD):")
            st.latex(r"""
            BFD_{SOC} = EFL_{SOC} - \frac{T_{SOC}}{n_{SOC}}
            """)
            
            st.markdown("---")
            st.markdown("### 辅助公式")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**有效焦距计算:**")
                st.latex(r"EFL = \frac{R}{n - 1}")
            
            with col2:
                st.markdown("**曲率半径计算:**")
                st.latex(r"R = EFL \times (n - 1)")
            
            st.markdown("---")
            st.markdown("### 📋 符号说明")
            
            symbols_data = {
                "符号": ["EFL", "R", "T", "n", "BFD", "FOC", "SOC"],
                "含义": [
                    "有效焦距 (Effective Focal Length)",
                    "曲率半径 (Radius of Curvature)",
                    "中心厚度 (Thickness)",
                    "材料折射率 (Refractive Index)",
                    "后焦距 (Back Focal Distance)",
                    "快轴 (Fast Axis)",
                    "慢轴 (Slow Axis)"
                ],
                "单位": ["mm", "mm", "mm", "无量纲", "mm", "-", "-"]
            }
            
            import pandas as pd
            st.dataframe(pd.DataFrame(symbols_data), use_container_width=True, hide_index=True)
    
    # 精度设置
    precision = st.slider("计算精度（小数位数）", 1, 6, st.session_state.inputs["precision"], key="precision_slider")
    st.session_state.inputs["precision"] = precision
    
    st.markdown("---")
    
    # 快轴和慢轴参数
    col_fast, col_slow = st.columns(2)
    
    materials_list = ["Custom"] + sorted(st.session_state.materials.keys())
    
    # === 快轴参数 ===
    with col_fast:
        st.markdown("### 🟡 快轴参数")
        
        material_fast = st.selectbox(
            "材质",
            options=materials_list,
            index=materials_list.index(st.session_state.inputs["material_fast"]) if st.session_state.inputs["material_fast"] in materials_list else 0,
            key="material_fast_select"
        )
        
        # 更新折射率
        if material_fast != "Custom" and material_fast in st.session_state.materials:
            default_re_fast = st.session_state.materials[material_fast]
            st.session_state.inputs["re_index_fast"] = default_re_fast
        else:
            default_re_fast = st.session_state.inputs["re_index_fast"]
        
        re_index_fast = st.text_input("折射率", value=default_re_fast, key="re_index_fast_input")
        
        # 自动计算并更新 R 或 EFL
        try:
            n_fast = float(re_index_fast) if re_index_fast else None
            
            # 获取当前输入值
            curvature_fast_input = st.session_state.inputs.get("curvature_fast", "")
            efl_fast_input = st.session_state.inputs.get("efl_fast", "")
            
            # 如果有 R 但没有 EFL，自动计算 EFL
            if n_fast and n_fast > 1 and curvature_fast_input and not efl_fast_input:
                r_fast = float(curvature_fast_input)
                calc_efl = r_fast / (n_fast - 1)
                st.session_state.inputs["efl_fast"] = f"{calc_efl:.{precision}f}"
            
            # 如果有 EFL 但没有 R，自动计算 R
            elif n_fast and n_fast > 1 and efl_fast_input and not curvature_fast_input:
                efl_fast_val = float(efl_fast_input)
                calc_r = efl_fast_val * (n_fast - 1)
                st.session_state.inputs["curvature_fast"] = f"{calc_r:.{precision}f}"
        except (ValueError, ZeroDivisionError):
            pass
        
        curvature_fast = st.text_input(
            "曲率半径 (R) [mm]", 
            value=st.session_state.inputs["curvature_fast"], 
            key="curvature_fast_input",
            help="输入曲率半径，或留空由 EFL 自动计算"
        )
        
        efl_fast = st.text_input(
            "有效焦距 (EFL) [mm]", 
            value=st.session_state.inputs["efl_fast"], 
            key="efl_fast_input",
            help="输入有效焦距，或留空由 R 自动计算"
        )
        
        thickness_fast = st.text_input("中心厚度 (T) [mm]", value=st.session_state.inputs["thickness_fast"], key="thickness_fast_input")
        
        # 更新 session state
        st.session_state.inputs["curvature_fast"] = curvature_fast
        st.session_state.inputs["efl_fast"] = efl_fast
        st.session_state.inputs["thickness_fast"] = thickness_fast
        st.session_state.inputs["re_index_fast"] = re_index_fast
        st.session_state.inputs["material_fast"] = material_fast
    
    # === 慢轴参数 ===
    with col_slow:
        st.markdown("### 🔵 慢轴参数")
        
        material_slow = st.selectbox(
            "材质",
            options=materials_list,
            index=materials_list.index(st.session_state.inputs["material_slow"]) if st.session_state.inputs["material_slow"] in materials_list else 0,
            key="material_slow_select"
        )
        
        # 更新折射率
        if material_slow != "Custom" and material_slow in st.session_state.materials:
            default_re_slow = st.session_state.materials[material_slow]
            st.session_state.inputs["re_index_slow"] = default_re_slow
        else:
            default_re_slow = st.session_state.inputs["re_index_slow"]
        
        re_index_slow = st.text_input("折射率", value=default_re_slow, key="re_index_slow_input")
        
        # 自动计算并更新 R 或 EFL
        try:
            n_slow = float(re_index_slow) if re_index_slow else None
            
            # 获取当前输入值
            curvature_slow_input = st.session_state.inputs.get("curvature_slow", "")
            efl_slow_input = st.session_state.inputs.get("efl_slow", "")
            
            # 如果有 R 但没有 EFL，自动计算 EFL
            if n_slow and n_slow > 1 and curvature_slow_input and not efl_slow_input:
                r_slow = float(curvature_slow_input)
                calc_efl = r_slow / (n_slow - 1)
                st.session_state.inputs["efl_slow"] = f"{calc_efl:.{precision}f}"
            
            # 如果有 EFL 但没有 R，自动计算 R
            elif n_slow and n_slow > 1 and efl_slow_input and not curvature_slow_input:
                efl_slow_val = float(efl_slow_input)
                calc_r = efl_slow_val * (n_slow - 1)
                st.session_state.inputs["curvature_slow"] = f"{calc_r:.{precision}f}"
        except (ValueError, ZeroDivisionError):
            pass
        
        curvature_slow = st.text_input(
            "曲率半径 (R) [mm]", 
            value=st.session_state.inputs["curvature_slow"], 
            key="curvature_slow_input",
            help="输入曲率半径，或留空由 EFL 自动计算"
        )
        
        efl_slow = st.text_input(
            "有效焦距 (EFL) [mm]", 
            value=st.session_state.inputs["efl_slow"], 
            key="efl_slow_input",
            help="输入有效焦距，或留空由 R 自动计算"
        )
        
        thickness_slow = st.text_input("中心厚度 (T) [mm]", value=st.session_state.inputs["thickness_slow"], key="thickness_slow_input")
        
        # 更新 session state
        st.session_state.inputs["curvature_slow"] = curvature_slow
        st.session_state.inputs["efl_slow"] = efl_slow
        st.session_state.inputs["thickness_slow"] = thickness_slow
        st.session_state.inputs["re_index_slow"] = re_index_slow
        st.session_state.inputs["material_slow"] = material_slow
    
    st.markdown("---")
    
    # 计算按钮
    if st.button("🧮 计算 BFD", type="primary", use_container_width=True):
        # 验证输入
        errors = []
        
        # 验证快轴
        valid, msg = validate_float(re_index_fast, lambda x: x > 1, "必须大于1", "快轴折射率")
        if not valid:
            errors.append(msg)
        
        valid, msg = validate_float(thickness_fast, lambda x: x >= 0, "必须为非负数", "快轴厚度")
        if not valid:
            errors.append(msg)
        
        if not curvature_fast and not efl_fast:
            errors.append("快轴参数中，曲率半径 (R) 或有效焦距 (EFL) 必须至少输入一个")
        
        # 验证慢轴
        valid, msg = validate_float(re_index_slow, lambda x: x > 1, "必须大于1", "慢轴折射率")
        if not valid:
            errors.append(msg)
        
        valid, msg = validate_float(thickness_slow, lambda x: x >= 0, "必须为非负数", "慢轴厚度")
        if not valid:
            errors.append(msg)
        
        if not curvature_slow and not efl_slow:
            errors.append("慢轴参数中，曲率半径 (R) 或有效焦距 (EFL) 必须至少输入一个")
        
        if errors:
            for error in errors:
                st.error(error)
        else:
            # 执行计算
            try:
                n_fast = float(re_index_fast)
                t_fast = float(thickness_fast)
                n_slow = float(re_index_slow)
                t_slow = float(thickness_slow)
                
                # 计算或获取 EFL 和 R
                if efl_fast:
                    efl_fast_val = float(efl_fast)
                else:
                    r_fast_val = float(curvature_fast)
                    efl_fast_val = r_fast_val / (n_fast - 1)
                
                if efl_slow:
                    efl_slow_val = float(efl_slow)
                else:
                    r_slow_val = float(curvature_slow)
                    efl_slow_val = r_slow_val / (n_slow - 1)
                
                # 计算 BFD
                bfd_fast = efl_fast_val - (t_fast / n_fast) + (t_slow * (n_slow - 1) / n_slow)
                bfd_slow = efl_slow_val - (t_slow / n_slow)
                
                # 显示结果
                st.success("✅ 计算完成！")
                
                result_col1, result_col2 = st.columns(2)
                with result_col1:
                    st.markdown(f"### 🟡 快轴后焦距 (FOC BFD)")
                    st.markdown(f"# {bfd_fast:.{precision}f} mm")
                
                with result_col2:
                    st.markdown(f"### 🔵 慢轴后焦距 (SOC BFD)")
                    st.markdown(f"# {bfd_slow:.{precision}f} mm")
                
                # 保存输入
                st.session_state.inputs.update({
                    "material_fast": material_fast,
                    "re_index_fast": re_index_fast,
                    "curvature_fast": curvature_fast,
                    "efl_fast": efl_fast,
                    "thickness_fast": thickness_fast,
                    "material_slow": material_slow,
                    "re_index_slow": re_index_slow,
                    "curvature_slow": curvature_slow,
                    "efl_slow": efl_slow,
                    "thickness_slow": thickness_slow,
                    "precision": precision,
                })
                
                save_json({
                    "material_fast": material_fast,
                    "re_index_fast": re_index_fast,
                    "foc_curvature": curvature_fast,
                    "foc_efl": efl_fast,
                    "foc_thickness": thickness_fast,
                    "material_slow": material_slow,
                    "re_index_slow": re_index_slow,
                    "soc_curvature": curvature_slow,
                    "soc_efl": efl_slow,
                    "soc_thickness": thickness_slow,
                    "precision": precision,
                }, INPUT_FILE)
                
            except Exception as e:
                st.error(f"计算错误: {str(e)}")


if __name__ == "__main__":
    main()
