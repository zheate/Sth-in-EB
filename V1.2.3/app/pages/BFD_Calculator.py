import json
import streamlit as st

from config import get_config_path

# 配置页面（仅在独立运行时使用）
try:
    st.set_page_config(page_title="后焦距计算器", page_icon="🔧", layout="wide")
except:
    pass  # 如果已经配置过，忽略错误

# 文件路径
MATERIAL_FILE = get_config_path("material.json")
INPUT_FILE = get_config_path("BFD_Calculator_input.json")


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
            "has_endcap": saved_inputs.get("has_endcap", False),
            "endcap_material": saved_inputs.get("endcap_material", "SK1310_976"),
            "endcap_length": saved_inputs.get("endcap_length") or "5.0",
        }
    
    if "show_material_manager" not in st.session_state:
        st.session_state.show_material_manager = False
    
    ensure_axis_state_defaults()


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


AXIS_CONFIG = {
    "fast": {
        "refr": "re_index_fast",
        "curvature": "curvature_fast",
        "efl": "efl_fast",
        "curvature_widget": "curvature_fast_input",
        "efl_widget": "efl_fast_input",
    },
    "slow": {
        "refr": "re_index_slow",
        "curvature": "curvature_slow",
        "efl": "efl_slow",
        "curvature_widget": "curvature_slow_input",
        "efl_widget": "efl_slow_input",
    },
}


def _to_float(value):
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _set_axis_field(axis, field, value):
    cfg = AXIS_CONFIG[axis]
    storage_key = cfg[field]
    widget_key = cfg.get(f"{field}_widget")
    string_value = "" if value in ("", None) else str(value)
    st.session_state.inputs[storage_key] = string_value
    if widget_key:
        st.session_state[widget_key] = string_value


def ensure_axis_state_defaults():
    axis_state = st.session_state.setdefault("axis_state", {})
    for axis, cfg in AXIS_CONFIG.items():
        state = axis_state.setdefault(axis, {"last": None})
        if state.get("last") is None:
            state["last"] = {
                "refr": st.session_state.inputs.get(cfg["refr"], ""),
                "curvature": st.session_state.inputs.get(cfg["curvature"], ""),
                "efl": st.session_state.inputs.get(cfg["efl"], ""),
            }
        for field in ("curvature", "efl"):
            widget_key = cfg[f"{field}_widget"]
            storage_key = cfg[field]
            st.session_state.setdefault(widget_key, st.session_state.inputs.get(storage_key, ""))


def sync_axis_fields(axis, precision):
    axis_state = st.session_state.setdefault("axis_state", {})
    state = axis_state.setdefault(axis, {"last": None})
    cfg = AXIS_CONFIG[axis]

    current = {}
    current["refr"] = st.session_state.inputs.get(cfg["refr"], "")
    for field in ("curvature", "efl"):
        widget_key = cfg[f"{field}_widget"]
        current[field] = st.session_state.get(
            widget_key,
            st.session_state.inputs.get(cfg[field], "")
        )

    last = state.get("last")
    if not last:
        for field, value in current.items():
            _set_axis_field(axis, field, value)
        state["last"] = current.copy()
        return

    changed = [field for field in current if current[field] != last.get(field)]

    for field, value in current.items():
        _set_axis_field(axis, field, value)

    if not changed:
        state["last"] = current.copy()
        return
    if len(changed) > 1:
        state["last"] = current.copy()
        return

    source = changed[0]
    n = _to_float(current["refr"])
    r = _to_float(current["curvature"])
    efl = _to_float(current["efl"])

    if n is None or n <= 1:
        if source == "refr":
            _set_axis_field(axis, "curvature", "")
            _set_axis_field(axis, "efl", "")
            current["curvature"] = ""
            current["efl"] = ""
        state["last"] = current.copy()
        return

    n_minus_1 = n - 1
    if abs(n_minus_1) < 1e-12:
        if source == "curvature":
            _set_axis_field(axis, "efl", "")
            current["efl"] = ""
        elif source == "efl":
            _set_axis_field(axis, "curvature", "")
            current["curvature"] = ""
        elif source == "refr":
            _set_axis_field(axis, "efl", "")
            _set_axis_field(axis, "curvature", "")
            current["efl"] = ""
            current["curvature"] = ""
        state["last"] = current.copy()
        return

    formatter = f"{{:.{precision}f}}"
    if source == "curvature" and r is not None:
        new_efl = formatter.format(r / n_minus_1)
        _set_axis_field(axis, "efl", new_efl)
        current["efl"] = new_efl
    elif source == "efl" and efl is not None:
        new_r = formatter.format(efl * n_minus_1)
        _set_axis_field(axis, "curvature", new_r)
        current["curvature"] = new_r
    elif source == "refr":
        if r is not None:
            new_efl = formatter.format(r / n_minus_1)
            _set_axis_field(axis, "efl", new_efl)
            current["efl"] = new_efl
        elif efl is not None:
            new_r = formatter.format(efl * n_minus_1)
            _set_axis_field(axis, "curvature", new_r)
            current["curvature"] = new_r

    state["last"] = current.copy()


def material_manager():
    """材料管理界面"""
    st.subheader("📦 材料管理")
    
    materials = st.session_state.materials
    editor_state = st.session_state.setdefault(
        "material_editor_state",
        {"selected": "", "name": "", "index": ""}
    )
    
    # 显示材料列表
    if materials:
        selected_material = st.selectbox(
            "选择材料进行编辑",
            options=[""] + sorted(materials.keys()),
            key="bfd_selected_material_edit"
        )
    else:
        selected_material = ""
        st.info("暂无可用材料，请先新增。")
    
    st.markdown("---")
    
    # 编辑区域
    col1, col2 = st.columns(2)
    
    with col1:
        if selected_material:
            if editor_state.get("selected") != selected_material:
                editor_state["selected"] = selected_material
                editor_state["name"] = selected_material
                editor_state["index"] = str(materials.get(selected_material, "1.5"))
                st.session_state["bfd_edit_material_name"] = editor_state["name"]
                st.session_state["bfd_edit_re_index"] = editor_state["index"]
            material_name = st.text_input("材料名称", key="bfd_edit_material_name")
            re_index = st.text_input("折射率", key="bfd_edit_re_index")
            editor_state["name"] = material_name
            editor_state["index"] = re_index
        else:
            editor_state["selected"] = ""
            material_name = st.text_input("材料名称", key="bfd_new_material_name")
            re_index = st.text_input("折射率", value="1.5", key="bfd_new_re_index")
    
    with col2:
        st.write("")  # 占位
        st.write("")  # 占位
        
        btn_col1, btn_col2, btn_col3 = st.columns(3)
        
        with btn_col1:
            if st.button("💾 保存", key="bfd_save_material"):
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
            if st.button("🗑️ 删除", disabled=not selected_material, key="bfd_delete_material"):
                if selected_material in materials:
                    del materials[selected_material]
                    save_json(materials, MATERIAL_FILE)
                    st.success(f"材料 '{selected_material}' 已删除")
                    st.rerun()
        
        with btn_col3:
            if st.button("❌ 关闭", key="bfd_close_material"):
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
        if st.button("📦 管理材料", key="bfd_manage_material"):
            st.session_state.show_material_manager = not st.session_state.show_material_manager
            st.rerun()
    
    with col2:
        if st.button("ℹ️ 公式说明", key="bfd_formula_help"):
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
            BFD_{FOC} = EFL_{FOC} - \frac{T_{FOC}}{n_{FOC}} + \frac{T_{SOC} \times (n_{SOC} - 1)}{n_{SOC}} + \Delta_{端帽}
            """)
            
            st.markdown("#### 🔵 慢轴后焦距 (SOC BFD):")
            st.latex(r"""
            BFD_{SOC} = EFL_{SOC} - \frac{T_{SOC}}{n_{SOC}} + \Delta_{端帽}
            """)
            
            st.markdown("#### 🔬 端帽影响:")
            st.latex(r"""
            \Delta_{端帽} = L_{端帽} \times \frac{n_{端帽} - 1}{n_{端帽}}
            """)
            st.markdown("其中 $L_{端帽}$ 为端帽长度，$n_{端帽}$ 为端帽折射率")
            
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
    
    # 固定精度为3位小数
    precision = 3
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
        
        # 检测材质是否改变，如果改变则更新折射率并重新渲染
        if material_fast != st.session_state.inputs.get("material_fast"):
            st.session_state.inputs["material_fast"] = material_fast
            if material_fast != "Custom" and material_fast in st.session_state.materials:
                st.session_state.inputs["re_index_fast"] = st.session_state.materials[material_fast]
            st.rerun()
        
        # 显示折射率
        if material_fast != "Custom" and material_fast in st.session_state.materials:
            # 预设材质：只读显示（使用 markdown）
            re_index_fast = st.session_state.materials[material_fast]
            st.session_state.inputs["re_index_fast"] = re_index_fast
            st.markdown("**折射率**")
            st.markdown(f'<div style="background-color: #f0f2f6; padding: 10px; border-radius: 5px; color: #666;">{re_index_fast}</div>', unsafe_allow_html=True)
        else:
            # Custom：可编辑
            re_index_fast = st.text_input("折射率", value=st.session_state.inputs["re_index_fast"], key="re_index_fast_input")
            st.session_state.inputs["re_index_fast"] = re_index_fast

        sync_axis_fields("fast", precision)

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
        
        # 检测材质是否改变，如果改变则更新折射率并重新渲染
        if material_slow != st.session_state.inputs.get("material_slow"):
            st.session_state.inputs["material_slow"] = material_slow
            if material_slow != "Custom" and material_slow in st.session_state.materials:
                st.session_state.inputs["re_index_slow"] = st.session_state.materials[material_slow]
            st.rerun()
        
        # 显示折射率
        if material_slow != "Custom" and material_slow in st.session_state.materials:
            # 预设材质：只读显示（使用 markdown）
            re_index_slow = st.session_state.materials[material_slow]
            st.session_state.inputs["re_index_slow"] = re_index_slow
            st.markdown("**折射率**")
            st.markdown(f'<div style="background-color: #f0f2f6; padding: 10px; border-radius: 5px; color: #666;">{re_index_slow}</div>', unsafe_allow_html=True)
        else:
            # Custom：可编辑
            re_index_slow = st.text_input("折射率", value=st.session_state.inputs["re_index_slow"], key="re_index_slow_input")
            st.session_state.inputs["re_index_slow"] = re_index_slow

        sync_axis_fields("slow", precision)

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
    
    # 计算按钮和端帽设置并排
    calc_col1, calc_col2 = st.columns([1, 3])
    
    with calc_col1:
        calculate_button = st.button("🧮 计算 BFD", type="primary", key="bfd_calculate", use_container_width=True)
    
    with calc_col2:
        endcap_sub_col1, endcap_sub_col2, endcap_sub_col3 = st.columns([1, 1.5, 1.5])
        
        with endcap_sub_col1:
            has_endcap = st.checkbox(
                "包含端帽",
                value=st.session_state.inputs.get("has_endcap", False),
                key="bfd_has_endcap_checkbox",
                help="勾选此项以考虑端帽对焦距的影响"
            )
            st.session_state.inputs["has_endcap"] = has_endcap
        
        endcap_material = None
        endcap_n = None
        endcap_length_val = None
        
        if has_endcap:
            with endcap_sub_col2:
                # 端帽材料选择
                endcap_materials_list = sorted(st.session_state.materials.keys())
                current_endcap_material = st.session_state.inputs.get("endcap_material", "SK1310_976")
                
                if current_endcap_material not in endcap_materials_list:
                    if "SK1310_976" in endcap_materials_list:
                        current_endcap_material = "SK1310_976"
                    else:
                        current_endcap_material = endcap_materials_list[0] if endcap_materials_list else "air"
                
                # 获取端帽折射率
                endcap_n = float(st.session_state.materials.get(current_endcap_material, 1.45))
                
                endcap_material = st.selectbox(
                    "端帽材料",
                    options=endcap_materials_list,
                    index=endcap_materials_list.index(current_endcap_material) if current_endcap_material in endcap_materials_list else 0,
                    key="bfd_endcap_material_select",
                    help=f"选择端帽材料 (当前折射率: {endcap_n})"
                )
                st.session_state.inputs["endcap_material"] = endcap_material
                
                # 更新折射率
                endcap_n = float(st.session_state.materials.get(endcap_material, 1.45))
            
            with endcap_sub_col3:
                existing_length = st.session_state.inputs.get("endcap_length") or "5.0"
                endcap_length = st.text_input(
                    "端帽长度 [mm]",
                    value=existing_length,
                    key="bfd_endcap_length_input",
                    help="输入端帽的长度"
                )
                if not endcap_length.strip():
                    endcap_length = "5.0"
                st.session_state.inputs["endcap_length"] = endcap_length
                
                try:
                    endcap_length_val = float(endcap_length)
                except ValueError:
                    endcap_length_val = 5.0
                    st.session_state.inputs["endcap_length"] = "5.0"
    
    if calculate_button:
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
                
                # 计算端帽影响
                endcap_correction = 0
                # 从session_state获取端帽设置
                has_endcap_calc = st.session_state.inputs.get("has_endcap", False)
                if has_endcap_calc:
                    endcap_material_calc = st.session_state.inputs.get("endcap_material", "")
                    endcap_length_calc = st.session_state.inputs.get("endcap_length", "")
                    
                    if endcap_material_calc and endcap_length_calc:
                        try:
                            endcap_n_calc = float(st.session_state.materials.get(endcap_material_calc, 1.45))
                            endcap_length_val_calc = float(endcap_length_calc)
                            
                            if endcap_length_val_calc > 0:
                                # 端帽带来的焦距影响：厚度 * (端帽折射率 - 1) / 端帽折射率
                                endcap_correction = endcap_length_val_calc * (endcap_n_calc - 1) / endcap_n_calc
                        except (ValueError, TypeError):
                            pass
                
                # 计算 BFD（加上端帽影响）
                bfd_fast = efl_fast_val - (t_fast / n_fast) + (t_slow * (n_slow - 1) / n_slow) + endcap_correction
                bfd_slow = efl_slow_val - (t_slow / n_slow) + endcap_correction
                
                # 显示结果
                st.success("✅ 计算完成！")
                
                result_col1, result_col2 = st.columns(2)
                with result_col1:
                    st.markdown(f"### 🟡 快轴后焦距 (FOC BFD)")
                    st.markdown(f"# {bfd_fast:.{precision}f} mm")
                
                with result_col2:
                    st.markdown(f"### 🔵 慢轴后焦距 (SOC BFD)")
                    st.markdown(f"# {bfd_slow:.{precision}f} mm")
                
                # 显示端帽影响
                if has_endcap_calc and abs(endcap_correction) > 0.001:
                    st.markdown("---")
                    st.info(f"🔬 端帽影响: {endcap_correction:+.{precision}f} mm (材料: {endcap_material_calc}, 长度: {endcap_length_val_calc:.{precision}f} mm, 折射率: {endcap_n_calc})")
                
                # 保存输入（端帽信息已经在输入时保存到session_state）
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
                    "has_endcap": has_endcap_calc,
                    "endcap_material": endcap_material_calc if has_endcap_calc else "",
                    "endcap_length": endcap_length_calc if has_endcap_calc else "",
                }, INPUT_FILE)
                
            except Exception as e:
                st.error(f"计算错误: {str(e)}")


if __name__ == "__main__":
    main()
