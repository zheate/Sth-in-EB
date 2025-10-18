import json
import math
import streamlit as st
from pathlib import Path
from typing import Any, Dict

# 配置页面
st.set_page_config(page_title="NA 计算器", page_icon="🔬", layout="wide")

# 文件路径
MATERIAL_FILE = Path("material.json")
INPUT_FILE = Path("NA_Calculator_input.json")

# 默认与常量配置
DEFAULT_MATERIALS: Dict[str, float] = {"air": 1.0003}
_FLOAT_TOLERANCE = 1e-9


def _try_float(value: Any) -> float | None:
    """尝试将任意值转换为 float，失败返回 None。"""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_json(filename: Path, default_data: Any) -> Any:
    """从 JSON 文件加载数据，失败时返回 default_data 的拷贝。"""
    try:
        with open(filename, "r", encoding="utf-8") as file_obj:
            return json.load(file_obj)
    except (FileNotFoundError, json.JSONDecodeError):
        return json.loads(json.dumps(default_data))


def _sanitize_material_map(raw: Any, *, emit_warning: bool) -> Dict[str, float]:
    """清洗材料字典并在需要时输出提示。"""
    sanitized: Dict[str, float] = {}
    if not isinstance(raw, dict):
        return sanitized

    for name, value in raw.items():
        index = _try_float(value)
        if index is None:
            if emit_warning:
                st.warning(f"材料“{name}”的折射率无法解析，已忽略。")
            continue
        if index <= 0:
            if emit_warning:
                st.warning(f"材料“{name}”的折射率必须大于 0，已忽略。")
            continue
        sanitized[name] = index

    return sanitized


def load_materials(filename: Path) -> Dict[str, float]:
    """加载材料字典，确保折射率为有效正浮点数。"""
    raw_materials = load_json(filename, DEFAULT_MATERIALS)
    sanitized = _sanitize_material_map(raw_materials, emit_warning=True)

    if not sanitized:
        st.warning("未找到有效的材料数据，已使用默认材料列表。")
        return DEFAULT_MATERIALS.copy()

    return sanitized


def _format_index(value: float, digits: int = 4) -> str:
    """格式化折射率显示，去除多余的尾随零。"""
    formatted = f"{value:.{digits}f}"
    formatted = formatted.rstrip("0").rstrip(".")
    return formatted if formatted else f"{value:.{digits}f}"


def _normalize_text(value: Any) -> str:
    """将输入转换为去除首尾空白的字符串。"""
    if value is None:
        return ""
    return str(value).strip()


def _clear_last_result() -> None:
    """重置最近一次计算的显示结果。"""
    st.session_state.pop("na_last_mode", None)
    st.session_state.pop("na_last_result", None)


def _on_input_change(field_key: str) -> None:
    """标记最近修改的输入字段，用于确定计算方向。"""
    suppressed = st.session_state.pop("na_suppress_on_change_for", None)
    if suppressed == field_key:
        return
    st.session_state["na_active_input"] = field_key
    _clear_last_result()


def _trigger_rerun() -> bool:
    """在支持的 Streamlit 版本上触发重新运行，返回是否成功。"""
    rerun_fn = getattr(st, "experimental_rerun", None)
    if rerun_fn is None:
        rerun_fn = getattr(st, "rerun", None)
    if rerun_fn is not None:
        rerun_fn()
        return True
    st.session_state["na_manual_refresh_required"] = True
    return False


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
    text_str = "" if text is None else str(text).strip()
    if not text_str:
        return False, f"{field_name}不能为空"
    try:
        value = float(text_str)
        if not condition(value):
            return False, f"{field_name}{error_msg}"
        return True, ""
    except ValueError:
        return False, f"{field_name}必须是数字"


def init_session_state():
    """初始化 session state"""
    if "materials" not in st.session_state:
        st.session_state.materials = load_materials(MATERIAL_FILE)
    else:
        sanitized_existing = _sanitize_material_map(
            st.session_state.materials, emit_warning=False
        )
        if sanitized_existing:
            st.session_state.materials = sanitized_existing
        else:
            st.session_state.materials = load_materials(MATERIAL_FILE)
    
    if "na_inputs" not in st.session_state:
        saved_inputs = load_json(INPUT_FILE, {})
        st.session_state.na_inputs = {
            "radius": saved_inputs.get("radius", "1.005"),
            "length": saved_inputs.get("length", "4.457"),
            "material": saved_inputs.get("material", "air"),
            "refractive_index": saved_inputs.get("refractive_index", "1.0003"),
            "na": saved_inputs.get("na", ""),
            "theta": saved_inputs.get("theta", ""),
        }
    
    if "show_material_manager" not in st.session_state:
        st.session_state.show_material_manager = False
    
    # 初始化计算状态
    if "na_calc_state" not in st.session_state:
        st.session_state.na_calc_state = {
            "last_radius": st.session_state.na_inputs["radius"],
            "last_length": st.session_state.na_inputs["length"],
            "last_na": st.session_state.na_inputs["na"],
            "last_refractive_index": st.session_state.na_inputs["refractive_index"],
        }


def calculate_na(radius, length, refractive_index):
    """根据半径、长度和折射率计算 NA。"""
    try:
        if math.isclose(length, 0.0):
            raise ValueError("长度为零，无法计算 NA 值")

        half_angle_rad = math.atan(radius / length)
        na = math.sin(half_angle_rad) * refractive_index
        full_angle_deg = math.degrees(half_angle_rad) * 2

        if na > refractive_index + _FLOAT_TOLERANCE:
            # 浮点精度导致的细小超界，按照桌面版逻辑夹紧到折射率
            na = refractive_index
        elif na < 0:
            raise ValueError("计算出的 NA 为负，异常")

        return na, full_angle_deg, None
    except ValueError as error:
        return None, None, str(error)
    except Exception as error:
        return None, None, f"计算 NA 时发生未知错误：{str(error)}"


def calculate_length(radius, na, refractive_index):
    """根据半径和 NA 计算长度。"""
    try:
        if na > refractive_index + _FLOAT_TOLERANCE:
            raise ValueError(f"NA值 ({na:.4f}) 不能大于折射率 ({refractive_index:.4f})")
        if na < 0:
            raise ValueError("NA值不能为负")
        if math.isclose(na, 0.0):
            return None, None, "NA值为零，无法计算有限长度"

        asin_arg = na / refractive_index
        if not (-1.0 <= asin_arg <= 1.0):
            raise ValueError(f"计算角度时出错：arcsin 的参数 ({asin_arg:.4f}) 超出范围 [-1, 1]")

        theta1 = math.asin(asin_arg)
        tan_theta = math.tan(theta1)
        length = radius / tan_theta
        if length < 0:
            return None, None, "计算出的长度为负，异常"

        full_angle_deg = math.degrees(theta1) * 2
        return length, full_angle_deg, None
    except ValueError as error:
        return None, None, str(error)
    except ZeroDivisionError:
        return None, None, "计算长度时发生除零错误（这不应发生）"
    except Exception as error:
        return None, None, f"计算长度时发生未知错误：{str(error)}"


def update_angle(na, refractive_index):
    """根据 NA 更新角度显示"""
    try:
        if not (0 <= na <= refractive_index):
            return None
        
        if math.isclose(na, 0.0):
            return 0.0
        
        asin_arg = na / refractive_index
        if asin_arg > 1.0:
            asin_arg = 1.0
        if asin_arg < -1.0:
            asin_arg = -1.0
        
        angle_rad_half = math.asin(asin_arg)
        angle_deg = math.degrees(angle_rad_half) * 2
        
        return angle_deg
    
    except Exception:
        return None


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
            key="selected_material_edit"
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
                editor_state["index"] = _format_index(materials.get(selected_material, 1.5))
                st.session_state["edit_material_name"] = editor_state["name"]
                st.session_state["edit_re_index"] = editor_state["index"]
            material_name = st.text_input("材料名称", key="edit_material_name")
            re_index = st.text_input("折射率", key="edit_re_index")
            editor_state["name"] = material_name
            editor_state["index"] = re_index
        else:
            editor_state["selected"] = ""
            material_name = st.text_input("材料名称", key="new_material_name")
            re_index = st.text_input("折射率", value="1.5", key="new_re_index")
    
    with col2:
        st.write("")
        st.write("")
        
        btn_col1, btn_col2, btn_col3 = st.columns(3)
        
        with btn_col1:
            if st.button("💾 保存", use_container_width=True):
                valid, msg = validate_float(re_index, lambda x: x > 0, "必须大于0", "折射率")
                if not material_name.strip():
                    st.error("材料名称不能为空")
                elif not valid:
                    st.error(msg)
                else:
                    index_value = _try_float(re_index)
                    if index_value is None:
                        st.error("折射率必须是有效数字")
                    else:
                        materials[material_name] = index_value
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
        material_data = [{"材料名称": k, "折射率": _format_index(v)} for k, v in sorted(materials.items())]
        st.dataframe(material_data, use_container_width=True, hide_index=True)
    else:
        st.info("暂无材料数据")


def main():
    init_session_state()

    pending_updates = st.session_state.pop("na_pending_widget_updates", None)

    if "radius_input" not in st.session_state:
        st.session_state["radius_input"] = _normalize_text(st.session_state.na_inputs.get("radius", ""))
    if "length_input" not in st.session_state:
        st.session_state["length_input"] = _normalize_text(st.session_state.na_inputs.get("length", ""))
    if "na_input" not in st.session_state:
        st.session_state["na_input"] = _normalize_text(st.session_state.na_inputs.get("na", ""))

    if pending_updates:
        suppress_key = pending_updates.pop("_suppress", None)
        if suppress_key:
            st.session_state["na_suppress_on_change_for"] = suppress_key
        for key, value in pending_updates.items():
            st.session_state[key] = value
    
    manual_refresh_required = st.session_state.pop("na_manual_refresh_required", False)
    
    st.title("🔬 NA 计算器")
    
    if manual_refresh_required:
        st.info("请再次执行任意操作以刷新输入框显示。")
    
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
            st.markdown("### 数值孔径 (NA) 计算公式")
            
            st.markdown("#### 根据几何参数计算 NA:")
            st.latex(r"""
            NA = n \times \sin(\theta_1)
            """)
            st.latex(r"""
            \theta_1 = \arctan\left(\frac{r}{L}\right)
            """)
            
            st.markdown("#### 根据 NA 计算长度:")
            st.latex(r"""
            L = \frac{r}{\tan(\theta_1)}
            """)
            st.latex(r"""
            \theta_1 = \arcsin\left(\frac{NA}{n}\right)
            """)
            
            st.markdown("---")
            st.markdown("### 📋 符号说明")
            
            symbols_data = {
                "符号": ["NA", "n", "r", "L", "θ₁", "2θ₁"],
                "含义": [
                    "数值孔径 (Numerical Aperture)",
                    "材料折射率 (Refractive Index)",
                    "小孔半径 (Aperture Radius)",
                    "光纤端面到小孔的距离 (Distance)",
                    "半角 (Half Angle)",
                    "全角 (Full Angle)"
                ],
                "单位": ["无量纲", "无量纲", "mm", "mm", "弧度/度", "弧度/度"]
            }
            
            import pandas as pd
            st.dataframe(pd.DataFrame(symbols_data), use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # 参数输入区域
    st.markdown("### 📊 参数输入")
    
    col1, col2 = st.columns(2)
    
    with col1:
        radius = st.text_input(
            "小孔半径 (r) [mm]",
            key="radius_input",
            help="输入小孔半径",
            on_change=_on_input_change,
            args=("radius_input",),
        )
        
        length = st.text_input(
            "光纤端面到小孔的距离 (L) [mm]",
            key="length_input",
            help="输入长度，或留空由 NA 自动计算",
            on_change=_on_input_change,
            args=("length_input",),
        )
    
    material_changed = False
    refractive_index_value = DEFAULT_MATERIALS["air"]
    refractive_index_display = _format_index(refractive_index_value)
    
    with col2:
        materials_list = sorted(st.session_state.materials.keys())
        current_material = st.session_state.na_inputs.get("material", "")
        
        if not materials_list:
            materials_list = [current_material or "air"]
        
        if current_material not in materials_list:
            current_material = "air" if "air" in materials_list else materials_list[0]
            st.session_state.na_inputs["material"] = current_material
        
        material = st.selectbox(
            "选择材料",
            options=materials_list,
            index=materials_list.index(current_material) if current_material in materials_list else 0,
            key="material_select"
        )
        
        material_changed = material != st.session_state.na_inputs.get("material")
        if material_changed:
            st.session_state.na_inputs["material"] = material
        
        raw_index_value = st.session_state.materials.get(material, DEFAULT_MATERIALS["air"])
        refractive_index_value = _try_float(raw_index_value) or DEFAULT_MATERIALS["air"]
        if refractive_index_value <= 0:
            refractive_index_value = DEFAULT_MATERIALS["air"]
        
        refractive_index_display = _format_index(refractive_index_value)
        st.session_state.na_inputs["refractive_index"] = refractive_index_display
        st.markdown("**折射率 (n)**")
        st.markdown(
            f'<div style="background-color: #f0f2f6; padding: 10px; border-radius: 5px; color: #666;">{refractive_index_display}</div>',
            unsafe_allow_html=True,
        )
        
        na = st.text_input(
            "NA 值",
            key="na_input",
            help="输入 NA 值，或留空由长度自动计算",
            on_change=_on_input_change,
            args=("na_input",),
        )
    
    st.markdown("---")
    
    # 同步输入状态，便于与桌面版逻辑保持一致
    radius = _normalize_text(radius)
    length = _normalize_text(length)
    na = _normalize_text(na)

    last_mode_snapshot = st.session_state.get("na_last_mode")
    last_result_snapshot = st.session_state.get("na_last_result")
    if last_mode_snapshot == "length_to_na" and last_result_snapshot:
        if na != last_result_snapshot.get("na"):
            _clear_last_result()
    elif last_mode_snapshot == "na_to_length" and last_result_snapshot:
        if length != last_result_snapshot.get("length"):
            _clear_last_result()

    st.session_state.na_inputs["radius"] = radius
    st.session_state.na_inputs["length"] = length
    st.session_state.na_inputs["na"] = na

    calc_state = st.session_state.get("na_calc_state", {})
    last_length_cached = calc_state.get("last_length", "")
    last_na_cached = calc_state.get("last_na", "")
    length_changed = length != last_length_cached
    na_changed = na != last_na_cached

    radius_val = _try_float(radius)
    length_val = _try_float(length)
    na_val_existing = _try_float(na)

    if material_changed:
        if na_val_existing is not None and na_val_existing - refractive_index_value > _FLOAT_TOLERANCE:
            st.warning(
                f"当前 NA 值 ({na_val_existing:.4f}) 大于所选材料的折射率 ({refractive_index_value:.4f})，已按桌面版逻辑清除。"
            )
            st.session_state["na_suppress_on_change_for"] = "na_input"
            st.session_state["na_input"] = ""
            st.session_state.na_inputs["na"] = ""
            st.session_state.na_inputs["theta"] = ""
            _clear_last_result()
            na = ""
            na_val_existing = None
            if radius_val and radius_val > 0 and length_val and length_val > 0:
                recalculated_na, theta_val, error = calculate_na(radius_val, length_val, refractive_index_value)
                if error:
                    st.warning(error)
                else:
                    new_na_str = f"{recalculated_na:.4f}"
                    st.session_state["na_suppress_on_change_for"] = "na_input"
                    st.session_state["na_input"] = new_na_str
                    st.session_state.na_inputs["na"] = new_na_str
                    st.session_state.na_inputs["theta"] = f"{theta_val:.3f}"
                    st.session_state.na_calc_state.update({
                        "last_radius": radius,
                        "last_na": new_na_str,
                        "last_refractive_index": refractive_index_display,
                    })
                    na = new_na_str
                    na_val_existing = recalculated_na
        else:
            na = _normalize_text(st.session_state.get("na_input", na))

    # 计算按钮
    if st.button("🧮 计算", type="primary", use_container_width=True):
        # 验证输入
        errors = []
        
        # 验证半径
        valid, msg = validate_float(radius, lambda x: x > 0, "必须大于0", "小孔半径")
        if not valid:
            errors.append(msg)
        
        # 验证折射率
        valid, msg = validate_float(refractive_index_display, lambda x: x > 0, "必须大于0", "折射率")
        if not valid:
            errors.append(msg)
        
        # 检查是否至少输入了长度或 NA
        if not length and not na:
            errors.append("长度 (L) 或 NA 值必须至少输入一个")
        
        if errors:
            _clear_last_result()
            for error in errors:
                st.error(error)
        else:
            try:
                radius_val = _try_float(radius)
                if radius_val is None or radius_val <= 0:
                    raise ValueError("小孔半径必须大于0")

                refractive_index_val = refractive_index_value

                active_input = st.session_state.get("na_active_input")

                mode = None
                if na and not length:
                    mode = "na"
                elif length and not na:
                    mode = "length"
                elif na and length:
                    if active_input == "na_input":
                        mode = "na"
                    elif active_input == "length_input":
                        mode = "length"
                    elif na_changed and not length_changed:
                        mode = "na"
                    elif length_changed and not na_changed:
                        mode = "length"
                    elif na_changed and length_changed:
                        mode = "na"
                    else:
                        last_mode = st.session_state.get("na_last_mode")
                        if last_mode == "na_to_length":
                            mode = "na"
                        elif last_mode == "length_to_na":
                            mode = "length"
                if mode is None and (na or length):
                    mode = "na" if na else "length"

                calculate_from_na = mode == "na"
                calculate_from_length = mode == "length"

                if calculate_from_length:
                    length_val = _try_float(length)
                    if length_val is None or length_val <= 0:
                        _clear_last_result()
                        st.error("长度必须大于0")
                    else:
                        na_val, theta_val, error = calculate_na(radius_val, length_val, refractive_index_val)

                        if error:
                            _clear_last_result()
                            st.error(error)
                        else:
                            na_str = f"{na_val:.4f}"
                            theta_str = f"{theta_val:.3f}"

                            st.session_state.na_inputs.update({
                                "radius": radius,
                                "length": length,
                                "material": material,
                                "refractive_index": refractive_index_display,
                                "na": na_str,
                                "theta": theta_str,
                            })

                            save_json({
                                "radius": radius,
                                "length": length,
                                "material": material,
                                "refractive_index": refractive_index_display,
                                "na": na_str,
                                "theta": theta_str,
                            }, INPUT_FILE)

                            st.session_state.na_calc_state.update({
                                "last_radius": radius,
                                "last_length": length,
                                "last_na": na_str,
                                "last_refractive_index": refractive_index_display,
                            })

                            st.session_state["na_last_mode"] = "length_to_na"
                            st.session_state["na_last_result"] = {
                                "na": na_str,
                                "theta": theta_str,
                            }
                            st.session_state["na_active_input"] = None
                            st.session_state["na_pending_widget_updates"] = {
                                "na_input": na_str,
                                "_suppress": "na_input",
                            }
                            if not _trigger_rerun():
                                st.session_state["na_suppress_on_change_for"] = "na_input"
                                st.session_state["na_input"] = na_str

                elif calculate_from_na:
                    na_val = _try_float(na)
                    if na_val is None:
                        _clear_last_result()
                        st.error("NA 值必须是数字")
                    elif na_val < 0:
                        _clear_last_result()
                        st.error("NA 值不能为负")
                    elif na_val > refractive_index_val + _FLOAT_TOLERANCE:
                        _clear_last_result()
                        st.error(f"NA 值 ({na_val:.4f}) 不能大于折射率 ({refractive_index_val:.4f})")
                    else:
                        length_val, theta_val, error = calculate_length(radius_val, na_val, refractive_index_val)

                        if error:
                            _clear_last_result()
                            st.error(error)
                        else:
                            length_str = f"{length_val:.4f}"
                            theta_str = f"{theta_val:.3f}"

                            st.session_state.na_inputs.update({
                                "radius": radius,
                                "length": length_str,
                                "material": material,
                                "refractive_index": refractive_index_display,
                                "na": na,
                                "theta": theta_str,
                            })

                            save_json({
                                "radius": radius,
                                "length": length_str,
                                "material": material,
                                "refractive_index": refractive_index_display,
                                "na": na,
                                "theta": theta_str,
                            }, INPUT_FILE)

                            st.session_state.na_calc_state.update({
                                "last_radius": radius,
                                "last_length": length_str,
                                "last_na": na,
                                "last_refractive_index": refractive_index_display,
                            })

                            st.session_state["na_last_mode"] = "na_to_length"
                            st.session_state["na_last_result"] = {
                                "length": length_str,
                                "theta": theta_str,
                            }
                            st.session_state["na_active_input"] = None
                            st.session_state["na_pending_widget_updates"] = {
                                "length_input": length_str,
                                "_suppress": "length_input",
                            }
                            if not _trigger_rerun():
                                st.session_state["na_suppress_on_change_for"] = "length_input"
                                st.session_state["length_input"] = length_str
                else:
                    _clear_last_result()
                    st.error("长度 (L) 或 NA 值必须至少输入一个")
            
            except ValueError as e:
                _clear_last_result()
                st.error(f"输入值无效: {str(e)}")
            except Exception as e:
                _clear_last_result()
                st.error(f"计算错误: {str(e)}")

    result_mode = st.session_state.get("na_last_mode")
    result_data = st.session_state.get("na_last_result")
    if result_mode == "length_to_na" and result_data:
        st.success("✅ 计算完成！")
        result_col1, result_col2 = st.columns(2)
        with result_col1:
            st.markdown("### 🎯 NA 值")
            st.markdown(f"# {result_data.get('na', '')}")
        with result_col2:
            st.markdown("### 📐 光纤端面可接受全角")
            st.markdown(f"# {result_data.get('theta', '')} °")
    elif result_mode == "na_to_length" and result_data:
        st.success("✅ 计算完成！")
        result_col1, result_col2 = st.columns(2)
        with result_col1:
            st.markdown("### 📏 光纤端面到小孔的距离")
            st.markdown(f"# {result_data.get('length', '')} mm")
        with result_col2:
            st.markdown("### 📐 光纤端面可接受全角")
            st.markdown(f"# {result_data.get('theta', '')} °")


if __name__ == "__main__":
    main()
