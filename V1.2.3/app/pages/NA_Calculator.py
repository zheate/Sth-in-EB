import json
import math
import streamlit as st
from pathlib import Path
from typing import Any, Dict

from config import get_config_path

# 配置页面（仅在独立运行时使用）
try:
    st.set_page_config(page_title="NA 计算器", page_icon="🔬", layout="wide")
except:
    pass  # 如果已经配置过，忽略错误

# 文件路径
MATERIAL_FILE = get_config_path("material.json")
INPUT_FILE = get_config_path("NA_Calculator_input.json")

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
            "length": saved_inputs.get("length", ""),
            "material": saved_inputs.get("material", "air"),
            "refractive_index": saved_inputs.get("refractive_index", "1.0003"),
            "na": saved_inputs.get("na", "0.2"),
            "theta": saved_inputs.get("theta", ""),
        }
    
    if "show_material_manager" not in st.session_state:
        st.session_state.show_material_manager = False
    
    # 初始化计算模式
    if "calculation_mode" not in st.session_state:
        st.session_state.calculation_mode = "standard"
    
    # 初始化端帽输入
    if "endcap_inputs" not in st.session_state:
        st.session_state.endcap_inputs = {
            "na": "0.2",
            "endcap_material": "SK1310_976",
            "endcap_length": "5",
            "air_distance": "2",
            "aperture_radius": "",
        }
    
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


def calculate_endcap_aperture(na, endcap_refractive_index, endcap_length, air_distance):
    """
    根据空气传播距离计算端帽光阑半径
    
    参数:
        na: 所需数值孔径
        endcap_refractive_index: 端帽折射率
        endcap_length: 端帽长度 (mm)
        air_distance: 空气中传播距离 (mm)
    
    返回:
        (aperture_radius, endcap_radius, endcap_angle_deg, endcap_refraction_angle_deg, air_refraction_angle_deg, error_msg)
    """
    try:
        if na <= 0:
            raise ValueError("NA值必须大于0")
        if na >= 1.0:
            raise ValueError("NA值必须小于1.0")
        if endcap_refractive_index <= 0:
            raise ValueError("端帽折射率必须大于0")
        if endcap_length <= 0:
            raise ValueError("端帽长度必须大于0")
        if air_distance < 0:
            raise ValueError("空气中传播距离不能为负")
        
        # 端帽入射角
        endcap_angle = math.asin(na)
        
        # 端帽折射角
        endcap_refraction_angle = math.asin(na / endcap_refractive_index)
        
        # 端帽上光斑半径
        endcap_radius = endcap_length * math.tan(endcap_refraction_angle)
        
        # 空气入射角
        air_refraction_angle = math.asin(endcap_refractive_index * math.sin(endcap_refraction_angle))
        
        # 光阑半径
        aperture_radius = endcap_radius + air_distance * math.tan(air_refraction_angle)
        
        # 转换角度为度数
        endcap_angle_deg = math.degrees(endcap_angle)
        endcap_refraction_angle_deg = math.degrees(endcap_refraction_angle)
        air_refraction_angle_deg = math.degrees(air_refraction_angle)
        
        return (aperture_radius, endcap_radius, endcap_angle_deg, 
                endcap_refraction_angle_deg, air_refraction_angle_deg, None)
    
    except ValueError as error:
        return None, None, None, None, None, str(error)
    except Exception as error:
        return None, None, None, None, None, f"计算端帽光阑时发生错误：{str(error)}"


def calculate_endcap_air_distance(na, endcap_refractive_index, endcap_length, aperture_radius):
    """
    根据光阑半径计算空气传播距离
    
    参数:
        na: 所需数值孔径
        endcap_refractive_index: 端帽折射率
        endcap_length: 端帽长度 (mm)
        aperture_radius: 光阑半径 (mm)
    
    返回:
        (air_distance, endcap_radius, endcap_angle_deg, endcap_refraction_angle_deg, air_refraction_angle_deg, error_msg)
    """
    try:
        if na <= 0:
            raise ValueError("NA值必须大于0")
        if na >= 1.0:
            raise ValueError("NA值必须小于1.0")
        if endcap_refractive_index <= 0:
            raise ValueError("端帽折射率必须大于0")
        if endcap_length <= 0:
            raise ValueError("端帽长度必须大于0")
        if aperture_radius <= 0:
            raise ValueError("光阑半径必须大于0")
        
        # 端帽入射角
        endcap_angle = math.asin(na)
        
        # 端帽折射角
        endcap_refraction_angle = math.asin(na / endcap_refractive_index)
        
        # 端帽上光斑半径
        endcap_radius = endcap_length * math.tan(endcap_refraction_angle)
        
        # 检查光阑半径是否小于端帽半径
        if aperture_radius < endcap_radius:
            raise ValueError(f"光阑半径 ({aperture_radius:.3f} mm) 不能小于端帽上光斑半径 ({endcap_radius:.3f} mm)")
        
        # 空气入射角
        air_refraction_angle = math.asin(endcap_refractive_index * math.sin(endcap_refraction_angle))
        
        # 计算空气传播距离
        air_distance = (aperture_radius - endcap_radius) / math.tan(air_refraction_angle)
        
        if air_distance < 0:
            raise ValueError("计算出的空气传播距离为负，异常")
        
        # 转换角度为度数
        endcap_angle_deg = math.degrees(endcap_angle)
        endcap_refraction_angle_deg = math.degrees(endcap_refraction_angle)
        air_refraction_angle_deg = math.degrees(air_refraction_angle)
        
        return (air_distance, endcap_radius, endcap_angle_deg, 
                endcap_refraction_angle_deg, air_refraction_angle_deg, None)
    
    except ValueError as error:
        return None, None, None, None, None, str(error)
    except ZeroDivisionError:
        return None, None, None, None, None, "计算空气传播距离时发生除零错误"
    except Exception as error:
        return None, None, None, None, None, f"计算空气传播距离时发生错误：{str(error)}"


def calculate_endcap_na(endcap_refractive_index, endcap_length, air_distance, aperture_radius):
    """
    根据端帽参数、空气传播距离和光阑半径计算NA值
    
    参数:
        endcap_refractive_index: 端帽折射率
        endcap_length: 端帽长度 (mm)
        air_distance: 空气中传播距离 (mm)
        aperture_radius: 光阑半径 (mm)
    
    返回:
        (na, endcap_radius, endcap_angle_deg, endcap_refraction_angle_deg, air_refraction_angle_deg, error_msg)
    """
    try:
        if endcap_refractive_index <= 0:
            raise ValueError("端帽折射率必须大于0")
        if endcap_length <= 0:
            raise ValueError("端帽长度必须大于0")
        if air_distance < 0:
            raise ValueError("空气中传播距离不能为负")
        if aperture_radius <= 0:
            raise ValueError("光阑半径必须大于0")
        
        # 使用迭代法求解NA值
        # 初始猜测值
        na_guess = 0.2
        tolerance = 1e-6
        max_iterations = 100
        
        for _ in range(max_iterations):
            # 计算当前NA对应的光阑半径
            endcap_angle = math.asin(na_guess)
            endcap_refraction_angle = math.asin(na_guess / endcap_refractive_index)
            endcap_radius = endcap_length * math.tan(endcap_refraction_angle)
            air_refraction_angle = math.asin(endcap_refractive_index * math.sin(endcap_refraction_angle))
            calculated_aperture = endcap_radius + air_distance * math.tan(air_refraction_angle)
            
            # 检查误差
            error = calculated_aperture - aperture_radius
            if abs(error) < tolerance:
                # 转换角度为度数
                endcap_angle_deg = math.degrees(endcap_angle)
                endcap_refraction_angle_deg = math.degrees(endcap_refraction_angle)
                air_refraction_angle_deg = math.degrees(air_refraction_angle)
                
                return (na_guess, endcap_radius, endcap_angle_deg, 
                        endcap_refraction_angle_deg, air_refraction_angle_deg, None)
            
            # 使用牛顿法更新NA值
            # 计算导数（数值微分）
            delta = 1e-8
            na_plus = na_guess + delta
            if na_plus >= 1.0:
                na_plus = na_guess - delta
                delta = -delta
            
            endcap_refraction_angle_plus = math.asin(na_plus / endcap_refractive_index)
            endcap_radius_plus = endcap_length * math.tan(endcap_refraction_angle_plus)
            air_refraction_angle_plus = math.asin(endcap_refractive_index * math.sin(endcap_refraction_angle_plus))
            calculated_aperture_plus = endcap_radius_plus + air_distance * math.tan(air_refraction_angle_plus)
            
            derivative = (calculated_aperture_plus - calculated_aperture) / delta
            
            if abs(derivative) < 1e-10:
                raise ValueError("无法收敛到解")
            
            # 更新NA值
            na_guess = na_guess - error / derivative
            
            # 确保NA在有效范围内
            if na_guess <= 0:
                na_guess = 0.01
            elif na_guess >= 1.0:
                na_guess = 0.99
        
        raise ValueError("迭代未收敛，无法计算NA值")
    
    except ValueError as error:
        return None, None, None, None, None, str(error)
    except Exception as error:
        return None, None, None, None, None, f"计算NA值时发生错误：{str(error)}"


def calculate_endcap_length(na, endcap_refractive_index, air_distance, aperture_radius):
    """
    根据NA值、空气传播距离和光阑半径计算端帽长度
    
    参数:
        na: 所需数值孔径
        endcap_refractive_index: 端帽折射率
        air_distance: 空气中传播距离 (mm)
        aperture_radius: 光阑半径 (mm)
    
    返回:
        (endcap_length, endcap_radius, endcap_angle_deg, endcap_refraction_angle_deg, air_refraction_angle_deg, error_msg)
    """
    try:
        if na <= 0:
            raise ValueError("NA值必须大于0")
        if na >= 1.0:
            raise ValueError("NA值必须小于1.0")
        if endcap_refractive_index <= 0:
            raise ValueError("端帽折射率必须大于0")
        if air_distance < 0:
            raise ValueError("空气中传播距离不能为负")
        if aperture_radius <= 0:
            raise ValueError("光阑半径必须大于0")
        
        # 端帽入射角
        endcap_angle = math.asin(na)
        
        # 端帽折射角
        endcap_refraction_angle = math.asin(na / endcap_refractive_index)
        
        # 空气折射角
        air_refraction_angle = math.asin(endcap_refractive_index * math.sin(endcap_refraction_angle))
        
        # 计算端帽上光斑半径
        # aperture_radius = endcap_radius + air_distance * tan(air_refraction_angle)
        # endcap_radius = endcap_length * tan(endcap_refraction_angle)
        # 因此: aperture_radius = endcap_length * tan(endcap_refraction_angle) + air_distance * tan(air_refraction_angle)
        # 解出: endcap_length = (aperture_radius - air_distance * tan(air_refraction_angle)) / tan(endcap_refraction_angle)
        
        tan_endcap = math.tan(endcap_refraction_angle)
        tan_air = math.tan(air_refraction_angle)
        
        if abs(tan_endcap) < 1e-10:
            raise ValueError("端帽折射角过小，无法计算端帽长度")
        
        endcap_length = (aperture_radius - air_distance * tan_air) / tan_endcap
        
        if endcap_length <= 0:
            raise ValueError(f"计算出的端帽长度为负或零 ({endcap_length:.4f} mm)，请检查输入参数")
        
        # 计算端帽上光斑半径
        endcap_radius = endcap_length * tan_endcap
        
        # 转换角度为度数
        endcap_angle_deg = math.degrees(endcap_angle)
        endcap_refraction_angle_deg = math.degrees(endcap_refraction_angle)
        air_refraction_angle_deg = math.degrees(air_refraction_angle)
        
        return (endcap_length, endcap_radius, endcap_angle_deg, 
                endcap_refraction_angle_deg, air_refraction_angle_deg, None)
    
    except ValueError as error:
        return None, None, None, None, None, str(error)
    except ZeroDivisionError:
        return None, None, None, None, None, "计算端帽长度时发生除零错误"
    except Exception as error:
        return None, None, None, None, None, f"计算端帽长度时发生错误：{str(error)}"


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
            key="na_selected_material_edit"
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
                st.session_state["na_edit_material_name"] = editor_state["name"]
                st.session_state["na_edit_re_index"] = editor_state["index"]
            material_name = st.text_input("材料名称", key="na_edit_material_name")
            re_index = st.text_input("折射率", key="na_edit_re_index")
            editor_state["name"] = material_name
            editor_state["index"] = re_index
        else:
            editor_state["selected"] = ""
            material_name = st.text_input("材料名称", key="na_new_material_name")
            re_index = st.text_input("折射率", value="1.5", key="na_new_re_index")
    
    with col2:
        st.write("")
        st.write("")
        
        btn_col1, btn_col2, btn_col3 = st.columns(3)
        
        with btn_col1:
            if st.button("💾 保存", use_container_width=True, key="na_save_material"):
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
            if st.button("🗑️ 删除", use_container_width=True, disabled=not selected_material, key="na_delete_material"):
                if selected_material in materials:
                    del materials[selected_material]
                    save_json(materials, MATERIAL_FILE)
                    st.success(f"材料 '{selected_material}' 已删除")
                    st.rerun()
        
        with btn_col3:
            if st.button("❌ 关闭", use_container_width=True, key="na_close_material"):
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



def render_formula_help():
    """显示公式说明"""
    if not st.session_state.get("show_formula", False):
        return

    import pandas as pd

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
        st.markdown("### 端帽光阑计算公式")

        st.markdown("#### 1. 端帽入射角:")
        st.latex(r"""
        \theta_{入射} = \arcsin(NA)
        """)

        st.markdown("#### 2. 端帽折射角 (斯涅尔定律):")
        st.latex(r"""
        \theta_{端帽} = \arcsin\left(\frac{NA}{n_{端帽}}\right)
        """)

        st.markdown("#### 3. 端帽上光斑半径:")
        st.latex(r"""
        r_{端帽} = L_{端帽} \times \tan(\theta_{端帽})
        """)

        st.markdown("#### 4. 空气中折射角:")
        st.latex(r"""
        \theta_{空气} = \arcsin(n_{端帽} \times \sin(\theta_{端帽}))
        """)

        st.markdown("#### 5. 最终光阑半径:")
        st.latex(r"""
        R_{光阑} = r_{端帽} + d_{空气} \times \tan(\theta_{空气})
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
                "全角 (Full Angle)",
            ],
            "单位": ["无量纲", "无量纲", "mm", "mm", "弧度/度", "弧度/度"],
        }
        st.dataframe(pd.DataFrame(symbols_data), use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("### 📋 端帽光阑符号说明")

        endcap_symbols_data = {
            "符号": ["NA", "n₍端帽₎", "L₍端帽₎", "d₍空气₎", "θ₍入射₎", "θ₍端帽₎", "θ₍空气₎", "r₍端帽₎", "R₍光阑₎"],
            "含义": [
                "所需数值孔径",
                "端帽材料折射率",
                "端帽长度",
                "空气中传播距离",
                "端帽入射角",
                "端帽折射角",
                "空气折射角",
                "端帽上光斑半径",
                "最终光阑半径",
            ],
            "单位": ["无量纲", "无量纲", "mm", "mm", "弧度/度", "弧度/度", "弧度/度", "mm", "mm"],
        }
        st.dataframe(pd.DataFrame(endcap_symbols_data), use_container_width=True, hide_index=True)

        st.dataframe(pd.DataFrame(symbols_data), use_container_width=True, hide_index=True)

def main():
    init_session_state()

    pending_updates = st.session_state.pop("na_pending_widget_updates", None)

    if "radius_input" not in st.session_state:
        st.session_state["radius_input"] = _normalize_text(st.session_state.na_inputs.get("radius", "1.005"))
    if "length_input" not in st.session_state:
        st.session_state["length_input"] = _normalize_text(st.session_state.na_inputs.get("length", ""))
    if "na_input" not in st.session_state:
        st.session_state["na_input"] = _normalize_text(st.session_state.na_inputs.get("na", "0.2"))

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
    
    # 计算模式选择
    calc_mode = st.radio(
        "选择计算模式",
        options=["标准NA计算", "端帽光阑计算"],
        horizontal=True,
        key="calc_mode_radio"
    )
    
    if calc_mode == "端帽光阑计算":
        st.session_state.calculation_mode = "endcap"
    else:
        st.session_state.calculation_mode = "standard"
    
    # 顶部按钮
    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        if st.button("📦 管理材料", use_container_width=True, key="na_manage_material"):
            st.session_state.show_material_manager = not st.session_state.show_material_manager
            st.rerun()
    
    with col2:
        if st.button("ℹ️ 公式说明", use_container_width=True, key="na_formula_help"):
            st.session_state.show_formula = not st.session_state.get("show_formula", False)

    if st.session_state.get("show_formula", False):
        render_formula_help()
        st.markdown("---")
    
    # 显示材料管理器
    if st.session_state.show_material_manager:
        material_manager()
        return
    
    # 端帽光阑计算模式
    if st.session_state.calculation_mode == "endcap":
        # 初始化端帽输入框状态
        if "endcap_na_input" not in st.session_state:
            st.session_state["endcap_na_input"] = _normalize_text(st.session_state.endcap_inputs.get("na", "0.2"))
        if "endcap_length_input" not in st.session_state:
            st.session_state["endcap_length_input"] = _normalize_text(st.session_state.endcap_inputs.get("endcap_length", "5"))
        if "endcap_air_distance_input" not in st.session_state:
            st.session_state["endcap_air_distance_input"] = _normalize_text(st.session_state.endcap_inputs.get("air_distance", ""))
        if "endcap_aperture_radius_input" not in st.session_state:
            st.session_state["endcap_aperture_radius_input"] = _normalize_text(st.session_state.endcap_inputs.get("aperture_radius", ""))
        
        # 处理pending updates（用于计算后更新输入框）
        if pending_updates:
            for key, value in pending_updates.items():
                if key in ["endcap_air_distance_input", "endcap_aperture_radius_input", "endcap_na_input", "endcap_length_input"]:
                    st.session_state[key] = value
        
        # 显示端帽光阑示意图（居中）
        try:
            from pathlib import Path
            import base64
            endcap_image_path = Path("app/data/endcap.png")
            if endcap_image_path.exists():
                # 使用HTML居中图片
                with open(endcap_image_path, "rb") as img_file:
                    img_data = base64.b64encode(img_file.read()).decode()
                st.markdown(
                    f"""
                    <div style="display: flex; justify-content: center; align-items: center; flex-direction: column;">
                        <img src="data:image/png;base64,{img_data}" width="800" />
                        <p style="text-align: center; color: #888; font-size: 14px; margin-top: 8px;">端帽光阑示意图</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        except Exception:
            pass
        
        st.markdown("---")
        st.markdown("### 🔍 端帽光阑计算")
        st.markdown("根据已知参数计算未知参数（留空字段将被计算，默认计算光阑半径）")
        
        col1, col2 = st.columns(2)
        
        with col1:
            endcap_na = st.text_input(
                "所需 NA 值",
                
                key="endcap_na_input",
                help="输入NA值，或留空由其他参数自动计算",
                on_change=_on_input_change,
                args=("endcap_na_input",),
            )
            
            # 端帽材料选择
            materials_list = sorted(st.session_state.materials.keys())
            current_endcap_material = st.session_state.endcap_inputs.get("endcap_material", "SK1310_976")
            
            if current_endcap_material not in materials_list:
                if "SK1310_976" in materials_list:
                    current_endcap_material = "SK1310_976"
                else:
                    current_endcap_material = materials_list[0] if materials_list else "air"
            
            endcap_material = st.selectbox(
                "端帽材料",
                options=materials_list,
                index=materials_list.index(current_endcap_material) if current_endcap_material in materials_list else 0,
                key="endcap_material_select",
                help="选择端帽材料"
            )
            
            # 获取端帽折射率
            endcap_refraction_value = st.session_state.materials.get(endcap_material, 1.55)
            endcap_refraction_display = _format_index(endcap_refraction_value)
            
            # 使用禁用的文本输入框显示折射率，保持对齐
            st.text_input(
                "端帽折射率",
                value=endcap_refraction_display,
                disabled=True,
                key="endcap_refraction_display"
            )
        
        with col2:
            endcap_len = st.text_input(
                "端帽长度 [mm]",
                key="endcap_length_input",
                help="输入端帽长度，或留空由其他参数自动计算",
                on_change=_on_input_change,
                args=("endcap_length_input",),
            )
            
            air_dist = st.text_input(
                "空气中传播距离 [mm]",
                key="endcap_air_distance_input",
                help="输入空气传播距离，或留空由光阑半径自动计算",
                on_change=_on_input_change,
                args=("endcap_air_distance_input",),
            )
            
            aperture_r = st.text_input(
                "光阑半径 [mm]",
                key="endcap_aperture_radius_input",
                help="输入光阑半径，或留空由空气传播距离自动计算",
                on_change=_on_input_change,
                args=("endcap_aperture_radius_input",),
            )
        
        st.markdown("---")
        
        # 规范化输入
        air_dist = _normalize_text(air_dist)
        aperture_r = _normalize_text(aperture_r)
        
        # 检查输入是否改变，如果改变则清除结果
        last_mode_snapshot = st.session_state.get("na_last_mode")
        last_result_snapshot = st.session_state.get("na_last_result")
        if last_mode_snapshot in ["endcap_distance_to_aperture", "endcap_aperture_to_distance"] and last_result_snapshot:
            if last_mode_snapshot == "endcap_distance_to_aperture":
                # 如果上次是从距离计算光阑，检查光阑是否被修改
                try:
                    current_aperture = float(aperture_r) if aperture_r else None
                    last_aperture = float(last_result_snapshot.get("aperture_radius", "0"))
                    if current_aperture is not None and abs(current_aperture - last_aperture) > 0.0001:
                        _clear_last_result()
                except (ValueError, TypeError):
                    if aperture_r:  # 如果有输入但无法转换，清除结果
                        _clear_last_result()
            elif last_mode_snapshot == "endcap_aperture_to_distance":
                # 如果上次是从光阑计算距离，检查距离是否被修改
                try:
                    current_distance = float(air_dist) if air_dist else None
                    last_distance = float(last_result_snapshot.get("air_distance", "0"))
                    if current_distance is not None and abs(current_distance - last_distance) > 0.0001:
                        _clear_last_result()
                except (ValueError, TypeError):
                    if air_dist:  # 如果有输入但无法转换，清除结果
                        _clear_last_result()
        
        if st.button("🧮 计算", type="primary", use_container_width=True, key="endcap_calculate"):
            errors = []
            
            # 统计空值数量
            empty_fields = []
            if not endcap_na:
                empty_fields.append("NA值")
            if not endcap_len:
                empty_fields.append("端帽长度")
            if not air_dist:
                empty_fields.append("空气传播距离")
            if not aperture_r:
                empty_fields.append("光阑半径")
            
            # 验证端帽折射率
            if endcap_refraction_value <= 0:
                errors.append("端帽折射率必须大于0")
            
            # 如果没有空字段，默认计算光阑半径
            if len(empty_fields) == 0:
                calc_target = "光阑半径"
            elif len(empty_fields) == 1:
                calc_target = empty_fields[0]
            else:
                errors.append(f"留空字段过多，当前留空了: {', '.join(empty_fields)}")
                calc_target = None
            
            if errors:
                _clear_last_result()
                for error in errors:
                    st.error(error)
            else:
                try:
                    refraction_val = endcap_refraction_value
                    
                    # 根据空字段确定计算模式
                    if calc_target == "NA值":
                        # 计算NA值
                        length_val = float(endcap_len)
                        distance_val = float(air_dist)
                        aperture_val = float(aperture_r)
                        
                        na_result, endcap_r, endcap_angle, endcap_refr_angle, air_refr_angle, error = calculate_endcap_na(
                            refraction_val, length_val, distance_val, aperture_val
                        )
                        
                        if error:
                            _clear_last_result()
                            st.error(error)
                        else:
                            na_str = f"{na_result:.4f}"
                            
                            st.session_state.endcap_inputs.update({
                                "na": na_str,
                                "endcap_material": endcap_material,
                                "endcap_length": endcap_len,
                                "air_distance": air_dist,
                                "aperture_radius": aperture_r,
                            })
                            
                            st.session_state["na_last_mode"] = "endcap_calc_na"
                            st.session_state["na_last_result"] = {
                                "na": na_str,
                                "endcap_radius": f"{endcap_r:.4f}",
                                "endcap_angle": f"{endcap_angle:.3f}",
                                "endcap_refr_angle": f"{endcap_refr_angle:.3f}",
                                "air_refr_angle": f"{air_refr_angle:.3f}",
                            }
                            st.session_state["na_pending_widget_updates"] = {
                                "endcap_na_input": na_str,
                                "_suppress": "endcap_na_input",
                            }
                            if not _trigger_rerun():
                                st.session_state["na_suppress_on_change_for"] = "endcap_na_input"
                                st.session_state["endcap_na_input"] = na_str
                    
                    elif calc_target == "端帽长度":
                        # 计算端帽长度
                        na_val = float(endcap_na)
                        distance_val = float(air_dist)
                        aperture_val = float(aperture_r)
                        
                        length_result, endcap_r, endcap_angle, endcap_refr_angle, air_refr_angle, error = calculate_endcap_length(
                            na_val, refraction_val, distance_val, aperture_val
                        )
                        
                        if error:
                            _clear_last_result()
                            st.error(error)
                        else:
                            length_str = f"{length_result:.4f}"
                            
                            st.session_state.endcap_inputs.update({
                                "na": endcap_na,
                                "endcap_material": endcap_material,
                                "endcap_length": length_str,
                                "air_distance": air_dist,
                                "aperture_radius": aperture_r,
                            })
                            
                            st.session_state["na_last_mode"] = "endcap_calc_length"
                            st.session_state["na_last_result"] = {
                                "endcap_length": length_str,
                                "endcap_radius": f"{endcap_r:.4f}",
                                "endcap_angle": f"{endcap_angle:.3f}",
                                "endcap_refr_angle": f"{endcap_refr_angle:.3f}",
                                "air_refr_angle": f"{air_refr_angle:.3f}",
                            }
                            st.session_state["na_pending_widget_updates"] = {
                                "endcap_length_input": length_str,
                                "_suppress": "endcap_length_input",
                            }
                            if not _trigger_rerun():
                                st.session_state["na_suppress_on_change_for"] = "endcap_length_input"
                                st.session_state["endcap_length_input"] = length_str
                    
                    elif calc_target == "空气传播距离":
                        # 计算空气传播距离
                        na_val = float(endcap_na)
                        length_val = float(endcap_len)
                        aperture_val = float(aperture_r)
                        
                        distance_result, endcap_r, endcap_angle, endcap_refr_angle, air_refr_angle, error = calculate_endcap_air_distance(
                            na_val, refraction_val, length_val, aperture_val
                        )
                        
                        if error:
                            _clear_last_result()
                            st.error(error)
                        else:
                            distance_str = f"{distance_result:.4f}"
                            
                            st.session_state.endcap_inputs.update({
                                "na": endcap_na,
                                "endcap_material": endcap_material,
                                "endcap_length": endcap_len,
                                "air_distance": distance_str,
                                "aperture_radius": aperture_r,
                            })
                            
                            st.session_state["na_last_mode"] = "endcap_calc_distance"
                            st.session_state["na_last_result"] = {
                                "air_distance": distance_str,
                                "endcap_radius": f"{endcap_r:.4f}",
                                "endcap_angle": f"{endcap_angle:.3f}",
                                "endcap_refr_angle": f"{endcap_refr_angle:.3f}",
                                "air_refr_angle": f"{air_refr_angle:.3f}",
                            }
                            st.session_state["na_pending_widget_updates"] = {
                                "endcap_air_distance_input": distance_str,
                                "_suppress": "endcap_air_distance_input",
                            }
                            if not _trigger_rerun():
                                st.session_state["na_suppress_on_change_for"] = "endcap_air_distance_input"
                                st.session_state["endcap_air_distance_input"] = distance_str
                    
                    elif calc_target == "光阑半径":
                        # 计算光阑半径
                        na_val = float(endcap_na)
                        length_val = float(endcap_len)
                        distance_val = float(air_dist)
                        
                        aperture_result, endcap_r, endcap_angle, endcap_refr_angle, air_refr_angle, error = calculate_endcap_aperture(
                            na_val, refraction_val, length_val, distance_val
                        )
                        
                        if error:
                            _clear_last_result()
                            st.error(error)
                        else:
                            aperture_str = f"{aperture_result:.4f}"
                            
                            st.session_state.endcap_inputs.update({
                                "na": endcap_na,
                                "endcap_material": endcap_material,
                                "endcap_length": endcap_len,
                                "air_distance": air_dist,
                                "aperture_radius": aperture_str,
                            })
                            
                            st.session_state["na_last_mode"] = "endcap_calc_aperture"
                            st.session_state["na_last_result"] = {
                                "aperture_radius": aperture_str,
                                "endcap_radius": f"{endcap_r:.4f}",
                                "endcap_angle": f"{endcap_angle:.3f}",
                                "endcap_refr_angle": f"{endcap_refr_angle:.3f}",
                                "air_refr_angle": f"{air_refr_angle:.3f}",
                            }
                            st.session_state["na_pending_widget_updates"] = {
                                "endcap_aperture_radius_input": aperture_str,
                                "_suppress": "endcap_aperture_radius_input",
                            }
                            if not _trigger_rerun():
                                st.session_state["na_suppress_on_change_for"] = "endcap_aperture_radius_input"
                                st.session_state["endcap_aperture_radius_input"] = aperture_str
                
                except ValueError as e:
                    _clear_last_result()
                    st.error(f"输入值无效: {str(e)}")
                except Exception as e:
                    _clear_last_result()
                    st.error(f"计算错误: {str(e)}")
        
        # 显示结果
        result_mode = st.session_state.get("na_last_mode")
        result_data = st.session_state.get("na_last_result")
        
        if result_mode and result_mode.startswith("endcap_calc_") and result_data:
            st.success("✅ 计算完成！")
            
            result_col1, result_col2 = st.columns(2)
            
            with result_col1:
                if result_mode == "endcap_calc_na":
                    st.markdown("### 🎯 NA 值")
                    st.markdown(f"# {result_data.get('na', '')}")
                elif result_mode == "endcap_calc_length":
                    st.markdown("### 📏 端帽长度")
                    st.markdown(f"# {result_data.get('endcap_length', '')} mm")
                elif result_mode == "endcap_calc_distance":
                    st.markdown("### 📏 空气传播距离")
                    st.markdown(f"# {result_data.get('air_distance', '')} mm")
                elif result_mode == "endcap_calc_aperture":
                    st.markdown("### 🎯 光阑半径")
                    st.markdown(f"# {result_data.get('aperture_radius', '')} mm")
                st.markdown(f"**端帽上光斑半径:** {result_data.get('endcap_radius', '')} mm")
            
            with result_col2:
                st.markdown("### 📐 角度信息")
                st.markdown(f"**端帽入射角:** {result_data.get('endcap_angle', '')}°")
                st.markdown(f"**端帽折射角:** {result_data.get('endcap_refr_angle', '')}°")
                st.markdown(f"**空气折射角:** {result_data.get('air_refr_angle', '')}°")
            
            # 显示计算详情
            st.markdown("---")
            st.markdown("### 📋 计算详情")
            
            # 获取当前输入值用于显示
            current_na = st.session_state.endcap_inputs.get("na", "")
            current_material = st.session_state.endcap_inputs.get("endcap_material", "")
            current_length = st.session_state.endcap_inputs.get("endcap_length", "")
            current_air_dist = st.session_state.endcap_inputs.get("air_distance", "")
            current_aperture = st.session_state.endcap_inputs.get("aperture_radius", "")
            
            detail_col1, detail_col2 = st.columns(2)
            
            with detail_col1:
                st.markdown(f"""
                - **所需 NA 值:** {current_na}
                - **端帽材料:** {current_material}
                - **端帽折射率:** {endcap_refraction_display}
                - **端帽长度:** {current_length} mm
                - **空气传播距离:** {current_air_dist} mm
                """)
            
            with detail_col2:
                st.markdown(f"""
                - **光阑半径:** {current_aperture} mm
                - **端帽入射角:** {result_data.get('endcap_angle', '')}°
                - **端帽折射角:** {result_data.get('endcap_refr_angle', '')}°
                - **端帽上光斑半径:** {result_data.get('endcap_radius', '')} mm
                - **空气折射角:** {result_data.get('air_refr_angle', '')}°
                """)
        
    
    else:
        # 参数输入区域
        if not st.session_state.get("show_formula", False):
            st.markdown("---")
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
            
            # 材料选择
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
            
            # 获取折射率
            raw_index_value = st.session_state.materials.get(material, DEFAULT_MATERIALS["air"])
            refractive_index_value = _try_float(raw_index_value) or DEFAULT_MATERIALS["air"]
            if refractive_index_value <= 0:
                refractive_index_value = DEFAULT_MATERIALS["air"]
            
            refractive_index_display = _format_index(refractive_index_value)
            st.session_state.na_inputs["refractive_index"] = refractive_index_display
            # 使用与输入框相同的标签样式
            st.markdown(
                f'<label style="font-size: 0.875rem; font-weight: 400; margin-bottom: 0.25rem;">折射率 (n)</label>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div style="background-color: #f0f2f6; padding: 10px; border-radius: 5px; color: #666; font-size: 1rem;">{refractive_index_display}</div>',
                unsafe_allow_html=True,
            )
        
        material_changed = False
        refractive_index_value = DEFAULT_MATERIALS["air"]
        refractive_index_display = _format_index(refractive_index_value)
        
        with col2:
            length = st.text_input(
                "光纤端面到小孔的距离 (L) [mm]",
                key="length_input",
                help="输入长度，或留空由 NA 自动计算",
                on_change=_on_input_change,
                args=("length_input",),
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
        if st.button("🧮 计算", type="primary", use_container_width=True, key="na_calculate"):
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
