import math
from PyQt6.QtWidgets import (
    QApplication, QWidget, QComboBox, QGridLayout, QLabel, QVBoxLayout, QLineEdit, QMessageBox, QPushButton, QGroupBox
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QDoubleValidator
import sys
import json


class Na_Calculator(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('NA 计算器')
        self.material_list = self.load_json()

        self.layout = QGridLayout()

        self.groupbox1 = QGroupBox('NA 计算器')
        self.vbox = QVBoxLayout()

        # 小孔半径
        self.radius_label = QLabel('小孔半径 (单位: mm)')
        self.radius_input = QLineEdit()
        self.radius_input.setPlaceholderText("请输入小孔半径")
        self.radius_input.setText("1.005")  # 设置默认值
        self.vbox.addWidget(self.radius_label)
        self.vbox.addWidget(self.radius_input)

        # 光纤端面到小孔的距离
        self.length_label = QLabel('光纤端面到小孔的距离 (单位: mm)')
        self.length_input = QLineEdit()
        self.length_input.setPlaceholderText("请输入长度")
        self.length_input.setText("4.457") # 设置默认值
        self.vbox.addWidget(self.length_label)
        self.vbox.addWidget(self.length_input)

        # 材质
        self.material_combobox = QComboBox()
        self.refractive_index = 1.0 # 默认空气折射率，以防json加载失败

        if self.material_list:
            for material in self.material_list:
                self.material_combobox.addItem(material)
            default_material = 'air'
            if default_material in self.material_list:
                self.material_combobox.setCurrentText(default_material)
                self.refractive_index = self.material_list[default_material] # load_json已确保是float
            else:
                # 如果'air'不在，则使用列表中的第一个材料
                first_material = self.material_combobox.currentText()
                if first_material and first_material in self.material_list:
                     self.refractive_index = self.material_list[first_material]
                     self.show_error_message('默认材料缺失', f"材料列表中缺少默认材料 '{default_material}'。已选择 '{first_material}' 作为默认材料。")
                else:
                     # 如果列表为空或第一个材料无效（理论上不应发生，因为load_json会过滤），则保持默认值1.0
                     self.show_error_message('材料错误', "无法从材料列表加载有效的默认材料。使用默认折射率 1.0。")


        self.material_combobox.currentTextChanged.connect(self.update_material)

        self.vbox.addWidget(QLabel("选择材料："))
        self.vbox.addWidget(self.material_combobox)


        # NA值
        self.na_label = QLabel('NA值')
        self.na_input = QLineEdit()
        self.na_input.setPlaceholderText("请输入NA值")
        self.vbox.addWidget(self.na_label)
        self.vbox.addWidget(self.na_input)

        # 角度
        self.theta = QLabel('光纤端面可接受全角(°):')
        self.theta_value = QLineEdit('')  # 重命名为 theta_value
        self.theta_value.setReadOnly(True)
        self.vbox.addWidget(self.theta)
        self.vbox.addWidget(self.theta_value)

        self.groupbox1.setLayout(self.vbox)
        self.layout.addWidget(self.groupbox1, 0, 0)

        # 计算按钮
        self.calculate_button = QPushButton("计算", self)
        self.calculate_button.setStyleSheet('background: green')
        self.calculate_button.clicked.connect(self.on_calculate_button_clicked)
        self.layout.addWidget(self.calculate_button)

        self.setLayout(self.layout)

        # 设置焦点策略
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        # 添加输入框验证
        double_validator_non_negative = QDoubleValidator()
        double_validator_non_negative.setBottom(0)  # 小孔半径和长度必须大于等于0
        self.radius_input.setValidator(double_validator_non_negative)
        self.length_input.setValidator(double_validator_non_negative)

        # 创建并设置NA验证器
        self.na_validator = QDoubleValidator()
        self.na_validator.setBottom(0)  # NA值必须为非负值
        # 初始上限设置为当前折射率，会在update_material中更新
        self.na_validator.setTop(self.refractive_index if self.refractive_index > 0 else 1.1)
        self.na_validator.setDecimals(4) # 限制小数位数
        self.na_input.setValidator(self.na_validator)

        # 初始化参数 (从默认文本获取)
        self.radius = None
        self.length = None
        self.na = None

        try:
            self.radius = float(self.radius_input.text())
        except ValueError:
            self.show_error("默认半径值无效。")
            self.radius = 1.0 # 提供一个备用默认值

        try:
            self.length = float(self.length_input.text())
        except ValueError:
             self.show_error("默认长度值无效。")
             self.length = 10.0 # 提供一个备用默认值


        # 标记是否正在进行自动更新，防止递归
        self.updating = False

        # 连接输入框的 editingFinished 信号以实现自动计算
        self.length_input.editingFinished.connect(self.on_length_finished)
        self.na_input.editingFinished.connect(self.on_na_finished)
        self.radius_input.editingFinished.connect(self.on_radius_finished)

        # 执行初始计算
        if self.radius is not None and self.length is not None:
            self.calculate_based_on_length_or_na()


    def load_json(self):
        valid_materials = {}
        try:
            with open('material.json', 'r', encoding='utf-8') as f: # 指定utf-8编码
                data = json.load(f)
                if not isinstance(data, dict):
                    self.show_error_message('格式错误', "材料文件应为JSON对象（字典）")
                    return {}

                for material, index_str in data.items():
                    try:
                        index_float = float(index_str)
                        if index_float <= 0:
                             print(f"警告：材料 '{material}' 的折射率 ({index_float}) 无效（非正值），已忽略。")
                             continue # 跳过无效条目
                        valid_materials[material] = index_float
                    except (ValueError, TypeError):
                         print(f"警告：材料 '{material}' 的值 '{index_str}' 不是有效的数字，已忽略。")

                if not valid_materials:
                     self.show_error_message('数据错误', "材料文件中未找到有效的材料数据。")

                return valid_materials

        except FileNotFoundError:
            self.show_error_message('文件未找到', "无法加载 'material.json'。请确保文件存在于程序目录下。")
            return {}
        except json.JSONDecodeError as e:
            self.show_error_message('解析错误', f"材料文件的JSON格式无效: {e}")
            return {}
        except Exception as e: 
             self.show_error_message('加载错误', f"加载材料文件时发生未知错误: {e}")
             return {}


    def update_material(self):
        """根据选择的材料更新折射率，并调整NA验证器上限"""
        material = self.material_combobox.currentText()
        refractive_index = self.material_list.get(material, None)

        if refractive_index is not None: # load_json已确保是float
            self.refractive_index = refractive_index
            print(f"选择的材料：{material}, 折射率：{self.refractive_index}")

            # 更新NA验证器的上限
            self.na_validator.setTop(self.refractive_index)

            # 检查当前的NA值是否超过新的上限
            if self.na is not None and self.na > self.refractive_index:
                self.show_error_message("NA值调整", f"当前NA值 ({self.na:.4f}) 大于新材料 '{material}' 的折射率 ({self.refractive_index:.4f})。NA值将被清除或重新计算。")
                self.na = None
                self.na_input.clear()
                self.theta_value.clear()
                # 如果长度已知，则基于长度重新计算NA
                if self.length is not None:
                    self.calculate_based_on_length_or_na()

            # 如果NA值有效或已被清除/重新计算，则基于当前状态更新（可能会重新计算长度或NA）
            elif self.radius is not None: # 确保半径存在以进行计算
                 self.calculate_based_on_length_or_na()

        else:
            # 理论上不应发生，因为combobox只包含有效材料
            self.show_error(f"内部错误：未在加载的材料列表中找到 '{material}'")


    def on_radius_finished(self):
        """当半径输入完成时，更新半径并重新计算"""
        if self.updating:
            return

        try:
            radius_text = self.radius_input.text().strip().replace(',', '.') # 允许逗号作为小数点
            if not radius_text:
                # 如果清空输入，我们可能需要决定如何处理，例如清除依赖项或保留旧值
                # 这里暂时不做操作，等待有效输入
                return
            radius = float(radius_text)
            if radius <= 0:
                raise ValueError("小孔半径必须大于零。")
            self.radius = radius
            # 根据新的半径重新计算
            self.calculate_based_on_length_or_na()
        except ValueError as e:
            self.show_error(f"半径输入无效: {e}")
            # 出错时恢复上一个有效值或清空？这里暂时不清空
            if self.radius is not None:
                 self.radius_input.setText(str(self.radius)) # 恢复显示旧值
            else:
                 self.radius_input.clear() # 如果之前没有有效值则清空

        except Exception as e:  # 捕获所有其他异常
            self.show_error(f"处理半径输入时发生未知错误：{str(e)}")

    def on_length_finished(self):
        """当长度输入完成时，重新计算 NA"""
        if self.updating:
            return
        try:
            length_text = self.length_input.text().strip().replace(',', '.') # 允许逗号
            if not length_text:
                self.length = None # 如果清空，则将内部值设为None
                self.na_input.clear() # 清除依赖的NA值
                self.theta_value.clear() # 清除角度
                return

            length = float(length_text)
            if length <= 0:
                raise ValueError("长度必须大于零。")
            self.length = length
            self.na = None # 清除旧的NA，因为长度变了
            self.na_input.clear()
            self.theta_value.clear()

            if self.radius is not None: # 只有半径存在时才能计算NA
                 calculated_na = self.calculate_na() # calculate_na内部会更新角度
                 if calculated_na is not None:
                     self.updating = True
                     self.na = calculated_na
                     self.na_input.setText(f"{self.na:.4f}")
                     self.updating = False

        except ValueError as e:
            self.show_error(f"长度输入无效: {e}")
            if self.length is not None:
                 self.length_input.setText(str(self.length)) # 恢复旧值
            else:
                 self.length_input.clear()
        except Exception as e:
            self.show_error(f"处理长度输入时发生未知错误: {str(e)}")


    def on_na_finished(self):
        """当 NA 输入完成时，重新计算长度并更新角度"""
        if self.updating:
            return

        try:
            na_text = self.na_input.text().strip().replace(',', '.') # 允许逗号
            if not na_text:
                self.na = None # 如果清空，则将内部值设为None
                self.length_input.clear() # 清除依赖的长度值
                self.theta_value.clear() # 清除角度
                return

            na = float(na_text)

            # 验证器应该已经阻止了大于折射率的值，但再次检查以防万一
            if na > self.refractive_index:
                # 理论上不会到这里，因为validator会阻止
                raise ValueError(f"NA值 ({na:.4f}) 不能大于当前材料折射率 ({self.refractive_index:.4f})。")
            if na < 0:
                 raise ValueError("NA值不能为负。") # Validator也应阻止


            self.na = na
            self.length = None # 清除旧的长度，因为NA变了
            self.length_input.clear()
            self.theta_value.clear()

            if self.radius is not None: # 只有半径存在才能计算长度
                calculated_length = self.calculate_length() # calculate_length内部会处理NA为0的情况
                if calculated_length is not None:
                    self.updating = True
                    # 只有当计算值与当前值不同时才更新
                    if self.length is None or not math.isclose(self.length, calculated_length, abs_tol=1e-5):
                         self.length = calculated_length
                         self.length_input.setText(f"{self.length:.4f}")
                    # 确保角度是基于当前NA更新的
                    self.update_angle(self.na) # 确保基于新NA更新角度
                    self.updating = False
                else:
                     # 如果calculate_length返回None (例如NA为0)，确保角度也被清除
                     self.theta_value.clear()

        except ValueError as e:
            self.show_error(f"NA输入无效: {e}")
            if self.na is not None:
                 self.na_input.setText(str(self.na)) # 恢复旧值
            else:
                 self.na_input.clear()
            self.theta_value.clear() # 出错时清除角度
        except Exception as e: # 捕获其他异常
            self.show_error(f"处理NA输入时发生未知错误: {str(e)}")
            self.theta_value.clear() # 出错时清除角度


    def calculate_based_on_length_or_na(self):
        """
        根据当前已知的半径、长度或 NA 状态，重新计算缺失的值。
        优先级：
        1. 如果长度已知，用长度计算 NA。
        2. 如果 NA 已知（且长度未知），用 NA 计算长度。
        """
        if self.radius is None:
             # self.show_error("请输入有效的半径才能进行计算。") # 可以取消注释以提示
             return # 半径未知，无法计算

        if self.length is not None:
            # 长度已知，计算或重新计算 NA
            try:
                calculated_na = self.calculate_na() # calculate_na 内部会更新角度
                if calculated_na is not None:
                    # 检查新计算的NA是否超过当前折射率（理论上不应发生，除非折射率本身无效）
                    if calculated_na > self.refractive_index:
                         # 这通常意味着 length 或 radius 相对于 refractive_index 不合理
                         self.show_error(f"计算得到的 NA ({calculated_na:.4f}) 超出当前折射率 ({self.refractive_index:.4f})。请检查输入值。")
                         self.na = None
                         self.na_input.clear()
                         self.theta_value.clear()
                         return

                    self.updating = True
                    # 只有当计算值与当前值不同时才更新，避免不必要的信号触发
                    if self.na is None or not math.isclose(self.na, calculated_na, abs_tol=1e-5):
                         self.na = calculated_na
                         self.na_input.setText(f"{self.na:.4f}")
                    self.updating = False
                else:
                    # 计算失败（例如长度为0），清除NA
                    self.na = None
                    self.na_input.clear()
                    self.theta_value.clear()
            except Exception as e:
                 self.show_error(f"根据长度计算NA时出错: {e}")
                 self.na = None
                 self.na_input.clear()
                 self.theta_value.clear()

        elif self.na is not None:
            # NA 已知，长度未知，计算长度
            try:
                # 先检查NA是否有效
                if self.na > self.refractive_index:
                     self.show_error(f"当前 NA ({self.na:.4f}) 大于折射率 ({self.refractive_index:.4f})，无法计算长度。")
                     self.length = None
                     self.length_input.clear()
                     self.theta_value.clear() # 确保角度也清除
                     return

                calculated_length = self.calculate_length()
                if calculated_length is not None:
                    self.updating = True
                    # 只有当计算值与当前值不同时才更新
                    if self.length is None or not math.isclose(self.length, calculated_length, abs_tol=1e-5):
                         self.length = calculated_length
                         self.length_input.setText(f"{self.length:.4f}")
                    # 确保角度是基于当前NA更新的
                    self.update_angle(self.na)
                    self.updating = False
                else:
                    # 计算失败（例如NA为0），清除长度
                    self.length = None
                    self.length_input.clear()
                    self.theta_value.clear()
            except Exception as e:
                 self.show_error(f"根据NA计算长度时出错: {e}")
                 self.length = None
                 self.length_input.clear()
                 self.theta_value.clear()
        else:
            # 长度和 NA 都未知，清除显示
            self.na_input.clear()
            self.length_input.clear()
            self.theta_value.clear()


    def calculate_na(self):
        """根据半径、长度和折射率计算 NA。内部会更新角度。"""
        if self.radius is None or self.length is None:
            # self.show_error("需要半径和长度才能计算NA。") # 可以取消注释
            return None

        try:
            if math.isclose(self.length, 0.0):
                raise ValueError("长度为零，无法计算 NA 值。") # 改为ValueError更合适
            # 计算半角 in radians
            half_angle_rad = math.atan(self.radius / self.length)
            # 计算 NA
            na = math.sin(half_angle_rad) * self.refractive_index

            # 在这里直接更新角度显示，因为它依赖于NA的计算过程
            full_angle_deg = math.degrees(half_angle_rad) * 2
            self.theta_value.setText(f"{full_angle_deg:.3f} °")

            # 检查计算出的NA是否有效（虽然理论上基于有效输入计算出的NA应该总是有效）
            # 但以防万一，例如折射率非常小的情况
            if na > self.refractive_index:
                 print(f"警告：计算出的NA {na} 略大于折射率 {self.refractive_index}，可能由于浮点精度问题。将NA限制为折射率。")
                 na = self.refractive_index # 限制
            elif na < 0:
                 print(f"警告：计算出的NA {na} 为负，异常。")
                 return None # 返回None表示计算失败

            return na

        except ValueError as e: # 捕获 length 为 0 的情况
            self.show_error(str(e))
            self.theta_value.clear() # 计算失败，清除角度
            return None
        except Exception as e:  # 捕获所有其他异常
            self.show_error(f"计算NA时发生未知错误：{str(e)}")
            self.theta_value.clear() # 计算失败，清除角度
            return None

    def calculate_length(self):
        """根据半径和 NA 计算长度。"""
        if self.radius is None or self.na is None:
             # self.show_error("需要半径和NA才能计算长度。") # 可以取消注释
             return None

        try:
            # 首先检查 NA 的有效性相对于折射率
            if self.na > self.refractive_index:
                raise ValueError(f"NA值 ({self.na:.4f}) 不能大于折射率 ({self.refractive_index:.4f})。")
            if self.na < 0:
                 raise ValueError("NA值不能为负。")

            # 处理 NA 接近零的情况
            if math.isclose(self.na, 0.0):
                # NA 为 0 意味着角度为 0，理论上长度应为无穷大
                # 在实际应用中，返回一个错误或一个非常大的数可能更合适
                self.show_error("NA值为零，无法计算有限长度。")
                return None # 返回 None 表示无法计算

            # 计算 theta1 (半角 in radians)
            # 添加检查确保 asin 的参数在 [-1, 1] 范围内，尽管之前的检查应已覆盖
            asin_arg = self.na / self.refractive_index
            if not (-1.0 <= asin_arg <= 1.0):
                 # 这理论上不应发生，除非 refractive_index 无效或 na > refractive_index 被绕过
                 raise ValueError(f"计算角度时出错：arcsin 的参数 ({asin_arg:.4f}) 超出范围 [-1, 1]。")

            theta1 = math.asin(asin_arg) # 使用 math.asin，因为 numpy 不是必须的

            # 检查 theta1 是否接近零 (对应 NA 接近零的情况，上面已处理)
            # 但也检查 tan(theta1) 是否接近零，以防 theta1 接近 pi/2 (NA接近折射率) 导致长度非常小
            if math.isclose(math.tan(theta1), 0.0):
                 # 如果 tan(theta1) 为 0 (theta1=0)，说明 NA=0，上面已处理
                 # 如果 tan(theta1) 非常大 (theta1 接近 pi/2)，长度会非常小
                 # 这里我们允许计算，因为 tan(theta1) 不会精确为0，除非 theta1=0
                 pass # 允许继续计算


            length = self.radius / math.tan(theta1)

            if length < 0:
                 # 正半径 / 正tan(theta1) 不应为负，除非出现意外情况
                 print(f"警告：计算出的长度 {length} 为负，异常。")
                 return None

            return length

        except ValueError as e: # 捕获 NA 无效或计算中的数学错误
            self.show_error(str(e))
            return None
        except ZeroDivisionError:
             # 理论上 math.tan(theta1) 只有在 theta1 = k*pi 时为0 (k为整数)
             # 对于 0 < theta1 < pi/2, tan(theta1) > 0
             # 如果 theta1=0 (NA=0), 上面已经处理了
             self.show_error("计算长度时发生除零错误（这不应发生）。")
             return None
        except Exception as e:  # 捕获所有其他异常
            self.show_error(f"计算长度时发生未知错误：{str(e)}")
            return None

    def update_angle(self, na):
        """根据NA更新角度显示。仅更新显示，不执行计算。"""
        if na is None:
            self.theta_value.clear()
            return

        try:
             # 再次检查 NA 有效性
            if not (0 <= na <= self.refractive_index):
                 # 如果 NA 无效（可能来自直接输入但绕过了验证，或来自错误计算）
                 self.theta_value.clear()
                 # 可以选择显示错误，但可能过于频繁
                 # self.show_error(f"无法更新角度：NA值 ({na:.4f}) 无效。")
                 return

            # 处理 NA 为 0 的情况
            if math.isclose(na, 0.0):
                 angle_deg = 0.0
            else:
                 # 计算半角（弧度）
                 asin_arg = na / self.refractive_index
                 # 添加安全检查，尽管理论上不应超出范围
                 if asin_arg > 1.0: asin_arg = 1.0
                 if asin_arg < -1.0: asin_arg = -1.0 # NA非负，所以这个不需要
                 angle_rad_half = math.asin(asin_arg)
                 # 计算全角（度）
                 angle_deg = math.degrees(angle_rad_half) * 2

            self.theta_value.setText(f"{angle_deg:.3f} °")

        except ValueError as e: # 主要捕获 math.asin 的域错误（理论上不应发生）
            # self.show_error(f"更新角度时出错：{e}")
            print(f"更新角度时出错：{e}, NA={na}, n={self.refractive_index}") # 打印调试信息
            self.theta_value.clear()
        except Exception as e:
             self.show_error(f"更新角度时发生未知错误: {e}")
             self.theta_value.clear()


    def show_error(self, message):
        """弹出错误提示框（简单错误）"""
        QMessageBox.critical(self, "错误", message)

    def show_error_message(self, title, message):
        """弹出带标题的错误提示框"""
        QMessageBox.critical(self, title, message)

    def on_calculate_button_clicked(self):
        """点击“计算”按钮时，根据当前输入自动判断计算NA或长度"""
        if self.radius_input.text().strip() and self.length_input.text().strip():
            # 优先根据半径和长度计算NA
            self.radius = float(self.radius_input.text())
            self.length = float(self.length_input.text())
            self.na = None
            self.calculate_based_on_length_or_na()
        elif self.radius_input.text().strip() and self.na_input.text().strip():
            # 根据半径和NA计算长度
            self.radius = float(self.radius_input.text())
            self.na = float(self.na_input.text())
            self.length = None
            self.calculate_based_on_length_or_na()
        else:
            self.show_error("请至少输入半径和长度，或半径和NA值。")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Na_Calculator()
    window.show()
    sys.exit(app.exec())