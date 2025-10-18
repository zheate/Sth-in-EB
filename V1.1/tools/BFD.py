import sys
import os
os.environ['QT_LOGGING_RULES'] = '*.debug=false;qt.svg.warning=false'
from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QFormLayout, QLineEdit, QLabel,
                             QPushButton, QComboBox, QGroupBox, QHBoxLayout, QMessageBox, QInputDialog,
                             QDialog, QListWidget, QListWidgetItem, QSpinBox)
from PyQt6.QtCore import Qt, pyqtSignal
import json

class InputValidator:
    """输入验证类，用于检查输入值的有效性"""
    @staticmethod
    def validate_float(text, condition, error_msg_base, field_name=""):
        # 验证浮点数输入
        # Args:
        #   text (str): 输入的文本.
        #   condition (function): 一个接受浮点数并返回布尔值的函数.
        #   error_msg_base (str): 基础错误信息.
        #   field_name (str, optional): 字段名称，用于错误信息. Defaults to "".
        # Returns:
        #   tuple: (bool, str) 验证结果和错误信息.
        if not text:
            return False, f"{field_name}值不能为空" if field_name else "值不能为空"
        try:
            value = float(text)
            if not condition(value):
                return False, f"{field_name}{error_msg_base}" if field_name else error_msg_base
            return True, ""
        except ValueError:
            return False, f"{field_name}必须是数字" if field_name else "必须是数字"

class MaterialManager(QDialog):
    """优化后的材料管理界面"""
    # 定义信号
    # material_saved: 当材料被保存（新增或更新）时发射
    #   参数: str (旧名称, 如果是更新), str (新名称), str (折射率)
    material_saved = pyqtSignal(str, str, str)
    # material_deleted: 当材料被删除时发射
    #   参数: str (被删除的材料名称)
    material_deleted = pyqtSignal(str)

    def __init__(self, materials, parent=None):
        super().__init__(parent)
        self.materials = materials # 引用外部材料字典
        self.setWindowTitle("材料管理")
        self.current_item = None  # 当前选中项
        self.init_ui()

    def init_ui(self):
        # 初始化用户界面
        main_layout = QHBoxLayout()

        # 左侧材料列表
        left_widget = QWidget()
        left_layout = QVBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("搜索材料...")
        self.search_input.textChanged.connect(self.filter_materials)
        left_layout.addWidget(self.search_input)

        self.material_list = QListWidget()
        self.material_list.itemClicked.connect(self.show_material_details)
        self.update_material_list()
        left_layout.addWidget(self.material_list)

        add_button = QPushButton("新增材料")
        add_button.clicked.connect(self.add_material)
        left_layout.addWidget(add_button)
        left_widget.setLayout(left_layout)
        main_layout.addWidget(left_widget, 1)

        # 右侧编辑区域
        self.right_widget = QWidget()
        self.right_layout = QVBoxLayout()
        self.name_input = QLineEdit()
        self.re_index_input = QLineEdit()
        self.right_layout.addWidget(QLabel("材料名称:"))
        self.right_layout.addWidget(self.name_input)
        self.right_layout.addWidget(QLabel("折射率:"))
        self.right_layout.addWidget(self.re_index_input)

        button_layout = QHBoxLayout()
        self.save_button = QPushButton("保存")
        self.delete_button = QPushButton("删除")
        self.save_button.clicked.connect(self.save_material)
        self.delete_button.clicked.connect(self.delete_material)
        button_layout.addWidget(self.save_button)
        button_layout.addWidget(self.delete_button)
        self.right_layout.addLayout(button_layout)
        self.right_widget.setLayout(self.right_layout)
        main_layout.addWidget(self.right_widget, 2)

        self.setLayout(main_layout)

        # 美化样式
        self.setStyleSheet("""
            QDialog { background-color: #f5f5f5; }
            QListWidget { background-color: #2E2E2E; color: white; border: 1px solid #555; }
            QListWidget::item { padding: 10px; border-bottom: 1px solid #444; }
            QListWidget::item:hover { background-color: #4CAF50; }
            QListWidget::item:selected { background-color: #4CAF50; color: white; }
            QPushButton { background-color: #4CAF50; color: white; border-radius: 5px; padding: 5px; min-height: 20px;}
            QPushButton:hover { background-color: #45a049; }
            QLineEdit { border: 1px solid #ccc; padding: 4px; border-radius: 3px; background-color: white; color: black;}
            QLabel { color: black; }
        """)

    def update_material_list(self):
        """更新左侧材料列表"""
        self.material_list.clear()
        for name, re_index in sorted(self.materials.items()):
            item = QListWidgetItem(f"{name} - {re_index}")
            self.material_list.addItem(item)

    def show_material_details(self, item):
        """显示选中材料的详细信息"""
        self.current_item = item
        text = item.text()
        try:
            name, re_index = text.split(" - ", 1)
            self.name_input.setText(name)
            self.re_index_input.setText(re_index)
        except ValueError:
            # 处理 " - " 分隔符不存在或格式不正确的情况
            self.name_input.setText(text) # 假设整个文本是名称
            self.re_index_input.clear()
            QMessageBox.warning(self, "格式错误", f"材料 '{text}' 的格式无法解析。")


    def filter_materials(self):
        """根据搜索框过滤材料"""
        search_text = self.search_input.text().lower()
        self.material_list.clear()
        for name, re_index in sorted(self.materials.items()):
            if search_text in name.lower() or search_text in str(re_index):
                self.material_list.addItem(f"{name} - {re_index}")

    def add_material(self):
        """新增材料"""
        self.current_item = None
        self.name_input.clear()
        self.re_index_input.setText("1.5") # 默认折射率
        self.name_input.setFocus()

    def save_material(self):
        """保存材料信息"""
        name = self.name_input.text().strip()
        re_index_str = self.re_index_input.text().strip()

        valid, msg = InputValidator.validate_float(re_index_str, lambda x: x > 1, "必须大于1", "折射率")
        if not name:
            QMessageBox.warning(self, "错误", "材料名称不能为空")
            return
        if not valid:
            QMessageBox.warning(self, "错误", msg)
            return

        old_name = None
        if self.current_item:
            try:
                old_name = self.current_item.text().split(" - ", 1)[0]
            except ValueError: # 万一格式不对
                old_name = self.current_item.text()


        # 检查名称是否重复 (排除正在编辑的项本身名称未改变的情况)
        if name in self.materials and (not self.current_item or old_name != name):
            QMessageBox.warning(self, "错误", f"材料 '{name}' 已存在。")
            return

        # 如果是编辑现有材料且名称改变，先删除旧的
        if old_name and old_name != name and old_name in self.materials:
            del self.materials[old_name]

        self.materials[name] = re_index_str
        self.material_saved.emit(old_name if old_name and old_name != name else None, name, re_index_str)
        self.update_material_list() # 更新列表以反映更改
        # 选中刚保存/更新的项
        items = self.material_list.findItems(f"{name} - {re_index_str}", Qt.MatchFlag.MatchExactly)
        if items:
            self.material_list.setCurrentItem(items[0])
            self.show_material_details(items[0]) # 确保右侧同步
        QMessageBox.information(self, "成功", "材料保存成功")


    def delete_material(self):
        """删除选中材料"""
        if not self.current_item:
            QMessageBox.warning(self, "警告", "请先选择一种材料")
            return

        try:
            name = self.current_item.text().split(" - ", 1)[0]
        except ValueError:
            name = self.current_item.text()


        reply = QMessageBox.question(self, "确认删除", f"确定删除材料 '{name}' 吗？",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            if name in self.materials:
                del self.materials[name]
                self.material_deleted.emit(name)
                self.update_material_list()
                self.name_input.clear()
                self.re_index_input.clear()
                self.current_item = None
                QMessageBox.information(self, "成功", "材料删除成功")
            else:
                QMessageBox.warning(self, "错误", f"材料 '{name}' 未在数据中找到，可能已被删除。")


class Calculator(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('后焦距计算器')
        self.material_refractive_indices = self.load_json("material.json", {"ZF52-976": "1.8145"}) # 材料数据
        self._updating_fast = False # 防止快轴参数更新递归
        self._updating_slow = False # 防止慢轴参数更新递归
        self.init_ui()
        self.load_input_settings() # 加载上次的输入设置
        self.update_re_index_fields_from_combobox() # 根据下拉框初始化折射率

    def load_json(self, filename, default_data):
        # 从JSON文件加载数据
        # Args:
        #   filename (str): JSON文件名.
        #   default_data (dict): 如果文件不存在或无效，返回的默认数据.
        # Returns:
        #   dict: 加载的数据.
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            self.show_error_message('文件未找到', f"无法加载 {filename}，将使用默认数据。")
            return default_data
        except json.JSONDecodeError:
            self.show_error_message('文件格式错误', f"{filename} 格式错误，将使用默认数据。")
            return default_data
        except PermissionError:
            self.show_error_message('无权限', f"无法读取 {filename}，将使用默认数据。")
            return default_data
        except Exception as e:
            self.show_error_message('加载错误', f"加载 {filename} 时发生未知错误: {str(e)}。将使用默认数据。")
            return default_data

    def save_json(self, data, filename):
        # 将数据保存到JSON文件
        # Args:
        #   data (dict): 要保存的数据.
        #   filename (str): JSON文件名.
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            return True
        except PermissionError:
            self.show_error_message("保存失败", f"无权限保存文件 {filename}。")
        except Exception as e:
            self.show_error_message("保存失败", f"保存 {filename} 时发生未知错误: {str(e)}。")
        return False

    def init_ui(self):
        # 初始化主用户界面
        main_layout = QVBoxLayout()

        # --- 快轴参数组 ---
        foc_group = QGroupBox("快轴参数")
        foc_layout = QFormLayout()
        
        (self.material_combobox_fast, 
         self.save_button_fast, 
         self.re_index_input_fast, 
         material_container_fast) = self._create_material_section(
            self.on_material_changed_fast,
            lambda: self._save_new_material(self.re_index_input_fast, self.material_combobox_fast),
            lambda: self._update_params_on_text_change(self.re_index_input_fast, "re_index", "fast"),
            "fast")

        # 连接信号（这些信号的回调函数可能会用到 self.material_combobox_fast 等，所以在它们被赋值后连接）
        self.material_combobox_fast.currentTextChanged.connect(self.on_material_changed_fast)
        self.save_button_fast.clicked.connect(lambda: self._save_new_material(self.re_index_input_fast, self.material_combobox_fast))
        # _update_params_on_text_change for re_index_input is connected via the lambda passed to _create_material_section
        # and also the _handle_re_index_text_changed is connected inside _create_material_section.

        self.curvature_fast_widget, self.curvature_fast_input = self.create_input_with_unit("mm")
        self.curvature_fast_input.textChanged.connect(lambda: self._update_params_on_text_change(self.curvature_fast_input, "curvature", "fast"))
        self.efl_fast_widget, self.efl_fast_input = self.create_input_with_unit("mm")
        self.efl_fast_input.textChanged.connect(lambda: self._update_params_on_text_change(self.efl_fast_input, "efl", "fast"))
        self.thickness_fast_widget, self.thickness_fast_input = self.create_input_with_unit("mm")

        foc_layout.addRow("材质:", material_container_fast) 
        foc_layout.addRow("折射率:", self.re_index_input_fast)
        foc_layout.addRow("曲率半径 (R):", self.curvature_fast_widget)
        foc_layout.addRow("有效焦距 (EFL):", self.efl_fast_widget)
        foc_layout.addRow("中心厚度 (T):", self.thickness_fast_widget)
        foc_group.setLayout(foc_layout)

        # --- 慢轴参数组 ---
        soc_group = QGroupBox("慢轴参数")
        soc_layout = QFormLayout()
        (self.material_combobox_slow, 
         self.save_button_slow, 
         self.re_index_input_slow, 
         material_container_slow) = self._create_material_section(
            self.on_material_changed_slow,
            lambda: self._save_new_material(self.re_index_input_slow, self.material_combobox_slow),
            lambda: self._update_params_on_text_change(self.re_index_input_slow, "re_index", "slow"),
            "slow")

        self.material_combobox_slow.currentTextChanged.connect(self.on_material_changed_slow)
        self.save_button_slow.clicked.connect(lambda: self._save_new_material(self.re_index_input_slow, self.material_combobox_slow))


        self.curvature_slow_widget, self.curvature_slow_input = self.create_input_with_unit("mm")
        self.curvature_slow_input.textChanged.connect(lambda: self._update_params_on_text_change(self.curvature_slow_input, "curvature", "slow"))
        self.efl_slow_widget, self.efl_slow_input = self.create_input_with_unit("mm")
        self.efl_slow_input.textChanged.connect(lambda: self._update_params_on_text_change(self.efl_slow_input, "efl", "slow"))
        self.thickness_slow_widget, self.thickness_slow_input = self.create_input_with_unit("mm")

        soc_layout.addRow("材质:", material_container_slow)
        soc_layout.addRow("折射率:", self.re_index_input_slow)
        soc_layout.addRow("曲率半径 (R):", self.curvature_slow_widget)
        soc_layout.addRow("有效焦距 (EFL):", self.efl_slow_widget)
        soc_layout.addRow("中心厚度 (T):", self.thickness_slow_widget)
        soc_group.setLayout(soc_layout)

        # --- 输出和控制按钮 ---
        self.bfd_fast_label = QLabel("快轴后焦距 (FOC BFD): -- mm")
        self.bfd_fast_label.setStyleSheet("color: #F1A208; font-size: 18px; font-weight: bold;")
        self.bfd_slow_label = QLabel("慢轴后焦距 (SOC BFD): -- mm")
        self.bfd_slow_label.setStyleSheet("color: #05C3F7; font-size: 18px; font-weight: bold;") 

        output_layout = QVBoxLayout()
        output_layout.addWidget(self.bfd_fast_label)
        output_layout.addWidget(self.bfd_slow_label)

        self.calculate_button = QPushButton("计算 BFD")
        self.calculate_button.clicked.connect(self.perform_bfd_calculation)
        self.info_button = QPushButton("i") 
        self.info_button.setFixedSize(25, 25)
        self.info_button.setStyleSheet("font-weight: bold; border-radius: 12px;")
        self.info_button.setToolTip("显示计算公式")
        self.info_button.clicked.connect(self.show_formula_dialog)
        self.manage_button = QPushButton("管理材料")
        self.manage_button.clicked.connect(self.open_material_manager_dialog)

        self.precision_spinbox = QSpinBox()
        self.precision_spinbox.setRange(1, 6) 
        self.precision_spinbox.setValue(3)
        self.precision_spinbox.setToolTip("设置结果显示的小数位数")

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.calculate_button)
        button_layout.addStretch(1)
        button_layout.addWidget(QLabel("精度:"))
        button_layout.addWidget(self.precision_spinbox)
        button_layout.addWidget(self.manage_button)
        button_layout.addWidget(self.info_button)


        main_layout.addWidget(foc_group)
        main_layout.addWidget(soc_group)
        main_layout.addLayout(output_layout)
        main_layout.addLayout(button_layout)

        self.setLayout(main_layout)
        self.setStyleSheet("""
            QWidget { background-color: #2E2E2E; color: white; font-size: 14px; }
            QGroupBox {
                font-weight: bold;
                border: 1px solid #555;
                border-radius: 8px;
                padding: 10px;
                margin-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 5px 0 5px;
                color: #4CAF50;
            }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border-radius: 5px;
                padding: 8px 12px;
                min-height: 20px; 
            }
            QPushButton:hover { background-color: #45a049; }
            QPushButton:disabled { background-color: #555; color: #aaa; }
            QLineEdit {
                background-color: #1E1E1E;
                color: white;
                border: 1px solid #555;
                border-radius: 4px;
                padding: 5px;
            }
            QComboBox {
                background-color: #1E1E1E;
                color: white;
                border: 1px solid #555;
                border-radius: 4px;
                padding: 5px;
            }
            QComboBox::drop-down { border: none; }
            QComboBox QAbstractItemView { background-color: #1E1E1E; color: white; selection-background-color: #4CAF50; }
            QLabel { padding-top: 3px; }
            QSpinBox {
                background-color: #1E1E1E;
                color: white;
                border: 1px solid #555;
                border-radius: 4px;
                padding: 4px;
            }
        """)

    def _create_material_section(self, on_material_changed_callback, save_material_callback, on_re_index_change_callback, axis_type):
        # 创建材质选择和折射率输入的通用部分
        # Args:
        #   on_material_changed_callback (function): 当材质下拉框变化时调用的回调. (Will be connected externally)
        #   save_material_callback (function): 当保存新材质按钮点击时调用的回调. (Will be connected externally)
        #   on_re_index_change_callback (function): 当折射率输入框文本变化时调用的回调. (Will be connected externally for _update_params)
        #   axis_type (str): "fast" 或 "slow"，用于区分控件.
        # Returns:
        #   tuple: (QComboBox, QPushButton, QLineEdit, QWidget) 创建的控件和它们的容器.
        combobox = QComboBox()
        combobox.addItem("Custom") 
        for material in sorted(self.material_refractive_indices.keys()):
            combobox.addItem(material)
        combobox.setCurrentText("ZF52-976" if "ZF52-976" in self.material_refractive_indices else "Custom")
        # on_material_changed_callback will be connected in init_ui

        save_button = QPushButton("保存材质")
        save_button.setEnabled(False) 
        save_button.setToolTip("当选择 'Custom' 或修改预设材质的折射率时，可保存为新材质或覆盖现有材质。")
        # save_material_callback will be connected in init_ui

        material_layout = QHBoxLayout()
        material_layout.addWidget(combobox, 2) 
        material_layout.addWidget(save_button, 1)
        material_layout.setContentsMargins(0,0,0,0)
        material_container = QWidget()
        material_container.setLayout(material_layout)
        # Explicit setParent calls removed here. Layout handles parenting.

        re_index_input = QLineEdit()
        re_index_input.setAlignment(Qt.AlignmentFlag.AlignRight)
        # This internal handler for UI logic (enabling save button, switching to custom) is connected here.
        re_index_input.textChanged.connect(lambda text: self._handle_re_index_text_changed(text, combobox, save_button, axis_type))
        # The on_re_index_change_callback (for _update_params_on_text_change) will be connected in init_ui.
        
        # The passed on_re_index_change_callback is for the main parameter update logic.
        # We still need to connect it to the re_index_input that is returned.
        # This is now done in init_ui after self.re_index_input_fast/slow is assigned.
        # However, the lambda passed to _create_material_section for on_re_index_change_callback
        # already captures self.re_index_input_fast/slow.
        # To be absolutely clear and avoid potential issues with when self.re_index_input_fast is available
        # for the lambda, it's better to connect the on_re_index_change_callback (which is a lambda itself)
        # in init_ui after self.re_index_input_fast is set.
        # The original code connected it via the argument:
        # lambda: self._update_params_on_text_change(self.re_index_input_fast, "re_index", "fast")
        # This lambda is created in init_ui and passed. When re_index_input.textChanged emits, this lambda is called.
        # At that point, self.re_index_input_fast should be valid.
        # So, the original connection method for on_re_index_change_callback is likely fine.
        # Let's ensure the passed callback is connected to the correct re_index_input instance.
        re_index_input.textChanged.connect(on_re_index_change_callback)


        return combobox, save_button, re_index_input, material_container

    def _handle_re_index_text_changed(self, text, combobox, save_button, axis_type_unused):
        # 当折射率输入框文本改变时，处理UI联动
        current_material = combobox.currentText()
        is_custom = (current_material == "Custom")
        expected_re_index = self.material_refractive_indices.get(current_material, "")

        if not is_custom and text.strip() != expected_re_index:
            combobox.setCurrentText("Custom") 
            save_button.setEnabled(True)
        elif is_custom:
            save_button.setEnabled(True) 
        else: 
            save_button.setEnabled(False)


    def create_input_with_unit(self, unit_text):
        # 创建带单位标签的输入框
        input_field = QLineEdit()
        input_field.setAlignment(Qt.AlignmentFlag.AlignRight)
        unit_label = QLabel(unit_text)
        unit_label.setStyleSheet("padding-left: 5px;") 

        layout = QHBoxLayout()
        layout.addWidget(input_field, 1) 
        layout.addWidget(unit_label)
        layout.setContentsMargins(0, 0, 0, 0) 
        container = QWidget()
        container.setLayout(layout)
        return container, input_field

    def update_re_index_fields_from_combobox(self):
        # 根据当前下拉框选项更新折射率输入框
        self.on_material_changed_fast(self.material_combobox_fast.currentText())
        self.on_material_changed_slow(self.material_combobox_slow.currentText())

    def on_material_changed_fast(self, material_name):
        # 快轴材质下拉框变化时的处理
        self._update_re_index_for_axis(material_name, self.re_index_input_fast, self.save_button_fast, "fast")

    def on_material_changed_slow(self, material_name):
        # 慢轴材质下拉框变化时的处理
        self._update_re_index_for_axis(material_name, self.re_index_input_slow, self.save_button_slow, "slow")

    def _update_re_index_for_axis(self, material_name, re_index_input, save_button, axis_type):
        # 通用方法：更新指定轴的折射率输入框和保存按钮状态
        if material_name != "Custom":
            re_index_value = self.material_refractive_indices.get(material_name, "1.5") 
            re_index_input.setText(re_index_value)
            save_button.setEnabled(False) 
        else:
            save_button.setEnabled(True) 
        
        self._update_params_on_text_change(re_index_input, "re_index", axis_type)


    def _save_new_material(self, re_index_input, combobox):
        # 通用方法：保存新材质（或覆盖现有材质）
        current_re_index = re_index_input.text().strip()
        valid, msg = InputValidator.validate_float(current_re_index, lambda x: x > 1, "必须大于1", "折射率")
        if not valid:
            self.show_error_message("输入错误", msg)
            return

        name, ok = QInputDialog.getText(self, "保存材质", "请输入材质名称:", text=combobox.currentText() if combobox.currentText() != "Custom" else "")
        if ok and name:
            name = name.strip()
            if not name:
                self.show_error_message("输入错误", "材质名称不能为空。")
                return

            if name == "Custom":
                self.show_error_message("输入错误", "不能将材质命名为 'Custom'。")
                return

            if name in self.material_refractive_indices and name != combobox.currentText(): 
                reply = QMessageBox.question(self, "确认覆盖", f"材质 '{name}' 已存在，其折射率为 {self.material_refractive_indices[name]}。\n是否要将其折射率更新为 {current_re_index}？",
                                             QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
                if reply == QMessageBox.StandardButton.No:
                    return
            
            self.material_refractive_indices[name] = current_re_index
            if self.save_json(self.material_refractive_indices, "material.json"):
                self._update_combobox_item(self.material_combobox_fast, name)
                self._update_combobox_item(self.material_combobox_slow, name)
                
                combobox.setCurrentText(name) 
                if combobox == self.material_combobox_fast:
                    self.save_button_fast.setEnabled(False)
                else:
                    self.save_button_slow.setEnabled(False)
                QMessageBox.information(self, "成功", f"材质 '{name}' 已保存。")


    def _update_combobox_item(self, combobox, item_name):
        # 更新下拉框，如果项不存在则添加
        if combobox.findText(item_name) == -1:
            combobox.addItem(item_name)

    def open_material_manager_dialog(self):
        # 打开材料管理对话框
        dialog = MaterialManager(self.material_refractive_indices, self)
        dialog.material_saved.connect(self.handle_material_saved_from_manager)
        dialog.material_deleted.connect(self.handle_material_deleted_from_manager)
        dialog.exec()

    def handle_material_saved_from_manager(self, old_name, new_name, new_re_index):
        # 处理从MaterialManager保存的材料
        if old_name and old_name != new_name and old_name in self.material_refractive_indices:
            del self.material_refractive_indices[old_name]
        self.material_refractive_indices[new_name] = new_re_index
        self.save_json(self.material_refractive_indices, "material.json")

        if old_name and old_name != new_name:
            idx_fast = self.material_combobox_fast.findText(old_name)
            if idx_fast != -1: self.material_combobox_fast.removeItem(idx_fast)
            idx_slow = self.material_combobox_slow.findText(old_name)
            if idx_slow != -1: self.material_combobox_slow.removeItem(idx_slow)

        self._update_combobox_item(self.material_combobox_fast, new_name)
        self._update_combobox_item(self.material_combobox_slow, new_name)

        if self.material_combobox_fast.currentText() == old_name:
            self.material_combobox_fast.setCurrentText(new_name)
        elif self.material_combobox_fast.currentText() == new_name: 
            self.on_material_changed_fast(new_name) 

        if self.material_combobox_slow.currentText() == old_name:
            self.material_combobox_slow.setCurrentText(new_name)
        elif self.material_combobox_slow.currentText() == new_name:
            self.on_material_changed_slow(new_name)


    def handle_material_deleted_from_manager(self, deleted_name):
        # 处理从MaterialManager删除的材料
        self.save_json(self.material_refractive_indices, "material.json")

        current_fast_text = self.material_combobox_fast.currentText()
        idx_fast = self.material_combobox_fast.findText(deleted_name)
        if idx_fast != -1:
            self.material_combobox_fast.removeItem(idx_fast)
            if current_fast_text == deleted_name: 
                self.material_combobox_fast.setCurrentText("Custom") 

        current_slow_text = self.material_combobox_slow.currentText()
        idx_slow = self.material_combobox_slow.findText(deleted_name)
        if idx_slow != -1:
            self.material_combobox_slow.removeItem(idx_slow)
            if current_slow_text == deleted_name:
                self.material_combobox_slow.setCurrentText("Custom")


    def perform_bfd_calculation(self):
        # 执行BFD计算
        inputs_config = [
            (self.re_index_input_fast, "快轴折射率", lambda x: x > 1, "必须大于1", True),
            (self.curvature_fast_input, "快轴曲率半径", lambda x: True, "", False), 
            (self.efl_fast_input, "快轴EFL", lambda x: True, "", False), 
            (self.thickness_fast_input, "快轴厚度", lambda x: x >= 0, "必须为非负数", True),

            (self.re_index_input_slow, "慢轴折射率", lambda x: x > 1, "必须大于1", True),
            (self.curvature_slow_input, "慢轴曲率半径", lambda x: True, "", False),
            (self.efl_slow_input, "慢轴EFL", lambda x: True, "", False),
            (self.thickness_slow_input, "慢轴厚度", lambda x: x >= 0, "必须为非负数", True),
        ]

        values = {}
        has_error = False
        for field, name, condition, error_suffix, is_required in inputs_config:
            text = field.text().strip()
            if is_required:
                valid, msg = InputValidator.validate_float(text, condition, error_suffix, name)
                if not valid:
                    self.show_error_message("输入错误", msg)
                    has_error = True
                    break
                values[name] = float(text)
            else: 
                if text: 
                    valid, msg = InputValidator.validate_float(text, condition, error_suffix, name)
                    if not valid:
                        self.show_error_message("输入错误", msg)
                        has_error = True
                        break
                    values[name] = float(text)
                else:
                    values[name] = None 
        if has_error: return

        if values["快轴EFL"] is None and values["快轴曲率半径"] is None:
            self.show_error_message("输入错误", "快轴参数中，有效焦距 (EFL) 或曲率半径 (R) 必须至少输入一个。")
            return
        if values["慢轴EFL"] is None and values["慢轴曲率半径"] is None:
            self.show_error_message("输入错误", "慢轴参数中，有效焦距 (EFL) 或曲率半径 (R) 必须至少输入一个。")
            return

        n_fast = values["快轴折射率"]
        r_fast_val = values["快轴曲率半径"]
        efl_fast_val = values["快轴EFL"]
        t_fast = values["快轴厚度"]

        n_slow = values["慢轴折射率"]
        r_slow_val = values["慢轴曲率半径"]
        efl_slow_val = values["慢轴EFL"]
        t_slow = values["慢轴厚度"]

        if efl_fast_val is None: 
            if r_fast_val is None or (n_fast - 1) == 0: 
                self.show_error_message("计算错误", "快轴无法计算EFL：R为空或折射率为1。")
                return
            efl_fast_val = r_fast_val / (n_fast - 1)
            self._updating_fast = True
            self.efl_fast_input.setText(f"{efl_fast_val:.{self.precision_spinbox.value()}f}")
            self._updating_fast = False
        elif r_fast_val is None: 
            if efl_fast_val is None or (n_fast - 1) == 0 : 
                 self.show_error_message("计算错误", "快轴无法计算R：EFL为空或折射率为1。")
                 return
            r_fast_val = efl_fast_val * (n_fast - 1)
            self._updating_fast = True
            self.curvature_fast_input.setText(f"{r_fast_val:.{self.precision_spinbox.value()}f}")
            self._updating_fast = False
        
        if efl_slow_val is None:
            if r_slow_val is None or (n_slow - 1) == 0:
                self.show_error_message("计算错误", "慢轴无法计算EFL：R为空或折射率为1。")
                return
            efl_slow_val = r_slow_val / (n_slow - 1)
            self._updating_slow = True
            self.efl_slow_input.setText(f"{efl_slow_val:.{self.precision_spinbox.value()}f}")
            self._updating_slow = False
        elif r_slow_val is None:
            if efl_slow_val is None or (n_slow - 1) == 0:
                self.show_error_message("计算错误", "慢轴无法计算R：EFL为空或折射率为1。")
                return
            r_slow_val = efl_slow_val * (n_slow - 1)
            self._updating_slow = True
            self.curvature_slow_input.setText(f"{r_slow_val:.{self.precision_spinbox.value()}f}")
            self._updating_slow = False

        if n_fast == 0 or n_slow == 0: 
            self.show_error_message("计算错误", "折射率不能为零。")
            return

        bfd_fast_value = efl_fast_val - (t_fast / n_fast) + (t_slow * (n_slow - 1) / n_slow if n_slow !=0 else 0)
        bfd_slow_value = efl_slow_val - (t_slow / n_slow if n_slow !=0 else 0)

        precision = self.precision_spinbox.value()
        self.bfd_fast_label.setText(f"快轴后焦距 (FOC BFD): {bfd_fast_value:.{precision}f} mm")
        self.bfd_slow_label.setText(f"慢轴后焦距 (SOC BFD): {bfd_slow_value:.{precision}f} mm")

        self.save_input_settings() 

    def show_error_message(self, title, message):
        # 显示错误消息框
        msg_box = QMessageBox(self) 
        msg_box.setIcon(QMessageBox.Icon.Warning) 
        msg_box.setWindowTitle(title)
        msg_box.setText(message)
        msg_box.setStyleSheet("QMessageBox { background-color: #333; color: white; } "
                              "QMessageBox QLabel { color: white; } "
                              "QMessageBox QPushButton { background-color: #4CAF50; color: white; padding: 5px 10px; border-radius: 3px; }")
        msg_box.exec()

    def keyPressEvent(self, event):
        # 键盘事件：回车键触发计算
        if event.key() == Qt.Key.Key_Return or event.key() == Qt.Key.Key_Enter:
            self.perform_bfd_calculation()
        else:
            super().keyPressEvent(event)


    def save_input_settings(self):
        # 保存当前所有输入字段的值到JSON文件
        inputs = {
            'material_fast': self.material_combobox_fast.currentText(),
            're_index_fast': self.re_index_input_fast.text(),
            'foc_curvature': self.curvature_fast_input.text(),
            'foc_efl': self.efl_fast_input.text(),
            'foc_thickness': self.thickness_fast_input.text(),

            'material_slow': self.material_combobox_slow.currentText(),
            're_index_slow': self.re_index_input_slow.text(),
            'soc_curvature': self.curvature_slow_input.text(),
            'soc_efl': self.efl_slow_input.text(),
            'soc_thickness': self.thickness_slow_input.text(),

            'precision': self.precision_spinbox.value()
        }
        self.save_json(inputs, 'BFD_Calculator_input.json')

    def load_input_settings(self):
        # 从JSON文件加载上次保存的输入
        inputs = self.load_json('BFD_Calculator_input.json', {})
        if not inputs: return 

        self.material_combobox_fast.setCurrentText(inputs.get('material_fast', "Custom"))
        self.re_index_input_fast.setText(inputs.get('re_index_fast', '1.5'))
        self.curvature_fast_input.setText(inputs.get('foc_curvature', ''))
        self.efl_fast_input.setText(inputs.get('foc_efl', ''))
        self.thickness_fast_input.setText(inputs.get('foc_thickness', ''))

        self.material_combobox_slow.setCurrentText(inputs.get('material_slow', "Custom"))
        self.re_index_input_slow.setText(inputs.get('re_index_slow', '1.5'))
        self.curvature_slow_input.setText(inputs.get('soc_curvature', ''))
        self.efl_slow_input.setText(inputs.get('soc_efl', ''))
        self.thickness_slow_input.setText(inputs.get('soc_thickness', ''))

        self.precision_spinbox.setValue(inputs.get('precision', 3))

        self._handle_re_index_text_changed(self.re_index_input_fast.text(), self.material_combobox_fast, self.save_button_fast, "fast")
        self._handle_re_index_text_changed(self.re_index_input_slow.text(), self.material_combobox_slow, self.save_button_slow, "slow")
        
        self._update_params_on_text_change(self.re_index_input_fast, "re_index", "fast")
        self._update_params_on_text_change(self.re_index_input_slow, "re_index", "slow")


    def show_formula_dialog(self):
        # 显示包含计算公式的信息对话框
        formula_text = (
            "<html><body style='font-size:14px; color: #333;'>"
            "<h3 style='color:#005A9C;'>后焦距 (BFD) 计算公式:</h3>"
            "<p><b>快轴后焦距 (FOC BFD):</b><br>"
            "&nbsp;&nbsp;BFD<sub>FOC</sub> = EFL<sub>FOC</sub> - (T<sub>FOC</sub> / n<sub>FOC</sub>) + (T<sub>SOC</sub> &times; (n<sub>SOC</sub> - 1) / n<sub>SOC</sub>)</p>"
            "<p><b>慢轴后焦距 (SOC BFD):</b><br>"
            "&nbsp;&nbsp;BFD<sub>SOC</sub> = EFL<sub>SOC</sub> - (T<sub>SOC</sub> / n<sub>SOC</sub>)</p>"
            "<h3 style='color:#005A9C;'>辅助公式:</h3>"
            "<p><b>有效焦距 (EFL) 与 曲率半径 (R):</b><br>"
            "&nbsp;&nbsp;EFL = R / (n - 1)<br>"
            "&nbsp;&nbsp;R = EFL &times; (n - 1)</p>"
            "<hr><p style='font-size:12px; color:#555;'>"
            "<b>符号说明:</b><br>"
            "EFL: 有效焦距 (Effective Focal Length)<br>"
            "T: 中心厚度 (Thickness)<br>"
            "n: 材料折射率 (Refractive Index)<br>"
            "R: 曲率半径 (Radius of Curvature)<br>"
            "FOC: 快轴 (Fast Axis)<br>"
            "SOC: 慢轴 (Slow Axis)"
            "</p></body></html>"
        )
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("计算公式说明")
        msg_box.setTextFormat(Qt.TextFormat.RichText) 
        msg_box.setText(formula_text)
        msg_box.setIcon(QMessageBox.Icon.Information)
        ok_button = msg_box.addButton("确定", QMessageBox.ButtonRole.AcceptRole)
        msg_box.exec()

    @staticmethod
    def is_float(value_str):
        # 检查字符串是否可以转换为浮点数
        try:
            float(value_str)
            return True
        except ValueError:
            return False

    def _update_params_on_text_change(self, changed_field, source_field_type, axis_type):
        # 当折射率、曲率半径或EFL文本框内容改变时，联动更新相关参数
        if axis_type == "fast":
            if self._updating_fast: return
            self._updating_fast = True
            n_input, r_input, efl_input = self.re_index_input_fast, self.curvature_fast_input, self.efl_fast_input
        elif axis_type == "slow":
            if self._updating_slow: return
            self._updating_slow = True
            n_input, r_input, efl_input = self.re_index_input_slow, self.curvature_slow_input, self.efl_slow_input
        else:
            return 

        try:
            n_text = n_input.text().strip()
            r_text = r_input.text().strip()
            efl_text = efl_input.text().strip()

            n = float(n_text) if n_text and self.is_float(n_text) else None
            r = float(r_text) if r_text and self.is_float(r_text) else None
            efl = float(efl_text) if efl_text and self.is_float(efl_text) else None

            precision = self.precision_spinbox.value() 

            if n is not None and n > 1: 
                n_minus_1 = n - 1
                if n_minus_1 == 0: 
                    if source_field_type == "curvature" and r is not None: efl_input.clear()
                    elif source_field_type == "efl" and efl is not None: r_input.clear()
                    elif source_field_type == "re_index":
                        if r is not None: efl_input.clear()
                        if efl is not None: r_input.clear()
                    return 

                if source_field_type == "curvature" and r is not None:
                    calculated_efl = r / n_minus_1
                    if changed_field != efl_input: efl_input.setText(f"{calculated_efl:.{precision}f}")
                elif source_field_type == "efl" and efl is not None:
                    calculated_r = efl * n_minus_1
                    if changed_field != r_input: r_input.setText(f"{calculated_r:.{precision}f}")
                elif source_field_type == "re_index":
                    if r is not None: 
                        calculated_efl = r / n_minus_1
                        if changed_field != efl_input: efl_input.setText(f"{calculated_efl:.{precision}f}")
                    elif efl is not None: 
                        calculated_r = efl * n_minus_1
                        if changed_field != r_input: r_input.setText(f"{calculated_r:.{precision}f}")
            else: 
                if source_field_type == "re_index":
                    if changed_field != efl_input: efl_input.clear()
                    if changed_field != r_input: r_input.clear()
        except ValueError:
            pass
        finally:
            if axis_type == "fast": self._updating_fast = False
            elif axis_type == "slow": self._updating_slow = False


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Calculator()
    window.show() 
    sys.exit(app.exec())
