# 光耦测试数据分析系统

一个基于 Streamlit 的光耦测试数据分析和计算工具集。

## 功能模块

### 📚 数据提取
- 光耦测试数据的提取和处理
- 支持 Excel 文件导入

### 📊 进度追踪
- 模块测试进度可视化
- 实时数据监控

### 📈 测试数据分析
- 测试数据的统计分析
- 数据可视化展示

### 🔧 后焦距计算器 (BFD Calculator)
- 快轴和慢轴后焦距计算
- 支持多种材料折射率
- 自动计算曲率半径和有效焦距
- 材料管理功能

### 🔬 NA 计算器 (Numerical Aperture Calculator)
- 数值孔径 (NA) 计算
- 根据几何参数计算 NA 值
- 根据 NA 值反算光纤距离
- 光纤端面接受角度计算

## 安装

### 环境要求
- Python 3.8+
- pip

### 安装依赖

```bash
pip install -r requirements.txt
```

## 运行

### Windows
双击运行 `run.bat` 文件，或在命令行中执行：

```bash
cd V1.2
streamlit run app/app.py
```

### Linux/Mac
```bash
cd V1.2
streamlit run app/app.py
```

## 项目结构

```
V1.2/
├── app/
│   ├── .streamlit/          # Streamlit 配置
│   │   ├── config.toml      # 应用配置
│   │   └── pages.toml       # 页面配置
│   ├── pages/               # 页面模块
│   │   ├── Data_fetch.py    # 数据提取
│   │   ├── Progress.py      # 进度追踪
│   │   ├── TestAnalysis.py  # 测试分析
│   │   ├── BFD_Calculator.py # 后焦距计算器
│   │   └── NA_Calculator.py  # NA 计算器
│   ├── utils/               # 工具函数
│   ├── data/                # 数据文件
│   ├── material.json        # 材料数据库
│   └── app.py               # 主应用入口
├── tools/                   # PyQt6 工具（原始版本）
│   ├── BFD.py
│   └── NA.py
└── run.bat                  # Windows 启动脚本
```

## 使用说明

### 后焦距计算器
1. 选择或管理材料
2. 输入快轴和慢轴参数（曲率半径或有效焦距）
3. 输入中心厚度
4. 点击"计算 BFD"获取结果

### NA 计算器
1. 选择材料（自动加载折射率）
2. 输入小孔半径
3. 输入长度或 NA 值（二选一）
4. 点击"计算"获取结果

## 技术栈

- **前端框架**: Streamlit
- **数据处理**: Pandas, NumPy
- **可视化**: Plotly, Matplotlib
- **Excel 处理**: openpyxl

## 开发

### 添加新页面
1. 在 `V1.2/app/pages/` 目录下创建新的 Python 文件
2. 在 `V1.2/app/.streamlit/pages.toml` 中注册新页面

### 材料管理
材料数据存储在 `material.json` 文件中，可通过界面进行管理。

## 许可证

MIT License

## 作者

长光华芯软件团队
