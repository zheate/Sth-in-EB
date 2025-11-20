# 测试数据分析系统 V1.2.3

一个基于 Streamlit 的综合性光学测试数据分析与计算工具集，专为光纤和光学器件的测试、分析和计算而设计。

## 📋 项目简介

本系统集成了多个功能模块，包括数据提取、测试分析、光学计算器等，为光学器件的研发和生产提供全方位的数据支持。

## ✨ 主要功能
### 1. 数据提取 (Data_fetch)
- 📊 从 Excel 文件中提取和处理测试数据
- 🔄 支持批量数据处理和缓存机制
- 📈 数据清洗和格式化
- 💾 支持多种数据源格式

### 2. 测试分析 (TestAnalysis)

- 📉 测试数据的统计分析
- 📊 数据可视化展示
- 🔍 异常数据检测
- 📋 生成分析报告

### 3. 进度跟踪 (Progress)

- 📅 WIP（在制品）进度监控
- 📊 生产进度可视化
- 🎯 关键指标追踪
- 📈 趋势分析

### 4. 光学计算器 (Optical_Calculators)

#### NA 计算器 (NA_Calculator)

- 🔬 数值孔径 (NA) 计算
- 📏 根据几何参数计算 NA 值
- 🎯 根据 NA 值反算光纤端面到小孔的距离
- 🔄 端帽光阑计算（支持多参数互算）
- 📐 角度和折射率计算
- 💾 材料库管理

#### BFD 计算器 (BFD_Calculator)

- 📏 后焦距 (Back Focal Distance) 计算
- 🔍 光学系统参数优化
- 📊 多种计算模式支持

### 5. COS 滤波器 (COS_Filter)

- 🔧 COS 数据处理和滤波
- 📊 数据质量分析
- 🎯 参数优化

## 🚀 快速开始

### 环境要求

- Python 3.8+
- Streamlit
- Pandas
- NumPy
- 其他依赖见 requirements.txt

### 安装步骤

1. 克隆项目

```bash
git clone https://github.com/zheate/V1.2.git
cd V1.2.3
```

2. 安装依赖

```bash
pip install -r requirements.txt
```

3. 运行应用

**Windows 用户：**

```bash
run.bat
```

**或直接使用 Streamlit：**

```bash
streamlit run app/app.py
```

## 🖥 使用 streamlit-desktop-app 打包

1. 安装桌面打包依赖（`requirements.txt` 已包含 `streamlit-desktop-app` 与 `pyinstaller`）：
   ```bash
   pip install -r requirements.txt
   ```
2. 在仓库根目录本地验证桌面模式：
   ```bash
   python desktop_app.py
   ```
3. 生成 Windows 可执行文件（携带 metadata，避免 `PackageNotFoundError: streamlit`）：
   ```bash
   pyinstaller --noconfirm --windowed --clean ^
     --add-data "app;app" ^
     --collect-all streamlit ^
     --copy-metadata streamlit ^
     --name SthInEB desktop_app.py
   ```
   - 产物路径：`dist/SthInEB/SthInEB.exe`，首次启动会自动解压临时文件夹。
   - 如需图标，追加 `--icon your_icon.ico`。
   - 如遇中文路径导致异常，可先将项目移到英文路径后再打包。
   - 窗口标题、尺寸和 Streamlit 选项可在 `desktop_app.py` 中调整。
4. 也可直接运行 `build_desktop.bat`（参数与上面命令一致）。

## 📁 项目结构

```
V1.2.3/
├── app/
│   ├── .streamlit/          # Streamlit 配置
│   ├── config/              # 配置文件
│   │   ├── material.json    # 材料数据库
│   │   ├── NA_Calculator_input.json
│   │   ├── BFD_Calculator_input.json
│   │   └── ...
│   ├── data/                # 数据文件
│   │   ├── endcap.png       # 端帽光阑示意图
│   │   └── *.xlsx           # 测试数据文件
│   ├── pages/               # 功能页面
│   │   ├── Data_fetch.py
│   │   ├── TestAnalysis.py
│   │   ├── Progress.py
│   │   ├── NA_Calculator.py
│   │   ├── BFD_Calculator.py
│   │   ├── COS_Filter.py
│   │   └── Optical_Calculators.py
│   ├── utils/               # 工具函数
│   └── app.py               # 主应用入口
├── .gitignore
├── run.bat                  # Windows 启动脚本
└── README.md
```

## 🔧 配置说明

### 材料数据库

材料折射率数据存储在 `app/config/material.json` 中，可通过 NA 计算器的材料管理界面进行编辑。

### 数据文件

测试数据文件放置在 `app/data/` 目录下，支持 `.xlsx` 和 `.xls` 格式。

## 📖 使用说明

### NA 计算器使用

1. **标准 NA 计算模式**

   - 输入小孔半径 (r)
   - 选择材料或输入折射率 (n)
   - 输入长度 (L) 或 NA 值
   - 点击计算获得结果
2. **端帽光阑计算模式**

   - 选择端帽材料
   - 输入已知参数（NA、端帽长度、空气传播距离、光阑半径）
   - 留空需要计算的参数
   - 系统自动计算未知参数

### 数据提取使用

1. 上传或选择测试数据文件
2. 配置数据提取参数
3. 执行数据提取和处理
4. 导出处理后的数据

## 🎯 特色功能

- ✅ **智能缓存**：自动缓存处理结果，提升性能
- ✅ **多路径支持**：灵活的文件路径配置
- ✅ **友好界面**：基于 Streamlit 的现代化 UI
- ✅ **实时计算**：即时反馈计算结果
- ✅ **数据可视化**：丰富的图表展示
- ✅ **错误提示**：详细的错误信息和使用提示

## 🔄 更新日志

### V1.2.3 (最新)

- ✨ 优化端帽光阑示意图显示功能
- 🔧 支持多路径图片查找
- 🐛 修复图片加载错误提示
- 📦 更新 .gitignore 配置
- 📊 添加测试数据文件

### V1.2.2

- 🎨 改进 UI 界面
- 🔧 优化计算性能
- 📝 完善文档

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

本项目仅供内部使用。

## 👥 联系方式

- 项目维护：leo_z
- GitHub：https://github.com/zheate/V1.2

## 🙏 致谢

感谢所有为本项目做出贡献的开发者和测试人员。

---

**注意事项：**

- 请确保 Python 环境正确配置
- 数据文件较大时可能需要较长加载时间
- 建议使用 Chrome 或 Edge 浏览器以获得最佳体验
