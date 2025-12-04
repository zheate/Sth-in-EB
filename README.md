# 📊 测试数据分析系统

一个基于 Streamlit 的综合性光学测试数据分析与计算工具集，专为光纤和光学器件的测试、分析和计算而设计。

## 📋 项目简介

本系统集成了多个功能模块，包括数据提取、测试分析、工程分析、光学计算器、折射率查询等，为光学器件的研发和生产提供全方位的数据支持。

## ✨ 主要功能

### 1. 📊 数据提取 (Data_fetch)
- 从 Excel 文件中提取和处理测试数据
- 支持批量数据处理和缓存机制
- 多站点数据对比分析
- 功率、效率、波长趋势分析
- 数据拟合与预测功能
- 箱线图分布分析
- 数据清洗和格式化
- 支持多种数据源格式

### 2. 📈 测试分析 (TestAnalysis)
- 测试数据的统计分析
- 数据可视化展示
- 异常数据检测
- 生成分析报告

### 3. 📅 进度跟踪 (Progress)
- WIP（在制品）进度监控
- 生产进度可视化
- 关键指标追踪
- 趋势分析

### 4. 📊 工程分析 (Engineering_Analysis) ⭐ 新增
- **数据概览**：总不良数、唯一SN数、涉及站点、不良类型统计
- **不良分析**：
  - 站点分析：各站点不良数量分布、Top 5 排名
  - 现象分析：各现象不良数量分布、Top 5 排名
  - 原因分析：不良原因饼图与统计表
- **趋势分析**：
  - 按日/周/月查看不良趋势
  - 移动平均线分析
  - 峰值、最低值、最近变化统计
- **交叉分析**：站点-现象交叉热力图
- **帕累托分析**：80/20法则分析，快速定位关键问题
- **筛选功能**：
  - 快速日期选择（今天、最近7天、最近30天）
  - 生产线筛选
  - 工单类型筛选
  - 关键词搜索
- **数据导出**：支持导出 CSV 和 Excel 格式

### 5. 🔬 光学计算器 (Optical_Calculators)

#### NA 计算器 (NA_Calculator)
- 数值孔径 (NA) 计算
- 根据几何参数计算 NA 值
- 根据 NA 值反算光纤端面到小孔的距离
- 端帽光阑计算（支持多参数互算）
- 角度和折射率计算
- 材料库管理

#### BFD 计算器 (BFD_Calculator)
- 后焦距 (Back Focal Distance) 计算
- 光学系统参数优化
- 多种计算模式支持

### 6. 🔍 折射率查询 (Refractive_Index) ⭐ 新增
- **材料数据库**：集成 refractiveindex.info 数据库
- **材料搜索**：支持按材料名称、厂家、代号搜索
- **折射率计算**：输入波长即可查询折射率 n 和消光系数 k
- **可视化展示**：折射率与波长关系曲线图
- **支持厂家**：
  - Schott（肖特）光学玻璃
  - Ohara（小原）光学玻璃
  - Sumita（住田）光学玻璃
  - Vitron 红外材料
  - 以及更多材料数据

### 7. 🔧 COS 滤波器 (COS_Filter)
- COS 数据处理和滤波
- 数据质量分析
- 参数优化

## 🚀 快速开始

### 环境要求

- Python 3.8+
- Streamlit
- Pandas
- NumPy
- Altair（数据可视化）
- PyYAML（折射率数据解析）
- openpyxl（Excel 文件处理）
- 其他依赖见 requirements.txt

### 安装步骤

1. 克隆项目

```bash
git clone https://github.com/zheate/Sth-in-EB.git
cd Sth-in-EB
```

2. 安装依赖

```bash
pip install -r requirements.txt
```

3. 运行应用

**方式一：使用启动脚本（推荐）**

```bash
run.bat
```

**方式二：直接使用 Streamlit**

```bash
streamlit run app/app.py
```

**方式三：使用打包后的可执行文件**

直接运行 `dist/Sth-in-EB/Sth-in-EB.exe`（需先打包）

## 🔐 登录认证

- 应用启动后默认呈现登录界面，所有页面在通过验证前不可访问。
- 账号信息保存在 `app/config/users.json` 中，仅包含用户名、角色与 bcrypt 哈希。
- 使用 `python -m app.auth_cli add --username <name>` 可创建账号并交互式输入密码。
- 只有系统用户名属于 `AUTH_ADMIN_OS_USERS`（默认在 `app/config/config.py` 中配置）时，才能执行 `add/password/delete`；可通过设置环境变量 `AUTH_ADMIN_OS_USERS` 临时扩展授权列表。
- 更多账号管理说明请参见 [`docs/auth.md`](docs/auth.md)。

## 📦 打包为可执行文件

### 使用 PyInstaller 打包

1. 确保已安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

2. 运行打包命令：
   ```bash
   pyinstaller Sth-in-EB.spec
   ```

3. 打包完成后，可执行文件位于：
   ```
   dist/Sth-in-EB/
   ├── Sth-in-EB.exe    # 主程序
   └── _internal/        # 依赖文件（必须一起分发）
   ```

### 分发说明

**重要：分发时必须包含以下内容**
- ✅ `Sth-in-EB.exe` - 主程序
- ✅ `_internal/` 文件夹 - 所有依赖库和资源文件

**不能只分发 .exe 文件！** 程序运行需要 `_internal/` 文件夹中的依赖。

建议将整个 `dist/Sth-in-EB/` 文件夹打包成 zip 或 7z 分发给用户。

## 📁 项目结构

```
Sth-in-EB/
├── app/
│   ├── .streamlit/                      # Streamlit 配置
│   │   └── config.toml                  # 主题和服务器配置
│   ├── config/                          # 配置文件
│   │   ├── material.json                # 材料数据库
│   │   ├── NA_Calculator_input.json     # NA 计算器配置
│   │   ├── BFD_Calculator_input.json    # BFD 计算器配置
│   │   └── Rayleigh_Calculator_input.json
│   ├── data/                            # 数据文件
│   │   ├── endcap.png                   # 端帽光阑示意图
│   │   ├── refractiveindex.info-database/  # 折射率数据库
│   │   ├── 工程分析明细报表*.xlsx       # 工程分析数据
│   │   └── *.xlsx                       # 其他测试数据文件
│   ├── pages/                           # 功能页面
│   │   ├── Data_fetch.py                # 数据提取
│   │   ├── TestAnalysis.py              # 测试分析
│   │   ├── Progress.py                  # 进度跟踪
│   │   ├── Engineering_Analysis.py      # 工程分析 ⭐ 新增
│   │   ├── NA_Calculator.py             # NA 计算器
│   │   ├── BFD_Calculator.py            # BFD 计算器
│   │   ├── Refractive_Index.py          # 折射率查询 ⭐ 新增
│   │   ├── COS_Filter.py                # COS 滤波器
│   │   └── Optical_Calculators.py       # 光学计算器入口
│   ├── utils/                           # 工具函数
│   │   ├── data_cleaning.py             # 数据清洗工具
│   │   ├── refractive_index_helper.py   # 折射率查询辅助 ⭐ 新增
│   │   └── __init__.py
│   └── app.py                           # 主应用入口
├── dist/                                # 打包输出目录
│   └── Sth-in-EB/
│       ├── Sth-in-EB.exe               # 可执行文件
│       └── _internal/                   # 依赖文件
├── build/                               # 打包临时文件
├── .gitignore
├── Sth-in-EB.spec                      # PyInstaller 配置
├── run.bat                              # Windows 启动脚本
├── run_app.py                           # Python 启动脚本
├── requirements.txt                     # 依赖列表
└── README.md                            # 项目说明
```

## 🔧 配置说明

### 材料数据库

1. **NA 计算器材料库**：存储在 `app/config/material.json`，可通过 NA 计算器的材料管理界面进行编辑
2. **折射率数据库**：位于 `app/data/refractiveindex.info-database/`，包含数千种光学材料的折射率数据

### 数据文件

1. **测试数据**：放置在 `app/data/` 目录下，支持 `.xlsx` 和 `.xls` 格式
2. **工程分析数据**：文件名格式为 `工程分析明细报表*.xlsx`，系统会自动加载最新的文件
3. **WIP 数据**：用于进度跟踪的在制品数据

### Streamlit 配置

配置文件位于 `app/.streamlit/config.toml`，可自定义：
- 主题颜色
- 服务器端口
- 浏览器行为
- 文件上传限制等

## 📖 使用说明

### 数据提取 (Data_fetch)

1. 选择测试数据文件（支持多个壳体号）
2. 选择电流点进行数据提取
3. 查看多站点对比分析：
   - 功率对比图和拟合预测
   - 效率对比图和拟合预测
   - 波长对比图和拟合预测
4. 查看箱线图分布分析
5. 查看单壳体详细分析
6. 导出处理后的数据

### 工程分析 (Engineering_Analysis)

1. 系统自动加载最新的工程分析数据
2. 使用侧边栏筛选条件：
   - 快速选择日期范围
   - 筛选生产线和工单类型
   - 搜索关键词
3. 查看数据概览和关键指标
4. 切换标签页查看不同维度分析：
   - 站点分析
   - 现象分析
   - 原因分析
5. 查看趋势分析（支持按日/周/月）
6. 查看交叉分析热力图
7. 使用帕累托分析找出关键问题
8. 导出分析结果（CSV 或 Excel）

### NA 计算器

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

### 折射率查询 (Refractive_Index)

1. 在搜索框输入材料关键词（如：BK7、S-TIH53、Si）
2. 从匹配结果中选择目标材料
3. 输入目标波长（nm）
4. 查看折射率 n 和消光系数 k
5. 查看折射率-波长关系曲线图

## 🎯 特色功能

- ✅ **智能缓存**：自动缓存处理结果，提升性能
- ✅ **多路径支持**：灵活的文件路径配置
- ✅ **友好界面**：基于 Streamlit 的现代化 UI，渐变色设计
- ✅ **实时计算**：即时反馈计算结果
- ✅ **数据可视化**：丰富的图表展示（Altair 交互式图表）
- ✅ **数据拟合**：支持多项式拟合和预测
- ✅ **帕累托分析**：快速定位关键问题（80/20法则）
- ✅ **交叉分析**：多维度数据关联分析
- ✅ **趋势分析**：支持按日/周/月查看，含移动平均线
- ✅ **材料数据库**：集成数千种光学材料折射率数据
- ✅ **错误提示**：详细的错误信息和使用提示
- ✅ **数据导出**：支持 CSV 和 Excel 格式导出

## 🔄 更新日志

### 最新版本 (2024-11)

- ✨ **新增工程分析模块**：
  - 数据概览和关键指标统计
  - 站点、现象、原因多维度分析
  - 趋势分析（支持按日/周/月）
  - 交叉分析热力图
  - 帕累托分析（80/20法则）
  - 快速筛选和搜索功能
  - 支持导出 CSV 和 Excel

- ✨ **新增折射率查询模块**：
  - 集成 refractiveindex.info 数据库
  - 支持数千种光学材料查询
  - 折射率-波长关系可视化
  - 支持 Schott、Ohara、Sumita 等厂家材料

- 🎨 **界面优化**：
  - 渐变色主题设计
  - 改进图表样式和布局
  - 优化移动端显示

- 🔧 **功能增强**：
  - Data_fetch 新增数据拟合预测功能
  - 新增箱线图分布分析
  - 优化多站点对比分析
  - 改进数据清洗工具

- 📦 **打包优化**：
  - 更新 PyInstaller 配置
  - 优化依赖管理
  - 改进启动脚本

### V1.2.3

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
- GitHub：https://github.com/zheate/Sth-in-EB

## 🙏 致谢

- 感谢所有为本项目做出贡献的开发者和测试人员
- 折射率数据来源：[refractiveindex.info](https://refractiveindex.info/)

---

## ⚠️ 注意事项

- 请确保 Python 环境正确配置（Python 3.8+）
- 数据文件较大时可能需要较长加载时间
- 建议使用 Chrome 或 Edge 浏览器以获得最佳体验
- 工程分析数据文件需放置在 `app/data/` 目录下，文件名格式：`工程分析明细报表*.xlsx`
- 折射率数据库需完整保留在 `app/data/refractiveindex.info-database/` 目录
- 打包分发时必须包含 `_internal/` 文件夹

## 📸 功能截图

### 数据提取与分析
- 多站点数据对比
- 功率/效率/波长趋势分析
- 数据拟合与预测

### 工程分析
- 不良数据统计与可视化
- 帕累托分析
- 交叉分析热力图

### 光学计算器
- NA 计算器
- BFD 计算器
- 折射率查询

### 进度跟踪
- WIP 进度监控
- 生产进度可视化

---

**本项目专为光学器件测试与分析设计，持续更新中...**
