# Requirements Document

## Introduction

本功能为 Streamlit 数据分析应用增加本地数据保存能力，允许用户将数据提取（Data_fetch）、进度追踪（Progress）和工程分析（Engineering_Analysis）三个模块的数据统一保存到本地，并支持后续加载、管理和跨模块共享。

## Glossary

- **DataStore**: 本地数据存储管理器，负责数据的保存、加载、列表和删除操作
- **Dataset**: 一个保存的数据集，包含 DataFrame 数据和元数据信息
- **Metadata**: 数据集的描述信息，包括来源模块、保存时间、数据摘要等
- **Category**: 数据类别，分为 extraction（数据提取）、progress（进度追踪）、analysis（工程分析）

## Requirements

### Requirement 1: 数据保存功能

**User Story:** As a 数据分析人员, I want to 将当前分析结果保存到本地, so that 我可以在之后快速加载使用而无需重新提取数据。

#### Acceptance Criteria

1. WHEN 用户在数据提取页面点击保存按钮 THEN THE DataStore SHALL 将当前 result_df 和 lvi/rth 绘图数据保存为本地文件
2. WHEN 用户在进度追踪页面点击保存按钮 THEN THE DataStore SHALL 将当前 progress_df 和筛选条件保存为本地文件
3. WHEN 用户在工程分析页面点击保存按钮 THEN THE DataStore SHALL 将当前筛选后的 DataFrame 保存为本地文件
4. WHEN 保存数据时 THEN THE DataStore SHALL 自动生成包含时间戳和数据摘要的文件名
5. WHEN 保存成功 THEN THE System SHALL 显示成功提示并返回保存的文件标识

### Requirement 2: 数据加载功能

**User Story:** As a 数据分析人员, I want to 加载之前保存的数据集, so that 我可以快速恢复之前的分析状态。

#### Acceptance Criteria

1. WHEN 用户选择加载已保存的数据集 THEN THE DataStore SHALL 读取文件并还原 DataFrame 和元数据
2. WHEN 加载数据提取类数据集 THEN THE System SHALL 恢复 session_state 中的 extraction_state 和绘图数据源
3. WHEN 加载进度追踪类数据集 THEN THE System SHALL 恢复 session_state 中的 progress_df 和筛选条件
4. WHEN 加载工程分析类数据集 THEN THE System SHALL 将数据加载到当前分析视图
5. WHEN 加载的文件不存在或损坏 THEN THE DataStore SHALL 返回明确的错误信息

### Requirement 3: 数据集列表与管理

**User Story:** As a 数据分析人员, I want to 查看和管理已保存的数据集, so that 我可以找到需要的数据并清理不需要的旧数据。

#### Acceptance Criteria

1. WHEN 用户打开数据管理界面 THEN THE System SHALL 显示所有已保存数据集的列表
2. WHEN 显示数据集列表 THEN THE System SHALL 展示文件名、保存时间、数据类别、记录数等摘要信息
3. WHEN 用户选择删除某个数据集 THEN THE DataStore SHALL 删除对应文件并更新列表
4. WHEN 用户筛选数据集类别 THEN THE System SHALL 仅显示对应类别的数据集

### Requirement 4: 存储格式与性能

**User Story:** As a 系统管理员, I want to 数据以高效的格式存储, so that 存储空间占用小且加载速度快。

#### Acceptance Criteria

1. WHEN 保存 DataFrame 数据 THEN THE DataStore SHALL 使用 Parquet 格式存储以获得高压缩率
2. WHEN 保存元数据 THEN THE DataStore SHALL 使用 JSON 格式存储以便于人工查看
3. WHEN 数据集包含复杂对象（如绘图数据源字典） THEN THE DataStore SHALL 将其序列化为可恢复的格式
4. WHEN 加载大型数据集 THEN THE DataStore SHALL 在 3 秒内完成加载（10万行以内）

### Requirement 5: 跨模块数据共享

**User Story:** As a 数据分析人员, I want to 在不同模块间共享数据, so that 我可以将数据提取的结果用于工程分析。

#### Acceptance Criteria

1. WHEN 用户在数据管理页面选择数据集 THEN THE System SHALL 提供"发送到"选项以将数据加载到指定模块
2. WHEN 数据格式与目标模块不完全匹配 THEN THE System SHALL 进行必要的格式转换或提示用户
3. WHEN 共享数据到其他模块 THEN THE System SHALL 保留原始数据的关键列和元数据

### Requirement 6: 数据导出功能

**User Story:** As a 数据分析人员, I want to 将数据导出为常用格式, so that 我可以在其他工具中使用或分享给同事。

#### Acceptance Criteria

1. WHEN 用户选择导出数据 THEN THE System SHALL 提供 Excel (.xlsx) 和 CSV 两种导出格式
2. WHEN 导出 Excel 格式 THEN THE System SHALL 支持多 Sheet 导出（如主数据、统计摘要分开）
3. WHEN 导出 CSV 格式 THEN THE System SHALL 使用 UTF-8-BOM 编码以确保中文兼容性
4. WHEN 导出数据集 THEN THE System SHALL 自动生成包含日期和数据类型的文件名
5. WHEN 用户从数据管理页面导出 THEN THE System SHALL 允许批量选择多个数据集合并导出

### Requirement 7: 用户界面集成

**User Story:** As a 用户, I want to 在现有页面中方便地访问保存和加载功能, so that 我不需要离开当前工作流程。

#### Acceptance Criteria

1. WHEN 用户在任意数据页面 THEN THE System SHALL 在侧边栏或工具栏提供保存按钮
2. WHEN 用户点击保存 THEN THE System SHALL 弹出对话框允许用户输入可选的备注信息
3. WHEN 用户需要加载数据 THEN THE System SHALL 提供下拉选择或独立的数据管理页面
4. WHEN 当前有未保存的数据 THEN THE System SHALL 在页面显示提示标记
