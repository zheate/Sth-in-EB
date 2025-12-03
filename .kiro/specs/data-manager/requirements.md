# Requirements Document

## Introduction

本功能为 Streamlit 数据分析应用提供产品数据管理与分析系统（Zh's DataBase），实现三层数据架构：
- 第一层：产品类型管理（从Progress保存的数据，支持重命名和附件管理）
- 第二层：壳体进度分析（甘特图展示壳体号和最新站别信息）
- 第三层：数据分析（调用Data_fetch逻辑进行多站别分析，支持指标筛选）

## Glossary

- **ProductType**: 产品类型，如 M20-AM-C，从 Progress 页面保存时指定的名称
- **ProductionOrder**: 生产订单，格式如 WO-MP-M20-HX-25090251
- **ShellID**: 壳体号，唯一标识一个壳体
- **Station**: 站别，生产流程中的工序节点
- **Attachment**: 附件，与产品类型关联的 PDF 或 Excel 文件
- **GanttChart**: 甘特图，用于展示壳体进度的可视化图表
- **DataFetchLogic**: Data_fetch 页面的数据提取和分析逻辑
- **Metric**: 指标，用于数据分析的测量值，如功率、效率等
- **Threshold**: 阈值，用户设定的指标合格范围

## Requirements

### Requirement 1: 产品类型管理（第一层）

**User Story:** As a 数据分析人员, I want to 从 Progress 页面保存数据并指定产品类型名称, so that 我可以按产品类型组织和管理生产数据。

#### Acceptance Criteria

1. WHEN 用户在 Progress 页面点击"另存为"按钮 THEN THE System SHALL 弹出对话框允许用户输入产品类型名称（如 M20-AM-C）
2. WHEN 用户保存产品类型 THEN THE System SHALL 将该产品类型下的所有壳体数据和生产订单信息保存到本地数据库
3. WHEN 用户在 Data_Manager 页面选择产品类型 THEN THE System SHALL 显示该产品类型下的所有生产订单列表
4. WHEN 用户选择产品类型 THEN THE System SHALL 在产品类型旁边显示附件上传按钮
5. WHEN 用户重命名产品类型 THEN THE System SHALL 更新数据库中的产品类型名称并保持关联数据不变

### Requirement 2: 附件管理

**User Story:** As a 数据分析人员, I want to 为产品类型上传和管理附件, so that 我可以将相关的技术文档与产品数据关联。

#### Acceptance Criteria

1. WHEN 用户点击产品类型旁的上传按钮 THEN THE System SHALL 允许上传 PDF 或 Excel 格式的附件
2. WHEN 附件上传成功 THEN THE System SHALL 将附件保存到 data/zh_database 目录并关联到对应产品类型
3. WHEN 用户查看附件列表 THEN THE System SHALL 默认折叠附件预览区域
4. WHEN 用户展开附件预览 THEN THE System SHALL 显示 PDF 或 Excel 文件的预览内容
5. WHEN 用户删除附件 THEN THE System SHALL 从文件系统和数据库中移除该附件

### Requirement 3: 生产订单选择

**User Story:** As a 数据分析人员, I want to 选择一个或多个生产订单进行分析, so that 我可以灵活地分析不同批次的生产数据。

#### Acceptance Criteria

1. WHEN 用户选择产品类型后 THEN THE System SHALL 显示该产品类型下的所有生产订单（格式如 WO-MP-M20-HX-25090251）
2. WHEN 用户选择生产订单 THEN THE System SHALL 支持单选和多选模式
3. WHEN 显示生产订单列表 THEN THE System SHALL 同时显示每个订单的时间信息
4. WHEN 用户未选择时间 THEN THE System SHALL 默认选择最新的时间记录
5. WHEN 用户选择多个生产订单 THEN THE System SHALL 合并显示所有选中订单的壳体数据

### Requirement 4: 壳体进度分析（第二层）

**User Story:** As a 数据分析人员, I want to 查看选中订单的壳体进度, so that 我可以了解每个壳体的当前生产状态。

#### Acceptance Criteria

1. WHEN 用户选择生产订单后 THEN THE System SHALL 显示该订单下所有壳体号和最新站别信息
2. WHEN 显示壳体进度 THEN THE System SHALL 使用甘特图展示每个壳体的进度状态
3. WHEN 甘特图渲染 THEN THE System SHALL 按站别顺序显示每个壳体的完成情况
4. WHEN 用户悬停在甘特图上 THEN THE System SHALL 显示该壳体的详细站别时间信息
5. WHEN 壳体数量超过显示限制 THEN THE System SHALL 提供分页或滚动功能

### Requirement 5: 数据分析（第三层）

**User Story:** As a 数据分析人员, I want to 对选中的壳体进行多站别数据分析, so that 我可以评估产品质量并识别异常。

#### Acceptance Criteria

1. WHEN 用户进入数据分析层 THEN THE System SHALL 调用 Data_fetch 的逻辑获取壳体的测试数据
2. WHEN 数据获取完成 THEN THE System SHALL 显示多站别分析结果表格
3. WHEN 显示分析结果 THEN THE System SHALL 支持按指标列进行筛选
4. WHEN 用户设定指标阈值 THEN THE System SHALL 高亮显示超出阈值的数据
5. WHEN 用户修改阈值 THEN THE System SHALL 实时更新筛选结果

### Requirement 6: 指标阈值设定

**User Story:** As a 数据分析人员, I want to 自定义指标的合格范围, so that 我可以根据产品规格筛选合格和不合格的壳体。

#### Acceptance Criteria

1. WHEN 用户打开阈值设定界面 THEN THE System SHALL 显示所有可用的数值指标列
2. WHEN 用户设定某指标的最小值和最大值 THEN THE System SHALL 将该范围作为合格标准
3. WHEN 应用阈值筛选 THEN THE System SHALL 将数据分为符合标准和不符合标准两组
4. WHEN 显示筛选结果 THEN THE System SHALL 显示合格率统计和不合格原因分析
5. WHEN 用户保存阈值配置 THEN THE System SHALL 将配置与产品类型关联以便下次使用

### Requirement 7: 数据持久化

**User Story:** As a 数据分析人员, I want to 数据能够持久化保存, so that 我可以在不同会话中访问相同的数据。

#### Acceptance Criteria

1. WHEN 保存产品类型数据 THEN THE System SHALL 使用 JSON 格式存储元数据到 data/zh_database 目录
2. WHEN 保存壳体数据 THEN THE System SHALL 使用 Parquet 格式存储以获得高压缩率
3. WHEN 应用启动 THEN THE System SHALL 自动加载已保存的产品类型列表
4. WHEN 数据文件损坏 THEN THE System SHALL 提供错误提示并允许用户删除损坏的数据
5. WHEN 用户删除产品类型 THEN THE System SHALL 同时删除关联的壳体数据和附件文件

