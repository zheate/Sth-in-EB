# 错误处理和用户反馈系统

本文档描述了为数据收集和分析系统实现的错误处理和用户反馈功能。

## 实现的功能

### 1. 全局错误处理 (error_handler.py)

#### 核心功能
- **统一的错误处理器 (ErrorHandler)**
  - 日志记录系统初始化
  - 错误消息格式化（用户友好 vs 详细日志）
  - 错误分类和记录
  - 安全执行函数包装

#### 日志系统
- 自动创建日志目录和文件
- 按日期分割日志文件
- 文件日志：DEBUG级别，包含完整堆栈跟踪
- 控制台日志：WARNING级别，仅显示重要信息
- 日志格式：时间戳 - 模块名 - 级别 - 消息

#### 错误分类
定义了专门的异常类型：
- `DataCollectionError`: 数据收集错误基类
- `DataValidationError`: 数据验证错误
- `DataStorageError`: 数据存储错误
- `DataLoadError`: 数据加载错误
- `ConfigurationError`: 配置错误

#### 使用示例

```python
from utils.error_handler import ErrorHandler, handle_exceptions

# 初始化日志系统
ErrorHandler.initialize_logger()

# 方式1: 使用装饰器
@handle_exceptions(context="数据收集")
def collect_data():
    # 函数实现
    pass

# 方式2: 手动处理
try:
    # 执行操作
    result = some_operation()
except Exception as e:
    ErrorHandler.log_error(e, "操作失败")
    success, message = ErrorHandler.handle_error(e, "操作失败")

# 方式3: 安全执行
success, result, error_msg = ErrorHandler.safe_execute(
    some_function,
    arg1, arg2,
    context="执行某操作",
    default_return=None
)
```

### 2. 操作反馈系统 (feedback_handler.py)

#### 核心功能
- **用户反馈处理器 (FeedbackHandler)**
  - 成功/错误/警告/信息消息显示
  - 加载旋转器（spinner）
  - 进度条显示
  - 数据收集进度显示
  - 操作结果显示
  - 验证反馈显示

#### 主要方法

##### 基础消息显示
```python
FeedbackHandler.show_success("操作成功！")
FeedbackHandler.show_error("操作失败")
FeedbackHandler.show_warning("注意事项")
FeedbackHandler.show_info("提示信息")
```

##### 加载状态
```python
# 使用上下文管理器
with FeedbackHandler.show_spinner("正在加载..."):
    load_data()

# 或使用占位符
status = FeedbackHandler.show_loading_status("正在处理...")
# 执行操作
status.empty()  # 清除状态
```

##### 进度条
```python
# 通用进度条
results = FeedbackHandler.show_progress_bar(
    items=shell_ids,
    process_func=collect_shell_data,
    message_template="处理中... {current}/{total}",
    success_message="处理完成！"
)

# 数据收集专用进度条
results = FeedbackHandler.show_collection_progress(
    shell_ids=shell_ids,
    data_sources={'data_fetch': True, 'test_analysis': True},
    collect_func=collect_function
)
```

##### 验证反馈
```python
# 基础验证反馈
FeedbackHandler.show_validation_feedback(
    is_valid=True,
    messages=["警告1", "警告2"],
    title="数据验证"
)

# 详细验证反馈（带分类和建议）
FeedbackHandler.show_detailed_validation_feedback(
    is_valid=False,
    messages=error_list,
    title="数据验证",
    show_suggestions=True
)
```

### 3. 增强的数据模块

#### data_collector.py
- 在所有数据收集方法中添加了错误日志记录
- 捕获并记录每个壳体的收集错误
- 错误信息包含在返回结果中

#### data_storage.py
- 保存操作的详细日志记录
- 加载操作的详细日志记录
- 增强的验证反馈，包括：
  - 数据完整性统计
  - 分类的错误和警告
  - 修复建议
- 新增 `save_dataset_with_validation` 方法，支持保存前验证

#### data_validator.py
- 导入错误处理模块
- 为未来的验证增强做准备

#### DataAnalysis.py 页面
- 所有加载操作使用 `FeedbackHandler.show_spinner`
- 所有成功/错误消息使用 `FeedbackHandler` 方法
- 错误记录到日志系统

## 错误处理流程

### 数据收集流程
```
用户触发收集
    ↓
显示进度条 (FeedbackHandler)
    ↓
收集每个壳体数据
    ├─ 成功: 记录INFO日志
    └─ 失败: 记录ERROR日志 + 标记数据不可用
    ↓
显示收集结果
```

### 数据保存流程
```
用户触发保存
    ↓
验证数据集 (可选)
    ├─ 有错误: 显示详细错误 + 修复建议
    └─ 有警告: 显示警告 + 允许继续
    ↓
创建目录（如需要）
    ├─ 成功: 记录INFO日志
    └─ 失败: 记录ERROR日志 + 返回错误
    ↓
保存JSON文件
    ├─ 成功: 记录INFO日志 + 显示成功消息
    └─ 失败: 记录ERROR日志 + 显示错误消息
```

### 数据加载流程
```
用户选择文件
    ↓
显示加载状态 (spinner)
    ↓
验证文件格式
    ├─ 失败: 记录ERROR + 显示错误
    └─ 成功: 继续
    ↓
读取JSON文件
    ├─ 失败: 记录ERROR + 显示错误
    └─ 成功: 继续
    ↓
验证数据集结构
    ├─ 失败: 显示分类错误 + 修复建议
    ├─ 有警告: 显示警告 + 加载数据
    └─ 成功: 显示成功消息 + 数据统计
```

## 日志文件

### 位置
- 默认目录: `logs/`
- 文件名格式: `data_collection_YYYYMMDD.log`
- 每天一个日志文件

### 日志级别
- **DEBUG**: 详细的调试信息
- **INFO**: 一般信息（操作成功、状态变化）
- **WARNING**: 警告信息（数据验证警告）
- **ERROR**: 错误信息（操作失败、异常）

### 日志内容示例
```
2025-10-18 10:30:00 - DataCollection - INFO - 开始加载数据集: C:/data/dataset.json
2025-10-18 10:30:01 - DataCollection - INFO - JSON文件解析成功: C:/data/dataset.json
2025-10-18 10:30:01 - DataCollection - WARNING - 数据验证警告: 壳体'Shell001'的data_fetch缺少data_available字段
2025-10-18 10:30:01 - DataCollection - INFO - 数据集加载成功: C:/data/dataset.json, 壳体数量: 50, 完整数据: 45, 部分数据: 5
```

## 用户体验改进

### 1. 清晰的错误消息
- 用户友好的错误描述
- 避免技术术语
- 提供具体的错误位置

### 2. 详细的反馈
- 操作进度实时显示
- 成功/失败状态明确
- 数据统计信息

### 3. 修复建议
- 针对错误类型的具体建议
- 分类显示（关键/字段/数据错误）
- 可展开的详细信息

### 4. 非阻塞式警告
- 警告不阻止操作继续
- 可选择查看详情
- 记录到日志供后续查看

## 最佳实践

### 1. 错误处理
```python
# ✅ 推荐：使用装饰器
@handle_exceptions(context="数据收集")
def collect_data():
    # 实现
    pass

# ✅ 推荐：记录详细上下文
try:
    result = operation()
except Exception as e:
    ErrorHandler.log_error(e, f"处理壳体'{shell_id}'时出错")
```

### 2. 用户反馈
```python
# ✅ 推荐：使用进度条
with FeedbackHandler.show_spinner("正在处理..."):
    process_data()

# ✅ 推荐：显示详细结果
FeedbackHandler.show_operation_result(
    success=True,
    success_message="操作成功",
    error_message="",
    details="处理了50个项目"
)
```

### 3. 日志记录
```python
# ✅ 推荐：记录关键操作
logger = ErrorHandler.get_logger()
logger.info(f"开始处理: {item_count} 个项目")
logger.debug(f"详细参数: {params}")

# ✅ 推荐：记录错误时包含上下文
ErrorHandler.log_error(e, f"处理项目 {item_id} 时失败")
```

## 未来改进

1. **错误恢复机制**
   - 自动重试失败的操作
   - 部分失败时的继续处理选项

2. **更详细的进度信息**
   - 估计剩余时间
   - 当前处理速度

3. **错误统计和报告**
   - 错误趋势分析
   - 常见错误汇总

4. **用户配置**
   - 可配置的日志级别
   - 可配置的反馈详细程度
