## 目标
- 并发读取/解析测量文件，提升批量壳体/芯片处理速度
- 为模块目录索引增加缓存封装，与芯片目录一致
- 抽象重复的数值清洗为共享库，并配套单元测试
- 增加进度与异常监控、缓存命中率监控与手动刷新能力

## 并发读取/解析
- 线程池：`ThreadPoolExecutor(max_workers = clamp(4, 2*cpu_count, 16))`，IO+解析混合场景，限制上限避免资源争用
- 任务封装：为每个“测量文件”构建任务结构 `(entry_id, test_category, measurement_label, file_path, file_mtime)`
- 线程安全：
  - 解析在工作线程执行，返回经 `align_output_columns` 的 DataFrame 与信息对象
  - 仅在主线程汇总到 `combined_frames`、`lvi_plot_sources`、`rth_plot_sources`（避免并发写 `session_state`）
- 进度监控：
  - 已提交任务总数 N，完成数 k，实时更新 `progress_bar` 与状态文本
  - 失败捕获后将异常写入 `error_messages`
- 等价性保证：
  - 保持原串行流程的排序与合并（按`TEST_TYPE_COLUMN`、`CURRENT_COLUMN`排序），确保结果一致
- 代码改动点：
  - 任务收集与提交：`app/pages/Data_fetch.py:2391-2453`
  - 进度刷新：`app/pages/Data_fetch.py:2488-2501`

## 模块目录索引缓存
- 缓存键：`(test_folder_str, mtime)`；值：`Dict[token, List[(Path, timestamp, mtime)]]`
- 封装函数：`build_module_measurement_index_cached(folder_str, mtime)`（`@st.cache_data`）
- 调用处用缓存封装替代直接 `_build_measurement_file_index(test_folder, files_in_folder)`
- 命中率监控：在缓存封装中返回命中标志，累计到 `st.session_state.module_index_cache_hits/misses`
- 手动刷新：沿用现有“♻️ 强制刷新缓存”，同时清空模块索引缓存计数与内容
- 代码改动点：
  - 缓存封装与调用：`app/pages/Data_fetch.py:2412-2413` 附近新增并替换

## 数据清洗共享库
- 新增模块：`app/utils/data_cleaning.py`
- 接口设计：
  - `ensure_numeric(df, columns, *, strict=True) -> pd.DataFrame`：批量转数值，严格校验失败抛异常
  - `drop_zero_current(df, current_col, *, tol=1e-6) -> pd.DataFrame`：过滤近零电流
  - `clean_current_metric(df, current_col, metric_col) -> pd.DataFrame`：组合清洗与排序，返回列名统一为 `current/value`
- 日志：抛出 `ValueError` 时携带详细列名与样本行位置信息；在调用处捕获并汇总到 `error_messages`
- 替换点：
  - `_prepare_metric_series`、`build_multi_shell_chart`、`build_station_metric_chart` 与 `_clean_metric_dataframe` 使用共享库（`app/pages/Data_fetch.py:501-516`、`613-637`、`1015-1020`、`708-721`）

## 错误与进度监控
- 异常分类：路径错误、文件缺失、解析失败、清洗失败；在结果区按类别展开显示（沿用现有 `render_extraction_results_section`）
- 进度：基于 `futures.as_completed` 更新，失败任务不阻塞整体进度
- 线程池资源：提交批次控制，避免瞬时提交过多任务导致内存/句柄压力

## API与UI改动
- 对外函数签名保持不变；新增内部缓存函数：
  - `build_module_measurement_index_cached(folder_str: str, mtime: float) -> Dict[...]`
- UI不新增控件，仅复用“强制刷新缓存”；在侧栏“当前状态”区新增缓存命中率指标（命中/未命中）

## 单元测试
- 位置：`tests/test_data_cleaning.py`
- 内容：覆盖边界条件（空列、非数值、混合类型、近零电流、排序与重命名一致性）
- 运行：为 Windows 环境提供 `python -m unittest -v` 说明

## 性能评估
- 基准方法：对 20/50/100 壳体的典型目录，记录索引构建与解析耗时（`time.perf_counter`）
- 指标：总耗时、每任务平均耗时、缓存命中率、错误率
- 输出：在页面 `info_messages` 中追加简要统计，附详细日志到控制台/文件（可选）

## 推进步骤
1) 引入数据清洗库与替换调用点
2) 增加模块目录索引缓存封装并接入命中率统计
3) 并发提取任务封装与主线程汇总，完善异常与进度
4) 编写与运行单元测试，完成性能基准记录

请确认后我将开始实施以上改造，并提交代码改动与测试结果。