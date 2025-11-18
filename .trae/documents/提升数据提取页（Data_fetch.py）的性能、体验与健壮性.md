## 性能优化

* 并行读取测试文件：在批量遍历壳体/芯片时，使用 ThreadPool 并发读取和解析 Excel（find_measurement_file 与 extract_* 流程），显著缩短总耗时。入口：handle_measurement 与外层 for 循环，建议封装并发执行器（e.g. concurrent.futures）。参考：app/pages/Data_fetch.py:2391-2453。

* 缓存目录索引：为 _build_measurement_file_index 与 build_chip_measurement_index 增加 @st.cache_data，缓存以“目录路径+最新修改时间”为 key，减少重复扫描磁盘。参考：app/pages/Data_fetch.py:1451-1491、1257-1268。

* 统一数值清洗：避免多处重复 .apply(pd.to_numeric)，添加小工具函数后在 _prepare_metric_series、build_multi_shell_chart、build_station_metric_chart 等统一调用。参考：app/pages/Data_fetch.py:501-516、613-637、1015-1020。

* CSV 回退更鲁棒：read_excel_with_engine 在 CSV 分支传 on_bad_lines='skip'，并为典型列指定 dtype，减少解析错误与二次类型转换。参考：app/pages/Data_fetch.py:1588-1604。

* 进度条去重与更细粒度更新：移除重复更新（app/pages/Data_fetch.py:2498-2499），并在每个测量完成后刷新，带来更平滑的进度显示。

## 交互与布局

* 侧边栏按钮自适应：将 width='stretch' 改为 use_container_width=True，提高兼容性。参考：app/pages/Data_fetch.py:2014、2022、2030、2038。

* 下载按钮与表单控件自适应：下载区的下载按钮与输入控件统一 use_container_width=True，减少留白。参考：app/pages/Data_fetch.py:1974-1981、1920-1924。

* 表格与图表高度自适应：已在 Progress 页实施的自适应高度策略复用到“抽取结果明细”和图表区域，进一步减少大面积空白。参考：app/pages/Data_fetch.py:1917-1924、699-705、968-971。

* 锚点滚动的 UX 强化：当分析区打开时自动滚动到对应锚点，并在结果区顶部提供“返回顶部/返回输入”快速链接。参考：app/pages/Data_fetch.py:1858-1876、2635-2641、2932-2937。

## 健壮性与错误处理

* 统一错误消息：将错误分级（路径错误、文件缺失、解析失败），在 render_extraction_results_section 中按类别展示。参考：app/pages/Data_fetch.py:1987-2001、2398-2443。

* 文件解析健壮：对 _extract_lvi_data_impl/_extract_rth_data_impl 的 skiprows/usecols 参数做健全性检查并在失败时回退到 _extract_generic_excel_impl。参考：app/pages/Data_fetch.py:1652-1703、1726-1811。

* 可选依赖检测提示统一：将拟合依赖提示统一在页顶一次性展示，并在功能位点简化为“依赖未安装”短提示。参考：app/pages/Data_fetch.py:2849-2860、2785-2789。

## 代码结构与复用

* 提取逻辑下沉：把 interpret_*、find_*、extract_* 组装为 service 层，UI 只负责收集参数与展示，便于单元测试与复用。参考：app/pages/Data_fetch.py:1209-1831、2189-2616。

* 统一颜色与站别映射：将 PLOT_ORDER、STATION_COLORS 提取到 config 或共享模块，避免重复定义，便于跨页保持一致。参考：app/pages/Data_fetch.py:63-74。

* 数据模型：使用轻量 dataclass 定义提取结果与预测结果结构，减少 dict 拼装与键名错误风险。参考：app/pages/Data_fetch.py:548-598、2603-2616。

## 预期收益

* 批量提取性能提升（并发+缓存）：对于 20+ 壳体的批量场景，整体耗时预计下降 30–60%。

* 解析失败率降低：CSV 回退与健壮性增强减少用户重试次数。

* UI 紧凑度提升：按钮/表格/图表自适应宽高，减少空白，提高可读性。

请确认是否按以上方案实施；确认后我将分阶段提交具体代码修改（先性能与缓存，再交互布局，最后结构整理）。