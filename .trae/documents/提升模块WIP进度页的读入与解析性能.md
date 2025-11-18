## 性能瓶颈定位

* 文**件读取：Excel 读取本身较慢；CSV 未指定高效参数会**多次类型推断。当前在 Progress.py:964-972 读取后再解析，读入耗时与解析耗时都可优化。

* 行级解析：extract\_progress\_data 在 295 行使用 iterrows，并在每行内遍历所有站别时间列，产生大量 Python 循环开销。

* 时间解析：最新测试时间在 1216-1264 行逐行逐值做 to\_datetime，重复且分支多。

## 读取阶段优化

1. *CSV 加速（Progress.py:964-972）*

* 继续两段式读取：先读表头，再计算 usecols。

* 预先收集时间列：`time_cols = [f"{excel_col}时间" for excel_col in STATION_MAPPING if f"{excel_col}时间" in header_df.columns]`

* 使用更高效参数：

  * `pd.read_csv(path, usecols=usecols, dtype={"壳体号": "string", "料号": "string"}, parse_dates=time_cols, infer_datetime_format=True, low_memory=False)`

  * 如果 CSV 很大，可用 `chunksize=200_000` 流式读，将解析改为增量（见解析阶段优化-3）。

1. Excel 加速（Progress.py:969-971）

* 指定引擎：`engine="openpyxl"`（xlsx）或 `engine="xlrd"`（xls；如版本支持）。

* 同样传 `usecols` 与 `parse_dates=time_cols`，减少后续逐值时间解析。

* 可选高级方案：超大表用 openpyxl 的 `load_workbook(read_only=True, data_only=True)` 流式提取，再组装 DataFrame（代码量略大，后续按需加）。

## 解析阶段优化

1. 行迭代替换（Progress.py:295）

* 用 `itertuples(index=False)` 替代 `iterrows()`，并用列索引映射访问，减少 Pandas/Series 对象创建。

* 示例：

  * 在进入循环前：`cols = list(df.columns); idx = {c: i for i, c in enumerate(cols)}`

  * 循环：`for row in df.itertuples(index=False, name=None): shell_id = row[idx['壳体号']] ...`

1. 完成站别组装降开销（Progress.py:376-386）

* 先把 `existing_station_time_cols` 拆为并行列表：`std_names = [...]; time_cols = [...]`。

* 逐行时只遍历 `time_cols`，用位置索引读取 `row[idx[col]]` 判定是否非空；命中则将 `std_names[j]` 追加并记录时间。

* 可进一步：把“非空”条件向量化为 `notna & str.strip().ne('')` 的布尔矩阵，用其行索引生成列表，减少 Python 判断分支次数。

1. 支持流式解析（结合 CSV chunksize）

* 当 `chunksize` 被设置时：对每个 chunk 调用同样的行迭代逻辑生成 `progress_data` 子列表，并在末尾 `pd.DataFrame(progress_data_all)` 合并。

* 好处：降低峰值内存，通常总体耗时也下降；大数据文件更稳定。

1. 最新时间聚合（Progress.py:1216-1264）

* 依赖读取阶段已 `parse_dates`：直接在原始 df 上做全局聚合。

* 例：`latest_time = pd.to_datetime(df[time_cols]).max().max()`；避免逐壳体逐站别地转时间。

## 交互与缓存优化

* 结果缓存：保持现有 `progress_data_cache`；在解析完后将 `time_cols` 存入 `progress_df.attrs['time_cols']` 供统计时复用。

* “仅统计模式”开关：新增单选或复选，启用时只生成计数与最新时间，不构建每壳体的 `完成站别/站别时间`，加快首屏（后续用户打开表格时再补算或切换完整模式）。

* 预览行数：原始数据 expander 限制显示行数以避免大表渲染卡顿（读取速度不受影响，但交互更顺畅）。

## 改动位置一览

* 读取逻辑：`app/pages/Progress.py:964-972` 引入 `time_cols`、`parse_dates`、`dtype`，可选 `chunksize`。

* Excel 读取：`app/pages/Progress.py:969-971` 增加 `engine` 与 `parse_dates`。

* 解析迭代：`app/pages/Progress.py:295` 用 `itertuples`；`376-386` 改为索引驱动的判定与组装。

* 最新时间计算：替换 `app/pages/Progress.py:1216-1264` 的逐项解析为列聚合。

* 新增开关：在“选择文件”下方加入“仅统计模式”控件，并在解析分支按开关切换轻/全模式。

## 预期收益

* CSV 读入：20-50% 提速（取决于列数与内容）。

* Excel 读入：10-30% 提速（openpyxl 指定与少列）。

* 解析阶段：2-4 倍提速（itertuples + 索引访问 + 向量化判定）。

* 首屏渲染：仅统计模式显著缩短到秒级。

请确认是否按以上方案实施；确认后我将提交具体代码改动。
