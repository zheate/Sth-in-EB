# Implementation Plan

- [x] 1. 创建核心存储模块





  - [x] 1.1 创建异常类和数据类型定义

    - 在 `app/utils/exceptions.py` 创建自定义异常类
    - 在 `app/utils/local_storage.py` 创建 `DataCategory` 枚举和 `DatasetMetadata` 数据类
    - _Requirements: 2.5, 3.2_


  - [x] 1.2 实现 LocalDataStore 基础结构





    - 创建 `LocalDataStore` 类，初始化存储目录
    - 实现目录自动创建逻辑
    - _Requirements: 4.1, 4.2_
  - [ ]* 1.3 编写属性测试：Save-Load Round Trip
    - **Property 1: Save-Load Round Trip**
    - **Validates: Requirements 1.1, 1.2, 1.3, 2.1**
-

- [x] 2. 实现保存功能


  - [x] 2.1 实现 save() 方法


    - 支持 DataFrame 保存为 Parquet 格式
    - 支持自定义文件名和自动生成文件名
    - 生成并保存元数据 JSON
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_


  - [x] 2.2 实现文件名生成逻辑

    - 自动生成：时间戳 + 数据摘要
    - 自定义：安全字符过滤（sanitize）
    - _Requirements: 1.4_
  - [ ]* 2.3 编写属性测试：Filename Generation
    - **Property 2: Filename Generation and Custom Filename**
    - **Validates: Requirements 1.4**
  - [ ]* 2.4 编写属性测试：Save Returns Valid Identifier
    - **Property 3: Save Returns Valid Identifier**
    - **Validates: Requirements 1.5**

- [x] 3. 实现加载功能



  - [x] 3.1 实现 load() 方法


    - 读取 Parquet 文件还原 DataFrame
    - 读取元数据 JSON
    - 读取扩展数据（如存在）
    - _Requirements: 2.1_

  - [x] 3.2 实现错误处理

    - 文件不存在时抛出 DatasetNotFoundError
    - 文件损坏时抛出 CorruptedDataError
    - _Requirements: 2.5_
  - [ ]* 3.3 编写属性测试：Load Non-existent File
    - **Property 4: Load Non-existent File Returns Error**
    - **Validates: Requirements 2.5**

- [x] 4. Checkpoint - 确保所有测试通过





  - Ensure all tests pass, ask the user if questions arise.

- [-] 5. 实现列表和删除功能











  - [x] 5.1 实现 list_datasets() 方法




    - 扫描存储目录获取所有数据集；
    - 支持按类别筛选
    - 返回元数据列表
    - _Requirements: 3.1, 3.2, 3.4_

  - [x] 5.2 实现 delete() 方法







    - 删除数据集的所有关联文件
    - _Requirements: 3.3_
  - [ ]* 5.3 编写属性测试：List Metadata Fields
    - **Property 5: List Datasets Contains Required Metadata Fields**
    - **Validates: Requirements 3.2**
  - [ ]* 5.4 编写属性测试：Delete Removes Files
    - **Property 6: Delete Removes All Associated Files**
    - **Validates: Requirements 3.3**
  - [ ]* 5.5 编写属性测试：Category Filter
    - **Property 7: Category Filter Returns Only Matching Datasets**
    - **Validates: Requirements 3.4**

- [x] 6. 实现序列化器（复杂对象支持）









  - [x] 6.1 实现绘图数据源序列化

    - 将 lvi_sources 和 rth_sources 字典序列化为 JSON 可存储格式
    - DataFrame 转为 Base64 编码的 Parquet
    - _Requirements: 4.3_
  - [x] 6.2 实现绘图数据源反序列化

    - 从 JSON 还原 lvi_sources 和 rth_sources 字典
    - _Requirements: 4.3_
  - [ ]* 6.3 编写属性测试：Complex Object Round Trip
    - **Property 8: Complex Object Serialization Round Trip**
    - **Validates: Requirements 4.3**

- [x] 7. 实现导出功能




  - [x] 7.1 实现 export_to_excel() 方法


    - 支持单个和批量数据集导出
    - 支持多 Sheet（数据 + 统计摘要）
    - 自动生成带日期的文件名
    - _Requirements: 6.1, 6.2, 6.4, 6.5_


  - [ ] 7.2 实现 export_to_csv() 方法
    - 使用 UTF-8-BOM 编码
    - 自动生成文件名
    - _Requirements: 6.1, 6.3, 6.4_
  - [ ]* 7.3 编写属性测试：Excel Multiple Sheets
    - **Property 9: Export Excel Contains Multiple Sheets**
    - **Validates: Requirements 6.2**
  - [ ]* 7.4 编写属性测试：Export Filename Format
    - **Property 10: Export Filename Contains Date and Type**
    - **Validates: Requirements 6.4**
  - [ ]* 7.5 编写属性测试：Batch Export
    - **Property 11: Batch Export Merges All Selected Datasets**
    - **Validates: Requirements 6.5**

- [x] 8. Checkpoint - 确保所有测试通过




  - Ensure all tests pass, ask the user if questions arise.

- [x] 9. 创建 UI 组件


  - [x] 9.1 创建保存按钮组件


    - 实现 `render_save_button()` 函数
    - 支持自定义文件名输入
    - 支持备注输入
    - _Requirements: 7.1, 7.2_

  - [x] 9.2 创建加载选择器组件

    - 实现 `render_load_selector()` 函数
    - 显示数据集列表和摘要信息
    - _Requirements: 7.3_
  - [x] 9.3 创建数据管理页面



    - 新建 `app/pages/Data_Manager.py`
    - 实现数据集列表、删除、导出功能
    - _Requirements: 3.1, 3.3, 6.5_

- [x] 10. 集成到现有页面






  - [x] 10.1 集成到 Data_fetch.py

    - 在侧边栏添加保存按钮
    - 添加加载历史数据选项
    - 保存时包含 lvi/rth 绘图数据源
    - _Requirements: 1.1, 2.2, 7.1_

  - [x] 10.2 集成到 Progress.py

    - 在侧边栏添加保存按钮
    - 添加加载历史数据选项
    - 保存时包含筛选条件
    - _Requirements: 1.2, 2.3, 7.1_


  - [ ] 10.3 集成到 Engineering_Analysis.py
    - 在侧边栏添加保存按钮
    - 添加加载历史数据选项
    - _Requirements: 1.3, 2.4, 7.1_

- [x] 11. 实现跨模块数据共享




  - [x] 11.1 实现数据格式转换


    - 检测源数据和目标模块的列兼容性
    - 进行必要的列映射和转换
    - _Requirements: 5.2, 5.3_
  - [x] 11.2 在数据管理页面添加"发送到"功能



    - 允许将数据集发送到指定模块
    - _Requirements: 5.1_

- [x] 12. Final Checkpoint - 确保所有测试通过




  - Ensure all tests pass, ask the user if questions arise.
