# Implementation Plan

- [x] 1. Set up project structure and data models





  - [x] 1.1 Create data models module


    - Create `app/pages/data_manager/models.py` with dataclasses for ProductType, ProductionOrder, ShellProgress, Attachment
    - Define type hints and validation methods
    - _Requirements: 7.1, 7.2_
  - [ ]* 1.2 Write property test for data model serialization
    - **Property 1: Product Type Data Round-Trip**
    - **Validates: Requirements 1.2, 7.1, 7.2**

  - [x] 1.3 Create constants module


    - Create `app/pages/data_manager/constants.py` with database paths, station lists, metric columns
    - _Requirements: 4.3_

- [x] 2. Implement ProductTypeService



  - [x] 2.1 Create ProductTypeService class


    - Create `app/pages/data_manager/product_type_service.py`
    - Implement `save_product_type()` method with JSON metadata and Parquet data storage
    - Implement `get_product_type()` and `list_product_types()` methods
    - _Requirements: 1.2, 7.1, 7.2_
  - [ ]* 2.2 Write property test for product type CRUD
    - **Property 1: Product Type Data Round-Trip**
    - **Validates: Requirements 1.2, 7.1, 7.2**

  - [x] 2.3 Implement rename and delete operations
    - Implement `rename_product_type()` preserving associated data
    - Implement `delete_product_type()` with cascading delete of shells and attachments
    - _Requirements: 1.5, 7.5_
  - [ ]* 2.4 Write property tests for rename and delete
    - **Property 2: Product Type Rename Preserves Data**
    - **Property 12: Cascading Delete**
    - **Validates: Requirements 1.5, 7.5**
  - [x] 2.5 Implement production order retrieval

    - Implement `get_production_orders()` returning orders with time information
    - _Requirements: 1.3, 3.3_
  - [ ]* 2.6 Write property test for production order retrieval
    - **Property 3: Production Orders Retrieval**
    - **Validates: Requirements 1.3, 3.3**

  - [x] 2.7 Implement attachment management

    - Implement `upload_attachment()` saving to `data/zh_database/attachments/`
    - Implement `list_attachments()` and `delete_attachment()`
    - _Requirements: 2.2, 2.5_
  - [ ]* 2.8 Write property test for attachment lifecycle
    - **Property 6: Attachment Lifecycle**
    - **Validates: Requirements 2.2, 2.5**

- [x] 3. Checkpoint - Ensure all tests pass





  - Ensure all tests pass, ask the user if questions arise.

- [x] 4. Implement ShellProgressService





  - [x] 4.1 Create ShellProgressService class


    - Create `app/pages/data_manager/shell_progress_service.py`
    - Implement `get_shells_by_orders()` with multi-select support
    - Implement default time selection (latest)
    - _Requirements: 3.4, 3.5, 4.1_
  - [ ]* 4.2 Write property tests for shell retrieval
    - **Property 4: Default Time Selection**
    - **Property 5: Multi-Order Shell Aggregation**
    - **Property 7: Shell Progress Retrieval**
    - **Validates: Requirements 3.4, 3.5, 4.1**
  - [x] 4.3 Implement Gantt chart data generation


    - Implement `generate_gantt_data()` with station ordering
    - Use station order from Progress.py BASE_STATIONS
    - _Requirements: 4.3_
  - [ ]* 4.4 Write property test for Gantt chart ordering
    - **Property 8: Gantt Chart Station Ordering**
    - **Validates: Requirements 4.3**

- [x] 5. Implement DataAnalysisService





  - [x] 5.1 Create DataAnalysisService class


    - Create `app/pages/data_manager/data_analysis_service.py`
    - Implement `fetch_test_data()` integrating with Data_fetch logic


    - _Requirements: 5.1_
  - [x] 5.2 Implement threshold filtering




    - Implement `apply_thresholds()` partitioning data into pass/fail groups
    - Calculate pass rate and failure reasons
    - _Requirements: 6.2, 6.3, 6.4_
  - [x]* 5.3 Write property tests for threshold filtering

    - **Property 9: Threshold Filtering Partition**
    - **Property 10: Pass Rate Calculation**
    - **Validates: Requirements 6.2, 6.3, 6.4**
  - [x] 5.4 Implement threshold config persistence

    - Implement `save_threshold_config()` and `load_threshold_config()`
    - _Requirements: 6.5_
  - [ ]* 5.5 Write property test for threshold config persistence
    - **Property 11: Threshold Config Persistence**
    - **Validates: Requirements 6.5**

- [x] 6. Checkpoint - Ensure all tests pass


  - Ensure all tests pass, ask the user if questions arise.

- [x] 7. Implement Progress.py integration





  - [x] 7.1 Add "Save As Product Type" button to Progress.py


    - Add button in sidebar or main area
    - Create dialog for entering product type name
    - _Requirements: 1.1_


  - [ ] 7.2 Implement save logic in Progress.py
    - Call ProductTypeService.save_product_type() with current data
    - Show success/error feedback
    - _Requirements: 1.2_

- [x] 8. Implement Data_Manager.py UI - Layer 1 (Product Type Management)





  - [x] 8.1 Create product type selector


    - Implement dropdown for selecting product type
    - Show product type list with shell count and order count
    - _Requirements: 1.3_
  - [x] 8.2 Implement product type rename UI

    - Add rename button and input dialog
    - _Requirements: 1.5_

  - [x] 8.3 Implement attachment upload UI
    - Add upload button next to product type selector
    - Support PDF and Excel file types
    - Save to `data/zh_database/attachments/`
    - _Requirements: 2.1, 2.2_
  - [x] 8.4 Implement attachment preview UI
    - Default collapsed state
    - Expandable preview for PDF and Excel
    - _Requirements: 2.3, 2.4_
  - [x] 8.5 Implement production order selector
    - Show orders for selected product type
    - Support single and multi-select modes
    - Show time information for each order
    - Default to latest time
    - _Requirements: 3.1, 3.2, 3.3, 3.4_

- [x] 9. Implement Data_Manager.py UI - Layer 2 (Shell Progress Analysis)
  - [x] 9.1 Implement shell list display
    - Show shell IDs and current stations for selected orders
    - _Requirements: 4.1_
  - [x] 9.2 Implement Gantt chart visualization
    - Use Altair or Plotly for Gantt chart
    - Show station progress for each shell
    - Add hover tooltips with station time details
    - _Requirements: 4.2, 4.4_
  - [x] 9.3 Implement pagination for large datasets

    - Add pagination or virtual scrolling for many shells
    - _Requirements: 4.5_

- [x] 10. Implement Data_Manager.py UI - Layer 3 (Data Analysis)




  - [x] 10.1 Implement test data fetch UI


    - Add button to fetch test data for selected shells
    - Show loading indicator during fetch
    - _Requirements: 5.1_
  - [x] 10.2 Implement analysis results table

    - Display multi-station analysis results
    - Support column filtering
    - _Requirements: 5.2, 5.3_

  - [x] 10.3 Implement threshold setting UI
    - Show available numeric metric columns
    - Allow setting min/max for each metric
    - _Requirements: 6.1, 6.2_
  - [x] 10.4 Implement threshold filtering display

    - Highlight out-of-threshold values
    - Show pass/fail statistics
    - Display failure reason analysis
    - _Requirements: 5.4, 6.3, 6.4_
  - [x] 10.5 Implement threshold config save/load

    - Add save button for current threshold config
    - Auto-load saved config when selecting product type
    - _Requirements: 6.5_

- [x] 11. Final Checkpoint - Ensure all tests pass





  - Ensure all tests pass, ask the user if questions arise.