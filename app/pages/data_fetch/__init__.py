# Data Fetch 模块
# 将原 Data_fetch.py 拆分为多个子模块以提高可维护性

from .constants import (
    PLOT_ORDER,
    SANITIZED_PLOT_ORDER,
    SANITIZED_ORDER_LOOKUP,
    STATION_COLORS,
    DEFAULT_PALETTE,
    OUTPUT_COLUMNS,
    SHELL_COLUMN,
    TEST_TYPE_COLUMN,
    CURRENT_COLUMN,
    POWER_COLUMN,
    VOLTAGE_COLUMN,
    EFFICIENCY_COLUMN,
    LAMBDA_COLUMN,
    SHIFT_COLUMN,
    WAVELENGTH_2A_COLUMN,
    WAVELENGTH_COLD_COLUMN,
    CURRENT_TOLERANCE,
    MODULE_MODE,
    CHIP_MODE,
    CHIP_TEST_CATEGORY,
    MEASUREMENT_OPTIONS,
    TEST_CATEGORY_OPTIONS,
)

from .file_utils import (
    read_excel_with_engine,
    find_measurement_file,
    find_chip_measurement_file,
    interpret_folder_input,
    interpret_chip_folder_input,
    resolve_test_folder,
    build_chip_measurement_index,
    build_module_measurement_index_cached,
    build_chip_measurement_index_cached,
)

from .data_extraction import (
    extract_lvi_data,
    extract_rth_data,
    extract_generic_excel,
    clear_extraction_caches,
    align_output_columns,
    merge_measurement_rows,
)

from .models import (
    EFFICIENCY_MODELS,
    HAS_PREDICTION_LIBS,
    ensure_prediction_libs_loaded,
)

from .charts import (
    build_multi_shell_chart,
    build_station_metric_chart,
    build_single_shell_dual_metric_chart,
    build_multi_shell_diff_band_charts,
    plot_multi_shell_prediction,
    compute_power_predictions,
)

__all__ = [
    # Constants
    "PLOT_ORDER",
    "SANITIZED_PLOT_ORDER", 
    "SANITIZED_ORDER_LOOKUP",
    "STATION_COLORS",
    "DEFAULT_PALETTE",
    "OUTPUT_COLUMNS",
    "SHELL_COLUMN",
    "TEST_TYPE_COLUMN",
    "CURRENT_COLUMN",
    "POWER_COLUMN",
    "VOLTAGE_COLUMN",
    "EFFICIENCY_COLUMN",
    "LAMBDA_COLUMN",
    "SHIFT_COLUMN",
    "WAVELENGTH_2A_COLUMN",
    "WAVELENGTH_COLD_COLUMN",
    "CURRENT_TOLERANCE",
    "MODULE_MODE",
    "CHIP_MODE",
    "CHIP_TEST_CATEGORY",
    "MEASUREMENT_OPTIONS",
    "TEST_CATEGORY_OPTIONS",
    # File utils
    "read_excel_with_engine",
    "find_measurement_file",
    "find_chip_measurement_file",
    "interpret_folder_input",
    "interpret_chip_folder_input",
    "resolve_test_folder",
    "build_chip_measurement_index",
    "build_module_measurement_index_cached",
    "build_chip_measurement_index_cached",
    # Data extraction
    "extract_lvi_data",
    "extract_rth_data",
    "extract_generic_excel",
    "clear_extraction_caches",
    "align_output_columns",
    "merge_measurement_rows",
    # Models
    "EFFICIENCY_MODELS",
    "HAS_PREDICTION_LIBS",
    "ensure_prediction_libs_loaded",
    # Charts
    "build_multi_shell_chart",
    "build_station_metric_chart",
    "build_single_shell_dual_metric_chart",
    "build_multi_shell_diff_band_charts",
    "plot_multi_shell_prediction",
    "compute_power_predictions",
]
