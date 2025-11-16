import io
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

# 将父目录加入 sys.path 以加载项目模块

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DEFAULT_DATA_FOLDER
from utils.data_storage import DataStorage
from utils.feedback_handler import FeedbackHandler
from utils.error_handler import ErrorHandler
from utils.ui_components import render_dataset_summary

# 初始化日志
ErrorHandler.initialize_logger()

COLUMN_CONFIG: List[Tuple[str, str]] = [
    ("shell_id", "壳体号"),
    ("current", "电流 (A)"),
    ("power", "功率 (W)"),
    ("efficiency", "效率 (%)"),
    ("wavelength", "波长 (nm)"),
    ("shift", "波长 shift"),
    ("na", "NA"),
    ("spectral_fwhm", "光谱全高宽"),
    ("thermal_resistance", "热阻 (K/W)"),
]
DEFAULT_COLUMNS = [key for key, _ in COLUMN_CONFIG]
NUMERIC_COLUMNS = [col for col in DEFAULT_COLUMNS if col != "shell_id"]
ROUNDING_RULES = {
    "current": 3,
    "power": 3,
    "efficiency": 3,
    "wavelength": 3,
    "shift": 3,
    "na": 4,
    "spectral_fwhm": 3,
    "thermal_resistance": 3,
}


def _ensure_session_state() -> None:
    if "loaded_dataset" not in st.session_state:
        st.session_state.loaded_dataset = None
    if "selected_shells" not in st.session_state:
        st.session_state.selected_shells = []
    if "current_filter" not in st.session_state:
        st.session_state.current_filter = None
    if "column_selection" not in st.session_state:
        st.session_state.column_selection = DEFAULT_COLUMNS.copy()


def _records_to_dataframe(dataset: Dict) -> pd.DataFrame:
    records = dataset.get("records", []) or []
    df = pd.DataFrame(records)

    if df.empty:
        for column, _ in COLUMN_CONFIG:
            df[column] = pd.Series(dtype="float64")
        df["shell_id"] = pd.Series(dtype="string")
        return df[DEFAULT_COLUMNS]

    # 确保包含所有列并保持既定顺序
    for column, _ in COLUMN_CONFIG:
        if column not in df.columns:
            df[column] = pd.NA

    df = df[DEFAULT_COLUMNS].copy()
    df["shell_id"] = df["shell_id"].astype(str).str.strip()

    for column in NUMERIC_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    return df


def _filter_records(
    df: pd.DataFrame,
    selected_shells: List[str],
    current_range: Optional[Tuple[float, float]],
    require_complete: bool,
) -> pd.DataFrame:
    filtered = df.copy()

    if selected_shells:
        filtered = filtered[filtered["shell_id"].isin(selected_shells)]

    if current_range is not None:
        lower, upper = current_range
        filtered = filtered[filtered["current"].between(lower, upper, inclusive="both")]

    if require_complete:
        filtered = filtered.dropna(
            subset=["power", "efficiency", "wavelength", "na", "thermal_resistance"]
        )

    return filtered


def _format_for_display(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    export_df = df[columns].copy()

    for column, decimals in ROUNDING_RULES.items():
        if column in export_df.columns:
            export_df[column] = export_df[column].round(decimals)

    column_labels = {key: label for key, label in COLUMN_CONFIG}
    display_df = export_df.rename(columns=column_labels)

    return export_df, display_df


def _render_filter_controls(df: pd.DataFrame) -> Tuple[List[str], Optional[Tuple[float, float]], bool]:
    st.subheader("🔍 数据筛选")

    shell_options = sorted(shell for shell in df["shell_id"].dropna().unique())
    default_shells = (
        st.session_state.selected_shells if st.session_state.selected_shells else shell_options
    )

    selected_shells = st.multiselect(
        "选择壳体号",
        options=shell_options,
        default=default_shells,
        help="选择需要分析的壳体号（若不选择则默认展示全部）",
    )

    numeric_current = df["current"].dropna()
    current_range: Optional[Tuple[float, float]] = None

    if not numeric_current.empty:
        min_current = float(numeric_current.min())
        max_current = float(numeric_current.max())
        stored_range = st.session_state.current_filter

        if (
            isinstance(stored_range, (tuple, list))
            and len(stored_range) == 2
            and stored_range[0] is not None
            and stored_range[1] is not None
        ):
            lower_bound = max(min_current, float(stored_range[0]))
            upper_bound = min(max_current, float(stored_range[1]))
        else:
            lower_bound, upper_bound = min_current, max_current

        if min_current == max_current:
            current_range = (min_current, max_current)
            st.info(f"当前数据集的电流值固定为 {min_current} A")
        else:
            current_range = st.slider(
                "电流范围 (A)",
                min_value=min_current,
                max_value=max_current,
                value=(lower_bound, upper_bound),
                step=max((max_current - min_current) / 200, 0.01),
            )

    require_complete = st.checkbox(
        "仅显示包含完整关键字段的记录",
        value=False,
        help="过滤掉缺少功率、效率、波长、NA 或热阻的记录",
    )

    st.session_state.selected_shells = selected_shells
    st.session_state.current_filter = current_range

    return selected_shells, current_range, require_complete


def _render_column_selector() -> List[str]:
    column_labels = {key: label for key, label in COLUMN_CONFIG}
    selected_columns = st.multiselect(
        "选择显示的列",
        options=DEFAULT_COLUMNS,
        default=st.session_state.column_selection,
        format_func=lambda key: column_labels.get(key, key),
        help="请选择需要展示和导出的列",
    )

    if not selected_columns:
        st.warning("至少需要选择一列数据进行展示")

    st.session_state.column_selection = selected_columns or DEFAULT_COLUMNS.copy()
    return st.session_state.column_selection


def _render_summary_metrics(df: pd.DataFrame) -> None:
    st.subheader("📈 汇总指标")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("记录数量", len(df))
    with col2:
        shell_count = df["shell_id"].nunique(dropna=True)
        st.metric("壳体数量", shell_count)
    with col3:
        if df["power"].notna().any():
            st.metric("平均功率 (W)", f"{df['power'].mean():.3f}")
        else:
            st.metric("平均功率 (W)", "N/A")
    with col4:
        if df["efficiency"].notna().any():
            st.metric("平均效率 (%)", f"{df['efficiency'].mean():.3f}")
        else:
            st.metric("平均效率 (%)", "N/A")


def _render_export_buttons(export_df: pd.DataFrame, display_df: pd.DataFrame) -> None:
    st.subheader("💾 数据导出")

    column_labels = {key: label for key, label in COLUMN_CONFIG}
    labeled_df = export_df.rename(columns=column_labels)

    csv_buffer = io.StringIO()
    labeled_df.to_csv(csv_buffer, index=False, encoding="utf-8-sig")
    csv_bytes = csv_buffer.getvalue()

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="⬇️ 下载 CSV",
            data=csv_bytes,
            file_name=f"data_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with col2:
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
            labeled_df.to_excel(writer, index=False, sheet_name="数据分析")
        st.download_button(
            label="⬇️ 下载 Excel",
            data=excel_buffer.getvalue(),
            file_name=f"data_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )


def _render_filtered_table(display_df: pd.DataFrame) -> None:
    st.subheader("📋 数据表格")
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        height=min(500, 60 + 35 * len(display_df)),
    )


def _render_dataset_analysis(dataset: Dict) -> None:
    render_dataset_summary(dataset)
    st.divider()

    df = _records_to_dataframe(dataset)

    if df.empty:
        st.info("数据集中没有可用的记录")
        return

    selected_shells, current_range, require_complete = _render_filter_controls(df)
    st.divider()

    filtered_df = _filter_records(df, selected_shells, current_range, require_complete)

    if filtered_df.empty:
        st.warning("筛选条件下没有匹配的记录，请调整筛选条件。")
        return

    _render_summary_metrics(filtered_df)
    st.divider()

    selected_columns = _render_column_selector()
    if not selected_columns:
        return

    export_df, display_df = _format_for_display(filtered_df, selected_columns)

    _render_filtered_table(display_df)
    st.caption(f"共 {len(display_df)} 条记录 | 指定电流: {dataset.get('metadata', {}).get('target_current', 'N/A')} A")

    _render_export_buttons(export_df, display_df)


def _render_file_loader() -> None:
    st.subheader("📁 加载数据集")

    load_method = st.radio(
        "选择加载方式",
        options=["从文件夹选择", "输入文件路径", "上传文件"],
        horizontal=True,
    )

    if load_method == "从文件夹选择":
        col1, col2 = st.columns([3, 1])

        with col1:
            folder_path = st.text_input(
                "数据集文件夹路径",
                value=DEFAULT_DATA_FOLDER,
                placeholder="输入数据集所在的文件夹路径",
            )

        with col2:
            st.write("")
            st.write("")
            st.button("📂 浏览", use_container_width=True, disabled=True, help="请直接在左侧输入路径")

        if not folder_path:
            return

        folder_path_obj = Path(folder_path)
        if not folder_path_obj.exists() or not folder_path_obj.is_dir():
            st.warning(f"⚠️ 文件夹路径无效或不存在: {folder_path}")
            return

        json_files = sorted(
            [f for f in folder_path_obj.glob("*.json")],
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )

        if not json_files:
            st.info("📂 文件夹中没有找到 JSON 数据集文件")
            return

        file_options = [f.name for f in json_files]
        selected_file = st.selectbox(
            f"选择数据集文件 (共 {len(json_files)} 个)",
            options=file_options,
        )

        if not selected_file:
            return

        selected_path = folder_path_obj / selected_file
        stat_info = selected_path.stat()
        st.caption(
            f"📄 文件大小: {stat_info.st_size / 1024:.2f} KB | "
            f"📅 修改时间: {datetime.fromtimestamp(stat_info.st_mtime).strftime('%Y-%m-%d %H:%M:%S')}"
        )

        if st.button("🔄 加载选中的数据集", type="primary", use_container_width=True):
            with FeedbackHandler.show_spinner("正在加载数据集..."):
                dataset, message = DataStorage.load_dataset(str(selected_path))
                if dataset:
                    st.session_state.loaded_dataset = dataset
                    FeedbackHandler.show_success(message)
                    st.experimental_rerun()
                else:
                    FeedbackHandler.show_error(message)

    elif load_method == "输入文件路径":
        col1, col2 = st.columns([3, 1])

        with col1:
            file_path = st.text_input(
                "数据集文件路径",
                value="",
                placeholder="输入完整的 JSON 文件路径",
            )

        with col2:
            st.write("")
            st.write("")
            load_button = st.button("🔄 加载数据集", type="primary", use_container_width=True)

        if load_button and file_path:
            with FeedbackHandler.show_spinner("正在加载数据集..."):
                dataset, message = DataStorage.load_dataset(file_path)
                if dataset:
                    st.session_state.loaded_dataset = dataset
                    FeedbackHandler.show_success(message)
                    st.experimental_rerun()
                else:
                    FeedbackHandler.show_error(message)

    else:
        uploaded_file = st.file_uploader(
            "选择 JSON 数据集文件",
            type=["json"],
            help="上传本地保存的数据集文件",
        )

        if uploaded_file is None:
            return

        file_size_kb = len(uploaded_file.getvalue()) / 1024
        st.caption(f"📄 文件名: {uploaded_file.name} | 📊 文件大小: {file_size_kb:.2f} KB")

        if st.button("🔄 加载上传的文件", type="primary", use_container_width=True):
            with FeedbackHandler.show_spinner("正在加载上传的文件..."):
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = Path(tmp_file.name)

                    dataset, message = DataStorage.load_dataset(str(tmp_path))
                    tmp_path.unlink(missing_ok=True)

                    if dataset:
                        st.session_state.loaded_dataset = dataset
                        FeedbackHandler.show_success(message)
                        st.experimental_rerun()
                    else:
                        FeedbackHandler.show_error(message)
                except Exception as exc:
                    ErrorHandler.log_error(exc, "上传数据集文件失败")
                    FeedbackHandler.show_error(f"文件上传失败: {exc}")


def main() -> None:
    st.title("数据分析")
    _ensure_session_state()
    _render_file_loader()
    st.divider()

    dataset = st.session_state.loaded_dataset
    if dataset is None:
        st.info("👆 请先加载一个数据集文件")
        return

    _render_dataset_analysis(dataset)


if __name__ == "__main__":
    main()
