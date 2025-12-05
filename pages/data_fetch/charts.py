# 图表构建模块
"""
包含各种 Altair 图表的构建函数
"""

from typing import Any, Callable, Dict, List, Optional, Tuple
import warnings

import numpy as np
import pandas as pd
import altair as alt

from .constants import (
    CURRENT_COLUMN,
    POWER_COLUMN,
    VOLTAGE_COLUMN,
    EFFICIENCY_COLUMN,
    LAMBDA_COLUMN,
    SHIFT_COLUMN,
    TEST_TYPE_COLUMN,
    SHELL_COLUMN,
    STATION_COLORS,
    DEFAULT_PALETTE,
    SANITIZED_PLOT_ORDER,
    CURRENT_TOLERANCE,
    PRIMARY_RED,
    PRIMARY_BLUE,
)
from .models import (
    ensure_prediction_libs_loaded,
    get_stats_module,
    fit_efficiency_model,
    fit_polynomial_model,
    EFFICIENCY_MODELS,
    HAS_PREDICTION_LIBS,
)

# 导入数据清洗工具
import sys
from pathlib import Path
parent_dir = str(Path(__file__).parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from utils.data_cleaning import drop_zero_current, clean_current_metric


# ============================================================================
# 辅助函数
# ============================================================================

def _exclude_zero_current(df: pd.DataFrame) -> pd.DataFrame:
    """排除零电流数据"""
    if CURRENT_COLUMN not in df.columns or df.empty:
        return df
    return drop_zero_current(df, CURRENT_COLUMN, tol=CURRENT_TOLERANCE)


def _prepare_metric_series(df: pd.DataFrame, metric_column: str) -> pd.DataFrame:
    """准备用于绘图的指标数据"""
    from .data_extraction import align_output_columns
    from utils.data_cleaning import ensure_numeric
    
    if CURRENT_COLUMN not in df.columns or metric_column not in df.columns:
        return pd.DataFrame(columns=[CURRENT_COLUMN, metric_column])
    
    numeric = ensure_numeric(
        df[[CURRENT_COLUMN, metric_column]],
        [CURRENT_COLUMN, metric_column],
        strict=False
    )
    numeric = numeric.dropna(subset=[CURRENT_COLUMN, metric_column])
    numeric = _exclude_zero_current(numeric)
    
    if numeric.empty:
        return numeric
    
    aggregated = numeric.groupby(CURRENT_COLUMN, as_index=False).mean()
    return aggregated.sort_values(CURRENT_COLUMN)


def _clean_metric_dataframe(df: pd.DataFrame, metric_column: str) -> pd.DataFrame:
    """清洗指标数据框"""
    return clean_current_metric(df, CURRENT_COLUMN, metric_column)


def _get_color_for_shells(shells: List[str]) -> List[str]:
    """为壳体列表生成颜色"""
    return [DEFAULT_PALETTE[i % len(DEFAULT_PALETTE)] for i in range(len(shells))]


# ============================================================================
# 多壳体对比图
# ============================================================================

def build_multi_shell_chart(
    series_data: List[Tuple[str, pd.DataFrame]],
    metric_column: str,
    metric_label: str,
    test_type: str,
) -> Optional[alt.Chart]:
    """
    构建多壳体单Y轴对比图。
    
    Args:
        series_data: [(壳体ID, DataFrame), ...] 列表
        metric_column: 指标列名
        metric_label: 指标显示标签
        test_type: 测试类型名称
        
    Returns:
        Altair 图表对象或 None
    """
    if not series_data:
        return None

    chart_frames: List[pd.DataFrame] = []
    
    for shell_id, numeric in series_data:
        if numeric is None or numeric.empty:
            continue
        
        numeric_clean = numeric.dropna(subset=[CURRENT_COLUMN, metric_column]).copy()
        numeric_clean = _exclude_zero_current(numeric_clean)
        
        if numeric_clean.empty:
            continue
        
        numeric_clean[CURRENT_COLUMN] = pd.to_numeric(
            numeric_clean[CURRENT_COLUMN], errors="coerce"
        )
        numeric_clean[metric_column] = pd.to_numeric(
            numeric_clean[metric_column], errors="coerce"
        )
        numeric_clean = numeric_clean.dropna(subset=[CURRENT_COLUMN, metric_column])
        
        if numeric_clean.empty:
            continue
        
        numeric_clean = numeric_clean.assign(series=shell_id)
        renamed = numeric_clean.rename(
            columns={CURRENT_COLUMN: "current", metric_column: "value"}
        )
        chart_frames.append(renamed[["series", "current", "value"]])

    if not chart_frames:
        return None

    chart_data = pd.concat(chart_frames, ignore_index=True)
    present_labels = list(dict.fromkeys(chart_data["series"].tolist()))
    color_range = _get_color_for_shells(present_labels)

    # 高亮选择（鼠标悬停）
    highlight = alt.selection_point(
        on="mouseover",
        fields=["series"],
        nearest=True,
        empty=True,
    )
    
    # 缩放选择（鼠标滚轮缩放，拖拽平移）
    zoom = alt.selection_interval(
        bind="scales",  # 绑定到坐标轴，支持缩放和平移
        encodings=["x", "y"],  # 同时支持 X 和 Y 轴缩放
    )

    base = (
        alt.Chart(chart_data)
        .encode(
            x=alt.X("current:Q", title="电流(A)"),
            y=alt.Y("value:Q", title=metric_label, scale=alt.Scale(zero=False)),
            color=alt.Color(
                "series:N",
                title="Shell",
                scale=alt.Scale(domain=present_labels, range=color_range),
            ),
            tooltip=[
                alt.Tooltip("series:N", title="Shell"),
                alt.Tooltip("current:Q", title="电流(A)", format=".3f"),
                alt.Tooltip("value:Q", title=metric_label, format=".3f"),
            ],
        )
    )

    points = (
        base.mark_circle()
        .encode(
            size=alt.condition(highlight, alt.value(120), alt.value(70)),
            opacity=alt.condition(highlight, alt.value(1.0), alt.value(0.5)),
        )
    )
    lines = base.mark_line().encode(
        size=alt.condition(highlight, alt.value(4), alt.value(2)),
        opacity=alt.condition(highlight, alt.value(1.0), alt.value(0.5)),
    )

    chart = (points + lines).add_params(
        highlight,
        zoom,  # 添加缩放交互
    ).properties(
        title={
            "text": f"{test_type.replace('测试', '')}{metric_label}对比",
            "subtitle": "滚轮缩放 | 拖拽平移 | 双击重置",
            "subtitleColor": "#888888",
            "subtitleFontSize": 11,
        },
        width=600,
        height=420,
    )

    return chart


# ============================================================================
# 多站别趋势图
# ============================================================================

def build_station_metric_chart(
    dataframe: pd.DataFrame,
    metric_column: str,
    metric_label: str,
) -> Optional[alt.Chart]:
    """
    构建多站别趋势图。
    
    Args:
        dataframe: 包含测试数据的 DataFrame
        metric_column: 指标列名
        metric_label: 指标显示标签
        
    Returns:
        Altair 图表对象或 None
    """
    if dataframe is None or dataframe.empty:
        return None

    required_columns = {TEST_TYPE_COLUMN, CURRENT_COLUMN, metric_column}
    if not required_columns.issubset(dataframe.columns):
        return None

    chart_data = dataframe[[TEST_TYPE_COLUMN, CURRENT_COLUMN, metric_column]].dropna()
    if chart_data.empty:
        return None

    chart_data = chart_data.rename(
        columns={
            TEST_TYPE_COLUMN: "站别",
            CURRENT_COLUMN: "current",
            metric_column: "value",
        }
    )

    chart_data["current"] = pd.to_numeric(chart_data["current"], errors="coerce")
    chart_data["value"] = pd.to_numeric(chart_data["value"], errors="coerce")
    chart_data = chart_data.dropna(subset=["current", "value"])
    
    if chart_data.empty:
        return None

    present_labels = list(dict.fromkeys(chart_data["站别"].tolist()))
    color_domain = [
        label for label in SANITIZED_PLOT_ORDER if label in present_labels
    ]
    extras = [label for label in present_labels if label not in color_domain]
    color_domain = color_domain + extras if color_domain else present_labels
    
    # 构建颜色映射
    fallback_index = 0
    color_range: List[str] = []
    for label in color_domain:
        color = STATION_COLORS.get(label)
        if color is None:
            color = DEFAULT_PALETTE[fallback_index % len(DEFAULT_PALETTE)]
            fallback_index += 1
        color_range.append(color)

    highlight = alt.selection_point(
        on="mouseover",
        fields=["站别"],
        nearest=True,
        empty=False,
    )

    base = (
        alt.Chart(chart_data)
        .encode(
            x=alt.X("current:Q", title="电流(A)"),
            y=alt.Y("value:Q", title=metric_label, scale=alt.Scale(zero=False)),
            color=alt.Color(
                "站别:N",
                title="Station",
                scale=alt.Scale(domain=color_domain, range=color_range),
            ),
            tooltip=[
                alt.Tooltip("站别:N", title="Station"),
                alt.Tooltip("current:Q", title="电流(A)", format=".3f"),
                alt.Tooltip("value:Q", title=metric_label, format=".3f"),
            ],
        )
    )

    points = (
        base.mark_point(filled=True)
        .encode(
            size=alt.condition(highlight, alt.value(120), alt.value(70)),
            opacity=alt.value(0.9),
        )
        .add_params(highlight)
    )
    lines = base.mark_line().encode(
        size=alt.condition(highlight, alt.value(4), alt.value(2))
    )

    chart = (points + lines).properties(width=700, height=420)
    chart = chart.configure_view(stroke="transparent", fill="transparent").configure(
        background="transparent"
    )

    return chart


# ============================================================================
# 单壳体双轴图
# ============================================================================

def build_single_shell_dual_metric_chart(
    plot_df: pd.DataFrame,
    selected_df: Optional[pd.DataFrame],
    shell_id: str,
    test_type: str,
) -> Optional[alt.Chart]:
    """
    构建单壳体功率/效率对比的双轴点线图。
    
    Args:
        plot_df: 绘图数据
        selected_df: 选中的数据点（用于高亮）
        shell_id: 壳体ID
        test_type: 测试类型
        
    Returns:
        Altair 图表对象或 None
    """
    if plot_df is None or plot_df.empty:
        return None

    numeric = plot_df[[CURRENT_COLUMN, POWER_COLUMN, EFFICIENCY_COLUMN]].copy()
    numeric[CURRENT_COLUMN] = pd.to_numeric(numeric[CURRENT_COLUMN], errors="coerce")
    numeric[POWER_COLUMN] = pd.to_numeric(numeric[POWER_COLUMN], errors="coerce")
    numeric[EFFICIENCY_COLUMN] = (
        pd.to_numeric(numeric[EFFICIENCY_COLUMN], errors="coerce") * 100.0
    )
    numeric = numeric.dropna(subset=[CURRENT_COLUMN, POWER_COLUMN, EFFICIENCY_COLUMN])
    
    if numeric.empty:
        return None

    numeric = numeric.rename(
        columns={
            CURRENT_COLUMN: "current",
            POWER_COLUMN: "power",
            EFFICIENCY_COLUMN: "efficiency",
        }
    )
    numeric = numeric.sort_values("current")
    numeric["is_selected"] = False

    # 标记选中的数据点
    if selected_df is not None and not selected_df.empty:
        selected = selected_df[[CURRENT_COLUMN, POWER_COLUMN, EFFICIENCY_COLUMN]].copy()
        selected[CURRENT_COLUMN] = pd.to_numeric(selected[CURRENT_COLUMN], errors="coerce")
        selected[POWER_COLUMN] = pd.to_numeric(selected[POWER_COLUMN], errors="coerce")
        selected[EFFICIENCY_COLUMN] = (
            pd.to_numeric(selected[EFFICIENCY_COLUMN], errors="coerce") * 100.0
        )
        selected = selected.dropna(subset=[CURRENT_COLUMN, POWER_COLUMN])
        
        for _, row in selected.iterrows():
            mask = (
                np.isclose(numeric["current"].to_numpy(), row[CURRENT_COLUMN]) &
                np.isclose(numeric["power"].to_numpy(), row[POWER_COLUMN])
            )
            if mask.any():
                numeric.loc[mask, "is_selected"] = True

    hover = alt.selection_point(
        on="mouseover",
        fields=["current"],
        nearest=True,
        empty=False,
    )

    # 功率轴
    power_base = alt.Chart(numeric).encode(
        x=alt.X("current:Q", title="电流(A)"),
        y=alt.Y(
            "power:Q",
            title="功率(W)",
            axis=alt.Axis(titleColor=PRIMARY_BLUE, labelColor=PRIMARY_BLUE, orient="left"),
            scale=alt.Scale(zero=False),
        ),
        color=alt.value(PRIMARY_BLUE),
    )

    # 效率轴
    efficiency_base = alt.Chart(numeric).encode(
        x=alt.X("current:Q", title="电流(A)"),
        y=alt.Y(
            "efficiency:Q",
            title="电光效率(%)",
            axis=alt.Axis(titleColor=PRIMARY_RED, labelColor=PRIMARY_RED, orient="right"),
            scale=alt.Scale(zero=False),
        ),
        color=alt.value(PRIMARY_RED),
    )

    power_line = power_base.mark_line(size=3)
    power_points = power_base.mark_circle(size=90).encode(
        size=alt.condition("datum.is_selected", alt.value(140), alt.value(80)),
        opacity=alt.condition(hover, alt.value(1.0), alt.value(0.85)),
        tooltip=[
            alt.Tooltip("current:Q", title="电流(A)", format=".3f"),
            alt.Tooltip("power:Q", title="功率(W)", format=".3f"),
            alt.Tooltip("efficiency:Q", title="电光效率(%)", format=".3f"),
        ],
    )

    efficiency_line = efficiency_base.mark_line(size=3)
    efficiency_points = efficiency_base.mark_circle(size=90).encode(
        size=alt.condition("datum.is_selected", alt.value(140), alt.value(80)),
        opacity=alt.condition(hover, alt.value(1.0), alt.value(0.85)),
        tooltip=[
            alt.Tooltip("current:Q", title="电流(A)", format=".3f"),
            alt.Tooltip("efficiency:Q", title="电光效率(%)", format=".3f"),
            alt.Tooltip("power:Q", title="功率(W)", format=".3f"),
        ],
    )

    chart = (
        power_line + power_points + efficiency_line + efficiency_points
    ).resolve_scale(y="independent").add_params(hover)

    return chart.properties(
        title=f"{shell_id} {test_type.replace('测试', '')}",
        width=700,
        height=420,
    )


# ============================================================================
# Delta Band 图表
# ============================================================================

def build_multi_shell_diff_band_charts(
    series_data: List[Tuple[str, pd.DataFrame]],
    metric_column: str,
    metric_label: str,
    test_type: str,
) -> Tuple[Optional[str], List[Tuple[str, alt.Chart]]]:
    """
    构建多壳体 Delta Band 图表列表。
    
    Args:
        series_data: [(壳体ID, DataFrame), ...] 列表
        metric_column: 指标列名
        metric_label: 指标显示标签
        test_type: 测试类型名称
        
    Returns:
        (基准壳体ID, [(对比壳体ID, 图表), ...])
    """
    if len(series_data) < 2:
        baseline_shell = series_data[0][0] if series_data else None
        return baseline_shell, []

    shell_ids = [shell_id for shell_id, _ in series_data]
    color_lookup = {
        shell_id: DEFAULT_PALETTE[idx % len(DEFAULT_PALETTE)]
        for idx, shell_id in enumerate(shell_ids)
    }

    baseline_shell, baseline_df = series_data[0]
    baseline_clean = _clean_metric_dataframe(baseline_df, metric_column)
    if baseline_clean.empty:
        return baseline_shell, []

    charts: List[Tuple[str, alt.Chart]] = []

    for shell_id, df in series_data[1:]:
        other_clean = _clean_metric_dataframe(df, metric_column)
        if other_clean.empty:
            continue

        currents = sorted(set(baseline_clean["current"]).union(other_clean["current"]))
        if not currents:
            continue

        current_grid = pd.DataFrame({"current": currents})
        base_interp = current_grid.merge(baseline_clean, on="current", how="left")
        comp_interp = current_grid.merge(other_clean, on="current", how="left")
        
        for frame in (base_interp, comp_interp):
            frame["value"] = frame["value"].interpolate(method="linear")
            frame["value"] = frame["value"].ffill().bfill()

        merged = current_grid.copy()
        merged["value_baseline"] = base_interp["value"]
        merged["value_comparison"] = comp_interp["value"]
        merged = merged.dropna(subset=["value_baseline", "value_comparison"])
        
        if merged.empty:
            continue

        merged["delta"] = merged["value_comparison"] - merged["value_baseline"]
        merged["delta_abs"] = merged["delta"].abs()

        # 分离正负差异
        positive_df = merged[merged["delta"] >= 0].copy()
        negative_df = merged[merged["delta"] < 0].copy()
        band_df = pd.concat([
            positive_df.assign(sign="delta>=0"),
            negative_df.assign(sign="delta<0"),
        ], ignore_index=True)

        # 轴配置
        base_axis = alt.Axis(
            title=metric_label,
            tickCount=10,
            domain=True,
            domainWidth=2,
            domainColor='black',
        )
        x_axis = alt.Axis(
            title="电流(A)",
            tickCount=10,
            domain=True,
            domainWidth=2,
            domainColor='black',
        )

        # 差异带
        if band_df.empty:
            band_chart = alt.Chart(merged.iloc[0:0]).mark_area(opacity=0.0).encode(
                x=alt.X("current:Q", axis=x_axis),
                y=alt.Y("value_baseline:Q", axis=base_axis),
                y2="value_comparison:Q",
            )
        else:
            band_chart = alt.Chart(band_df).mark_area(opacity=0.35).encode(
                x=alt.X("current:Q", axis=x_axis),
                y=alt.Y("value_baseline:Q", axis=base_axis),
                y2="value_comparison:Q",
                color=alt.Color(
                    "sign:N",
                    scale=alt.Scale(
                        domain=["delta>=0", "delta<0"],
                        range=["#F9D7A5", "#B8D7FF"],
                    ),
                    legend=alt.Legend(title="Lead Direction"),
                ),
            )

        # 线图数据
        line_data = pd.concat([
            merged[["current", "value_baseline", "delta_abs"]]
            .assign(series=baseline_shell)
            .rename(columns={"value_baseline": "value"}),
            merged[["current", "value_comparison", "delta_abs"]]
            .assign(series=shell_id)
            .rename(columns={"value_comparison": "value"}),
        ], ignore_index=True)

        line_chart = alt.Chart(line_data).mark_line().encode(
            x=alt.X("current:Q", axis=x_axis),
            y=alt.Y("value:Q", axis=base_axis),
            color=alt.Color(
                "series:N",
                title="Shell",
                scale=alt.Scale(
                    domain=[baseline_shell, shell_id],
                    range=[color_lookup[baseline_shell], color_lookup[shell_id]],
                ),
            ),
            strokeWidth=alt.value(3),
            tooltip=[
                alt.Tooltip("series:N", title="Shell"),
                alt.Tooltip("current:Q", title="电流(A)", format=".3f"),
                alt.Tooltip("value:Q", title=metric_label, format=".3f"),
                alt.Tooltip("delta_abs:Q", title="|Δ|", format=".3f"),
            ],
        )

        point_chart = alt.Chart(line_data).mark_circle(size=110).encode(
            x=alt.X("current:Q", axis=x_axis),
            y=alt.Y("value:Q", axis=base_axis),
            color=alt.Color(
                "series:N",
                scale=alt.Scale(
                    domain=[baseline_shell, shell_id],
                    range=[color_lookup[baseline_shell], color_lookup[shell_id]],
                ),
                legend=None,
            ),
        )

        # 端点标注
        end_annotations = pd.DataFrame([
            {
                "series": baseline_shell,
                "current": merged.iloc[-1]["current"],
                "value": merged.iloc[-1]["value_baseline"],
                "label": f"{baseline_shell}: {merged.iloc[-1]['value_baseline']:.2f}",
            },
            {
                "series": shell_id,
                "current": merged.iloc[-1]["current"],
                "value": merged.iloc[-1]["value_comparison"],
                "label": f"{shell_id}: {merged.iloc[-1]['value_comparison']:.2f}",
            },
        ])
        
        end_text = alt.Chart(end_annotations).mark_text(
            align="left", dx=6, dy=-4, fontSize=12
        ).encode(
            x="current:Q",
            y="value:Q",
            text="label:N",
            color=alt.Color(
                "series:N",
                scale=alt.Scale(
                    domain=[baseline_shell, shell_id],
                    range=[color_lookup[baseline_shell], color_lookup[shell_id]],
                ),
                legend=None,
            ),
        )

        # 最大差异点标注
        max_idx = merged["delta_abs"].idxmax()
        max_annotations = merged.loc[[max_idx]].copy()
        max_annotations["label_str"] = max_annotations["delta"].map(lambda v: f"Delta={v:.2f}")
        
        max_text = alt.Chart(max_annotations).mark_text(
            align="center", dy=-10, fontSize=12, fontStyle="italic"
        ).encode(
            x="current:Q",
            y="value_comparison:Q",
            text="label_str:N",
        )
        
        max_point = alt.Chart(max_annotations).mark_point(
            size=120, shape="triangle-up"
        ).encode(
            x="current:Q",
            y="value_comparison:Q",
            color=alt.value(color_lookup[shell_id]),
        )

        combined = (
            alt.layer(band_chart, line_chart, point_chart, end_text, max_point, max_text)
            .resolve_scale(color="independent")
            .properties(
                width=600,
                height=360,
                title=f"{baseline_shell} vs {shell_id} {metric_label}Delta Band",
            )
            .configure_axis(grid=True, gridOpacity=0.3, tickCount=10)
        )

        charts.append((shell_id, combined))

    return baseline_shell, charts


# ============================================================================
# 拟合预测图
# ============================================================================

def plot_multi_shell_prediction(
    series_data: List[Tuple[str, pd.DataFrame]],
    metric_column: str,
    metric_label: str,
    test_type: str,
    poly_degree: int = 2,
    fit_mode: str = "global",
    auto_poly: bool = False,
    max_degree: int = 3,
) -> Optional[alt.Chart]:
    """
    创建多壳体拟合预测图，包含95%预测带。
    
    Args:
        series_data: [(壳体ID, DataFrame), ...] 列表
        metric_column: 指标列名
        metric_label: 指标显示标签
        test_type: 测试类型名称
        poly_degree: 多项式阶数（非效率数据）
        fit_mode: "global" 或 "per_shell"
        auto_poly: 是否自动选择阶数
        max_degree: 自动选择时的最大阶数
        
    Returns:
        Altair 图表对象或 None
    """
    if not ensure_prediction_libs_loaded():
        return None
    
    if not series_data or len(series_data) < 2:
        return None
    
    stats = get_stats_module()
    if stats is None:
        return None
    
    # 收集所有数据点
    all_x: List[float] = []
    all_y: List[float] = []
    shell_points: List[Dict[str, Any]] = []
    
    for shell_id, df in series_data:
        numeric = (
            df[[CURRENT_COLUMN, metric_column]]
            .apply(pd.to_numeric, errors="coerce")
            .dropna(subset=[CURRENT_COLUMN, metric_column])
        )
        numeric = _exclude_zero_current(numeric)
        if numeric.empty:
            continue

        x_vals = numeric[CURRENT_COLUMN].to_numpy(dtype=float)
        y_vals = numeric[metric_column].to_numpy(dtype=float)
        all_x.extend(x_vals.tolist())
        all_y.extend(y_vals.tolist())

        for x, y in zip(x_vals, y_vals):
            shell_points.append({"current": x, "value": y, "shell": shell_id})
    
    all_x_arr = np.array(all_x, dtype=float)
    all_y_arr = np.array(all_y, dtype=float)

    if all_x_arr.size == 0 or all_y_arr.size == 0 or not shell_points:
        return None
    
    shells = [shell_id for shell_id, _ in series_data]
    color_range = _get_color_for_shells(shells)
    
    if fit_mode == "per_shell":
        return _build_per_shell_prediction_chart(
            series_data, metric_column, metric_label, test_type,
            poly_degree, auto_poly, max_degree, shells, color_range
        )
    
    # 全局拟合模式
    return _build_global_prediction_chart(
        all_x_arr, all_y_arr, shell_points, metric_column, metric_label,
        test_type, poly_degree, auto_poly, max_degree, series_data, shells, color_range
    )


def _build_global_prediction_chart(
    all_x: np.ndarray,
    all_y: np.ndarray,
    shell_points: List[Dict],
    metric_column: str,
    metric_label: str,
    test_type: str,
    poly_degree: int,
    auto_poly: bool,
    max_degree: int,
    series_data: List[Tuple[str, pd.DataFrame]],
    shells: List[str],
    color_range: List[str],
) -> Optional[alt.Chart]:
    """构建全局拟合预测图"""
    stats = get_stats_module()
    
    if metric_column == EFFICIENCY_COLUMN:
        # 效率使用专业模型拟合
        result = fit_efficiency_model(all_x, all_y)
        if result is None:
            return None
        
        model_name, best_func, best_popt, model_info = result
        
        x_min, x_max = all_x.min(), all_x.max()
        x_range = x_max - x_min
        x_pred = np.linspace(x_min - x_range * 0.05, x_max + x_range * 0.05, 200)
        y_pred = best_func(x_pred, *best_popt)
        y_fitted = best_func(all_x, *best_popt)
        
        residuals = all_y - y_fitted
        n = len(all_y)
        p = len(best_popt)
        mse = np.sum(residuals**2) / (n - p)
        std_error = np.sqrt(mse)
        r_squared = model_info['r2']
        model_display_name = model_info['display_name']
    else:
        # 其他指标使用多项式拟合
        result = fit_polynomial_model(all_x, all_y, poly_degree, auto_poly, max_degree)
        if result is None:
            return None
        
        poly, model, chosen_deg, r_squared = result
        
        x_min, x_max = all_x.min(), all_x.max()
        x_range = x_max - x_min
        x_pred = np.linspace(x_min - x_range * 0.05, x_max + x_range * 0.05, 200)
        X_pred_poly = poly.transform(x_pred.reshape(-1, 1))
        y_pred = model.predict(X_pred_poly)
        
        y_fitted = model.predict(poly.fit_transform(all_x.reshape(-1, 1)))
        residuals = all_y - y_fitted
        n = len(all_y)
        p = chosen_deg + 1
        mse = np.sum(residuals**2) / (n - p)
        std_error = np.sqrt(mse)
        model_display_name = f"{chosen_deg}次多项式" if not auto_poly else f"自动选择{chosen_deg}次多项式"
    
    # 计算95%预测带
    t_val = stats.t.ppf(0.975, n - p)
    prediction_std = std_error * np.sqrt(1 + 1/n)
    y_upper = y_pred + t_val * prediction_std
    y_lower = y_pred - t_val * prediction_std
    
    # 准备数据
    band_df = pd.DataFrame({
        'current': x_pred,
        'upper': y_upper,
        'lower': y_lower,
        'fitted': y_pred
    })
    points_df = pd.DataFrame(shell_points)
    
    # 构建图表
    band_chart = alt.Chart(band_df).mark_area(opacity=0.2, color='#87CEEB').encode(
        x=alt.X('current:Q', title='电流(A)'),
        y=alt.Y('lower:Q', title=metric_label, scale=alt.Scale(zero=False)),
        y2='upper:Q'
    )
    
    upper_line = alt.Chart(band_df).mark_line(
        strokeDash=[5, 5], color='#4169E1', opacity=0.6, size=2
    ).encode(x='current:Q', y='upper:Q')
    
    lower_line = alt.Chart(band_df).mark_line(
        strokeDash=[5, 5], color='#4169E1', opacity=0.6, size=2
    ).encode(x='current:Q', y='lower:Q')
    
    fit_line = alt.Chart(band_df).mark_line(color='#FF4500', size=3).encode(
        x='current:Q',
        y='fitted:Q',
        tooltip=[
            alt.Tooltip('current:Q', title='电流(A)', format='.3f'),
            alt.Tooltip('fitted:Q', title=f'{metric_label}(拟合)', format='.3f')
        ]
    )
    
    points_chart = alt.Chart(points_df).mark_circle(size=80, opacity=0.7).encode(
        x='current:Q',
        y=alt.Y('value:Q', scale=alt.Scale(zero=False)),
        color=alt.Color('shell:N', title='壳体', scale=alt.Scale(domain=shells, range=color_range)),
        tooltip=[
            alt.Tooltip('shell:N', title='壳体'),
            alt.Tooltip('current:Q', title='电流(A)', format='.3f'),
            alt.Tooltip('value:Q', title=metric_label, format='.3f')
        ]
    )
    
    chart = (band_chart + upper_line + lower_line + fit_line + points_chart).properties(
        width=800,
        height=500,
        title={
            "text": f"{test_type.replace('测试', '')}{metric_label}拟合预测（{len(series_data)}个壳体，{model_display_name}）",
            "subtitle": f"R² = {r_squared:.4f}, RMSE = {std_error:.4f}",
            "subtitleColor": "#666666"
        }
    ).configure_axis(
        labelFontSize=11, titleFontSize=12, grid=True, gridOpacity=0.3
    ).configure_legend(
        labelFontSize=11, titleFontSize=12, orient='right'
    ).configure_title(fontSize=14, anchor='start')
    
    return chart


def _build_per_shell_prediction_chart(
    series_data: List[Tuple[str, pd.DataFrame]],
    metric_column: str,
    metric_label: str,
    test_type: str,
    poly_degree: int,
    auto_poly: bool,
    max_degree: int,
    shells: List[str],
    color_range: List[str],
) -> Optional[alt.Chart]:
    """构建分壳体拟合预测图"""
    stats = get_stats_module()
    if stats is None:
        return None
    
    per_shell_frames: List[pd.DataFrame] = []
    shell_points: List[Dict[str, Any]] = []
    
    for shell_id, df in series_data:
        numeric = (
            df[[CURRENT_COLUMN, metric_column]]
            .apply(pd.to_numeric, errors="coerce")
            .dropna(subset=[CURRENT_COLUMN, metric_column])
        )
        numeric = _exclude_zero_current(numeric)
        if numeric.empty:
            continue
        
        sx = numeric[CURRENT_COLUMN].to_numpy(dtype=float)
        sy = numeric[metric_column].to_numpy(dtype=float)
        
        # 保存散点数据
        for x, y in zip(sx, sy):
            shell_points.append({"current": x, "value": y, "shell": shell_id})
        
        if metric_column == EFFICIENCY_COLUMN:
            result = fit_efficiency_model(sx, sy)
            if result is None:
                continue
            _, best_func, best_popt, _ = result
            
            xmin, xmax = sx.min(), sx.max()
            xr = xmax - xmin
            x_pred = np.linspace(xmin - xr*0.05, xmax + xr*0.05, 200)
            y_pred = best_func(x_pred, *best_popt)
            y_fit = best_func(sx, *best_popt)
            
            resid = sy - y_fit
            n_local = len(sy)
            p_local = len(best_popt)
        else:
            result = fit_polynomial_model(sx, sy, poly_degree, auto_poly, max_degree)
            if result is None:
                continue
            poly, model, chosen_deg, _ = result
            
            xmin, xmax = sx.min(), sx.max()
            xr = xmax - xmin
            x_pred = np.linspace(xmin - xr*0.05, xmax + xr*0.05, 200)
            X_pred_poly = poly.transform(x_pred.reshape(-1, 1))
            y_pred = model.predict(X_pred_poly)
            y_fit = model.predict(poly.fit_transform(sx.reshape(-1, 1)))
            
            resid = sy - y_fit
            n_local = len(sy)
            p_local = chosen_deg + 1
        
        mse = np.sum(resid**2) / max(1, (n_local - p_local))
        se = np.sqrt(mse)
        t_val_local = stats.t.ppf(0.975, max(1, n_local - p_local))
        pred_std = se * np.sqrt(1 + 1/max(1, n_local))
        y_upper = y_pred + t_val_local * pred_std
        y_lower = y_pred - t_val_local * pred_std
        
        per_shell_frames.append(pd.DataFrame({
            'shell': shell_id,
            'current': x_pred,
            'upper': y_upper,
            'lower': y_lower,
            'fitted': y_pred
        }))
    
    if not per_shell_frames:
        return None
    
    band_df = pd.concat(per_shell_frames, ignore_index=True)
    points_df = pd.DataFrame(shell_points)
    
    band_chart = alt.Chart(band_df).mark_area(opacity=0.15).encode(
        x=alt.X('current:Q', title='电流(A)'),
        y=alt.Y('lower:Q', title=metric_label, scale=alt.Scale(zero=False)),
        y2='upper:Q',
        color=alt.Color('shell:N', scale=alt.Scale(domain=shells, range=color_range), legend=None)
    )
    
    fit_line = alt.Chart(band_df).mark_line(size=2).encode(
        x='current:Q',
        y='fitted:Q',
        color=alt.Color('shell:N', scale=alt.Scale(domain=shells, range=color_range), title='壳体'),
    )
    
    points_chart = alt.Chart(points_df).mark_circle(size=80, opacity=0.7).encode(
        x='current:Q',
        y=alt.Y('value:Q', scale=alt.Scale(zero=False)),
        color=alt.Color('shell:N', scale=alt.Scale(domain=shells, range=color_range), title='壳体'),
        tooltip=[
            alt.Tooltip('shell:N', title='壳体'),
            alt.Tooltip('current:Q', title='电流(A)', format='.3f'),
            alt.Tooltip('value:Q', title=metric_label, format='.3f')
        ]
    )
    
    chart = (band_chart + fit_line + points_chart).properties(
        width=800,
        height=500,
        title=f"{test_type.replace('测试', '')}{metric_label}拟合预测（{len(series_data)}个壳体，分壳体拟合）"
    ).configure_axis(
        labelFontSize=11, titleFontSize=12, grid=True, gridOpacity=0.3
    ).configure_legend(
        labelFontSize=11, titleFontSize=12, orient='right'
    ).configure_title(fontSize=14, anchor='start')
    
    return chart


# ============================================================================
# 功率预测计算
# ============================================================================

def _predict_metric_at_current(
    df: pd.DataFrame,
    metric_column: str,
    target_current: float,
    degree: int,
) -> Optional[Dict[str, float]]:
    """在指定电流点预测指标值"""
    stats = get_stats_module()
    
    prepared = _prepare_metric_series(df, metric_column)
    if prepared.empty:
        return None
    
    x = prepared[CURRENT_COLUMN].to_numpy(dtype=float)
    y = prepared[metric_column].to_numpy(dtype=float)
    min_current = float(np.min(x))
    max_current = float(np.max(x))
    in_range = min_current <= target_current <= max_current
    
    if len(x) == 1:
        value = float(y[0])
        return {
            "value": value,
            "in_range": in_range,
            "std_error": float("nan"),
            "ci_lower": value,
            "ci_upper": value,
            "min_current": min_current,
            "max_current": max_current,
        }
    
    effective_degree = max(1, min(degree, len(x) - 1))
    
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", np.RankWarning)
            coeffs, residuals, _, _, _ = np.polyfit(x, y, effective_degree, full=True)
        
        value = float(np.polyval(coeffs, target_current))
        n = len(x)
        p = effective_degree + 1
        rss = float(residuals[0]) if residuals.size > 0 else 0.0
        mse = rss / max(1, (n - p))
        std_error = float(np.sqrt(mse))
        
        t_val = stats.t.ppf(0.975, max(1, n - p)) if stats else 1.96
        prediction_std = std_error * np.sqrt(1 + 1/max(1, n))
        ci_lower = float(value - t_val * prediction_std)
        ci_upper = float(value + t_val * prediction_std)
    except (np.linalg.LinAlgError, ValueError):
        value = float(np.interp(target_current, x, y))
        std_error = float("nan")
        ci_lower = value
        ci_upper = value
    
    return {
        "value": value,
        "in_range": in_range,
        "std_error": std_error,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "min_current": min_current,
        "max_current": max_current,
    }


def compute_power_predictions(
    series_data: List[Tuple[str, pd.DataFrame]],
    target_current: float,
    degree: int,
) -> List[Dict[str, object]]:
    """
    计算多壳体在目标电流下的功率预测。
    
    Args:
        series_data: [(壳体ID, DataFrame), ...] 列表
        target_current: 目标电流值
        degree: 多项式阶数
        
    Returns:
        预测结果列表
    """
    predictions: List[Dict[str, object]] = []
    
    for shell_id, df in series_data:
        current_values = (
            pd.to_numeric(df.get(CURRENT_COLUMN, pd.Series(dtype=float)), errors="coerce")
            .dropna()
        )
        current_values = current_values[current_values.abs() > CURRENT_TOLERANCE]
        if current_values.empty:
            continue
        
        v_pred = _predict_metric_at_current(df, VOLTAGE_COLUMN, target_current, degree)
        e_pred = _predict_metric_at_current(df, EFFICIENCY_COLUMN, target_current, degree)
        
        if v_pred is None or e_pred is None:
            continue
        
        min_current = float(current_values.min())
        max_current = float(current_values.max())
        in_range = min_current <= target_current <= max_current
        
        # 处理效率值（可能是小数或百分比）
        efficiency_ratio = float(e_pred["value"])
        if efficiency_ratio > 1.5:
            efficiency_ratio = efficiency_ratio / 100.0
        efficiency_ratio = max(efficiency_ratio, 0.0)
        
        predicted_power = target_current * float(v_pred["value"]) * efficiency_ratio
        
        # 计算置信区间
        v_lower = float(v_pred["ci_lower"]) if not np.isnan(v_pred["std_error"]) else float(v_pred["value"])
        v_upper = float(v_pred["ci_upper"]) if not np.isnan(v_pred["std_error"]) else float(v_pred["value"])
        e_lower_ratio = float(e_pred["ci_lower"]) if not np.isnan(e_pred["std_error"]) else float(e_pred["value"])
        e_upper_ratio = float(e_pred["ci_upper"]) if not np.isnan(e_pred["std_error"]) else float(e_pred["value"])
        
        if e_lower_ratio > 1.5:
            e_lower_ratio = e_lower_ratio / 100.0
        if e_upper_ratio > 1.5:
            e_upper_ratio = e_upper_ratio / 100.0
        
        p_lower = target_current * v_lower * max(0.0, e_lower_ratio)
        p_upper = target_current * v_upper * max(0.0, e_upper_ratio)
        
        predictions.append({
            "壳体": shell_id,
            "预测电压(V)": round(float(v_pred["value"]), 3),
            "预测效率(%)": round(efficiency_ratio * 100.0, 3),
            "预测功率(W)": round(predicted_power, 3),
            "预测区间(W)": f"{p_lower:.3f}~{p_upper:.3f}",
            "数据范围(A)": f"{min_current:.3f}~{max_current:.3f}",
            "目标电流在范围内": "是" if in_range else "否",
        })
    
    return predictions
