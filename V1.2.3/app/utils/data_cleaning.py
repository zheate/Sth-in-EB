import pandas as pd
import numpy as np

def ensure_numeric(df: pd.DataFrame, columns: list[str], strict: bool = True) -> pd.DataFrame:
    frame = df.copy()
    for col in columns:
        if col not in frame.columns:
            if strict:
                raise ValueError(f"missing column: {col}")
            frame[col] = pd.NA
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    if strict and any(frame[c].isna().all() for c in columns):
        raise ValueError("all values invalid in numeric conversion")
    return frame

def drop_zero_current(df: pd.DataFrame, current_col: str, tol: float = 1e-6) -> pd.DataFrame:
    if current_col not in df.columns or df.empty:
        return df
    numeric_current = pd.to_numeric(df[current_col], errors="coerce")
    return df.loc[numeric_current.abs() > tol]

def clean_current_metric(df: pd.DataFrame, current_col: str, metric_col: str) -> pd.DataFrame:
    prepared = df[[current_col, metric_col]].copy()
    prepared = ensure_numeric(prepared, [current_col, metric_col], strict=True)
    prepared = prepared.dropna(subset=[current_col, metric_col])
    if prepared.empty:
        return prepared
    prepared = prepared.sort_values(current_col)
    return prepared.rename(columns={current_col: "current", metric_col: "value"})