"""
Shared session-state helpers for cross-page data access.

This module centralizes the keys used to stash the latest datasets
prepared on each Streamlit page so other pages can reuse them when
building combined exports.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
import streamlit as st

# Keys used inside st.session_state for each page payload.
DATA_FETCH_RESULT_KEY = "shared_data_fetch_result"
TEST_ANALYSIS_FILTERED_KEY = "shared_test_analysis_filtered"
PROGRESS_FILTERED_KEY = "shared_progress_filtered"
PROGRESS_RAW_KEY = "shared_progress_raw"


def _store_dataframe(key: str, df: Optional[pd.DataFrame]) -> None:
    """Persist or clear a DataFrame in session state."""
    if df is None:
        st.session_state.pop(key, None)
    else:
        st.session_state[key] = df


def store_data_fetch_result(df: Optional[pd.DataFrame]) -> None:
    """Stash the latest combined Data_fetch result table."""
    _store_dataframe(DATA_FETCH_RESULT_KEY, df)


def get_data_fetch_result() -> Optional[pd.DataFrame]:
    """Return the cached Data_fetch result table if available."""
    return st.session_state.get(DATA_FETCH_RESULT_KEY)


def store_test_analysis_filtered(df: Optional[pd.DataFrame]) -> None:
    """Stash the filtered TestAnalysis dataset currently in view."""
    _store_dataframe(TEST_ANALYSIS_FILTERED_KEY, df)


def get_test_analysis_filtered() -> Optional[pd.DataFrame]:
    """Return the cached TestAnalysis dataset if present."""
    return st.session_state.get(TEST_ANALYSIS_FILTERED_KEY)


def store_progress_filtered(df: Optional[pd.DataFrame]) -> None:
    """Stash the Progress page filtered data (station summary)."""
    _store_dataframe(PROGRESS_FILTERED_KEY, df)


def get_progress_filtered() -> Optional[pd.DataFrame]:
    """Return the cached Progress filtered dataset."""
    return st.session_state.get(PROGRESS_FILTERED_KEY)


def store_progress_raw(df: Optional[pd.DataFrame]) -> None:
    """Stash the Progress page raw report data."""
    _store_dataframe(PROGRESS_RAW_KEY, df)


def get_progress_raw() -> Optional[pd.DataFrame]:
    """Return the cached Progress raw dataset if available."""
    return st.session_state.get(PROGRESS_RAW_KEY)

