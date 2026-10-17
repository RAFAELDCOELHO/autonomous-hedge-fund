"""Extra edge-case coverage for snapshot validation and helper adapters."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import _pr7_api as api
from tradingagents.backtest.snapshot import SnapshotIncomplete, validate_snapshot


def _us_frame() -> pd.DataFrame:
    idx = pd.to_datetime(["2023-12-29", "2024-01-02", "2024-01-03", "2024-01-04"])
    return pd.DataFrame({"Open": [100.0, 101.0, 102.0, 103.0], "Close": [99.0, 100.0, 101.0, 102.0]}, index=idx)


def test_validate_snapshot_accepts_empty_ticker_list():
    assert validate_snapshot([], "2024-01-02", "2024-01-03", prices_by_ticker={}) == {}


def test_validate_snapshot_accepts_unsorted_date_index_when_required_rows_exist():
    frame = _us_frame().iloc[[2, 0, 3, 1]]
    report = validate_snapshot(["AAPL"], "2024-01-02", "2024-01-03", prices_by_ticker={"AAPL": frame})
    assert report == {
        "AAPL": {
            "exchange": "XNYS",
            "n_sessions": 2,
            "first_d_minus_1": "2023-12-29",
            "exit_session": "2024-01-04",
        }
    }


def test_validate_snapshot_allows_nan_open_on_first_d_minus_1_bar():
    frame = _us_frame()
    frame.loc[pd.Timestamp("2023-12-29"), "Open"] = np.nan
    report = validate_snapshot(["AAPL"], "2024-01-02", "2024-01-03", prices_by_ticker={"AAPL": frame})
    assert report["AAPL"]["first_d_minus_1"] == "2023-12-29"


def test_snapshot_missing_normalizes_dates_from_mixed_timestamp_types():
    exc = SnapshotIncomplete([
        ("AAPL", np.datetime64("2024-01-02"), "Open"),
        ("AAPL", pd.Timestamp("2024-01-03"), "bar"),
    ])
    assert api.snapshot_missing(exc) == {("AAPL", "2024-01-02", "Open"), ("AAPL", "2024-01-03", "bar")}


def test_validate_snapshot_flags_nan_open_on_required_session_even_if_dates_are_unsorted():
    frame = _us_frame().iloc[[2, 0, 3, 1]]
    frame.loc[pd.Timestamp("2024-01-02"), "Open"] = np.nan
    with pytest.raises(SnapshotIncomplete) as info:
        validate_snapshot(["AAPL"], "2024-01-02", "2024-01-03", prices_by_ticker={"AAPL": frame})
    assert api.snapshot_missing(info.value) == {("AAPL", "2024-01-02", "Open")}
