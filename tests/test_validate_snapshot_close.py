"""validate_snapshot: the first session's D-1 bar needs a finite positive Close.

D-1 rule: the first decision uses data up to the close of D-1, so a bar there
with no usable Close is a defect, reported as FIELD_CLOSE ("Close").
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradingagents.backtest.snapshot import (
    FIELD_BAR,
    FIELD_CLOSE,
    FIELD_OPEN,
    SnapshotIncomplete,
    validate_snapshot,
)

START, END = "2024-01-02", "2024-01-31"
D_MINUS_1 = {"AAPL": "2023-12-29", "PETR4.SA": "2023-12-28"}


def _frame() -> pd.DataFrame:
    idx = pd.bdate_range("2023-12-01", "2024-02-29", name="Date")
    return pd.DataFrame({"Open": 100.0, "Close": 101.0}, index=idx)


def _check(prices):
    return validate_snapshot(list(prices), START, END, prices_by_ticker=prices)


def test_field_close_constant_and_description():
    assert FIELD_CLOSE == "Close"
    err = SnapshotIncomplete([("AAPL", "2023-12-29", FIELD_CLOSE)])
    assert "AAPL 2023-12-29: Close missing/NaN/<=0" in str(err)


def test_good_close_passes():
    report = _check({"AAPL": _frame(), "PETR4.SA": _frame()})
    assert report["AAPL"]["first_d_minus_1"] == D_MINUS_1["AAPL"]


@pytest.mark.parametrize("bad", [np.nan, 0.0, -1.0, np.inf, None])
@pytest.mark.parametrize("ticker", ["AAPL", "PETR4.SA"])
def test_bad_close_at_first_d_minus_1_is_a_defect(ticker, bad):
    frame = _frame().astype({"Close": object}) if bad is None else _frame()
    frame.loc[D_MINUS_1[ticker], "Close"] = bad
    with pytest.raises(SnapshotIncomplete) as exc:
        _check({ticker: frame})
    assert exc.value.missing == [(ticker, D_MINUS_1[ticker], FIELD_CLOSE)]


def test_missing_close_column_is_a_defect():
    frame = _frame().drop(columns="Close")
    with pytest.raises(SnapshotIncomplete) as exc:
        _check({"AAPL": frame})
    assert exc.value.missing == [("AAPL", "2023-12-29", FIELD_CLOSE)]


def test_close_defect_listed_with_other_defects_across_tickers():
    aapl = _frame()
    aapl.loc["2023-12-29", "Close"] = np.nan
    aapl.loc["2024-01-10", "Open"] = 0.0
    petr = _frame().drop(pd.Timestamp("2023-12-28"))
    with pytest.raises(SnapshotIncomplete) as exc:
        _check({"AAPL": aapl, "PETR4.SA": petr})
    assert exc.value.missing == [
        ("AAPL", "2023-12-29", FIELD_CLOSE),
        ("AAPL", "2024-01-10", FIELD_OPEN),
        ("PETR4.SA", "2023-12-28", FIELD_BAR),  # no bar: reported as bar, not Close
    ]


def test_bad_close_on_a_window_session_is_not_this_check():
    """Scope: only the first session's D-1 (same bar the runner requires)."""
    frame = _frame()
    frame.loc["2024-01-10", "Close"] = np.nan
    assert "AAPL" in _check({"AAPL": frame})


def test_older_bad_close_before_d_minus_1_is_ignored():
    frame = _frame()
    frame.loc["2023-12-28", "Close"] = np.nan
    assert "AAPL" in _check({"AAPL": frame})
