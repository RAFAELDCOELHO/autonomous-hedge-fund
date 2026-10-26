"""validate_snapshot: the D-1 bar of every decision session needs a finite positive Close.

D-1 rule: the decision for D uses data up to the close of D-1, so a bar there
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
    missing = exc.value.missing
    assert {f for _, _, f in missing} == {FIELD_CLOSE}
    days = [d for _, d, _ in missing]
    # first D-1 plus every January session except the last (21 NYSE sessions)
    assert days[0] == "2023-12-29" and days[-1] == "2024-01-30" and len(days) == 21


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


@pytest.mark.parametrize("day", ["2024-01-02", "2024-01-10", "2024-01-30"])
def test_bad_close_on_a_later_sessions_d_minus_1_is_a_defect(day):
    """Every decision session D's window ends at the Close of D-1."""
    frame = _frame()
    frame.loc[day, "Close"] = np.nan
    with pytest.raises(SnapshotIncomplete) as exc:
        _check({"AAPL": frame})
    assert exc.value.missing == [("AAPL", day, FIELD_CLOSE)]


def test_bad_close_on_every_d_minus_1_is_listed():
    frame = _frame()
    frame.loc["2024-01-05", "Close"] = 0.0
    frame.loc["2024-01-12", "Close"] = -3.0
    petr = _frame()
    petr.loc["2024-01-17", "Close"] = np.nan
    with pytest.raises(SnapshotIncomplete) as exc:
        _check({"AAPL": frame, "PETR4.SA": petr})
    assert exc.value.missing == [
        ("AAPL", "2024-01-05", FIELD_CLOSE),
        ("AAPL", "2024-01-12", FIELD_CLOSE),
        ("PETR4.SA", "2024-01-17", FIELD_CLOSE),
    ]


def test_bad_open_and_close_on_same_d_minus_1_both_listed():
    frame = _frame()
    frame.loc["2024-01-10", ["Open", "Close"]] = np.nan
    with pytest.raises(SnapshotIncomplete) as exc:
        _check({"AAPL": frame})
    assert exc.value.missing == [("AAPL", "2024-01-10", FIELD_OPEN), ("AAPL", "2024-01-10", FIELD_CLOSE)]


@pytest.mark.parametrize("day", ["2024-01-31", "2024-02-01"])
def test_close_of_last_session_and_exit_not_required(day):
    """The last session is D-1 only of the exit (no decision); the exit trades at the open."""
    frame = _frame()
    frame.loc[day, "Close"] = np.nan
    assert "AAPL" in _check({"AAPL": frame})


def test_older_bad_close_before_d_minus_1_is_ignored():
    frame = _frame()
    frame.loc["2023-12-28", "Close"] = np.nan
    assert "AAPL" in _check({"AAPL": frame})


@pytest.mark.parametrize("ticker,day", [("AAPL", "2024-01-10"), ("PETR4.SA", "2024-01-17")])
def test_missing_bar_on_middle_session_is_bar_not_close(ticker, day):
    """A middle session is a D-1 too, but with no bar it is reported once, as a bar."""
    frame = _frame().drop(pd.Timestamp(day))
    with pytest.raises(SnapshotIncomplete) as exc:
        _check({ticker: frame})
    assert exc.value.missing == [(ticker, day, FIELD_BAR)]


@pytest.mark.parametrize("ticker,day", [("AAPL", "2024-01-10"), ("PETR4.SA", "2024-01-17")])
def test_string_close_on_middle_d_minus_1_is_a_defect(ticker, day):
    frame = _frame().astype({"Close": object})
    frame.loc[day, "Close"] = "n/a"
    with pytest.raises(SnapshotIncomplete) as exc:
        _check({ticker: frame})
    assert exc.value.missing == [(ticker, day, FIELD_CLOSE)]
