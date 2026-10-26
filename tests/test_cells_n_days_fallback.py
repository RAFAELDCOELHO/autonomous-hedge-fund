"""Pin make_cell_row n_days: explicit attrs, or len(curve) when the key is missing.

The runner stores decisions (= returns) on equity.attrs["n_days"] because a B2
curve has one anchor point more than decisions. make_cell_row writes that
value when it is present. When the key is missing, the row keeps len(curve)
(the same count h1_cell_metrics reports for a curve with no NaNs).

Offline: dates sit inside the committed data/rf snapshot. No network.
"""

from __future__ import annotations

import pandas as pd

from tradingagents.backtest.cells import make_cell_row

# Five points, inside the FRED snapshot (2023-12-01 .. 2024-04-30). Length 5 is
# distinct from len-1 (returns) and from the failed-row constant 0.
_DATES = pd.DatetimeIndex(
    ["2024-01-12", "2024-01-16", "2024-01-17", "2024-01-18", "2024-01-19"]
)
_VALUES = [100.0, 101.0, 100.5, 102.0, 103.0]
_CURVE_LEN = 5


def _curve_without_n_days() -> pd.Series:
    equity = pd.Series(_VALUES, index=_DATES)
    equity.attrs.pop("n_days", None)
    assert "n_days" not in equity.attrs
    assert len(equity) == _CURVE_LEN
    return equity


def test_missing_n_days_falls_back_to_curve_length():
    """No attrs['n_days']: the row's n_days is exactly len(curve)."""
    equity = _curve_without_n_days()

    row = make_cell_row("AAPL", "baseline", 0, equity)

    assert row["status"] == "ok"
    assert row["ticker"] == "AAPL"
    assert row["n_days"] == "5"
    assert row["n_days"] == str(len(equity))
    assert row["n_days"] == str(_CURVE_LEN)


def test_explicit_n_days_is_written_exactly():
    """attrs['n_days'] wins even when it is not len(curve) and not len(curve)-1."""
    equity = _curve_without_n_days()
    explicit = 17
    assert explicit != len(equity)
    assert explicit != len(equity) - 1
    equity.attrs["n_days"] = explicit

    row = make_cell_row("AAPL", "macro", 1, equity)

    assert row["status"] == "ok"
    assert row["arm"] == "present"
    assert row["seed"] == "1"
    assert row["n_days"] == "17"
    assert row["n_days"] == str(explicit)
    assert row["n_days"] != str(len(equity))
    assert row["n_days"] != str(len(equity) - 1)
