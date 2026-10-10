"""validate_snapshot over the PREREG grid (9 tickers, 2024-01-02..2024-03-28). Synthetic, offline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradingagents.backtest.cells import (
    PREREG_TICKERS,
    PREREG_WINDOW_END,
    PREREG_WINDOW_START,
)
from tradingagents.backtest.snapshot import SnapshotIncomplete, validate_snapshot

TICKERS = [t + ".SA" if m == "BR" else t for t, m in PREREG_TICKERS.items()]


def _frame() -> pd.DataFrame:
    # Every weekday, holidays included: vendor bars on non-sessions must be ignored.
    idx = pd.bdate_range("2023-12-01", "2024-04-30", name="Date")
    return pd.DataFrame({"Open": 100.0, "Close": 101.0}, index=idx)


def _snapshot(tickers=TICKERS) -> dict[str, pd.DataFrame]:
    return {t: _frame() for t in tickers}


def _check(prices, tickers=TICKERS):
    return validate_snapshot(tickers, PREREG_WINDOW_START, PREREG_WINDOW_END, prices_by_ticker=prices)


def test_complete_prereg_snapshot_passes_with_61_sessions_per_exchange():
    report = _check(_snapshot())
    assert len(TICKERS) == len(report) == 9
    for ticker, item in report.items():
        br = ticker.endswith(".SA")
        assert item == {
            "exchange": "BVMF" if br else "XNYS",
            "n_sessions": 61,
            "first_d_minus_1": "2023-12-28" if br else "2023-12-29",
            "exit_session": "2024-04-01",  # Good Friday 2024-03-29 closed on both
        }


def test_every_missing_item_across_tickers_is_listed():
    prices = _snapshot()
    prices["AAPL"] = prices["AAPL"].drop(pd.Timestamp("2024-01-10"))
    prices["PETR4.SA"].loc["2024-03-01", "Open"] = np.nan
    prices["VALE3.SA"] = prices["VALE3.SA"].drop(pd.Timestamp("2024-04-01"))  # exit open
    prices["GOOGL"] = prices["GOOGL"].drop(pd.Timestamp("2023-12-29"))  # first D-1
    prices["ITUB4.SA"].loc["2024-02-14", "Open"] = 0.0
    del prices["WEGE3.SA"]
    prices["RADL3.SA"] = prices["RADL3.SA"].drop(columns="Open")

    with pytest.raises(SnapshotIncomplete) as exc:
        _check(prices)
    missing = exc.value.missing
    assert ("AAPL", "2024-01-10", "bar") in missing
    assert ("PETR4.SA", "2024-03-01", "Open") in missing
    assert ("VALE3.SA", "2024-04-01", "bar") in missing  # exit session
    assert ("GOOGL", "2023-12-29", "bar") in missing  # first D-1
    assert ("ITUB4.SA", "2024-02-14", "Open") in missing  # Open = 0
    assert sum(t == "WEGE3.SA" and f == "bar" for t, _, f in missing) == 61 + 2
    assert sum(t == "RADL3.SA" and f == "Open" for t, _, f in missing) == 61 + 1  # D-1 needs only a bar
    assert len(missing) == 5 + 63 + 62
    assert isinstance(exc.value, ValueError)


@pytest.mark.parametrize(
    ("holiday_of", "days", "flagged"),
    [("B3", ["2024-02-12", "2024-02-13"], "AAPL"), ("NYSE", ["2024-02-19"], "PETR4.SA")],
)
def test_one_exchange_holiday_is_flagged_only_on_the_other(holiday_of, days, flagged):
    tickers = ["AAPL", "PETR4.SA"]
    prices = {t: _frame().drop(pd.DatetimeIndex(days)) for t in tickers}

    with pytest.raises(SnapshotIncomplete) as exc:
        _check(prices, tickers)
    assert exc.value.missing == [(flagged, d, "bar") for d in days]
