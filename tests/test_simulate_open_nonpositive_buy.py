"""_simulate_open must not buy a non-positive open.

BUY at 0, a negative open, or NaN used to divide cash by that price and
take the position. The fill now requires a positive open: cash stays put
and the book stays flat, so a later positive open can still enter.
"""

import pandas as pd
import pytest

from tradingagents.backtest import runner


def _buy(_day, _cutoff):
    return "BUY", False


@pytest.mark.parametrize("bad_open", [0.0, -5.0, float("nan")])
def test_non_positive_open_buy_keeps_cash_and_fills_later(bad_open):
    days = pd.DatetimeIndex(["2024-01-03", "2024-01-04"])
    opens = pd.Series([bad_open, 10.0, 12.0])

    equity = runner._simulate_open("AAPL", days, opens, _buy, 1_000.0, no_rf=True)

    # Anchor, skipped fill (still cash at the next open), then the real fill
    # marked at 12. Old code buys the bad open and this curve is not cash.
    assert list(equity.values) == [1_000.0, 1_000.0, 1_200.0]
