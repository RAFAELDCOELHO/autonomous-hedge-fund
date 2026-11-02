"""_simulate_open must not sell a non-positive open.

SELL at 0 used to set cash to zero, a negative open used to make equity
negative, and NaN used to make equity NaN. The fill now requires a positive
open: the shares stay, and a later positive open still exits.
"""

import pandas as pd
import pytest

from tradingagents.backtest import runner


@pytest.mark.parametrize("bad_open", [0.0, -5.0, float("nan")])
def test_non_positive_open_sell_keeps_position_and_exits_later(bad_open):
    days = pd.DatetimeIndex(["2024-01-02", "2024-01-03", "2024-01-04"])
    # Entry at 10, attempted exit at the bad open, real exit at 20.
    opens = pd.Series([10.0, bad_open, 20.0, 20.0])
    actions = iter(["BUY", "SELL", "SELL"])

    def decide(_day, _cutoff):
        return next(actions), False

    equity = runner._simulate_open("AAPL", days, opens, decide, 1_000.0, no_rf=True)

    # 100 shares bought at 10. The bad open is only the mark of that entry
    # day. Skipping the sell leaves those shares to be marked at 20, and the
    # exit fills there. Old code realizes the bad open and the tail is
    # 0, -500, or NaN.
    assert equity.iloc[0] == 1_000.0
    assert list(equity.iloc[-2:].values) == [2_000.0, 2_000.0]
