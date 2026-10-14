"""_simulate_open requires the cash rate itself (follow-up of #68, Lingxi nit).

The public entry points validate market/no_rf before any load, but the
internal simulator used to default market=None to 0% cash. It now applies the
same _cash_rate rule, so an internal caller cannot silently skip the rf.
"""

import pandas as pd
import pytest

from tradingagents.backtest import runner

DAYS = pd.DatetimeIndex(["2024-01-03", "2024-01-04"])
OPENS = pd.Series([100.0, 101.0, 102.0])


def _hold(_day, _cutoff):
    return "HOLD", False


@pytest.mark.parametrize("kwargs", [{}, {"market": None}, {"market": None, "no_rf": False}])
def test_missing_cash_rate_raises(kwargs):
    with pytest.raises(TypeError, match="cash rate required"):
        runner._simulate_open("AAPL", DAYS, OPENS, _hold, 1_000.0, **kwargs)


def test_both_market_and_no_rf_raises():
    with pytest.raises(ValueError, match="ambiguous"):
        runner._simulate_open("AAPL", DAYS, OPENS, _hold, 1_000.0, "US", no_rf=True)


def test_explicit_no_rf_keeps_cash_flat():
    eq = runner._simulate_open("AAPL", DAYS, OPENS, _hold, 1_000.0, no_rf=True)
    assert (eq == 1_000.0).all()
    assert eq.attrs["rf_source"] == "none" and eq.attrs["rf_market"] is None
