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



@pytest.mark.parametrize("entry", ["run_strategy", "run_agent_strategy"])
@pytest.mark.parametrize("market,no_rf", [("US", False), (None, True)])
def test_entry_points_forward_callers_no_rf(monkeypatch, entry, market, no_rf):
    """no_rf reaches _simulate_open exactly as the caller passed it, never
    inferred from market, so the simulator is a second, independent guard."""
    from tradingagents.backtest import baselines

    frame = pd.DataFrame({"Close": [1.0] * 4}, index=pd.date_range("2024-01-01", periods=4))
    monkeypatch.setattr(runner, "_execution_plan", lambda t, s, e: (frame, DAYS, OPENS))
    seen = {}

    def fake_sim(*args, **kwargs):
        seen["market"], seen["no_rf"] = args[5], kwargs["no_rf"]
        raise RuntimeError("stop")

    monkeypatch.setattr(runner, "_simulate_open", fake_sim)
    first = baselines.BuyAndHold() if entry == "run_strategy" else (lambda d, w: "HOLD")
    with pytest.raises(RuntimeError, match="stop"):
        getattr(runner, entry)(first, "AAPL", "2024-01-03", "2024-01-04", 1_000.0,
                               market=market, no_rf=no_rf)
    assert seen == {"market": market, "no_rf": no_rf}


@pytest.mark.parametrize("entry", ["run_strategy", "run_agent_strategy"])
def test_entry_market_none_without_no_rf_raises(monkeypatch, entry):
    """market=None without no_rf=True raises at every public entry, and keeps
    raising (in _simulate_open) even if the entry-point check is removed."""
    from tradingagents.backtest import baselines

    frame = pd.DataFrame({"Close": [1.0] * 4}, index=pd.date_range("2024-01-01", periods=4))
    monkeypatch.setattr(runner, "_execution_plan", lambda t, s, e: (frame, DAYS, OPENS))

    first = baselines.BuyAndHold() if entry == "run_strategy" else (lambda d, w: "HOLD")
    with pytest.raises(TypeError, match="cash rate required"):
        getattr(runner, entry)(first, "AAPL", "2024-01-03", "2024-01-04", 1_000.0, market=None)
