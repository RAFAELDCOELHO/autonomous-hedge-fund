"""B2 (Lingxi, convention approved): the equity curve opens with an initial-capital point.

Curve = [CAPITAL at D-1 of the first session (previous session on the ticker's
exchange calendar), equity[D_0], ..., equity[D_n]], equity[D_i] = value at
Open(D_{i+1}) after executing D_i at Open(D_i), D_n marked at the open of the
session after `end`. Over the pre-registered window (61 sessions per exchange)
that is 62 points and 61 returns, one per decision, and every one of the 61
intervals accrues rf when in cash: rf for the return labelled D_i is
daily_rf(market, curve.index)[D_i], the rate h1_cell_metrics subtracts, so an
all-cash run has excess returns of exactly 0 and Sharpe 0.

Expected to FAIL on 1bfe41c (no initial point, no rf in the first interval)
and PASS once B2 lands. Offline (tests/_pr7_fakes.py, committed data/rf).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import _pr7_api as api
from _pr7_fakes import RecordingAgent, install_fake_yahoo

CAPITAL = 100_000.0
CASES = [("AAPL", "US", "2023-12-29"), ("PETR4.SA", "BR", "2023-12-28")]
EXIT_SESSION = "2024-04-01"  # 2024-03-29 Good Friday: closed on NYSE and B3


@pytest.fixture
def yahoo(monkeypatch, tmp_path):
    return install_fake_yahoo(monkeypatch, tmp_path)


def _prereg_sessions(ticker):
    from tradingagents.backtest.calendar import sessions

    start, end = api.prereg_window()
    return pd.DatetimeIndex(sessions(ticker, start, end))


@pytest.mark.parametrize(("ticker", "market", "d_minus_1"), CASES)
def test_all_cash_prereg_run_has_61_rf_intervals_and_zero_excess(yahoo, ticker, market, d_minus_1):
    from tradingagents.backtest.metrics import h1_cell_metrics
    from tradingagents.backtest.risk_free import daily_rf

    start, end = api.prereg_window()
    days = _prereg_sessions(ticker)
    assert len(days) == 61

    eq = api.run_agent(RecordingAgent(actions=lambda n, d: "HOLD"), ticker, start, end, CAPITAL, market=market)

    assert eq.index.tolist() == [pd.Timestamp(d_minus_1), *days]
    assert eq.iloc[0] == CAPITAL
    returns = eq.pct_change().iloc[1:]
    assert len(returns) == 61
    rf = daily_rf(market, eq.index)
    assert rf.index.equals(days)
    assert (rf > 0).all()
    # The first interval (initial point -> D_0) accrues rf like every other one.
    assert eq.iloc[1] == pytest.approx(CAPITAL * (1 + rf.iloc[0]), rel=1e-12)
    np.testing.assert_allclose((returns - rf).to_numpy(), 0.0, atol=1e-15)
    m = h1_cell_metrics(eq, market)
    assert m["sharpe"] == 0.0
    assert m["n_days"] == 61


@pytest.mark.parametrize(("ticker", "market", "d_minus_1"), CASES)
def test_prereg_curve_has_61_open_to_open_returns_for_61_decisions(yahoo, ticker, market, d_minus_1):
    start, end = api.prereg_window()
    days = _prereg_sessions(ticker)
    o = yahoo.frame(ticker, True)["Open"].astype(float)

    eq = api.run_agent(RecordingAgent(actions=lambda n, d: "BUY"), ticker, start, end, CAPITAL)
    bh = api.run_buy_and_hold(ticker, start, end, CAPITAL)

    marks = [*days, pd.Timestamp(EXIT_SESSION)]
    expected = [o[marks[i + 1]] / o[marks[i]] - 1 for i in range(61)]
    for curve in (eq, bh):
        assert len(curve) == 62
        assert curve.index[0] == pd.Timestamp(d_minus_1) and curve.iloc[0] == CAPITAL
        assert curve.index[1:].equals(days)
        pnl = api.pnl_by_decision_date(curve, CAPITAL)
        assert len(pnl) == 61 and pnl.index.equals(days)
        np.testing.assert_allclose(pnl.to_numpy(), expected, atol=1e-12)
        # Last decision marked at the open of the session after end, never at a close.
        assert curve.iloc[-1] == pytest.approx(CAPITAL * o[pd.Timestamp(EXIT_SESSION)] / o[days[0]], rel=1e-12)
    assert eq.attrs["n_days"] == 61 == len(api.decision_log(eq))
