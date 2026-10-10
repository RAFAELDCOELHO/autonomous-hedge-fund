"""Follow-up (Lingxi): classical baselines earn the same lagged daily rf as the agent.

At ad6a004 run_strategy called _simulate_open without `market`, so baseline
cash earned 0% while agent cash earned risk_free.daily_rf (report.py declared
the asymmetry). Now run_strategy/run_buy_and_hold take `market`, run_backtest.py
passes market_of(ticker) to every strategy, and the report shows the H1 Sharpe
for every curve whose cash earned that rf. Outside the rf snapshot every
strategy fails closed (DataDefectError); --no-rf zeroes cash for ALL strategies
(agent included), labels the output, and is refused with --cells-out. Offline.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

import _pr7_api as api
from _pr7_fakes import RecordingAgent, install_fake_yahoo
from tradingagents.backtest.baselines import _Strategy
from tradingagents.backtest.metrics import h1_cell_metrics
from tradingagents.backtest.report import H1_SHARPE_COL, build_comparison_table
from tradingagents.backtest.risk_free import daily_rf
from tradingagents.backtest.runner import DataDefectError, run_buy_and_hold, run_strategy

CAPITAL = 100_000.0
CASES = [("AAPL", "US"), ("PETR4.SA", "BR")]


class AlwaysCash(_Strategy):
    name = "always-cash"

    def signals(self, prices):
        return pd.Series(False, index=prices.index)


class LongFromFebruary(_Strategy):
    """Cash in January, long from the first February signal: both regimes in one run."""

    name = "long-from-feb"

    def signals(self, prices):
        return pd.Series(prices.index >= pd.Timestamp("2024-02-01"), index=prices.index)


@pytest.fixture
def yahoo(monkeypatch, tmp_path):
    return install_fake_yahoo(monkeypatch, tmp_path)


@pytest.mark.parametrize(("ticker", "market"), CASES)
def test_all_cash_baseline_earns_the_daily_rf_on_every_interval(yahoo, ticker, market):
    start, end = api.prereg_window()
    eq = run_strategy(AlwaysCash(), ticker, start, end, CAPITAL, market=market)

    rf = daily_rf(market, eq.index)
    assert len(eq) == 62 and len(rf) == 61
    np.testing.assert_allclose(eq.to_numpy(), CAPITAL * np.cumprod([1.0, *(1 + rf)]), rtol=1e-12)
    np.testing.assert_allclose((eq.pct_change().iloc[1:] - rf).to_numpy(), 0.0, atol=1e-15)
    assert h1_cell_metrics(eq, market)["sharpe"] == 0.0
    assert eq.attrs["rf_market"] == market
    # Without a market the old behaviour is kept: 0% cash.
    assert np.allclose(run_strategy(AlwaysCash(), ticker, start, end, CAPITAL, no_rf=True).to_numpy(), CAPITAL)


@pytest.mark.parametrize(("ticker", "market"), CASES)
def test_agent_in_cash_and_cash_baseline_are_identical(yahoo, ticker, market):
    start, end = api.prereg_window()
    agent = api.run_agent(RecordingAgent(actions=lambda n, d: "HOLD"), ticker, start, end, CAPITAL, market=market)
    base = run_strategy(AlwaysCash(), ticker, start, end, CAPITAL, market=market)
    assert agent.index.equals(base.index)
    np.testing.assert_array_equal(agent.to_numpy(), base.to_numpy())


@pytest.mark.parametrize(("ticker", "market"), CASES)
def test_baseline_and_agent_with_the_same_positions_are_identical(yahoo, ticker, market):
    start, end = api.prereg_window()
    base = run_strategy(LongFromFebruary(), ticker, start, end, CAPITAL, market=market)
    # The agent sees D-1 and holds the same position: long once the D-1 signal is on.
    agent = api.run_agent(
        lambda date, _w: "BUY" if pd.Timestamp(date) >= pd.Timestamp("2024-02-01") else "SELL",
        ticker, start, end, CAPITAL, market=market,
    )
    assert agent.index.equals(base.index)
    np.testing.assert_allclose(agent.to_numpy(), base.to_numpy(), rtol=1e-13)
    jan = base.loc[:"2024-01-31"]
    assert (jan.diff().iloc[1:] > 0).all()  # cash compounding at rf, not flat


@pytest.mark.parametrize(("ticker", "market"), CASES)
def test_buy_and_hold_values_do_not_depend_on_rf(yahoo, ticker, market):
    """Always invested from Open(D_0): no cash, so rf never accrues (PAPER Table 1 unaffected)."""
    start, end = api.prereg_window()
    with_rf = run_buy_and_hold(ticker, start, end, CAPITAL, market=market)
    without = run_buy_and_hold(ticker, start, end, CAPITAL, no_rf=True)
    np.testing.assert_array_equal(with_rf.to_numpy(), without.to_numpy())
    assert with_rf.attrs["rf_market"] == market and without.attrs["rf_market"] is None


def _run_backtest_curves(monkeypatch, ticker, start, end):
    module = api.run_backtest_module()
    monkeypatch.setattr(module, "load_dotenv", lambda *a, **k: None)
    captured = {}
    monkeypatch.setattr(module, "print_comparison", lambda curves, **_kw: captured.update(curves))
    rc = module.main(["--ticker", ticker, "--start", start, "--end", end,
                      "--capital", str(CAPITAL), "--skip-agents"])
    assert rc == 0
    return captured


@pytest.mark.parametrize(("ticker", "market"), CASES)
def test_run_backtest_baselines_earn_rf_inside_the_snapshot(yahoo, monkeypatch, ticker, market):
    start, end = api.prereg_window()
    curves = _run_backtest_curves(monkeypatch, ticker, start, end)
    assert set(curves) == {"Buy & Hold", "MACD(12,26,9)", "SMA(50/200)"}
    source = {"US": "FRED-DTB3", "BR": "BCB-SGS-12"}[market]
    for name, eq in curves.items():
        assert eq.attrs["rf_market"] == market, name
        assert eq.attrs["rf_source"] == source, name
    table = build_comparison_table(curves, market=market)
    assert (table[H1_SHARPE_COL] != "—").all(), table
    assert table.attrs["cash_rf"] == f"Cash rf: {source} (all strategies)"


OUTSIDE = ("2024-06-03", "2024-06-28")  # data/rf covers 2023-12-01..2024-04-30


@pytest.mark.parametrize(("ticker", "market"), CASES)
def test_runner_fails_closed_without_rf_outside_the_snapshot(yahoo, ticker, market):
    """rf needed and not available: baselines and agent alike raise, never 0% cash."""
    start, end = OUTSIDE
    with pytest.raises(DataDefectError, match="no daily rf"):
        run_strategy(AlwaysCash(), ticker, start, end, CAPITAL, market=market)
    with pytest.raises(DataDefectError, match="no daily rf"):
        run_buy_and_hold(ticker, start, end, CAPITAL, market=market)
    with pytest.raises(DataDefectError, match="no daily rf"):
        api.run_agent(RecordingAgent(actions=lambda n, d: "HOLD"), ticker, start, end, CAPITAL, market=market)


def test_cli_without_no_rf_fails_closed_outside_the_snapshot(yahoo, monkeypatch):
    module = api.run_backtest_module()
    monkeypatch.setattr(module, "load_dotenv", lambda *a, **k: None)
    printed = []
    monkeypatch.setattr(module, "print_comparison", lambda curves, **_kw: printed.append(curves))
    rc = module.main(["--ticker", "AAPL", "--start", OUTSIDE[0], "--end", OUTSIDE[1], "--skip-agents"])
    assert rc == 1 and printed == []
    # The agent arm on its own fails closed the same way (no --no-rf).
    monkeypatch.setattr(module, "make_decide_fn", lambda **_kw: RecordingAgent(actions=lambda n, d: "HOLD"))
    with pytest.raises(DataDefectError, match="no daily rf"):
        module._run_agent_decider("AAPL", *OUTSIDE, CAPITAL, run_config={}, raise_unavailable=True, market="US")


def _all_cash_baselines(monkeypatch, module):
    """Every CLI baseline holds cash all run, keeping its table name."""
    for attr, name in (("BuyAndHold", "Buy & Hold"), ("MACDStrategy", "MACD(12,26,9)"),
                       ("SMACrossStrategy", "SMA(50/200)")):
        monkeypatch.setattr(module, attr, type(attr, (AlwaysCash,), {"name": name}))


@pytest.mark.parametrize("window", ["prereg", "outside"])
def test_no_rf_zeroes_cash_for_every_strategy_agent_included(yahoo, monkeypatch, window):
    """--no-rf: all-cash agent, B&H, MACD and SMA give exactly flat equity, all labelled."""
    module = api.run_backtest_module()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-dummy")
    monkeypatch.setattr(module, "load_dotenv", lambda *a, **k: None)
    monkeypatch.setattr(module, "make_decide_fn", lambda **_kw: RecordingAgent(actions=lambda n, d: "HOLD"))
    _all_cash_baselines(monkeypatch, module)
    captured = {}
    monkeypatch.setattr(module, "print_comparison", lambda curves, **_kw: captured.update(curves))
    start, end = api.prereg_window() if window == "prereg" else OUTSIDE

    rc = module.main(["--ticker", "AAPL", "--start", start, "--end", end,
                      "--capital", str(CAPITAL), "--arms", "baseline", "--no-rf"])

    assert rc == 0
    assert set(captured) == {"Buy & Hold", "MACD(12,26,9)", "SMA(50/200)", "TradingAgents (baseline)"}
    for name, eq in captured.items():
        assert (eq.to_numpy() == CAPITAL).all(), name  # exactly flat: 0% cash
        assert eq.attrs["rf_market"] is None, name
        assert eq.attrs["rf_source"] == "none (--no-rf)", name
    table = build_comparison_table(captured, market="US")
    assert table.attrs["cash_rf"] == "Cash rf: none (--no-rf) (all strategies)"
    assert (table[H1_SHARPE_COL] == "—").all()


def test_no_rf_table_3_style_run_is_labelled_in_the_printed_output(yahoo, monkeypatch, capsys, caplog):
    """Outside the snapshot (like PAPER Table 3) --no-rf runs and says so in the output."""
    module = api.run_backtest_module()
    monkeypatch.setattr(module, "load_dotenv", lambda *a, **k: None)
    with caplog.at_level(logging.WARNING):
        rc = module.main(["--ticker", "AAPL", "--start", OUTSIDE[0], "--end", OUTSIDE[1],
                          "--skip-agents", "--no-rf"])
    assert rc == 0
    assert "Cash rf: none (--no-rf) (all strategies)" in capsys.readouterr().out
    assert "--no-rf: cash earns 0% in every strategy" in caplog.text


def test_no_rf_is_refused_with_cells_out(yahoo, monkeypatch, tmp_path):
    module = api.run_backtest_module()
    calls = []
    monkeypatch.setattr(module, "run_strategy", lambda *a, **k: calls.append(a))
    out = tmp_path / "cells.csv"
    start, end = api.prereg_window()
    with pytest.raises(SystemExit) as exc:
        module.main(["--ticker", "AAPL", "--start", start, "--end", end, "--skip-agents",
                     "--cells-out", str(out), "--no-rf"])
    assert exc.value.code == 2
    assert calls == [] and not out.exists()


def test_report_h1_sharpe_only_for_curves_that_earned_that_market_rf():
    idx = pd.DatetimeIndex(["2023-12-29", "2024-01-02", "2024-01-03", "2024-01-04"])
    def curve(rf_market):
        eq = pd.Series([100.0, 101.0, 100.5, 102.0], index=idx)
        eq.attrs["rf_market"] = rf_market
        return eq
    table = build_comparison_table(
        {"Buy & Hold": curve("US"), "MACD(12,26,9)": curve(None), "SMA(50/200)": curve("BR")}, market="US"
    ).set_index("Strategy")
    assert table.loc["Buy & Hold", H1_SHARPE_COL] != "—"
    assert table.loc["MACD(12,26,9)", H1_SHARPE_COL] == "—"
    assert table.loc["SMA(50/200)", H1_SHARPE_COL] == "—"


def test_cash_rf_label_names_each_source_when_they_differ():
    from tradingagents.backtest.report import cash_rf_label

    a, b = pd.Series([1.0, 1.0]), pd.Series([1.0, 1.0])
    a.attrs["rf_source"], b.attrs["rf_source"] = "FRED-DTB3", "none (--no-rf)"
    assert cash_rf_label({"A": a, "B": b}) == "Cash rf: A: FRED-DTB3; B: none (--no-rf)"
    assert cash_rf_label({"A": pd.Series([1.0])}) == ""


# --- Megabrain + Capataz: the cash rate is REQUIRED (market or no_rf=True), fail closed ---


@pytest.fixture
def no_data(monkeypatch):
    """The cash-rate check runs before any price load."""
    def boom(*_a, **_k):
        raise AssertionError("prices loaded before the cash-rate check")
    monkeypatch.setattr("tradingagents.backtest.runner.load_ohlcv", boom)


@pytest.mark.parametrize("call", [
    lambda: run_strategy(AlwaysCash(), "AAPL", "2024-01-02", "2024-03-28", CAPITAL),
    lambda: run_strategy(AlwaysCash(), "AAPL", "2024-01-02", "2024-03-28", CAPITAL, market=None),
    lambda: run_buy_and_hold("AAPL", "2024-01-02", "2024-03-28", CAPITAL),
    lambda: run_buy_and_hold("AAPL", "2024-01-02", "2024-03-28", CAPITAL, None),
], ids=["strategy", "strategy-market-None", "bh", "bh-market-None"])
def test_baseline_without_a_cash_rate_is_refused(no_data, call):
    with pytest.raises(TypeError, match="market=.*or no_rf=True"):
        call()


@pytest.mark.parametrize("call", [
    lambda: run_strategy(AlwaysCash(), "AAPL", "2024-01-02", "2024-03-28", CAPITAL, market="US", no_rf=True),
    lambda: run_buy_and_hold("PETR4.SA", "2024-01-02", "2024-03-28", CAPITAL, "BR", no_rf=True),
], ids=["strategy", "bh"])
def test_baseline_with_both_market_and_no_rf_is_refused(no_data, call):
    with pytest.raises(ValueError, match="ambiguous"):
        call()


def test_baseline_rejects_an_unknown_market_or_a_non_bool_no_rf(no_data):
    with pytest.raises(ValueError, match="unknown market"):
        run_strategy(AlwaysCash(), "AAPL", "2024-01-02", "2024-03-28", CAPITAL, market="EU")
    with pytest.raises(TypeError, match="no_rf"):
        run_strategy(AlwaysCash(), "AAPL", "2024-01-02", "2024-03-28", CAPITAL, no_rf="yes")


@pytest.mark.parametrize(("ticker", "market"), CASES)
def test_no_rf_true_gives_flat_cash_and_market_gives_rf(yahoo, ticker, market):
    start, end = api.prereg_window()
    flat = run_strategy(AlwaysCash(), ticker, start, end, CAPITAL, no_rf=True)
    assert (flat.to_numpy() == CAPITAL).all() and len(flat) == 62
    assert flat.attrs["rf_market"] is None and flat.attrs["rf_source"] == "none"
    earning = run_strategy(AlwaysCash(), ticker, start, end, CAPITAL, market=market)
    rf = daily_rf(market, earning.index)
    assert (earning.diff().iloc[1:] > 0).all()
    assert earning.iloc[-1] == pytest.approx(CAPITAL * float(np.prod(1 + rf)), rel=1e-12)


def test_public_entry_points_require_the_cash_rate(no_data):
    """tradingagents.backtest exposes the runner functions lazily (__getattr__): same rule."""
    import tradingagents.backtest as bt

    with pytest.raises(TypeError):
        bt.run_strategy(AlwaysCash(), "AAPL", "2024-01-02", "2024-03-28", CAPITAL)
    with pytest.raises(TypeError):
        bt.run_buy_and_hold("AAPL", "2024-01-02", "2024-03-28", CAPITAL)
