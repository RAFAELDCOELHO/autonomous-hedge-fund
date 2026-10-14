"""Follow-up (Megabrain + Lingxi): the agent follows the baselines' cash-rate rule.

Every agent entry point (runner.run_agent_strategy, its lazy export in
tradingagents.backtest, agent_integration.run_tradingagents_backtest and
run_backtest._run_agent_decider) REQUIRES the cash rate: market="BR"/"US"
(daily rf, fails closed outside the snapshot) or the keyword-only no_rf=True
(explicit 0%). Neither, or market=None, raises TypeError; both raise ValueError;
both checks run before prices load (and before the agent graph is built).

Defense in depth: the cells.csv writer (cells.make_cell_row, used by
run_backtest's --cells-out path) refuses a curve whose cash did not earn the
market's daily rf (rf_source "none (--no-rf)", "none" or another market's),
independently of run_backtest's CLI guard (--no-rf with --cells-out -> exit 2).
Offline.
"""

from __future__ import annotations

import argparse
import csv

import numpy as np
import pytest

import _pr7_api as api
from _pr7_fakes import RecordingAgent, install_fake_yahoo
from tradingagents.backtest import agent_integration
from tradingagents.backtest.cells import make_cell_row
from tradingagents.backtest.risk_free import RF_SOURCE, daily_rf
from tradingagents.backtest.runner import run_agent_strategy

CAPITAL = 100_000.0
CASES = [("AAPL", "US"), ("PETR4.SA", "BR")]
WINDOW = ("2024-01-02", "2024-03-28")


def _hold(_date, _window):
    return "HOLD"


@pytest.fixture
def yahoo(monkeypatch, tmp_path):
    return install_fake_yahoo(monkeypatch, tmp_path)


@pytest.fixture
def no_data(monkeypatch):
    """Fail the test if prices load or the agent graph is built before the cash-rate check."""
    def boom(*_a, **_k):
        raise AssertionError("prices loaded / graph built before the cash-rate check")
    monkeypatch.setattr("tradingagents.backtest.runner.load_ohlcv", boom)
    monkeypatch.setattr(agent_integration, "make_decide_fn", boom)


def _agent_entry_points():
    import tradingagents.backtest as bt

    module = api.run_backtest_module()
    return {
        "run_agent_strategy": lambda **rate: run_agent_strategy(_hold, "AAPL", *WINDOW, CAPITAL, **rate),
        "lazy-export": lambda **rate: bt.run_agent_strategy(_hold, "AAPL", *WINDOW, CAPITAL, **rate),
        "run_tradingagents_backtest": lambda **rate: agent_integration.run_tradingagents_backtest(
            "AAPL", *WINDOW, config={}, initial_capital=CAPITAL, propagate_fn=lambda *_a: ({}, "HOLD"), **rate),
        "_run_agent_decider": lambda **rate: module._run_agent_decider(
            "AAPL", *WINDOW, CAPITAL, run_config={}, raise_unavailable=True, **rate),
    }, module


ENTRY_IDS = ["run_agent_strategy", "lazy-export", "run_tradingagents_backtest", "_run_agent_decider"]


@pytest.mark.parametrize("entry", ENTRY_IDS)
@pytest.mark.parametrize("rate", [{}, {"market": None}, {"no_rf": False}, {"market": None, "no_rf": False}],
                         ids=["neither", "market-None", "no_rf-False", "None-and-False"])
def test_agent_without_a_cash_rate_is_refused(no_data, monkeypatch, entry, rate):
    entries, module = _agent_entry_points()
    monkeypatch.setattr(module, "make_decide_fn", lambda **_k: pytest.fail("graph built before the check"))
    with pytest.raises(TypeError, match="market=.*or no_rf=True"):
        entries[entry](**rate)


@pytest.mark.parametrize("entry", ENTRY_IDS)
@pytest.mark.parametrize("market", ["US", "BR"])
def test_agent_with_both_market_and_no_rf_is_refused(no_data, monkeypatch, entry, market):
    entries, module = _agent_entry_points()
    monkeypatch.setattr(module, "make_decide_fn", lambda **_k: pytest.fail("graph built before the check"))
    with pytest.raises(ValueError, match="ambiguous"):
        entries[entry](market=market, no_rf=True)


def test_agent_no_rf_is_keyword_only_and_validated(no_data):
    with pytest.raises(TypeError):
        run_agent_strategy(_hold, "AAPL", *WINDOW, CAPITAL, None, True)  # no_rf positional
    with pytest.raises(TypeError, match="no_rf"):
        run_agent_strategy(_hold, "AAPL", *WINDOW, CAPITAL, no_rf="yes")
    with pytest.raises(ValueError, match="unknown market"):
        run_agent_strategy(_hold, "AAPL", *WINDOW, CAPITAL, market="EU")


@pytest.mark.parametrize(("ticker", "market"), CASES)
def test_agent_no_rf_gives_flat_cash_and_market_gives_the_daily_rf(yahoo, ticker, market):
    start, end = api.prereg_window()
    flat = api.run_agent(RecordingAgent(actions=lambda n, d: "HOLD"), ticker, start, end, CAPITAL, no_rf=True)
    assert len(flat) == 62 and (flat.to_numpy() == CAPITAL).all()
    assert flat.attrs["rf_market"] is None and flat.attrs["rf_source"] == "none"

    earning = api.run_agent(RecordingAgent(actions=lambda n, d: "HOLD"), ticker, start, end, CAPITAL, market=market)
    rf = daily_rf(market, earning.index)
    assert len(rf) == 61
    np.testing.assert_allclose(earning.to_numpy(), CAPITAL * np.cumprod([1.0, *(1 + rf)]), rtol=1e-12)
    assert earning.attrs["rf_market"] == market and earning.attrs["rf_source"] == RF_SOURCE[market]


def test_run_tradingagents_backtest_passes_exactly_one_rate_to_the_runner(monkeypatch):
    seen = []
    monkeypatch.setattr(agent_integration, "make_decide_fn", lambda **_k: _hold)
    monkeypatch.setattr(agent_integration, "run_agent_strategy", lambda **kw: seen.append(kw) or "curve")
    for rate in ({"market": "BR"}, {"no_rf": True}):
        agent_integration.run_tradingagents_backtest("PETR4.SA", *WINDOW, config={}, **rate)
    assert [(kw["market"], kw["no_rf"]) for kw in seen] == [("BR", False), (agent_integration._REQUIRED, True)]


@pytest.mark.parametrize("rate", [{"market": "US"}, {"no_rf": True}])
def test_run_agent_decider_passes_exactly_one_rate_to_the_runner(monkeypatch, rate):
    module = api.run_backtest_module()
    seen = []
    monkeypatch.setattr(module, "make_decide_fn", lambda **_k: _hold)
    monkeypatch.setattr(module, "run_agent_strategy", lambda *a, **kw: seen.append(kw) or "curve")
    assert module._run_agent_decider("AAPL", *WINDOW, CAPITAL, run_config={}, **rate) == "curve"
    assert seen == [rate]


@pytest.mark.parametrize(("ticker", "market"), [("AAPL", "US"), ("PETR4.SA", "BR"), ("vale3.sa", "BR")])
@pytest.mark.parametrize("no_rf", [False, True])
def test_main_gives_the_agent_market_of_ticker_or_no_rf(yahoo, monkeypatch, ticker, market, no_rf):
    """run_backtest.main passes exactly one rate to the agent arm: market_of(ticker), or no_rf under --no-rf."""
    module = api.run_backtest_module()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-dummy")
    monkeypatch.setattr(module, "load_dotenv", lambda *a, **k: None)
    seen = []

    def fake_decider(*_a, **kw):
        seen.append({k: v for k, v in kw.items() if k in ("market", "no_rf")})
        return None  # "pipeline unavailable": main skips the arm

    monkeypatch.setattr(module, "_run_agent_decider", fake_decider)
    monkeypatch.setattr(module, "run_strategy", lambda *a, **k: None)
    monkeypatch.setattr(module, "print_comparison", lambda *a, **k: None)
    argv = ["--ticker", ticker, "--start", WINDOW[0], "--end", WINDOW[1], "--arms", "baseline"]
    module.main(argv + (["--no-rf"] if no_rf else []))
    assert seen == [{"no_rf": True} if no_rf else {"market": market}]


# --- Defense in depth: the cells.csv writer refuses a curve without the daily rf ---


def _curve(ticker, **rate):
    start, end = api.prereg_window()
    return api.run_agent(RecordingAgent(actions=lambda n, d: "HOLD"), ticker, start, end, CAPITAL, **rate)


@pytest.mark.parametrize(("ticker", "market"), CASES)
def test_cells_writer_refuses_a_no_rf_curve(yahoo, ticker, market):
    curve = _curve(ticker, no_rf=True)
    with pytest.raises(ValueError, match="refusing cells.csv row.*'none'"):
        make_cell_row(ticker, "baseline", 0, curve)
    curve.attrs["rf_source"] = "none (--no-rf)"  # run_backtest's --no-rf label
    with pytest.raises(ValueError, match=r"refusing cells.csv row.*none \(--no-rf\)"):
        make_cell_row(ticker, "baseline", 0, curve)


def test_cells_writer_refuses_another_markets_rf(yahoo):
    with pytest.raises(ValueError, match="refusing cells.csv row.*FRED-DTB3"):
        make_cell_row("PETR4.SA", "baseline", 0, _curve("PETR4.SA", market="US"))


@pytest.mark.parametrize(("ticker", "market"), CASES)
def test_cells_writer_accepts_the_markets_daily_rf_curve(yahoo, ticker, market):
    row = make_cell_row(ticker, "baseline", 0, _curve(ticker, market=market))
    assert row["status"] == "ok" and row["rf_source"] == RF_SOURCE[market] and row["n_days"] == "61"
    assert float(row["sharpe"]) == 0.0  # all cash at the daily rf: zero excess Sharpe


def test_cells_out_writer_path_refuses_no_rf_even_with_the_cli_guard_bypassed(yahoo, tmp_path):
    """run_backtest's own --cells-out writer (_append_cell_row), called directly: nothing is written."""
    module = api.run_backtest_module()
    start, end = api.prereg_window()
    out = tmp_path / "cells.csv"
    args = argparse.Namespace(ticker="AAPL", seed=0, start=start, end=end, cells_out=out)
    curve = module._label_no_rf(_curve("AAPL", no_rf=True))
    with pytest.raises(ValueError, match=r"refusing cells.csv row.*none \(--no-rf\)"):
        module._append_cell_row(args, "baseline", {}, curve, "ok")
    assert not out.exists() or not list(csv.DictReader(out.open()))

    module._append_cell_row(args, "baseline", {}, _curve("AAPL", market="US"), "ok")
    rows = list(csv.DictReader(out.open()))
    assert len(rows) == 1 and rows[0]["rf_source"] == "FRED-DTB3"


def test_failed_rows_are_unaffected_by_the_writer_refusal():
    row = make_cell_row("AAPL", "baseline", 0, None, status="failed")
    assert row["status"] == "failed" and row["sharpe"] == "" and row["rf_source"] == "FRED-DTB3"
