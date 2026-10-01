"""Unit tests for run_backtest agent decision normalization path."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types
from unittest.mock import patch


REPO = Path(__file__).resolve().parents[1]
TARGET = REPO / "run_backtest.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("run_backtest", TARGET)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_run_backtest_uses_map_signal_for_verbose_llm_output():
    run_backtest = _load_module()

    class FakeGraph:
        def __init__(self, *args, **kwargs):
            pass

        def propagate(self, ticker: str, curr_date: str):
            assert ticker == "AAPL"
            assert curr_date == "2024-01-03"
            return {}, "Rating: OVERWEIGHT."

    captured = {}

    def fake_runner(decide_fn, ticker, start, end, capital, market=None):
        captured["market"] = market
        captured["decision"] = decide_fn("2024-01-03", None)
        captured["args"] = (ticker, start, end, capital)
        return "equity-curve"

    with patch("tradingagents.graph.trading_graph.TradingAgentsGraph", FakeGraph), patch.object(
        run_backtest, "run_agent_strategy", side_effect=fake_runner
    ):
        result = run_backtest._run_agent_decider(
            ticker="AAPL",
            start="2024-01-01",
            end="2024-01-31",
            capital=100_000.0,
        )

    assert result == "equity-curve"
    assert captured["decision"] == "BUY"
    assert captured["args"] == ("AAPL", "2024-01-01", "2024-01-31", 100_000.0)
    assert captured["market"] == "US"


def test_run_backtest_counts_unrecognized_verbose_signal_as_decision_error():
    """#50 + #51: make_decide_fn raises UnparseableSignal; the runner counts it and holds."""
    import pandas as pd
    import pytest

    from tradingagents.backtest.agent_integration import UnparseableSignal

    run_backtest = _load_module()

    class FakeGraph:
        def __init__(self, *args, **kwargs):
            pass

        def propagate(self, ticker: str, curr_date: str):
            assert ticker == "AAPL"
            return {}, "Model uncertain: wait-and-see." if curr_date == "2024-01-03" else "HOLD"

    captured = {}

    def fake_runner(decide_fn, ticker, start, end, capital, market=None):
        captured["market"] = market
        with pytest.raises(UnparseableSignal):
            decide_fn("2024-01-03", None)
        return "equity-curve"

    with patch("tradingagents.graph.trading_graph.TradingAgentsGraph", FakeGraph), patch.object(
        run_backtest, "run_agent_strategy", side_effect=fake_runner
    ):
        assert run_backtest._run_agent_decider("AAPL", "2024-01-02", "2024-01-31", 100_000.0) == "equity-curve"
    assert captured["market"] == "US"

    prices = pd.DataFrame({"Date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]), "Close": 100.0})
    with patch("tradingagents.graph.trading_graph.TradingAgentsGraph", FakeGraph), patch(
        "tradingagents.backtest.runner.load_ohlcv", return_value=prices
    ):
        eq = run_backtest._run_agent_decider("AAPL", "2024-01-02", "2024-01-04", 100_000.0)
    assert eq.attrs["n_decision_errors"] == 1
    assert eq.attrs["n_days"] == 3
    assert eq.iloc[-1] > 100_000.0  # all-cash (HOLD) accrues DTB3


def test_run_backtest_returns_none_when_graph_import_unavailable():
    run_backtest = _load_module()
    fake_graph_mod = types.ModuleType("tradingagents.graph.trading_graph")

    with patch.dict(sys.modules, {"tradingagents.graph.trading_graph": fake_graph_mod}):
        result = run_backtest._run_agent_decider(
            ticker="AAPL",
            start="2024-01-01",
            end="2024-01-31",
            capital=100_000.0,
        )

    assert result is None

