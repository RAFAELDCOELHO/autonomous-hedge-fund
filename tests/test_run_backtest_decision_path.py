"""Unit tests for run_backtest agent decision normalization path."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types
from unittest.mock import ANY, patch


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

    def fake_runner(decide_fn, ticker, start, end, capital):
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


def test_run_backtest_defaults_to_hold_for_unrecognized_verbose_signal():
    run_backtest = _load_module()

    class FakeGraph:
        def __init__(self, *args, **kwargs):
            pass

        def propagate(self, ticker: str, curr_date: str):
            assert ticker == "AAPL"
            assert curr_date == "2024-01-03"
            return {}, "Model uncertain: wait-and-see."

    captured = {}

    def fake_runner(decide_fn, ticker, start, end, capital):
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
    assert captured["decision"] == "HOLD"
    assert captured["args"] == ("AAPL", "2024-01-01", "2024-01-31", 100_000.0)


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


def test_run_backtest_propagate_exception_returns_hold_and_continues():
    run_backtest = _load_module()

    class FakeGraph:
        def __init__(self, *args, **kwargs):
            self.calls = 0

        def propagate(self, ticker: str, curr_date: str):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("temporary model failure")
            return {}, "BUY"

    captured = {}

    def fake_runner(decide_fn, ticker, start, end, capital):
        captured["day1"] = decide_fn("2024-01-03", None)
        captured["day2"] = decide_fn("2024-01-04", None)
        captured["args"] = (ticker, start, end, capital)
        return "equity-curve"

    with patch("tradingagents.graph.trading_graph.TradingAgentsGraph", FakeGraph), patch.object(
        run_backtest, "run_agent_strategy", side_effect=fake_runner
    ), patch.object(run_backtest.logging, "warning") as warn_mock:
        result = run_backtest._run_agent_decider(
            ticker="AAPL",
            start="2024-01-01",
            end="2024-01-31",
            capital=100_000.0,
        )

    assert result == "equity-curve"
    assert captured["day1"] == "HOLD"
    assert captured["day2"] == "BUY"
    assert captured["args"] == ("AAPL", "2024-01-01", "2024-01-31", 100_000.0)
    warn_mock.assert_any_call("Agent decision failed on %s: %s", "2024-01-03", ANY)
