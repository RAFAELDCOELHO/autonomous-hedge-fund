"""Unit tests for run_backtest agent decision normalization path."""

from __future__ import annotations

from unittest.mock import patch

import run_backtest


def test_run_backtest_uses_map_signal_for_verbose_llm_output():
    class FakeGraph:
        def propagate(self, ticker: str, curr_date: str):
            assert ticker == "AAPL"
            assert curr_date == "2024-01-03"
            return {}, "Rating: OVERWEIGHT."

    captured = {}

    def fake_runner(decide_fn, ticker, start, end, capital):
        captured["decision"] = decide_fn("2024-01-03", None)
        captured["args"] = (ticker, start, end, capital)
        return "equity-curve"

    with patch("tradingagents.graph.trading_graph.TradingAgentsGraph", FakeGraph), patch(
        "run_backtest.run_agent_strategy", side_effect=fake_runner
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
