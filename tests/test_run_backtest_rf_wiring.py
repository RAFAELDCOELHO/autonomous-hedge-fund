"""P3.7: run_backtest's agent arm passes the ticker's market so cash earns the daily rf."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest.mock import patch

import pytest


REPO = Path(__file__).resolve().parents[1]
TARGET = REPO / "run_backtest.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("run_backtest", TARGET)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("ticker, market", [("PETR4.SA", "BR"), ("vale3.sa", "BR"), ("AAPL", "US")])
def test_agent_arm_passes_market_of_ticker(ticker, market):
    run_backtest = _load_module()

    class FakeGraph:
        def __init__(self, *args, **kwargs):
            pass

        def propagate(self, ticker: str, curr_date: str):
            return {}, "HOLD"

    captured = {}

    def fake_runner(decide_fn, ticker, start, end, capital, **kwargs):
        captured.update(kwargs)
        return "equity-curve"

    with patch("tradingagents.graph.trading_graph.TradingAgentsGraph", FakeGraph), patch.object(
        run_backtest, "run_agent_strategy", side_effect=fake_runner
    ):
        result = run_backtest._run_agent_decider(ticker, "2024-01-02", "2024-01-31", 100_000.0)

    assert result == "equity-curve"
    assert captured == {"market": market}
