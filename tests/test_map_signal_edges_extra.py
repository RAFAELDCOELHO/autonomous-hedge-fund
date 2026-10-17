"""Additional edge-case tests for map_signal normalization."""

from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
TARGET = REPO / "tradingagents" / "backtest" / "agent_integration.py"


def _load_module():
    """Load agent_integration with minimal stubs for isolated unit tests."""
    fake_pandas = types.ModuleType("pandas")
    fake_pandas.DataFrame = object
    fake_pandas.Series = object

    fake_backtest_pkg = types.ModuleType("tradingagents.backtest")
    fake_backtest_pkg.__path__ = []
    fake_cash_rate = types.ModuleType("tradingagents.backtest.cash_rate")
    fake_cash_rate._REQUIRED = object()
    fake_cash_rate._cash_rate = lambda *_, **__: None
    fake_runner = types.ModuleType("tradingagents.backtest.runner")
    fake_runner.run_agent_strategy = lambda **_: None

    module_name = "tradingagents.backtest.agent_integration"
    spec = importlib.util.spec_from_file_location(module_name, TARGET)
    module = importlib.util.module_from_spec(spec)

    stubbed = {
        "pandas": fake_pandas,
        "tradingagents.backtest": fake_backtest_pkg,
        "tradingagents.backtest.cash_rate": fake_cash_rate,
        "tradingagents.backtest.runner": fake_runner,
    }
    with patch.dict(sys.modules, stubbed):
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    return module


class MapSignalExtraEdgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = _load_module()

    def test_spaced_weight_aliases_are_detected(self):
        self.assertEqual(self.mod.map_signal("OVER WEIGHT"), "BUY")
        self.assertEqual(self.mod.map_signal("under weight"), "SELL")

    def test_never_keyword_negates_label(self):
        self.assertEqual(self.mod.map_signal("never BUY"), "HOLD")
        self.assertEqual(self.mod.map_signal("NEVER sell"), "HOLD")

    def test_newline_boundary_breaks_negation_without_prior_label(self):
        self.assertEqual(self.mod.map_signal("do not\nSELL"), "SELL")

    def test_negation_window_allows_at_most_two_words_before_label(self):
        # "NOT really very" (two words) still negates the label.
        self.assertEqual(self.mod.map_signal("not really very BUY"), "HOLD")


if __name__ == "__main__":
    unittest.main()
