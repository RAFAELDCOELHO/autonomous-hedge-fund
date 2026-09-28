"""Offline unit tests for backtest signal normalization helper."""

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
    """Load the helper module with lightweight stubs only."""
    fake_pandas = types.ModuleType("pandas")
    fake_pandas.DataFrame = object
    fake_pandas.Series = object

    fake_backtest_pkg = types.ModuleType("tradingagents.backtest")
    fake_backtest_pkg.__path__ = []
    fake_runner = types.ModuleType("tradingagents.backtest.runner")
    fake_runner.run_agent_strategy = lambda **_: None

    fake_graph_pkg = types.ModuleType("tradingagents.graph")
    fake_graph_pkg.__path__ = []
    fake_graph_mod = types.ModuleType("tradingagents.graph.trading_graph")

    class DummyGraph:
        def __init__(self, *args, **kwargs):
            pass

        def propagate(self, *_):
            return {}, "HOLD"

    fake_graph_mod.TradingAgentsGraph = DummyGraph

    module_name = "tradingagents.backtest.agent_integration"
    spec = importlib.util.spec_from_file_location(module_name, TARGET)
    module = importlib.util.module_from_spec(spec)

    stubbed = {
        "pandas": fake_pandas,
        "tradingagents.backtest": fake_backtest_pkg,
        "tradingagents.backtest.runner": fake_runner,
        "tradingagents.graph": fake_graph_pkg,
        "tradingagents.graph.trading_graph": fake_graph_mod,
    }
    with patch.dict(sys.modules, stubbed):
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    return module


class MapSignalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = _load_module()

    def test_none_falls_back_to_hold(self):
        self.assertEqual(self.mod.map_signal(None), "HOLD")

    def test_known_aliases_map_to_expected_trading_actions(self):
        self.assertEqual(self.mod.map_signal("OVERWEIGHT"), "BUY")
        self.assertEqual(self.mod.map_signal("UNDERWEIGHT"), "SELL")

    def test_known_canonical_signals_pass_through(self):
        self.assertEqual(self.mod.map_signal("BUY"), "BUY")
        self.assertEqual(self.mod.map_signal("HOLD"), "HOLD")
        self.assertEqual(self.mod.map_signal("SELL"), "SELL")

    def test_whitespace_and_case_are_normalized(self):
        self.assertEqual(self.mod.map_signal("  overweight "), "BUY")
        self.assertEqual(self.mod.map_signal("\nUnderWeight\t"), "SELL")

    def test_verbose_outputs_extract_first_valid_label(self):
        self.assertEqual(self.mod.map_signal("**BUY**"), "BUY")
        self.assertEqual(self.mod.map_signal("Rating: OVERWEIGHT."), "BUY")
        self.assertEqual(self.mod.map_signal("Action => underweight!"), "SELL")
        self.assertEqual(self.mod.map_signal("hOlD (confidence 0.54)"), "HOLD")

    def test_first_label_wins_when_multiple_labels_appear(self):
        self.assertEqual(self.mod.map_signal("SELL then BUY"), "SELL")
        self.assertEqual(self.mod.map_signal("Hold / then overweight"), "HOLD")

    def test_underscore_and_hyphen_are_treated_like_spaces(self):
        self.assertEqual(self.mod.map_signal("STRONG_BUY"), "BUY")
        self.assertEqual(self.mod.map_signal("STRONG-BUY"), "BUY")
        self.assertEqual(self.mod.map_signal("under-weight"), "SELL")

    def test_negated_labels_are_ignored(self):
        self.assertEqual(self.mod.map_signal("do not BUY"), "HOLD")
        self.assertEqual(self.mod.map_signal("don't sell"), "HOLD")
        self.assertEqual(self.mod.map_signal("don’t sell"), "HOLD")
        self.assertEqual(self.mod.map_signal("cannot BUY"), "HOLD")
        self.assertEqual(self.mod.map_signal("can’t SELL"), "HOLD")
        self.assertEqual(self.mod.map_signal("not a buy"), "HOLD")
        self.assertEqual(self.mod.map_signal("avoid BUY"), "HOLD")
        self.assertEqual(self.mod.map_signal("no BUY"), "HOLD")

    def test_negated_label_falls_through_to_next_valid_label(self):
        self.assertEqual(self.mod.map_signal("do not BUY, SELL"), "SELL")
        self.assertEqual(self.mod.map_signal("FINAL: HOLD, not SELL"), "HOLD")
        self.assertEqual(self.mod.map_signal("NOT A HOLD - SELL"), "SELL")
        self.assertEqual(self.mod.map_signal("Not a SELL, not a HOLD - BUY"), "BUY")

    def test_idioms_and_double_negation_are_not_treated_as_negation(self):
        self.assertEqual(self.mod.map_signal("no doubt BUY"), "BUY")
        self.assertEqual(self.mod.map_signal("not only BUY"), "BUY")
        self.assertEqual(self.mod.map_signal("no reason not to BUY"), "BUY")

    def test_ing_words_are_not_signal_labels(self):
        self.assertEqual(self.mod.map_signal("Buying the dip is risky. SELL"), "SELL")
        self.assertEqual(self.mod.map_signal("Selling pressure eased; BUY"), "BUY")
        self.assertEqual(
            self.mod.map_signal("Insider buying noted. Rating: HOLD"), "HOLD"
        )
        self.assertEqual(self.mod.map_signal("Still holding for now. SELL"), "SELL")

    def test_non_text_inputs_do_not_raise_and_fallback_reasonably(self):
        self.assertEqual(self.mod.map_signal(None), "HOLD")
        self.assertEqual(self.mod.map_signal([]), "HOLD")
        self.assertEqual(self.mod.map_signal({}), "HOLD")
        self.assertEqual(self.mod.map_signal(42), "HOLD")

    def test_loose_hyphen_keeps_following_label_searchable(self):
        self.assertEqual(self.mod.map_signal(" - SELL"), "SELL")

    def test_unknown_or_blank_signal_is_defensive_hold(self):
        self.assertEqual(self.mod.map_signal(""), "HOLD")
        self.assertEqual(self.mod.map_signal("   "), "HOLD")
        self.assertEqual(self.mod.map_signal("WAIT"), "HOLD")


if __name__ == "__main__":
    unittest.main()
