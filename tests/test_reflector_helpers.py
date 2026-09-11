"""Offline unit tests for reflection helper behavior."""

import importlib.util
import unittest
from pathlib import Path

_MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "tradingagents" / "graph" / "reflection.py"
)


class _DummyLLM:
    def invoke(self, _messages):
        class _Response:
            content = "HOLD"

        return _Response()


class _DummyMemory:
    def __init__(self):
        self.records = []

    def add_situations(self, situations):
        self.records.extend(situations)


class ReflectorHelperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location(
            "reflection_module_for_tests", _MODULE_PATH
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Unable to load module spec from {_MODULE_PATH}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cls.reflector_cls = module.Reflector

    def test_extract_current_situation_combines_reports_in_expected_order(self):
        reflector = self.reflector_cls(_DummyLLM())
        state = {
            "market_report": "market",
            "sentiment_report": "sentiment",
            "news_report": "news",
            "fundamentals_report": "fundamentals",
        }

        out = reflector._extract_current_situation(state)

        self.assertEqual(out, "market\n\nsentiment\n\nnews\n\nfundamentals")

    def test_reflect_trader_persists_joined_situation_and_model_output(self):
        reflector = self.reflector_cls(_DummyLLM())
        memory = _DummyMemory()
        state = {
            "market_report": "MKT: uptrend",
            "sentiment_report": "SENT: mixed",
            "news_report": "NEWS: earnings beat",
            "fundamentals_report": "FUND: FCF improving",
            "trader_investment_plan": "BUY 100% allocation",
        }

        reflector.reflect_trader(state, returns_losses="+2.4%", trader_memory=memory)

        self.assertEqual(len(memory.records), 1)
        situation, reflection = memory.records[0]
        self.assertEqual(
            situation,
            "MKT: uptrend\n\nSENT: mixed\n\nNEWS: earnings beat\n\nFUND: FCF improving",
        )
        self.assertEqual(reflection, "HOLD")

    def test_extract_current_situation_raises_keyerror_when_required_key_missing(self):
        reflector = self.reflector_cls(_DummyLLM())
        incomplete_state = {
            "market_report": "market",
            "sentiment_report": "sentiment",
            "news_report": "news",
        }

        with self.assertRaises(KeyError):
            reflector._extract_current_situation(incomplete_state)


if __name__ == "__main__":
    unittest.main()
