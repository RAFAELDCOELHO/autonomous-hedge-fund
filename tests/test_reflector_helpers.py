"""Offline unit tests for reflection helper behavior."""

import importlib.util
import unittest
from pathlib import Path

_MODULE_PATH = Path(__file__).resolve().parents[1] / "tradingagents" / "graph" / "reflection.py"
_SPEC = importlib.util.spec_from_file_location("reflection_module_for_tests", _MODULE_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC is not None and _SPEC.loader is not None
_SPEC.loader.exec_module(_MODULE)
Reflector = _MODULE.Reflector


class _DummyLLM:
    def invoke(self, _messages):
        class _Response:
            content = "HOLD"

        return _Response()


class ReflectorHelperTests(unittest.TestCase):
    def test_extract_current_situation_combines_reports_in_expected_order(self):
        reflector = Reflector(_DummyLLM())
        state = {
            "market_report": "market",
            "sentiment_report": "sentiment",
            "news_report": "news",
            "fundamentals_report": "fundamentals",
        }

        out = reflector._extract_current_situation(state)

        self.assertEqual(out, "market\n\nsentiment\n\nnews\n\nfundamentals")

    def test_extract_current_situation_preserves_each_report_content(self):
        reflector = Reflector(_DummyLLM())
        state = {
            "market_report": "MKT: uptrend",
            "sentiment_report": "SENT: mixed",
            "news_report": "NEWS: earnings beat",
            "fundamentals_report": "FUND: FCF improving",
        }

        out = reflector._extract_current_situation(state)

        self.assertIn("MKT: uptrend", out)
        self.assertIn("SENT: mixed", out)
        self.assertIn("NEWS: earnings beat", out)
        self.assertIn("FUND: FCF improving", out)

    def test_extract_current_situation_raises_keyerror_when_required_key_missing(self):
        reflector = Reflector(_DummyLLM())
        incomplete_state = {
            "market_report": "market",
            "sentiment_report": "sentiment",
            "news_report": "news",
        }

        with self.assertRaises(KeyError):
            reflector._extract_current_situation(incomplete_state)


if __name__ == "__main__":
    unittest.main()
