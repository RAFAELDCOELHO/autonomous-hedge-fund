"""`make qwen-n10` contract (P1.5): fixture-only prompt, honest stats, live call optional.

Offline tests never touch Ollama. The single live test issues one *warm*
generate call (it never kills the user's server) and skips when Ollama or
qwen2.5:7b is missing. Committed artifacts must be self-consistent: the
per-date summary must equal `summarise()` recomputed from runs.jsonl.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "qwen_coldstart_n10.py"
OUT = REPO / "benchmark" / "results" / "qwen_n10"


def _load():
    spec = importlib.util.spec_from_file_location("qwen_coldstart_n10", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class OfflineTests(unittest.TestCase):
    def setUp(self):
        self.m = _load()

    def test_no_download_no_key(self):
        src = SCRIPT.read_text(encoding="utf-8").lower()
        for banned in ("anthropic", "dotenv", "yfinance", "api_key"):
            self.assertNotIn(banned, src, banned)

    def test_prompt_from_fixture(self):
        dates, closes = self.m.load_prices()
        p = self.m.build_prompt(dates, closes, "2020-03-06")
        self.assertIn("DATE: 2020-03-06", p)
        self.assertIn("- close: 6.16", p)
        self.assertEqual(len(p.split("oldest to newest):\n")[1].split("\n")[0].split(", ")), 20)
        self.assertNotIn("volume", p)  # fixture has no volume column

    def test_parse_decision(self):
        ok = self.m.parse_decision('{"signal": "BUY", "confidence": 0.7, "reasoning": "x"}')
        self.assertEqual(ok, {"signal": "BUY", "confidence": 0.7, "reasoning": "x"})
        fenced = self.m.parse_decision('```json\n{"signal": "HOLD", "confidence": 1}\n```')
        self.assertEqual(fenced["signal"], "HOLD")
        for bad in ("not json", '{"signal": "LONG", "confidence": 0.5}', '{"signal": "BUY", "confidence": 1.5}', ""):
            self.assertEqual(self.m.parse_decision(bad)["signal"], None, bad)

    def test_summarise(self):
        rows = [dict(signal="BUY", confidence=0.6, response_hash="a", load_duration_s=1.0, total_duration_s=2.0),
                dict(signal="BUY", confidence=0.8, response_hash="b", load_duration_s=3.0, total_duration_s=4.0),
                dict(signal=None, confidence=None, response_hash="c", load_duration_s=1.0, total_duration_s=1.0)]
        s = self.m.summarise("d", rows)
        self.assertEqual((s["n"], s["n_parsed"], s["n_unique_responses"], s["majority_signal"]), (3, 2, 3, "BUY"))
        self.assertEqual((s["agreement"], s["confidence_mean"], s["confidence_std"]), (0.6667, 0.7, 0.1414))
        self.assertIsNone(self.m.summarise("d", rows[:1])["confidence_std"])  # n<2 -> no std, not 0

    def test_committed_artifacts_consistent(self):
        rows = [json.loads(l) for l in (OUT / "runs.jsonl").read_text().splitlines()]
        summary = json.loads((OUT / "summary.json").read_text())
        self.assertEqual(summary["model"], "qwen2.5:7b")
        self.assertEqual(summary["options"], {"temperature": 0})
        dates = [s["date"] for s in summary["per_date"]]
        for r in rows:
            self.assertEqual(list(r), list(self.m.RUN_FIELDS))
            self.assertTrue(r["cold"])
            self.assertGreater(r["load_duration_s"], 0, "cold run must have loaded the model")
        for s in summary["per_date"]:
            mine = [r for r in rows if r["date"] == s["date"]]
            self.assertGreaterEqual(len(mine), 10, s["date"])
            self.assertEqual(self.m.summarise(s["date"], mine), s)
        with (OUT / "summary.csv").open(newline="") as f:
            csv_rows = list(csv.DictReader(f))
        self.assertEqual([r["date"] for r in csv_rows], dates)
        self.assertEqual(list(csv_rows[0]), list(self.m.SUMMARY_FIELDS))


class LiveOllamaTest(unittest.TestCase):
    def test_one_warm_call_parses(self):
        m = _load()
        if not m.model_available():
            self.skipTest(f"ollama or {m.MODEL} not available at {m.OLLAMA}")
        dates, closes = m.load_prices()
        dec = m.parse_decision(m.call_ollama(m.build_prompt(dates, closes, "2020-03-06"))["response"])
        self.assertIn(dec["signal"], m.SIGNALS)
        self.assertTrue(0.0 <= dec["confidence"] <= 1.0)


if __name__ == "__main__":
    unittest.main()
