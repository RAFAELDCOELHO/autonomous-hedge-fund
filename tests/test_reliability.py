"""`make reliability` contract (P1.6): offline, schema-stable, byte-reproducible.

yfinance is poisoned for each test so any download attempt raises; the script
must not import it, anthropic, dotenv or urllib. Regenerating decisions.jsonl
and bins.csv into a temp dir must match the committed files byte-for-byte.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "reliability_diagram.py"
OUT = REPO / "benchmark" / "results" / "reliability"

SCHEMA = {
    "source": str, "run": str, "model": str, "ticker": str, "date": str,
    "date_inferred": bool, "signal": str, "confidence": (float, type(None)),
    "next_date": str, "next_ret": float, "win": (bool, type(None)),
}


def _load():
    spec = importlib.util.spec_from_file_location("reliability_diagram", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class ReliabilityTests(unittest.TestCase):
    def setUp(self):
        self._yf = sys.modules.get("yfinance")
        sys.modules["yfinance"] = None

    def tearDown(self):
        if self._yf is None:
            sys.modules.pop("yfinance", None)
        else:
            sys.modules["yfinance"] = self._yf

    def test_script_is_offline(self):
        src = SCRIPT.read_text(encoding="utf-8").lower()
        for banned in ("anthropic", "dotenv", "yfinance", "import urllib", "import requests", "import http", "import socket"):
            self.assertNotIn(banned, src, banned)

    def test_committed_jsonl_matches_schema(self):
        rows = [json.loads(l) for l in OUT.joinpath("decisions.jsonl").read_text().splitlines()]
        self.assertEqual(len(rows), 486)
        for r in rows:
            self.assertEqual(list(r), list(SCHEMA))
            for k, t in SCHEMA.items():
                self.assertIsInstance(r[k], t, k)
            self.assertIn(r["signal"], ("BUY", "SELL", "HOLD"))
            self.assertLess(r["date"], r["next_date"])
            if r["confidence"] is not None:
                self.assertTrue(0.0 <= r["confidence"] <= 1.0)
            self.assertEqual(r["win"] is None, r["signal"] == "HOLD")
        self.assertEqual(sum(r["confidence"] is not None for r in rows), 6)
        self.assertEqual(len({r["date"] for r in rows if r["confidence"] is not None}), 1)

    def test_regen_matches_committed(self):
        rd = _load()
        with tempfile.TemporaryDirectory() as tmp:
            rd.OUT_DIR = Path(tmp)
            rd.JSONL_PATH = rd.OUT_DIR / "decisions.jsonl"
            rd.BINS_CSV = rd.OUT_DIR / "bins.csv"
            rd.SVG_PATH = rd.OUT_DIR / "reliability.svg"
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(rd.main(), 0)
            for name in ("decisions.jsonl", "bins.csv", "reliability.svg"):
                self.assertEqual((rd.OUT_DIR / name).read_bytes(), (OUT / name).read_bytes(), name)

    def test_bins_only_count_scored_rows(self):
        rd = _load()
        rows = [
            {"ticker": "X", "date": "d1", "confidence": 0.95, "win": True},
            {"ticker": "X", "date": "d1", "confidence": 1.0, "win": False},
            {"ticker": "X", "date": "d2", "confidence": None, "win": True},
            {"ticker": "X", "date": "d3", "confidence": 0.5, "win": None},
        ]
        bins = rd.bin_stats(rows)
        self.assertEqual(bins, [dict(bin_lo=0.9, bin_hi=1.0, n=2, n_unique_prompts=1,
                                     mean_confidence=0.975, win_rate=0.5)])
        self.assertEqual(rd.ece(bins), 0.475)

    def test_runs_helper_prefixes_key_and_is_one_indexed(self):
        rd = _load()
        obj = {"warm_outputs": ["a", "b"], "cold_runs": ["x"]}
        self.assertEqual(
            list(rd._runs(obj, "warm_outputs")),
            [("warm_1", "a"), ("warm_2", "b")],
        )
        self.assertEqual(list(rd._runs(obj, "cold_runs")), [("cold_1", "x")])


if __name__ == "__main__":
    unittest.main()
