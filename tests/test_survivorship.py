"""`make survivorship` contract (P1.9): offline fixtures, stable schema, deterministic.

yfinance is poisoned so any download attempt raises. The script must not touch
the paper-five TICKERS list, and regenerating the CSVs must be byte-identical
to the committed artifacts when they are present.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "survivorship_distress.py"
OUT = REPO / "benchmark" / "results" / "survivorship"
FIXTURES = REPO / "benchmark" / "prices" / "paper"

PERCELL_COLS = ["universe", "agent", "regime", "ticker", "n_days", "total_return_pct"]
SUMMARY_COLS = ["agent", "regime", "liquid_mean_cr_pct", "distressed_mean_cr_pct", "gap_pp"]


def _load():
    sys.path.insert(0, str(REPO))
    spec = importlib.util.spec_from_file_location("survivorship_distress", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class SurvivorshipTests(unittest.TestCase):
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
        for banned in ("anthropic", "dotenv", "yfinance", "import urllib", "import requests"):
            self.assertNotIn(banned, src, banned)

    def test_fixtures_present_and_paper_five_untouched(self):
        sd = _load()
        self.assertEqual(sd.rl.TICKERS, ["PETR4", "VALE3", "ITUB4", "BBDC4", "^BVSP"])
        self.assertEqual(sd.rl.DISTRESSED, ["OIBR3", "MGLU3", "AMER3"])
        self.assertNotIn("GOLL4", sd.rl.DISTRESSED)  # unavailable on Yahoo; AMER3 substitutes
        for t in sd.rl.TICKERS + sd.rl.DISTRESSED:
            path = FIXTURES / f"{t.replace('^', 'IDX_')}.csv"
            self.assertTrue(path.exists(), path)
            self.assertEqual(path.read_text().splitlines()[0], "Date,Close")

    def test_compute_schema_and_determinism(self):
        sd = _load()
        percell, summary = sd.compute()
        self.assertEqual(list(percell.columns), PERCELL_COLS)
        self.assertEqual(list(summary.columns), SUMMARY_COLS)
        self.assertEqual(len(percell), 5 * 4 * (5 + 3))  # agents x regimes x tickers
        self.assertEqual(len(summary), 5 * 4)
        self.assertFalse(summary.isna().any().any())
        self.assertTrue((percell["n_days"] > 0).all())
        percell2, summary2 = sd.compute()
        self.assertTrue(percell.equals(percell2))
        self.assertTrue(summary.equals(summary2))

    def test_regen_matches_committed(self):
        if not (OUT / "summary.csv").exists():
            self.skipTest("artifacts not generated yet: run `make survivorship`")
        sd = _load()
        with tempfile.TemporaryDirectory() as tmp:
            sd.OUT_DIR = Path(tmp)
            sd.PERCELL_CSV = sd.OUT_DIR / "per_cell.csv"
            sd.SUMMARY_CSV = sd.OUT_DIR / "summary.csv"
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(sd.main(), 0)
            for name in ("per_cell.csv", "summary.csv"):
                self.assertEqual((sd.OUT_DIR / name).read_bytes(), (OUT / name).read_bytes(), name)


if __name__ == "__main__":
    unittest.main()
