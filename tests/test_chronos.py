"""P1.10 Chronos comparator: offline contract (no chronos install required).

Checks committed artifact schema, script skip/import gates, banned source
substrings, and presence of the paper fixtures the comparator reads.
"""

from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "chronos_comparator.py"
OUT = REPO / "benchmark" / "results" / "chronos"
PAPER = REPO / "benchmark" / "prices" / "paper"

PER_CELL_COLS = ["agent", "regime", "ticker", "total_return_pct"]
SUMMARY_COLS = [
    "agent", "ticker", "mean_cr_pct", "min_cr_pct", "max_cr_pct", "gap_vs_bh_pct",
]
AGENTS = {"chronos", "buy_and_hold", "momentum"}
TICKERS = {"PETR4", "^BVSP"}
REGIMES = {"bull_2019", "crisis_2020", "recovery_2021", "hi_rates_2022"}
BANNED = ("anthropic", "dotenv", "yfinance")


class ChronosOfflineTests(unittest.TestCase):
    def test_script_bans_and_gates(self):
        src = SCRIPT.read_text(encoding="utf-8")
        lower = src.lower()
        for banned in BANNED:
            self.assertNotIn(banned, lower, banned)
        self.assertIn("CHRONOS_SKIP", src)
        self.assertIn("ImportError", src)
        self.assertIn("chronos_comparator: skip", src)

    def test_paper_fixtures_present(self):
        for name in ("PETR4.csv", "IDX_BVSP.csv"):
            path = PAPER / name
            self.assertTrue(path.is_file(), path)
            df = pd.read_csv(path)
            self.assertIn("Date", df.columns)
            self.assertIn("Close", df.columns)
            self.assertGreater(len(df), 100)

    def test_committed_per_cell_schema(self):
        path = OUT / "per_cell.csv"
        self.assertTrue(path.is_file(), path)
        df = pd.read_csv(path)
        self.assertEqual(list(df.columns), PER_CELL_COLS)
        self.assertEqual(len(df), 3 * 4 * 2)  # agents × regimes × tickers
        self.assertEqual(set(df["agent"]), AGENTS)
        self.assertEqual(set(df["ticker"]), TICKERS)
        self.assertEqual(set(df["regime"]), REGIMES)
        self.assertTrue(pd.api.types.is_numeric_dtype(df["total_return_pct"]))
        self.assertFalse(df["total_return_pct"].isna().any())

    def test_committed_summary_schema(self):
        path = OUT / "summary.csv"
        self.assertTrue(path.is_file(), path)
        df = pd.read_csv(path)
        self.assertEqual(list(df.columns), SUMMARY_COLS)
        self.assertEqual(len(df), 3 * 2)
        self.assertEqual(set(df["agent"]), AGENTS)
        self.assertEqual(set(df["ticker"]), TICKERS)
        # buy_and_hold gap vs itself must be ~0
        bh = df[df["agent"] == "buy_and_hold"]
        self.assertTrue((bh["gap_vs_bh_pct"].abs() < 1e-9).all())

    def test_committed_readme_exists(self):
        text = (OUT / "README.md").read_text(encoding="utf-8")
        self.assertIn("Chronos-t5-tiny", text)
        self.assertIn("Honest limits", text)
        self.assertIn("per_cell.csv", text)

    def test_skip_env_exits_zero(self):
        env = {**os.environ, "CHRONOS_SKIP": "1"}
        proc = subprocess.run(
            [sys.executable, str(SCRIPT)],
            cwd=str(REPO),
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("skip", proc.stdout.lower())


if __name__ == "__main__":
    unittest.main()
