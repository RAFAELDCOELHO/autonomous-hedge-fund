"""`make reproduce` contract: regenerating the paper artifacts from committed
fixtures must reproduce the committed files byte-for-byte.

Offline by construction: yfinance is poisoned for the duration of each test so
any download attempt raises. No ANTHROPIC_API_KEY. CI fails here if the regen
drifts; if the drift is intended, run `make reproduce` and commit the result.
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
SCRIPTS = REPO / "scripts"
RESULTS = REPO / "benchmark" / "results"
DOCS = REPO / "docs"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class ReproduceGoldenTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self._yf = sys.modules.get("yfinance")
        sys.modules["yfinance"] = None  # `import yfinance` now raises

    def tearDown(self):
        if self._yf is None:
            sys.modules.pop("yfinance", None)
        else:
            sys.modules["yfinance"] = self._yf
        self._tmp.cleanup()

    def assert_same_bytes(self, produced: Path, committed: Path):
        self.assertTrue(committed.is_file(), f"missing committed artifact {committed}")
        self.assertEqual(
            produced.read_bytes(),
            committed.read_bytes(),
            f"{committed.relative_to(REPO)} drifted from `make reproduce` output",
        )

    def test_paper_fixtures_are_committed_and_scripts_are_keyless(self):
        rl = _load("regime_lib")
        self.assertEqual(rl.CACHE_DIR, REPO / "benchmark" / "prices" / "paper")
        for ticker in rl.TICKERS:
            self.assertTrue(rl._cache_path(ticker).is_file(), ticker)
        for name in ("regime_lib", "run_random_n100", "run_ew_portfolio", "run_brazilbench"):
            src = (SCRIPTS / f"{name}.py").read_text(encoding="utf-8").lower()
            self.assertNotIn("anthropic", src, name)
            self.assertNotIn("dotenv", src, name)

    def test_random_n100_regen_matches_committed(self):
        rn = _load("run_random_n100")
        rn.OUT_DIR = self.tmp
        rn.SUMMARY_CSV = self.tmp / "summary.csv"
        rn.PERSEED_CSV = self.tmp / "per_seed_returns.csv"
        rn.TEX_PATH = self.tmp / "paper_random_n100.tex"
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(rn.main(), 0)
        self.assert_same_bytes(rn.SUMMARY_CSV, RESULTS / "random_n100" / "summary.csv")
        self.assert_same_bytes(rn.PERSEED_CSV, RESULTS / "random_n100" / "per_seed_returns.csv")
        self.assert_same_bytes(rn.TEX_PATH, DOCS / "paper_random_n100.tex")

    def test_ew_portfolio_regen_matches_committed(self):
        ew = _load("run_ew_portfolio")
        ew.BASE_DIR = self.tmp
        ew.PERCELL_CSV = self.tmp / "per_cell_returns.csv"
        ew.EW_CSV = self.tmp / "ew_returns.csv"
        ew.TEX_PATH = self.tmp / "paper_ew_portfolio_baselines.tex"
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(ew.main(), 0)
        self.assert_same_bytes(ew.PERCELL_CSV, RESULTS / "baselines" / "per_cell_returns.csv")
        self.assert_same_bytes(ew.EW_CSV, RESULTS / "baselines" / "ew_returns.csv")
        self.assert_same_bytes(ew.TEX_PATH, DOCS / "paper_ew_portfolio_baselines.tex")

    def test_brazilbench_matrix_regen_matches_committed(self):
        from tradingagents.backtest import brazilbench as bb

        csv_path, md_path = self.tmp / "matrix.csv", self.tmp / "matrix.md"
        bb.write_outputs(bb.run_matrix(), csv_path, md_path)
        self.assert_same_bytes(csv_path, bb.MATRIX_CSV)
        self.assert_same_bytes(md_path, bb.MATRIX_MD)
        self.assertEqual(bb.MATRIX_CSV, RESULTS / "brazilbench" / "matrix.csv")
        self.assertEqual(bb.MATRIX_MD, DOCS / "brazilbench_baselines.md")


if __name__ == "__main__":
    unittest.main()
