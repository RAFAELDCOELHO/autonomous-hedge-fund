"""`make multi-asset` contract (P1.8): offline, no look-ahead, paper-consistent.

The downloader module is poisoned in sys.modules so any fetch attempt raises;
the script must not mention it, anthropic, dotenv or a network client. The
Appendix cross-check column must reproduce benchmark/results/baselines/
ew_returns.csv, and regenerating into a temp dir must match the committed
artifacts byte-for-byte once they exist.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "multi_asset_corr.py"
OUT = REPO / "benchmark" / "results" / "multi_asset"
EW_APPENDIX = REPO / "benchmark" / "results" / "baselines" / "ew_returns.csv"
POISON = "yf" + "inance"  # spelled so the banlist below cannot match this file


def _load():
    spec = importlib.util.spec_from_file_location("multi_asset_corr", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class MultiAssetTests(unittest.TestCase):
    def setUp(self):
        self._saved = sys.modules.get(POISON)
        sys.modules[POISON] = None

    def tearDown(self):
        if self._saved is None:
            sys.modules.pop(POISON, None)
        else:
            sys.modules[POISON] = self._saved

    def test_script_is_offline(self):
        src = SCRIPT.read_text(encoding="utf-8").lower()
        for banned in ("anthropic", "dotenv", POISON, "hmm" + "learn", "scipy", "sklearn",
                       "import urllib", "import requests", "import http", "import socket"):
            self.assertNotIn(banned, src, banned)

    def test_weights_from_warmup_only(self):
        m = _load()
        raw = m.load_raw()
        closes = m.aligned_closes(raw)
        self.assertEqual(list(closes.columns), m.rl.TICKERS)
        for regime, (start, _) in m.rl.REGIMES.items():
            warmup, window = m.regime_returns(closes, regime)
            self.assertLess(warmup.index.max(), pd.Timestamp(start), regime)
            self.assertLessEqual(len(warmup), m.WARMUP_DAYS)
            self.assertGreater(window.index.min(), pd.Timestamp(start))
            cov = warmup.cov().to_numpy()
            w = m.weights_for(warmup)
            self.assertEqual(list(w), m.METHODS)
            for name, v in w.items():
                self.assertAlmostEqual(v.sum(), 1.0, places=12, msg=name)
                self.assertTrue((v >= 0).all(), name)
            var = {k: float(v @ cov @ v) for k, v in w.items()}
            self.assertLessEqual(var["min_var"], var["equal_weight"] + 1e-15, regime)
            self.assertLessEqual(var["min_var"], var["inv_vol"] + 1e-15, regime)

    def test_min_var_long_only_is_exact(self):
        m = _load()
        np.testing.assert_allclose(m.min_var_long_only(np.diag([1.0, 4.0])), [0.8, 0.2])
        # sigma=(1,2), rho=0.95: unconstrained w = (1.75, -0.75) -> long-only puts all on asset 0
        cov = np.array([[1.0, 1.9], [1.9, 4.0]])
        np.testing.assert_allclose(m.min_var_long_only(cov), [1.0, 0.0])

    def test_appendix_cross_check_and_shapes(self):
        m = _load()
        with contextlib.redirect_stdout(io.StringIO()):
            corr, weights, summary = m.compute()
        self.assertEqual(len(corr), 4 * 15)
        self.assertEqual(len(weights), 4 * 3 * 5)
        self.assertEqual(len(summary), 5 * 4 * 3)
        diag = corr[corr.ticker_a == corr.ticker_b]["corr"]
        self.assertTrue((diag == 1.0).all())
        self.assertTrue(corr["corr"].between(-1.0, 1.0).all())
        appendix = pd.read_csv(EW_APPENDIX).set_index(["agent", "regime"])["ew_return_pct"]
        ours = summary.drop_duplicates(["agent", "regime"]).set_index(["agent", "regime"])["ew_of_crs_pct"]
        for key, want in appendix.items():
            self.assertAlmostEqual(float(ours[key]), float(want), places=2, msg=str(key))
        # Buy & Hold with equal weights is a plain 1/5 constant-mix: all five slices exposed.
        bh = summary[(summary.agent == "buy_and_hold") & (summary.method == "equal_weight")]
        self.assertTrue((bh["ann_vol_pct"] > 0).all())

    @unittest.skipUnless(OUT.joinpath("portfolio_summary.csv").exists(),
                         "artifacts not generated yet: run `make multi-asset` and commit them")
    def test_regen_matches_committed(self):
        m = _load()
        with tempfile.TemporaryDirectory() as tmp:
            m.OUT_DIR = Path(tmp)
            m.CORR_CSV = m.OUT_DIR / "corr_by_regime.csv"
            m.WEIGHTS_CSV = m.OUT_DIR / "weights.csv"
            m.SUMMARY_CSV = m.OUT_DIR / "portfolio_summary.csv"
            m.README_MD = m.OUT_DIR / "README.md"
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(m.main(), 0)
            for name in ("corr_by_regime.csv", "weights.csv", "portfolio_summary.csv", "README.md"):
                self.assertEqual((m.OUT_DIR / name).read_bytes(), (OUT / name).read_bytes(), name)


if __name__ == "__main__":
    unittest.main()
