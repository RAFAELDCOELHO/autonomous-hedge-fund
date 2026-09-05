"""`make hmm-regimes` contract (P1.7): offline, deterministic, byte-reproducible.

yfinance is poisoned so any download attempt raises; the script must not
import it, anthropic, dotenv or any network module. The NumPy Baum-Welch is
checked on a synthetic two-state series with a known switch. Regenerating the
artifacts into a temp dir must match the committed files byte-for-byte
(skipped until `make hmm-regimes` has been run and its outputs committed).
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

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "hmm_regimes.py"
OUT = REPO / "benchmark" / "results" / "hmm_regimes"
ARTIFACTS = ("states.csv", "alignment.csv", "breakpoints.csv", "README.md")


def _load():
    spec = importlib.util.spec_from_file_location("hmm_regimes", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class HmmRegimesTests(unittest.TestCase):
    def setUp(self):
        self._yf = sys.modules.get("yfinance")
        sys.modules["yfinance"] = None

    def tearDown(self):
        if self._yf is None:
            sys.modules.pop("yfinance", None)
        else:
            sys.modules["yfinance"] = self._yf

    def test_script_is_offline_and_numpy_only(self):
        src = SCRIPT.read_text(encoding="utf-8").lower()
        for banned in ("anthropic", "dotenv", "yfinance", "hmmlearn", "statsmodels", "scipy",
                       "sklearn", "import urllib", "import requests", "import http", "import socket"):
            self.assertNotIn(banned, src, banned)

    def test_fit_recovers_synthetic_regimes(self):
        hm = _load()
        rng = np.random.default_rng(0)
        calm = rng.normal(0.0005, 0.008, 400)
        turb = rng.normal(-0.001, 0.03, 120)
        x = np.concatenate([calm, turb, calm])
        truth = np.r_[np.zeros(400), np.ones(120), np.zeros(400)]
        fit = hm.fit_hmm(x)
        self.assertLess(fit["var"][0], fit["var"][1])  # canonical order: calm first
        state = hm.smooth_states(fit["gamma"].argmax(axis=1))
        self.assertGreater((state == truth).mean(), 0.95)
        self.assertGreater(hm.adjusted_rand(truth, state), 0.8)

    def test_fit_is_deterministic(self):
        hm = _load()
        x = np.random.default_rng(1).normal(0, 0.01, 300)
        a, b = hm.fit_hmm(x), hm.fit_hmm(x)
        np.testing.assert_array_equal(a["gamma"], b["gamma"])
        self.assertEqual(a["loglik"], b["loglik"])

    def test_smooth_states_absorbs_short_runs(self):
        hm = _load()
        s = np.array([0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
        out = hm.smooth_states(s, min_run=5)
        self.assertEqual(out.tolist(), [0] * 12 + [1] * 6)

    def test_adjusted_rand_bounds(self):
        hm = _load()
        a = np.array(["x", "x", "y", "y", "z", "z"])
        self.assertAlmostEqual(hm.adjusted_rand(a, a), 1.0)
        # contingency all-ones (3x2): sum_ij=0, sa=3, sb=6, expected=1.2, max=4.5
        self.assertAlmostEqual(hm.adjusted_rand(a, np.array([0, 1, 0, 1, 0, 1])), -1.2 / 3.3, places=6)

    @unittest.skipUnless(all((OUT / f).exists() for f in ARTIFACTS),
                         "run `make hmm-regimes` and commit benchmark/results/hmm_regimes/")
    def test_regen_matches_committed(self):
        hm = _load()
        with tempfile.TemporaryDirectory() as tmp:
            hm.OUT_DIR = Path(tmp)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(hm.main(), 0)
            for name in ARTIFACTS:
                self.assertEqual((hm.OUT_DIR / name).read_bytes(), (OUT / name).read_bytes(), name)

    @unittest.skipUnless((OUT / "states.csv").exists(), "run `make hmm-regimes`")
    def test_committed_states_cover_fixture_and_hand_windows(self):
        import pandas as pd

        hm = _load()
        st = pd.read_csv(OUT / "states.csv", keep_default_na=False)
        px = pd.read_csv(REPO / "benchmark" / "prices" / "paper" / "IDX_BVSP.csv")
        self.assertEqual(list(st.columns), ["date", "close", "logret", "p_turbulent", "state", "hand_regime"])
        self.assertEqual(len(st), len(px) - 1)
        self.assertTrue(set(st.state.unique()) <= {0, 1})
        self.assertTrue(((st.p_turbulent >= 0) & (st.p_turbulent <= 1)).all())
        for name, (start, end) in hm.REGIMES.items():
            self.assertTrue((st.loc[(st.date >= start) & (st.date <= end), "hand_regime"] == name).all(), name)
        al = pd.read_csv(OUT / "alignment.csv")
        self.assertEqual(al.regime.tolist()[:4], hm.REGIME_ORDER)
        self.assertTrue(((al.purity >= 0.5) & (al.purity <= 1.0)).all())


if __name__ == "__main__":
    unittest.main()
