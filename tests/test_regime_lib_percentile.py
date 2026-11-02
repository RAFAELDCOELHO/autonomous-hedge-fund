from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import unittest

import pandas as pd


REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"


def _load_regime_lib():
    spec = importlib.util.spec_from_file_location("regime_lib", SCRIPTS / "regime_lib.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PercentileOfTests(unittest.TestCase):
    def setUp(self):
        self.rl = _load_regime_lib()

    def test_midrank_percentile_for_duplicate_values(self):
        out = self.rl.percentile_of([1.0, 2.0, 2.0, 3.0], 2.0)
        # (<2) + 0.5*(==2) => (1 + 1) / 4 = 0.5
        self.assertAlmostEqual(out, 50.0, places=9)

    def test_ignores_nan_inputs_when_ranking(self):
        out = self.rl.percentile_of([float("nan"), 1.0, 3.0], 2.0)
        # NaN is dropped, leaving [1, 3]; 2 sits exactly between.
        self.assertAlmostEqual(out, 50.0, places=9)

    def test_all_nan_input_returns_nan(self):
        out = self.rl.percentile_of([float("nan"), float("nan")], 1.0)
        self.assertTrue(math.isnan(out))

    def test_non_positive_entry_does_not_wipe_cash(self):
        idx = pd.date_range("2024-01-01", periods=2, freq="B")
        closes = pd.Series([0.0, 10.0], index=idx)
        held = pd.Series([True, False], index=idx)

        equity = self.rl._simulate_signal(closes, held, 1_000.0)

        self.assertEqual(list(equity.values), [1_000.0, 1_000.0])

    def test_negative_entry_close_keeps_cash(self):
        # `p != 0` still buys a negative close. The flat bar must stay in cash.
        idx = pd.date_range("2024-01-01", periods=2, freq="B")
        closes = pd.Series([-5.0, 10.0], index=idx)
        held = pd.Series([True, False], index=idx)

        equity = self.rl._simulate_signal(closes, held, 1_000.0)

        self.assertEqual(list(equity.values), [1_000.0, 1_000.0])

    def test_entry_defers_to_the_next_positive_close(self):
        # A zero or NaN bar does not consume the entry. The fill happens on the
        # next positive close; the bar after that marks the position.
        idx = pd.date_range("2024-01-01", periods=3, freq="B")
        held = pd.Series([True, True, True], index=idx)
        for bad in (0.0, float("nan")):
            with self.subTest(bad=bad):
                closes = pd.Series([bad, 10.0, 12.0], index=idx)
                equity = self.rl._simulate_signal(closes, held, 1_000.0)
                if bad == 0.0:
                    self.assertEqual(list(equity.values), [1_000.0, 1_000.0, 1_200.0])
                else:
                    self.assertEqual(list(equity.iloc[1:].values), [1_000.0, 1_200.0])


if __name__ == "__main__":
    unittest.main()
