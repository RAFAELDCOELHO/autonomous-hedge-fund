from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import unittest


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


if __name__ == "__main__":
    unittest.main()
