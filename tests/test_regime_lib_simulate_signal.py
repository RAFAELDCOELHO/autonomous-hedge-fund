"""A non-positive close must not fill an exit in the signal book.

``_simulate_signal`` already skips a buy when the close is not positive.
An exit used to fill anyway: close 0 set cash to 0, a negative close made
cash negative, and NaN wiped the curve. Shares and cash have to stay so a
later positive close can exit. Positive closes are unchanged, so the
committed paper artifacts stay the same.
"""

from __future__ import annotations

import importlib.util
import math
import unittest
from pathlib import Path

import pandas as pd


def _load_regime_lib():
    path = Path(__file__).resolve().parents[1] / "scripts" / "regime_lib.py"
    spec = importlib.util.spec_from_file_location("regime_lib_signal_exit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SimulateSignalNonpositiveExitTests(unittest.TestCase):
    def setUp(self):
        self.rl = _load_regime_lib()

    def _equity(self, closes, in_pos, capital=1_000.0):
        index = pd.date_range("2024-01-02", periods=len(closes), freq="B")
        return self.rl._simulate_signal(
            pd.Series(closes, index=index), pd.Series(in_pos, index=index), capital
        )

    def test_positive_exit_still_exits_at_that_close(self):
        equity = self._equity([10.0, 12.5], [True, False])
        self.assertEqual([float(v) for v in equity], [1_000.0, 1_250.0])

    def test_exit_at_zero_keeps_shares_for_a_later_positive_close(self):
        # 100 shares bought at 10. The zero close does not exit. 12 does,
        # and the following 9 stays at that cash.
        equity = self._equity([10.0, 0.0, 12.0, 9.0], [True, False, False, False])
        self.assertEqual(float(equity.iloc[2]), 1_200.0)
        self.assertEqual(float(equity.iloc[-1]), 1_200.0)

    def test_exit_at_negative_close_does_not_realize_it(self):
        equity = self._equity([10.0, -4.0, 11.0, 7.0], [True, False, False, False])
        # Old book: cash = 100 * -4 = -400, and the position is gone.
        self.assertEqual(float(equity.iloc[-1]), 1_100.0)

    def test_exit_at_nan_close_does_not_wipe_the_curve(self):
        equity = self._equity(
            [10.0, float("nan"), 11.0, 7.0], [True, False, False, False]
        )
        self.assertTrue(math.isnan(float(equity.iloc[1])))
        self.assertEqual(float(equity.iloc[-1]), 1_100.0)


if __name__ == "__main__":
    unittest.main()
