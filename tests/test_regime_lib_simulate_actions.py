"""A non-positive close must not fill a SELL in the action-stream book.

``_simulate_actions`` already refuses a BUY when the close is not positive.
A SELL used to fill anyway: close 0 set cash to 0, a negative close made
cash negative, and NaN wiped the curve. The shares have to stay so a later
positive close can exit. Positive closes are unchanged, so the committed
random-agent artifacts stay the same.
"""

from __future__ import annotations

import importlib.util
import math
import unittest
from pathlib import Path

import pandas as pd


def _load_regime_lib():
    path = Path(__file__).resolve().parents[1] / "scripts" / "regime_lib.py"
    spec = importlib.util.spec_from_file_location("regime_lib", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SimulateActionsNonpositiveSellTests(unittest.TestCase):
    def setUp(self):
        self.rl = _load_regime_lib()

    def _equity(self, closes, actions, capital=1_000.0):
        index = pd.date_range("2024-01-02", periods=len(closes), freq="B")
        return self.rl._simulate_actions(pd.Series(closes, index=index), actions, capital)

    def test_positive_sell_still_exits_at_that_close(self):
        equity = self._equity([10.0, 12.5], ["BUY", "SELL"])
        self.assertEqual([float(v) for v in equity], [1_000.0, 1_250.0])

    def test_sell_at_zero_keeps_shares_for_a_later_positive_close(self):
        equity = self._equity([10.0, 0.0, 12.0], ["BUY", "SELL", "SELL"])
        # 100 shares bought at 10. The zero close does not exit. 12 does.
        self.assertEqual(float(equity.iloc[-1]), 1_200.0)

    def test_sell_at_negative_close_does_not_realize_it(self):
        equity = self._equity([10.0, -4.0, 11.0], ["BUY", "SELL", "HOLD"])
        # Old book: cash = 100 * -4 = -400, and the position is gone.
        self.assertEqual(float(equity.iloc[-1]), 1_100.0)

    def test_sell_at_nan_close_does_not_wipe_the_curve(self):
        equity = self._equity([10.0, float("nan"), 11.0], ["BUY", "SELL", "HOLD"])
        self.assertTrue(math.isnan(float(equity.iloc[1])))
        self.assertEqual(float(equity.iloc[-1]), 1_100.0)


if __name__ == "__main__":
    unittest.main()
