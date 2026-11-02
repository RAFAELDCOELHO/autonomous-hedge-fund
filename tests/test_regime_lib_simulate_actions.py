"""A non-positive close must not fill a BUY in the action-stream book.

``_simulate_actions`` refuses a BUY unless the close is positive (``p > 0``).
These tests kill the mutant that drops that guard. A fill at 0, a negative
close, or NaN would spend the cash — and for 0 or NaN poison the book with
inf or NaN — so a later BUY at a real close could not open a normal long.
"""

from __future__ import annotations

import importlib.util
import math
import unittest
from pathlib import Path

import pandas as pd


def _load_regime_lib():
    path = Path(__file__).resolve().parents[1] / "scripts" / "regime_lib.py"
    spec = importlib.util.spec_from_file_location("regime_lib_simulate_actions", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SimulateActionsBuyGuardTests(unittest.TestCase):
    def setUp(self):
        self.rl = _load_regime_lib()

    def _equity(self, closes, actions, capital=1_000.0):
        index = pd.date_range("2024-01-02", periods=len(closes), freq="B")
        return self.rl._simulate_actions(pd.Series(closes, index=index), actions, capital)

    def _assert_later_buy_fills_normally(self, equity):
        """BUY at 10 then HOLD at 15 is an all-in long: 1000, then 1500.

        That path exists only if the earlier BUY left shares at 0 and cash
        at 1000. Both marks are finite (no NaN, no inf).
        """
        self.assertEqual([float(v) for v in equity.iloc[1:]], [1_000.0, 1_500.0])
        for value in equity.iloc[1:]:
            self.assertTrue(math.isfinite(float(value)))

    def test_buy_at_zero_close_does_not_fill_and_later_buy_does(self):
        equity = self._equity([0.0, 10.0, 15.0], ["BUY", "BUY", "HOLD"])
        # Close 0 marks at the untouched cash. It must not be inf or NaN.
        self.assertEqual(float(equity.iloc[0]), 1_000.0)
        self.assertTrue(math.isfinite(float(equity.iloc[0])))
        self._assert_later_buy_fills_normally(equity)

    def test_buy_at_negative_close_does_not_fill_and_later_buy_does(self):
        equity = self._equity([-4.0, 10.0, 15.0], ["BUY", "BUY", "HOLD"])
        # A fill at -4 would also mark this bar at 1000, then mark 10 at -2500
        # and skip the later BUY. The untouched book stays at 1000 here.
        self.assertEqual(float(equity.iloc[0]), 1_000.0)
        self.assertTrue(math.isfinite(float(equity.iloc[0])))
        self._assert_later_buy_fills_normally(equity)

    def test_buy_at_nan_close_does_not_fill_and_later_buy_does(self):
        equity = self._equity([float("nan"), 10.0, 15.0], ["BUY", "BUY", "HOLD"])
        # A NaN close marks to NaN even while flat (0 * NaN). The book is not
        # poisoned: the next bars are a normal long, not NaN.
        self.assertTrue(math.isnan(float(equity.iloc[0])))
        self.assertFalse(math.isinf(float(equity.iloc[0])))
        self._assert_later_buy_fills_normally(equity)


if __name__ == "__main__":
    unittest.main()
