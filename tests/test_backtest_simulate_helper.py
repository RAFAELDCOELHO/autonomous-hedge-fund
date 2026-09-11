import unittest

import pandas as pd

from tradingagents.backtest.baselines import _simulate


class BaselineSimulateHelperTests(unittest.TestCase):
    def test_simulate_stays_flat_when_never_in_position(self):
        prices = pd.DataFrame({"Close": [10.0, 11.0, 12.0]}, index=pd.date_range("2024-01-01", periods=3, freq="B"))
        signals = pd.Series([False, False, False], index=prices.index)

        equity = _simulate(prices, signals, initial_capital=1_000.0)

        self.assertEqual(list(equity.values), [1_000.0, 1_000.0, 1_000.0])

    def test_simulate_buys_holds_and_sells_on_state_transitions(self):
        prices = pd.DataFrame({"Close": [10.0, 12.0, 11.0, 14.0]}, index=pd.date_range("2024-01-01", periods=4, freq="B"))
        signals = pd.Series([True, True, False, False], index=prices.index)

        equity = _simulate(prices, signals, initial_capital=1_000.0)

        self.assertEqual(list(equity.values), [1_000.0, 1_200.0, 1_100.0, 1_100.0])


if __name__ == "__main__":
    unittest.main()
