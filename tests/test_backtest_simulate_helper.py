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

    def test_non_positive_entry_does_not_wipe_cash(self):
        # price <= 0 cannot be bought; cash must stay put, including after the signal ends.
        idx = pd.date_range("2024-01-01", periods=2, freq="B")
        prices = pd.DataFrame({"Close": [0.0, 10.0]}, index=idx)
        signals = pd.Series([True, False], index=idx)

        equity = _simulate(prices, signals, initial_capital=1_000.0)

        self.assertEqual(list(equity.values), [1_000.0, 1_000.0])

    def test_negative_entry_close_keeps_cash(self):
        # `price != 0` still buys a negative close (the zero-price test does not
        # catch it). The flat bar must stay in cash, not mark that short.
        idx = pd.date_range("2024-01-01", periods=2, freq="B")
        prices = pd.DataFrame({"Close": [-5.0, 10.0]}, index=idx)
        signals = pd.Series([True, False], index=idx)

        equity = _simulate(prices, signals, initial_capital=1_000.0)

        self.assertEqual(list(equity.values), [1_000.0, 1_000.0])

    def test_entry_defers_to_the_next_positive_close(self):
        # A zero or NaN bar does not consume the entry. The fill happens on the
        # next positive close; the bar after that marks the position.
        idx = pd.date_range("2024-01-01", periods=3, freq="B")
        signals = pd.Series([True, True, True], index=idx)
        for bad in (0.0, float("nan")):
            with self.subTest(bad=bad):
                prices = pd.DataFrame({"Close": [bad, 10.0, 12.0]}, index=idx)
                equity = _simulate(prices, signals, initial_capital=1_000.0)
                if bad == 0.0:
                    self.assertEqual(list(equity.values), [1_000.0, 1_000.0, 1_200.0])
                else:
                    self.assertEqual(list(equity.iloc[1:].values), [1_000.0, 1_200.0])


if __name__ == "__main__":
    unittest.main()
