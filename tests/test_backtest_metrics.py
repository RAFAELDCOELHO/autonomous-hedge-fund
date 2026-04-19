"""Tests for the Sprint 6 backtest module.

All tests use synthetic data — no network, no yfinance, no real agent.
"""

from __future__ import annotations

import math
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from tradingagents.backtest import (
    BuyAndHold,
    ExtendedMetricsCalculator,
    MACDStrategy,
    SMACrossStrategy,
    build_comparison_table,
    format_table_markdown,
    run_agent_strategy,
    run_strategy,
)


def _linear_prices(n: int, start: float = 100.0, step: float = 1.0) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    close = np.arange(n, dtype=float) * step + start
    return pd.DataFrame({"Close": close}, index=idx)


class ExtendedMetricsTests(unittest.TestCase):
    def test_empty_series_returns_none(self):
        calc = ExtendedMetricsCalculator()
        out = calc.compute(pd.Series([], dtype=float))
        self.assertIsNone(out["cr"])
        self.assertIsNone(out["ar"])
        self.assertIsNone(out["sharpe"])
        self.assertEqual(out["n_days"], 0)

    def test_cr_and_ar_on_known_series(self):
        # 252 business days, equity goes 100 -> 110 -> linear-ish
        idx = pd.date_range("2024-01-01", periods=252, freq="B")
        equity = pd.Series(np.linspace(100.0, 110.0, 252), index=idx)
        calc = ExtendedMetricsCalculator()
        m = calc.compute(equity)
        self.assertAlmostEqual(m["cr"], 0.10, places=6)
        # n_days == 252 → ar == cr
        self.assertAlmostEqual(m["ar"], 0.10, places=6)

    def test_mdd_on_drawdown_series(self):
        idx = pd.date_range("2024-01-01", periods=5, freq="B")
        equity = pd.Series([100.0, 120.0, 90.0, 95.0, 110.0], index=idx)
        m = ExtendedMetricsCalculator().compute(equity)
        # Peak 120, trough 90 -> -0.25
        self.assertAlmostEqual(m["mdd"], -0.25, places=6)
        self.assertEqual(m["mdd_date"], idx[2].strftime("%Y-%m-%d"))

    def test_sharpe_is_number_on_noisy_series(self):
        rng = np.random.default_rng(42)
        rets = rng.normal(loc=0.001, scale=0.01, size=252)
        equity = pd.Series(100.0 * np.cumprod(1 + rets),
                           index=pd.date_range("2024-01-01", periods=252, freq="B"))
        m = ExtendedMetricsCalculator().compute(equity)
        self.assertIsNotNone(m["sharpe"])
        self.assertFalse(math.isnan(m["sharpe"]))


class BaselineStrategyTests(unittest.TestCase):
    def test_buy_and_hold_matches_price_ratio(self):
        prices = _linear_prices(50, start=100.0, step=1.0)
        eq = BuyAndHold().run(prices, 100_000.0)
        expected_final = 100_000.0 * (prices["Close"].iloc[-1] / prices["Close"].iloc[0])
        self.assertAlmostEqual(eq.iloc[-1], expected_final, places=2)
        self.assertEqual(len(eq), len(prices))

    def test_macd_returns_equity_curve(self):
        prices = _linear_prices(200, start=100.0, step=0.5)
        eq = MACDStrategy().run(prices, 10_000.0)
        self.assertEqual(len(eq), len(prices))
        self.assertTrue((eq >= 0).all())

    def test_sma_cross_no_signal_when_too_short(self):
        prices = _linear_prices(30)
        eq = SMACrossStrategy().run(prices, 10_000.0)
        # Not enough data for 200-day SMA; stays flat in cash
        self.assertTrue(np.allclose(eq.values, 10_000.0))

    def test_sma_cross_goes_long_on_uptrend(self):
        prices = _linear_prices(300, start=100.0, step=0.5)
        eq = SMACrossStrategy().run(prices, 10_000.0)
        # On a steady uptrend once both SMAs exist, fast>slow triggers BUY
        self.assertGreater(eq.iloc[-1], 10_000.0)


class RunnerTests(unittest.TestCase):
    def test_run_strategy_uses_loaded_prices(self):
        prices = _linear_prices(20, start=100.0, step=1.0)
        with patch("tradingagents.backtest.runner.load_ohlcv") as load:
            df = prices.reset_index().rename(columns={"index": "Date"})
            load.return_value = df
            eq = run_strategy(BuyAndHold(), "AAPL", "2024-01-01", "2024-12-31", 1_000.0)
        self.assertEqual(len(eq), 20)
        self.assertAlmostEqual(eq.iloc[-1], 1_000.0 * (119.0 / 100.0), places=2)

    def test_run_agent_strategy_full_position(self):
        prices = _linear_prices(10, start=100.0, step=1.0)
        with patch("tradingagents.backtest.runner.load_ohlcv") as load:
            df = prices.reset_index().rename(columns={"index": "Date"})
            load.return_value = df

            # BUY on day 0, SELL on day 9
            def decider(date_str, _window):
                if date_str == prices.index[0].strftime("%Y-%m-%d"):
                    return "BUY"
                if date_str == prices.index[-1].strftime("%Y-%m-%d"):
                    return "SELL"
                return "HOLD"

            eq = run_agent_strategy(decider, "AAPL", "2024-01-01", "2024-12-31", 1_000.0)
        # Bought at 100, sold at 109 → final cash = 1000 * 109/100 = 1090
        self.assertAlmostEqual(eq.iloc[-1], 1_090.0, places=2)

    def test_run_agent_strategy_hold_keeps_cash_flat(self):
        prices = _linear_prices(5, start=100.0, step=1.0)
        with patch("tradingagents.backtest.runner.load_ohlcv") as load:
            df = prices.reset_index().rename(columns={"index": "Date"})
            load.return_value = df
            eq = run_agent_strategy(lambda d, w: "HOLD", "AAPL", "2024-01-01", "2024-12-31", 500.0)
        self.assertTrue(np.allclose(eq.values, 500.0))


class ReportTests(unittest.TestCase):
    def test_build_comparison_table_columns(self):
        idx = pd.date_range("2024-01-01", periods=10, freq="B")
        curves = {
            "A": pd.Series(np.linspace(100, 110, 10), index=idx),
            "B": pd.Series(np.linspace(100, 90, 10), index=idx),
        }
        df = build_comparison_table(curves)
        self.assertEqual(list(df.columns), ["Strategy", "CR (%)", "AR (%)", "Sharpe", "MDD (%)"])
        self.assertEqual(len(df), 2)

    def test_format_table_markdown_has_header_separator(self):
        idx = pd.date_range("2024-01-01", periods=5, freq="B")
        curves = {"A": pd.Series(np.linspace(100, 105, 5), index=idx)}
        md = format_table_markdown(build_comparison_table(curves))
        self.assertIn("| Strategy |", md)
        self.assertIn("| --- |", md)
        self.assertIn("| A |", md)


if __name__ == "__main__":
    unittest.main()
