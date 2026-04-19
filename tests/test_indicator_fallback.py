"""Tests for stockstats → openbb.technical / pandas_ta fallback chain."""

import sys
import types
import unittest
from unittest.mock import patch

import pandas as pd
import numpy as np


def _make_ohlcv(n: int = 250) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    rng = np.random.default_rng(42)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    high = close + rng.uniform(0.1, 2.0, n)
    low = close - rng.uniform(0.1, 2.0, n)
    open_ = close + rng.normal(0, 0.5, n)
    volume = rng.integers(1_000_000, 10_000_000, n).astype(float)
    return pd.DataFrame({
        "Date": dates,
        "Open": open_,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": volume,
    })


class IndicatorFallbackTests(unittest.TestCase):
    def setUp(self):
        from tradingagents.dataflows import indicator_fallback
        self.mod = indicator_fallback
        self.df = _make_ohlcv()

    def test_happy_path_stockstats_returns_series(self):
        series = self.mod.compute_indicator_with_fallback(self.df, "close_50_sma")
        self.assertIsInstance(series, pd.Series)
        self.assertEqual(len(series), len(self.df))
        self.assertFalse(series.isna().all())

    def test_index_is_date_string(self):
        series = self.mod.compute_indicator_with_fallback(self.df, "rsi")
        first = series.index[0]
        self.assertRegex(first, r"^\d{4}-\d{2}-\d{2}$")

    def test_fallback_triggered_on_exception(self):
        with patch.object(self.mod, "_try_stockstats", side_effect=RuntimeError("boom")):
            with patch.object(self.mod, "_openbb_fallback", return_value=pd.Series([1, 2], index=["a", "b"])) as fb:
                out = self.mod.compute_indicator_with_fallback(self.df, "close_50_sma")
                fb.assert_called_once()
                self.assertEqual(list(out.values), [1, 2])

    def test_fallback_triggered_on_all_nan(self):
        nan_series = pd.Series([np.nan, np.nan], index=["a", "b"])
        with patch.object(self.mod, "_try_stockstats", return_value=nan_series):
            with patch.object(self.mod, "_openbb_fallback", return_value=pd.Series([5.0])) as fb:
                self.mod.compute_indicator_with_fallback(self.df, "rsi")
                fb.assert_called_once()

    def test_fallback_triggered_on_empty(self):
        empty = pd.Series([], dtype=float)
        with patch.object(self.mod, "_try_stockstats", return_value=empty):
            with patch.object(self.mod, "_openbb_fallback", return_value=pd.Series([1.0])) as fb:
                self.mod.compute_indicator_with_fallback(self.df, "macd")
                fb.assert_called_once()

    def test_vwma_routes_to_pandas_ta_fallback(self):
        with patch.object(self.mod, "_try_stockstats", side_effect=RuntimeError("boom")):
            with patch.object(self.mod, "_pandas_ta_fallback", return_value=pd.Series([1.0])) as ta_fb, \
                 patch.object(self.mod, "_openbb_fallback", return_value=pd.Series([9.0])) as obb_fb:
                self.mod.compute_indicator_with_fallback(self.df, "vwma")
                ta_fb.assert_called_once()
                obb_fb.assert_not_called()

    def test_mfi_routes_to_pandas_ta_fallback(self):
        with patch.object(self.mod, "_try_stockstats", side_effect=RuntimeError("boom")):
            with patch.object(self.mod, "_pandas_ta_fallback", return_value=pd.Series([1.0])) as ta_fb, \
                 patch.object(self.mod, "_openbb_fallback", return_value=pd.Series([9.0])) as obb_fb:
                self.mod.compute_indicator_with_fallback(self.df, "mfi")
                ta_fb.assert_called_once()
                obb_fb.assert_not_called()

    def test_column_map_covers_macd_and_bbands_variants(self):
        expected = {
            "macd": "MACD_12_26_9",
            "macds": "MACDs_12_26_9",
            "macdh": "MACDh_12_26_9",
            "boll": "BBM_20_2.0_2.0",
            "boll_ub": "BBU_20_2.0_2.0",
            "boll_lb": "BBL_20_2.0_2.0",
        }
        self.assertEqual(self.mod.COLUMN_MAP, expected)

    def test_openbb_fallback_macd_extracts_correct_column(self):
        series = self.mod._openbb_fallback(self.df, "macd")
        self.assertIsInstance(series, pd.Series)
        self.assertEqual(len(series), len(self.df))

    def test_openbb_fallback_bbands_upper(self):
        upper = self.mod._openbb_fallback(self.df, "boll_ub")
        middle = self.mod._openbb_fallback(self.df, "boll")
        valid = (~upper.isna()) & (~middle.isna())
        self.assertTrue((upper[valid] >= middle[valid]).all())

    def test_pandas_ta_fallback_vwma(self):
        series = self.mod._pandas_ta_fallback(self.df, "vwma")
        self.assertIsInstance(series, pd.Series)
        self.assertFalse(series.isna().all())

    def test_pandas_ta_fallback_mfi(self):
        series = self.mod._pandas_ta_fallback(self.df, "mfi")
        self.assertIsInstance(series, pd.Series)
        valid = series.dropna()
        self.assertTrue(((valid >= 0) & (valid <= 100)).all())


if __name__ == "__main__":
    unittest.main()
