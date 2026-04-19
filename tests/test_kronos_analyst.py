"""Tests for Kronos analyst integration.

No real model weights are downloaded; _load_model is mocked in every test
to either return a fake predictor or None.
"""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd


def _make_ohlcv(n: int = 120) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    rng = np.random.default_rng(7)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    df = pd.DataFrame({
        "Open": close + rng.normal(0, 0.3, n),
        "High": close + rng.uniform(0.1, 1.5, n),
        "Low": close - rng.uniform(0.1, 1.5, n),
        "Close": close,
        "Volume": rng.integers(1_000_000, 5_000_000, n).astype(float),
    }, index=dates)
    return df


def _fake_pred_df(last_close: float, pct_change: float, pred_len: int) -> pd.DataFrame:
    target = last_close * (1 + pct_change)
    closes = np.linspace(last_close, target, pred_len + 1)[1:]
    return pd.DataFrame({
        "open": closes,
        "high": closes * 1.01,
        "low": closes * 0.99,
        "close": closes,
        "volume": np.full(pred_len, 1_000_000.0),
        "amount": np.full(pred_len, 1_000_000.0),
    })


class KronosSignalTests(unittest.TestCase):
    def setUp(self):
        # Clear predictor cache between tests
        from tradingagents.dataflows import kronos_analyst
        kronos_analyst._PREDICTOR_CACHE.clear()
        self.mod = kronos_analyst
        self.df = _make_ohlcv()
        self.last_close = float(self.df["Close"].iloc[-1])

    def _fake_predictor(self, pct_change: float):
        predictor = MagicMock()
        predictor.predict.side_effect = lambda df, x_timestamp, y_timestamp, pred_len, verbose=False: (
            _fake_pred_df(self.last_close, pct_change, pred_len)
        )
        return predictor

    def test_signal_buy_when_forecast_above_threshold(self):
        with patch.object(self.mod, "_load_model", return_value=self._fake_predictor(0.03)):
            out = self.mod.get_kronos_signal(self.df, "AAPL", pred_len=5)
        self.assertEqual(out["signal"], "BUY")
        self.assertGreater(out["forecast_return_pct"], 2.0)
        self.assertEqual(out["model"], "kronos-small")
        self.assertEqual(out["pred_len"], 5)

    def test_signal_sell_when_forecast_below_negative_threshold(self):
        with patch.object(self.mod, "_load_model", return_value=self._fake_predictor(-0.03)):
            out = self.mod.get_kronos_signal(self.df, "AAPL", pred_len=5)
        self.assertEqual(out["signal"], "SELL")
        self.assertLess(out["forecast_return_pct"], -2.0)

    def test_signal_hold_when_forecast_within_threshold(self):
        with patch.object(self.mod, "_load_model", return_value=self._fake_predictor(0.01)):
            out = self.mod.get_kronos_signal(self.df, "AAPL", pred_len=5)
        self.assertEqual(out["signal"], "HOLD")

    def test_threshold_is_configurable(self):
        # 1% change, threshold=0.005 → BUY
        with patch.object(self.mod, "_load_model", return_value=self._fake_predictor(0.01)):
            out = self.mod.get_kronos_signal(self.df, "AAPL", pred_len=5, threshold=0.005)
        self.assertEqual(out["signal"], "BUY")

    def test_returns_none_when_model_load_fails(self):
        with patch.object(self.mod, "_load_model", return_value=None):
            out = self.mod.get_kronos_signal(self.df, "AAPL", pred_len=5)
        self.assertIsNone(out)

    def test_returns_none_when_predict_raises(self):
        predictor = MagicMock()
        predictor.predict.side_effect = RuntimeError("OOM")
        with patch.object(self.mod, "_load_model", return_value=predictor):
            out = self.mod.get_kronos_signal(self.df, "AAPL", pred_len=5)
        self.assertIsNone(out)

    def test_device_resolution_prefers_mps_when_available(self):
        fake_torch = MagicMock()
        fake_torch.backends.mps.is_available.return_value = True
        with patch.dict("sys.modules", {"torch": fake_torch}):
            self.assertEqual(self.mod._resolve_device(), "mps")

    def test_device_resolution_falls_back_to_cpu(self):
        fake_torch = MagicMock()
        fake_torch.backends.mps.is_available.return_value = False
        with patch.dict("sys.modules", {"torch": fake_torch}):
            self.assertEqual(self.mod._resolve_device(), "cpu")

    def test_ohlcv_column_case_insensitive(self):
        # DataFrame with mixed-case columns must still work
        df = self.df.copy()
        with patch.object(self.mod, "_load_model", return_value=self._fake_predictor(0.025)):
            out = self.mod.get_kronos_signal(df, "AAPL", pred_len=5)
        self.assertEqual(out["signal"], "BUY")

    def test_missing_required_columns_returns_none(self):
        df = self.df.drop(columns=["Open"])
        with patch.object(self.mod, "_load_model", return_value=self._fake_predictor(0.03)):
            out = self.mod.get_kronos_signal(df, "AAPL", pred_len=5)
        self.assertIsNone(out)

    def test_output_contains_expected_keys(self):
        with patch.object(self.mod, "_load_model", return_value=self._fake_predictor(0.03)):
            out = self.mod.get_kronos_signal(self.df, "AAPL", pred_len=5)
        for key in ("signal", "forecast_return_pct", "predicted_close", "current_close", "model", "pred_len", "device"):
            self.assertIn(key, out)


if __name__ == "__main__":
    unittest.main()
