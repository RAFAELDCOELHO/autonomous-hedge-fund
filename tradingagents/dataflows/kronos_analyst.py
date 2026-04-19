"""Kronos forecasting integration for TradingAgents.

Kronos (https://github.com/NeoQuasar/Kronos) is a foundation model for
time-series forecasting. This module exposes a single entry point,
`get_kronos_signal`, that turns a short-horizon price forecast into a
directional trading signal. It is designed to degrade gracefully: any
failure (missing checkpoint, HF offline, OOM, device error) returns
None and logs a warning so the TradingAgents pipeline can continue
with classical indicators only.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

_KRONOS_REPO = os.path.expanduser("~/Downloads/Kronos-master")
_MODEL_REPO_ID = "NeoQuasar/Kronos-small"
_TOKENIZER_REPO_ID = "NeoQuasar/Kronos-Tokenizer-base"

_PREDICTOR_CACHE: dict = {}


def _resolve_device() -> str:
    """Prefer MPS on Apple Silicon; fall back to CPU."""
    try:
        import torch

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    except Exception as e:
        logger.warning("torch unavailable (%s); Kronos will not run", e)
        return "cpu"


def _load_model():
    """Lazy-load KronosPredictor singleton. Returns None on any failure."""
    if "predictor" in _PREDICTOR_CACHE:
        return _PREDICTOR_CACHE["predictor"]

    try:
        if _KRONOS_REPO not in sys.path:
            sys.path.insert(0, _KRONOS_REPO)
        from model import Kronos, KronosPredictor, KronosTokenizer

        tokenizer = KronosTokenizer.from_pretrained(_TOKENIZER_REPO_ID)
        model = Kronos.from_pretrained(_MODEL_REPO_ID)
        device = _resolve_device()
        predictor = KronosPredictor(model, tokenizer, device=device)
        _PREDICTOR_CACHE["predictor"] = predictor
        _PREDICTOR_CACHE["device"] = device
        logger.info("Kronos loaded on device=%s", device)
        return predictor
    except Exception as e:
        logger.warning("Kronos model unavailable (%s); signal will be None", e)
        _PREDICTOR_CACHE["predictor"] = None
        return None


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with lowercase open/high/low/close/volume columns."""
    rename = {}
    for col in df.columns:
        lc = col.lower()
        if lc in ("open", "high", "low", "close", "volume"):
            rename[col] = lc
    out = df.rename(columns=rename).copy()
    required = {"open", "high", "low", "close"}
    missing = required - set(out.columns)
    if missing:
        raise ValueError(f"OHLCV missing columns: {sorted(missing)}")
    return out


def _extract_timestamps(df: pd.DataFrame) -> pd.DatetimeIndex:
    if isinstance(df.index, pd.DatetimeIndex):
        return df.index
    if "Date" in df.columns:
        return pd.DatetimeIndex(pd.to_datetime(df["Date"]))
    if "date" in df.columns:
        return pd.DatetimeIndex(pd.to_datetime(df["date"]))
    raise ValueError("OHLCV has no DatetimeIndex and no 'Date' column")


def get_kronos_signal(
    ohlcv_df: pd.DataFrame,
    ticker: str,
    pred_len: int = 5,
    threshold: float = 0.02,
) -> Optional[dict]:
    """Return a directional signal from a Kronos short-horizon forecast.

    Parameters
    ----------
    ohlcv_df : DataFrame with OHLCV columns (case-insensitive) and either a
        DatetimeIndex or a 'Date' column.
    ticker : used for logging only.
    pred_len : forecast horizon in business days.
    threshold : |forecast return| triggering BUY/SELL. Default 0.02 (±2%).

    Returns
    -------
    dict with keys: signal, forecast_return_pct, predicted_close,
    current_close, model, pred_len, device.  None if Kronos is unavailable
    or the forecast raises.
    """
    predictor = _load_model()
    if predictor is None:
        return None

    try:
        df_lower = _normalize_ohlcv(ohlcv_df)
        x_timestamp = _extract_timestamps(ohlcv_df)
        last_ts = x_timestamp[-1]
        y_timestamp = pd.bdate_range(
            start=last_ts + pd.Timedelta(days=1), periods=pred_len
        )

        x_ts_series = pd.Series(x_timestamp)
        y_ts_series = pd.Series(y_timestamp)

        pred_df = predictor.predict(
            df=df_lower[["open", "high", "low", "close", "volume"]]
            if "volume" in df_lower.columns
            else df_lower[["open", "high", "low", "close"]],
            x_timestamp=x_ts_series,
            y_timestamp=y_ts_series,
            pred_len=pred_len,
            verbose=False,
        )

        predicted_close = float(pred_df["close"].iloc[-1])
        current_close = float(df_lower["close"].iloc[-1])
        forecast_return = (predicted_close - current_close) / current_close

        if forecast_return > threshold:
            signal = "BUY"
        elif forecast_return < -threshold:
            signal = "SELL"
        else:
            signal = "HOLD"

        return {
            "signal": signal,
            "forecast_return_pct": round(forecast_return * 100, 4),
            "predicted_close": round(predicted_close, 4),
            "current_close": round(current_close, 4),
            "model": "kronos-small",
            "pred_len": pred_len,
            "device": _PREDICTOR_CACHE.get("device", "unknown"),
        }
    except Exception as e:
        logger.warning("Kronos forecast failed for %s (%s)", ticker, e)
        return None
