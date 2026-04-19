"""Indicator computation with stockstats → openbb.technical / pandas_ta fallback.

Design (Sprint 3):
- Primary engine: stockstats (existing behavior preserved).
- Fallback trigger: stockstats raises OR returns empty / all-NaN series.
- Standard 11 indicators fall back to openbb.technical (router direct import).
- vwma and mfi fall back to pandas_ta (openbb.technical has no equivalent).
- Fallback receives the already-fetched OHLCV dataframe; no re-fetch.
- Returns a pd.Series indexed by date string (YYYY-MM-DD).
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

_PANDAS_TA_ONLY = {"vwma", "mfi"}

COLUMN_MAP = {
    "macd": "MACD_12_26_9",
    "macds": "MACDs_12_26_9",
    "macdh": "MACDh_12_26_9",
    "boll": "BBM_20_2.0_2.0",
    "boll_ub": "BBU_20_2.0_2.0",
    "boll_lb": "BBL_20_2.0_2.0",
}


def _engine_for(indicator: str) -> str:
    return "pandas_ta" if indicator in _PANDAS_TA_ONLY else "openbb.technical"


def _date_series(df: pd.DataFrame) -> pd.Series:
    dates = pd.to_datetime(df["Date"], errors="coerce")
    return dates.dt.strftime("%Y-%m-%d")


def _try_stockstats(df: pd.DataFrame, indicator: str) -> pd.Series:
    from stockstats import wrap

    wrapped = wrap(df.copy())
    values = wrapped[indicator]
    series = pd.Series(values.values, index=_date_series(df).values, name=indicator)
    return series


def _openbb_fallback(df: pd.DataFrame, indicator: str) -> pd.Series:
    import pandas_ta as ta

    ohlcv = df[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
    ohlcv.columns = ["date", "open", "high", "low", "close", "volume"]
    ohlcv = ohlcv.set_index(pd.to_datetime(ohlcv["date"]))
    ohlcv = ohlcv.drop(columns=["date"])

    if indicator == "close_50_sma":
        out = ta.sma(ohlcv["close"], length=50)
    elif indicator == "close_200_sma":
        out = ta.sma(ohlcv["close"], length=200)
    elif indicator == "close_10_ema":
        out = ta.ema(ohlcv["close"], length=10)
    elif indicator == "rsi":
        out = ta.rsi(ohlcv["close"], length=14)
    elif indicator == "atr":
        out = ta.atr(ohlcv["high"], ohlcv["low"], ohlcv["close"], length=14)
    elif indicator in ("macd", "macds", "macdh"):
        macd_df = ta.macd(ohlcv["close"], fast=12, slow=26, signal=9)
        out = macd_df[COLUMN_MAP[indicator]]
    elif indicator in ("boll", "boll_ub", "boll_lb"):
        bb_df = ta.bbands(ohlcv["close"], length=20, std=2.0)
        out = bb_df[COLUMN_MAP[indicator]]
    else:
        raise ValueError(f"No openbb.technical fallback mapping for {indicator!r}")

    return _align_to_dates(out, df)


def _pandas_ta_fallback(df: pd.DataFrame, indicator: str) -> pd.Series:
    import pandas_ta as ta

    close = df["Close"]
    volume = df["Volume"]
    if indicator == "vwma":
        out = ta.vwma(close, volume, length=20)
    elif indicator == "mfi":
        out = ta.mfi(df["High"], df["Low"], close, volume, length=14)
    else:
        raise ValueError(f"No pandas_ta fallback mapping for {indicator!r}")

    return _align_to_dates(out, df)


def _align_to_dates(values: pd.Series, df: pd.DataFrame) -> pd.Series:
    dates = _date_series(df).values
    arr = values.values if hasattr(values, "values") else values
    if len(arr) != len(dates):
        arr = list(arr) + [float("nan")] * (len(dates) - len(arr))
        arr = arr[: len(dates)]
    return pd.Series(arr, index=dates, name=values.name if hasattr(values, "name") else None)


def _is_insufficient(series: Optional[pd.Series]) -> bool:
    if series is None:
        return True
    if len(series) == 0:
        return True
    return series.isna().all()


def compute_indicator_with_fallback(
    df: pd.DataFrame,
    indicator: str,
    symbol: str = "",
    curr_date: str = "",
) -> pd.Series:
    """Compute indicator with stockstats → fallback chain.

    df must contain columns: Date (datetime-like), Open, High, Low, Close, Volume.
    Returns a Series indexed by date string (YYYY-MM-DD).
    """
    engine = _engine_for(indicator)
    try:
        series = _try_stockstats(df, indicator)
        if not _is_insufficient(series):
            return series
        logger.warning(
            "stockstats returned empty/NaN for %s (symbol=%s, date=%s), falling back to %s",
            indicator, symbol, curr_date, engine,
        )
    except Exception as e:
        logger.warning(
            "stockstats failed for %s (symbol=%s, date=%s), falling back to %s (error: %s)",
            indicator, symbol, curr_date, engine, e,
        )

    if indicator in _PANDAS_TA_ONLY:
        return _pandas_ta_fallback(df, indicator)
    return _openbb_fallback(df, indicator)
