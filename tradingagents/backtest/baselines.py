"""Baseline long-only strategies with binary (all-in / all-cash) positions.

Each strategy exposes:
    .run(prices: pd.DataFrame, initial_capital: float) -> pd.Series

Input `prices` must have a DatetimeIndex and at least a 'Close' column.
Output is an equity curve indexed by the same dates.

Position sizing: full position (100% equity on BUY, 100% cash on SELL).
No shorting. Trades execute at the bar's Close price.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


def _simulate(prices: pd.DataFrame, signals: pd.Series, initial_capital: float) -> pd.Series:
    """Simulate equity curve from a boolean `in_position` signal.

    signals[t] is True iff we should hold the asset into the close of day t.
    Switches between True/False are executed at that day's close.
    """
    closes = prices["Close"].astype(float).values
    in_pos = signals.astype(bool).values
    n = len(closes)

    cash = float(initial_capital)
    shares = 0.0
    equity = np.empty(n, dtype=float)
    prev_state = False

    for i in range(n):
        state = in_pos[i]
        price = closes[i]
        if state and not prev_state:
            shares = cash / price if price > 0 else 0.0
            cash = 0.0
        elif prev_state and not state:
            cash = shares * price
            shares = 0.0
        equity[i] = cash + shares * price
        prev_state = state

    return pd.Series(equity, index=prices.index, name="equity")


class _Strategy(ABC):
    name: str = "base"

    @abstractmethod
    def signals(self, prices: pd.DataFrame) -> pd.Series:
        ...

    def run(self, prices: pd.DataFrame, initial_capital: float) -> pd.Series:
        sig = self.signals(prices)
        return _simulate(prices, sig, initial_capital)


class BuyAndHold(_Strategy):
    name = "Buy & Hold"

    def signals(self, prices: pd.DataFrame) -> pd.Series:
        return pd.Series(True, index=prices.index)


class MACDStrategy(_Strategy):
    """MACD(12,26,9): long when MACD line > signal line, else flat."""

    name = "MACD(12,26,9)"

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9) -> None:
        self.fast = fast
        self.slow = slow
        self.signal = signal

    def signals(self, prices: pd.DataFrame) -> pd.Series:
        close = prices["Close"].astype(float)
        ema_fast = close.ewm(span=self.fast, adjust=False).mean()
        ema_slow = close.ewm(span=self.slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        sig_line = macd.ewm(span=self.signal, adjust=False).mean()
        return (macd > sig_line).fillna(False)


class SMACrossStrategy(_Strategy):
    """SMA(50/200) golden-cross: long when SMA_fast > SMA_slow."""

    name = "SMA(50/200)"

    def __init__(self, fast: int = 50, slow: int = 200) -> None:
        self.fast = fast
        self.slow = slow

    def signals(self, prices: pd.DataFrame) -> pd.Series:
        close = prices["Close"].astype(float)
        sma_fast = close.rolling(self.fast).mean()
        sma_slow = close.rolling(self.slow).mean()
        return (sma_fast > sma_slow).fillna(False)
