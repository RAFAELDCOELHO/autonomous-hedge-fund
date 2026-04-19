"""Own-loop backtest runner.

Data is pulled via `tradingagents.dataflows.stockstats_utils.load_ohlcv`,
which is backed by yfinance and cached locally — no external API key
required. This keeps the academic demo frictionless.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

import pandas as pd

from tradingagents.dataflows.stockstats_utils import load_ohlcv

logger = logging.getLogger(__name__)


def load_price_window(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Load OHLCV for [start, end] indexed by date (ascending)."""
    df = load_ohlcv(ticker, end)
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").set_index("Date")
    df = df.loc[pd.to_datetime(start): pd.to_datetime(end)]
    if df.empty:
        raise ValueError(f"No price data for {ticker} in {start}..{end}")
    return df


def run_strategy(
    strategy,
    ticker: str,
    start: str,
    end: str,
    initial_capital: float = 100_000.0,
) -> pd.Series:
    prices = load_price_window(ticker, start, end)
    return strategy.run(prices, initial_capital)


def run_buy_and_hold(ticker: str, start: str, end: str, initial_capital: float = 100_000.0) -> pd.Series:
    from .baselines import BuyAndHold
    return run_strategy(BuyAndHold(), ticker, start, end, initial_capital)


def run_agent_strategy(
    decide_fn: Callable[[str, pd.DataFrame], str],
    ticker: str,
    start: str,
    end: str,
    initial_capital: float = 100_000.0,
) -> pd.Series:
    """Run a day-by-day agent loop with full-position sizing.

    decide_fn(curr_date, prices_up_to_date) -> {"BUY", "SELL", "HOLD"}.
    Signals are executed at the close of curr_date. Look-ahead is
    prevented because decide_fn only receives prices up to curr_date.
    """
    prices = load_price_window(ticker, start, end)
    closes = prices["Close"].astype(float)

    cash = float(initial_capital)
    shares = 0.0
    equity = []

    for i, (date, price) in enumerate(closes.items()):
        window = prices.iloc[: i + 1]
        try:
            action = decide_fn(date.strftime("%Y-%m-%d"), window)
        except Exception as e:
            logger.warning("agent decide_fn failed on %s: %s", date, e)
            action = "HOLD"
        action = (action or "HOLD").upper()

        if action == "BUY" and shares == 0.0 and price > 0:
            shares = cash / price
            cash = 0.0
        elif action == "SELL" and shares > 0.0:
            cash = shares * price
            shares = 0.0

        equity.append(cash + shares * price)

    return pd.Series(equity, index=prices.index, name="equity")
