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

from .baselines import _simulate
from .risk_free import daily_rf

logger = logging.getLogger(__name__)

# Exclusion rule E4 (docs/PREREGISTRATION.md); mirrors scripts/h1_stats.MAX_ERROR_RATE.
MAX_DECISION_ERROR_RATE = 0.05


def load_price_history(ticker: str, end: str) -> pd.DataFrame:
    """Load all cached OHLCV up to and including `end`, indexed by date (ascending)."""
    df = load_ohlcv(ticker, end)
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").set_index("Date")
    return df.loc[: pd.to_datetime(end)]


def load_price_window(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Load OHLCV for [start, end] indexed by date (ascending)."""
    df = load_price_history(ticker, end).loc[pd.to_datetime(start):]
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
    """Signals see pre-window history (indicator warm-up); equity covers [start, end] only.

    Same semantics as brazilbench.run_cell: the curve starts at `initial_capital`
    on the first window bar, so Buy & Hold still buys on `start`.
    """
    history = load_price_history(ticker, end)
    prices = history.loc[pd.to_datetime(start):]
    if prices.empty:
        raise ValueError(f"No price data for {ticker} in {start}..{end}")
    sig = strategy.signals(history).loc[prices.index]
    return _simulate(prices, sig, initial_capital)


def run_buy_and_hold(ticker: str, start: str, end: str, initial_capital: float = 100_000.0) -> pd.Series:
    from .baselines import BuyAndHold
    return run_strategy(BuyAndHold(), ticker, start, end, initial_capital)


def run_agent_strategy(
    decide_fn: Callable[[str, pd.DataFrame], str],
    ticker: str,
    start: str,
    end: str,
    initial_capital: float = 100_000.0,
    market: Optional[str] = None,
) -> pd.Series:
    """Run a day-by-day agent loop with full-position sizing.

    decide_fn(curr_date, prices_up_to_date) -> {"BUY", "SELL", "HOLD"}.
    Signals are executed at the close of curr_date. Look-ahead is
    prevented because decide_fn only receives prices up to curr_date.

    market ("BR"/"US"): cash earns the daily rf of risk_free.daily_rf;
    None keeps cash at 0%. A decide_fn that raises or returns anything
    else falls back to HOLD and is counted in equity.attrs["n_decision_errors"].
    attrs also carry n_days (one decision per trading day) and
    decision_errors_exceed_limit (E4: error rate > MAX_DECISION_ERROR_RATE).
    """
    prices = load_price_window(ticker, start, end)
    closes = prices["Close"].astype(float)
    rf = daily_rf(market, prices.index) if market else None

    cash = float(initial_capital)
    shares = 0.0
    equity = []
    n_decision_errors = 0

    for i, (date, price) in enumerate(closes.items()):
        if rf is not None and i > 0:
            cash *= 1.0 + rf[date]
        window = prices.iloc[: i + 1]
        try:
            action = str(decide_fn(date.strftime("%Y-%m-%d"), window)).upper()
            if action not in ("BUY", "SELL", "HOLD"):
                raise ValueError(f"invalid action {action!r}")
        except Exception as e:
            logger.warning("agent decide_fn failed on %s: %s", date, e)
            n_decision_errors += 1
            action = "HOLD"

        if action == "BUY" and shares == 0.0 and price > 0:
            shares = cash / price
            cash = 0.0
        elif action == "SELL" and shares > 0.0:
            cash = shares * price
            shares = 0.0

        equity.append(cash + shares * price)

    out = pd.Series(equity, index=prices.index, name="equity")
    n_days = len(out)
    exceeded = n_days == 0 or n_decision_errors / n_days > MAX_DECISION_ERROR_RATE
    out.attrs.update(
        n_days=n_days,
        n_decision_errors=n_decision_errors,
        decision_errors_exceed_limit=exceeded,
    )
    if exceeded:
        logger.warning(
            "%s: %d/%d decision errors exceed %.0f%% (exclusion rule E4)",
            ticker, n_decision_errors, n_days, 100 * MAX_DECISION_ERROR_RATE,
        )
    return out
