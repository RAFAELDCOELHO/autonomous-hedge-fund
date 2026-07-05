"""Shared library for the Brazilian regime benchmark (Tasks 1 & 2).

Zero-cost by construction: prices come from yfinance (free, no API key) and
every strategy is pure Python/NumPy. No LLM or paid API is ever called.

Design (fixed by the study):
    * Tickers: PETR4, VALE3, ITUB4, BBDC4 (B3 single names, queried with the
      Yahoo ``.SA`` suffix) and ^BVSP (the Ibovespa index).
    * Regimes: four dated windows (see ``REGIMES``).

Backtest semantics mirror ``tradingagents/backtest`` exactly: long-only,
binary positioning (100% equity on a long signal / 100% cash otherwise),
frictionless, and decisions execute at the close of the bar they observe.
"Total return" is the harness Cumulative Return, CR = (V_end - V_start)/V_start.

Indicator warmup: rule-based strategies (MACD, SMA, Momentum) are given a
pre-window buffer from ``FETCH_START`` so their indicators are warm at the
first bar of each regime. Warmup uses only backward-looking data (rolling /
ewm / shift), so it introduces no look-ahead; it merely avoids cold-start
artifacts that would otherwise pin short-window strategies to all-cash.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
CACHE_DIR = REPO / "benchmark" / "results" / "_price_cache"

# Exact study windows (inclusive), YYYY-MM-DD.
REGIMES: dict[str, tuple[str, str]] = {
    "bull_2019":     ("2019-01-02", "2019-12-31"),
    "crisis_2020":   ("2020-02-03", "2020-05-29"),
    "recovery_2021": ("2021-01-04", "2021-06-30"),
    "hi_rates_2022": ("2022-01-03", "2022-12-30"),
}
REGIME_ORDER: list[str] = list(REGIMES)

TICKERS: list[str] = ["PETR4", "VALE3", "ITUB4", "BBDC4", "^BVSP"]

# One download per ticker covers every regime plus >= 1 year of warmup.
FETCH_START = "2018-01-01"
FETCH_END = "2023-01-15"

INITIAL_CAPITAL = 100_000.0
N_SEEDS = 100
SEED42 = 42


# --------------------------------------------------------------------------- #
# Data access (yfinance, cached)
# --------------------------------------------------------------------------- #
def yahoo_symbol(ticker: str) -> str:
    """Map a study ticker to its Yahoo Finance symbol."""
    return ticker if ticker.startswith("^") else f"{ticker}.SA"


def _cache_path(ticker: str) -> Path:
    safe = ticker.replace("^", "IDX_")
    return CACHE_DIR / f"{safe}.csv"


def load_prices(ticker: str, *, force: bool = False) -> pd.DataFrame:
    """Return a DatetimeIndex DataFrame with a 'Close' column over the fetch
    window. Downloads once via yfinance, then serves from a CSV cache.
    """
    path = _cache_path(ticker)
    if path.exists() and not force:
        df = pd.read_csv(path, parse_dates=["Date"]).set_index("Date")
        return df

    import yfinance as yf

    sym = yahoo_symbol(ticker)
    raw = yf.download(
        sym, start=FETCH_START, end=FETCH_END, auto_adjust=True, progress=False
    )
    if raw is None or raw.empty:
        raise RuntimeError(f"yfinance returned no data for {sym}")
    # Flatten possible MultiIndex columns, e.g. ('Close', 'PETR4.SA') -> 'Close'.
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = raw[["Close"]].astype(float).copy()
    df.index.name = "Date"
    df = df.dropna()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.reset_index().to_csv(path, index=False)
    return df


def regime_window(df: pd.DataFrame, regime: str) -> pd.DataFrame:
    """Slice the price frame to a regime's exact [start, end] window."""
    start, end = REGIMES[regime]
    return df.loc[start:end]


def warm_window(df: pd.DataFrame, regime: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (warmed, window): `warmed` is all history up to the window end
    (for indicator computation); `window` is the regime slice itself.
    """
    start, end = REGIMES[regime]
    warmed = df.loc[:end]
    return warmed, warmed.loc[start:end]


# --------------------------------------------------------------------------- #
# Simulation (mirrors baselines._simulate and runner.run_agent_strategy)
# --------------------------------------------------------------------------- #
def _simulate_signal(closes: pd.Series, in_pos: pd.Series, capital: float) -> pd.Series:
    """Equity curve from a boolean 'hold into this close' signal."""
    c = closes.astype(float).to_numpy()
    s = in_pos.astype(bool).to_numpy()
    n = len(c)
    cash, shares, prev = float(capital), 0.0, False
    eq = np.empty(n, dtype=float)
    for i in range(n):
        st, p = bool(s[i]), c[i]
        if st and not prev:
            shares = cash / p if p > 0 else 0.0
            cash = 0.0
        elif prev and not st:
            cash = shares * p
            shares = 0.0
        eq[i] = cash + shares * p
        prev = st
    return pd.Series(eq, index=closes.index)


def _simulate_actions(closes: pd.Series, actions: list[str], capital: float) -> pd.Series:
    """Equity curve from a BUY/SELL/HOLD action stream (agent semantics)."""
    c = closes.astype(float).to_numpy()
    n = len(c)
    cash, shares = float(capital), 0.0
    eq = np.empty(n, dtype=float)
    for i in range(n):
        a = (actions[i] or "HOLD").upper()
        p = c[i]
        if a == "BUY" and shares == 0.0 and p > 0:
            shares = cash / p
            cash = 0.0
        elif a == "SELL" and shares > 0.0:
            cash = shares * p
            shares = 0.0
        eq[i] = cash + shares * p
    return pd.Series(eq, index=closes.index)


def total_return(equity: pd.Series) -> float:
    """Cumulative return CR = (V_end - V_start) / V_start (a fraction)."""
    equity = equity.dropna()
    if len(equity) < 2:
        return float("nan")
    v0, v1 = float(equity.iloc[0]), float(equity.iloc[-1])
    return (v1 - v0) / v0 if v0 > 0 else float("nan")


# --------------------------------------------------------------------------- #
# Random agent
# --------------------------------------------------------------------------- #
def random_actions(n: int, seed: int) -> list[str]:
    """A fixed uninformed policy: draw BUY/SELL/HOLD i.i.d. uniform per day.

    Seeded solely by ``seed`` so that policy #s is a well-defined object
    evaluated identically across cells (only the window length differs).
    """
    rng = random.Random(seed)
    return [rng.choice(("BUY", "SELL", "HOLD")) for _ in range(n)]


def run_random_cell(df: pd.DataFrame, regime: str, seed: int,
                    capital: float = INITIAL_CAPITAL) -> float:
    """Total return of RandomAgent(seed) on one (ticker, regime) cell."""
    win = regime_window(df, regime)
    acts = random_actions(len(win), seed)
    return total_return(_simulate_actions(win["Close"], acts, capital))


def run_strategy_cell(strategy, df: pd.DataFrame, regime: str,
                      capital: float = INITIAL_CAPITAL) -> float:
    """Total return of a deterministic rule-based strategy on one cell,
    with warm indicators and no look-ahead.
    """
    warmed, win = warm_window(df, regime)
    sig = strategy.signals(warmed).loc[win.index]
    return total_return(_simulate_signal(win["Close"], sig, capital))


# --------------------------------------------------------------------------- #
# Stats
# --------------------------------------------------------------------------- #
def percentile_of(values, x: float) -> float:
    """Percentile rank (0-100) of `x` within `values` (mean/mid-rank rule)."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    n = len(v)
    if n == 0:
        return float("nan")
    return 100.0 * (np.sum(v < x) + 0.5 * np.sum(v == x)) / n
