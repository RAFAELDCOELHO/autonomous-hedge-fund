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
from .calendar import exchange_for, previous_session, sessions
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
    """Run a day-by-day agent loop with full-position sizing (P4.11).

    Decision before the open: the window sessions D and their D-1 come from
    the ticker's fixed exchange calendar (calendar.py: B3 for .SA, NYSE
    otherwise), not from the vendor's bars. For each D,
    decide_fn(prev_date, history_up_to_prev) -> {"BUY", "SELL", "HOLD"} is
    called with prev_date = D-1 and the vendor rows dated <= D-1 (window.attrs
    carries decision_date=D and data_cutoff=D-1). The agent and every tool
    keyed on that date see information up to the close of D-1 only; no row
    dated D is passed. The order executes at Open[D].

    Fail closed (ValueError naming ticker and date): a window session without
    a vendor bar, a missing D-1 bar (never slides back to an older bar; for
    the first session D-1 lies before `start` and the look-back must cover
    it), or a vendor bar on a non-session in [D-1 of start, end].

    Equity is marked open-to-open: equity[D] = cash + shares * Open[D] after
    D's order, so D's decision earns Open[D] -> Open[D+1]. The last session's
    decision is still taken, but its return lies after `end`: the position is
    marked at Open[end]. Closes are not used. Opens and closes come from the
    same load_ohlcv frame (yfinance auto_adjust=True adjusts all of OHLC), so
    returns use one adjustment. A missing Open column or a NaN/non-positive
    Open in the window raises ValueError; there is no Close fallback.

    market ("BR"/"US"): cash earns the daily rf of risk_free.daily_rf (the
    overnight accrual between consecutive opens, applied before D's order);
    None keeps cash at 0%. A decide_fn that raises or returns anything
    else falls back to HOLD and is counted in equity.attrs["n_decision_errors"].
    attrs also carry n_days (one decision per trading day),
    decision_errors_exceed_limit (E4: error rate > MAX_DECISION_ERROR_RATE),
    execution="open", information_cutoff="previous session close",
    decision_log (one {"decision_date": D, "data_cutoff": D-1, "action",
    "error"} per session, ISO dates) and data_cutoff (D-1 of the last session,
    the latest close any decision of the run could see).
    """
    history = load_price_history(ticker, end)
    window_sessions = sessions(ticker, start, end)
    if window_sessions.empty:
        raise ValueError(f"{ticker}: no {exchange_for(ticker)} sessions in {start}..{end}")
    if "Open" not in history.columns:
        raise ValueError(f"{ticker}: no Open column; execution at the open needs opens")
    first_prev = previous_session(ticker, window_sessions[0])
    off = history.loc[first_prev:].index.difference(sessions(ticker, first_prev, end))
    if not off.empty:
        raise ValueError(
            f"{ticker}: vendor bar on {off[0]:%Y-%m-%d}, not an {exchange_for(ticker)} session"
        )
    if first_prev not in history.index:
        raise ValueError(
            f"{ticker}: no vendor bar on {first_prev:%Y-%m-%d}, D-1 of the first window "
            f"session {window_sessions[0]:%Y-%m-%d}; the look-back must cover it"
        )
    missing = window_sessions.difference(history.index)
    if not missing.empty:
        raise ValueError(f"{ticker}: no vendor bar on session {missing[0]:%Y-%m-%d}")
    window_sessions = history.index[history.index.isin(window_sessions)]  # vendor index name, no freq
    opens = history["Open"].loc[window_sessions].astype(float)
    bad = opens[~(opens > 0)]
    if not bad.empty:
        raise ValueError(f"{ticker}: missing or non-positive Open on {bad.index[0]:%Y-%m-%d}")
    rf = daily_rf(market, window_sessions) if market else None

    cash = float(initial_capital)
    shares = 0.0
    equity = []
    decision_log = []
    n_decision_errors = 0

    for i, (date, price) in enumerate(opens.items()):
        if rf is not None and i > 0:
            cash *= 1.0 + rf[date]
        prev = previous_session(ticker, date)  # has a vendor bar: checked above
        window = history.loc[:prev].copy()
        window.attrs = {
            "decision_date": date.strftime("%Y-%m-%d"),
            "data_cutoff": prev.strftime("%Y-%m-%d"),
        }
        error = False
        try:
            action = str(decide_fn(prev.strftime("%Y-%m-%d"), window)).upper()
            if action not in ("BUY", "SELL", "HOLD"):
                raise ValueError(f"invalid action {action!r}")
        except Exception as e:
            logger.warning("agent decide_fn failed on %s: %s", date, e)
            n_decision_errors += 1
            action = "HOLD"
            error = True
        decision_log.append({
            "decision_date": date.strftime("%Y-%m-%d"),
            "data_cutoff": prev.strftime("%Y-%m-%d"),
            "action": action,
            "error": error,
        })

        if action == "BUY" and shares == 0.0 and price > 0:
            shares = cash / price
            cash = 0.0
        elif action == "SELL" and shares > 0.0:
            cash = shares * price
            shares = 0.0

        equity.append(cash + shares * price)

    out = pd.Series(equity, index=window_sessions, name="equity")
    n_days = len(out)
    exceeded = n_days == 0 or n_decision_errors / n_days > MAX_DECISION_ERROR_RATE
    out.attrs.update(
        n_days=n_days,
        n_decision_errors=n_decision_errors,
        decision_errors_exceed_limit=exceeded,
        execution="open",
        information_cutoff="previous session close",
        decision_log=decision_log,
        data_cutoff=decision_log[-1]["data_cutoff"],
    )
    if exceeded:
        logger.warning(
            "%s: %d/%d decision errors exceed %.0f%% (exclusion rule E4)",
            ticker, n_decision_errors, n_days, 100 * MAX_DECISION_ERROR_RATE,
        )
    return out


def run_buy_and_hold_at_open(
    ticker: str,
    start: str,
    end: str,
    initial_capital: float = 100_000.0,
    market: Optional[str] = None,
) -> pd.Series:
    """Buy & Hold under the agent convention: buy at Open[start], mark at opens."""
    return run_agent_strategy(lambda _d, _w: "BUY", ticker, start, end, initial_capital, market)
