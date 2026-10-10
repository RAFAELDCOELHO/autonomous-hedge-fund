"""Own-loop backtest runner.

Data is pulled via `tradingagents.dataflows.stockstats_utils.load_ohlcv`,
which is backed by yfinance and cached locally — no external API key
required. This keeps the academic demo frictionless.

Every curve here follows one convention (P4.11, decision before the open):
decision sessions D come from the ticker's exchange calendar (calendar.py),
the decision for D sees data up to the close of D-1 only, executes at
Open[D] and is marked at the next session's open. The paper/BrazilBench
simulators (baselines._simulate, brazilbench, scripts/) stay close-to-close.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

import pandas as pd

from tradingagents.dataflows.stockstats_utils import load_ohlcv

from .calendar import exchange_for, next_session, previous_session, sessions
from .risk_free import daily_rf

logger = logging.getLogger(__name__)

# Exclusion rule E4 (docs/PREREGISTRATION.md); mirrors scripts/h1_stats.MAX_ERROR_RATE.
MAX_DECISION_ERROR_RATE = 0.05


def _execution_plan(ticker: str, start: str, end: str):
    """Vendor frame, decision sessions D_0..D_n and the opens of D_0..D_n plus the exit session.

    The frame is loaded once through the exit session (the calendar session
    after the last D) with fill_prices=False, so opens and closes share one
    auto_adjust=True download and a missing Open is never filled. Vendor bars
    on non-sessions are dropped with a warning: the calendar alone defines the
    sessions and D-1. Fail closed (ValueError naming ticker and date) on a
    missing D-1 bar of the first session (no slide back to an older bar), a
    missing bar for any D or the exit session, a missing Open column, or a
    NaN/non-positive Open on any D or the exit session (no Close fallback).
    """
    window = sessions(ticker, start, end)
    if window.empty:
        raise ValueError(f"{ticker}: no {exchange_for(ticker)} sessions in {start}..{end}")
    exit_day = next_session(ticker, window[-1])
    frame = load_ohlcv(ticker, exit_day.strftime("%Y-%m-%d"), fill_prices=False).copy()
    frame["Date"] = pd.to_datetime(frame["Date"])
    frame = frame.sort_values("Date").set_index("Date").loc[:exit_day]
    if "Open" not in frame.columns:
        raise ValueError(f"{ticker}: no Open column; execution at the open needs opens")
    if not frame.empty:
        off = frame.index.difference(sessions(ticker, frame.index[0], exit_day))
        if not off.empty:
            logger.warning(
                "%s: ignoring vendor bars on non-%s sessions: %s",
                ticker, exchange_for(ticker), ", ".join(f"{d:%Y-%m-%d}" for d in off),
            )
            frame = frame.drop(off)
    first_prev = previous_session(ticker, window[0])
    if first_prev not in frame.index:
        raise ValueError(
            f"{ticker}: no vendor bar on {first_prev:%Y-%m-%d}, D-1 of the first window "
            f"session {window[0]:%Y-%m-%d}; the look-back must cover it"
        )
    missing = window.difference(frame.index)
    if not missing.empty:
        raise ValueError(f"{ticker}: no vendor bar on session {missing[0]:%Y-%m-%d}")
    if exit_day not in frame.index:
        raise ValueError(
            f"{ticker}: no vendor bar on {exit_day:%Y-%m-%d}, the exit session after {end}"
        )
    exec_days = frame.index[frame.index.isin(window) | (frame.index == exit_day)]  # vendor index name
    opens = frame.loc[exec_days, "Open"].astype(float)
    bad = opens[~(opens > 0)]
    if not bad.empty:
        raise ValueError(f"{ticker}: missing or non-positive Open on {bad.index[0]:%Y-%m-%d}")
    return frame, exec_days[:-1], opens


def _simulate_open(ticker, days, opens, decide, initial_capital, market=None) -> pd.Series:
    """Full-position open-to-open simulation over the decision sessions `days`.

    decide(day, cutoff) -> (action, error) with cutoff = D-1. The order fills
    at Open[D_i]; equity[D_i] = cash + shares * Open[D_{i+1}] (D_{n+1} = the
    exit session), so the decision of D earns Open[D] -> Open[D+1], the last
    one included.

    market ("BR"/"US"): after D_i's order, cash accrues
    risk_free.daily_rf(market, days)[D_i] for i >= 1 and nothing for i = 0.
    That is the per-label rf h1_cell_metrics subtracts from the return
    equity[D_{i-1}] -> equity[D_i], so an all-cash run has exactly zero excess
    Sharpe. The rate is lagged one session (the CDI/DTB3 rate of
    D_{i-1} -> D_i, applied over Open[D_i] -> Open[D_{i+1}]) and the first
    interval earns none. None keeps cash at 0%.
    """
    cutoffs = [previous_session(ticker, days[0]), *days[:-1]]  # every session has a bar: checked
    rf = daily_rf(market, days) if market else None
    cash = float(initial_capital)
    shares = 0.0
    equity = []
    decision_log = []
    for i, (day, cutoff) in enumerate(zip(days, cutoffs)):
        action, error = decide(day, cutoff)
        price = opens.iloc[i]
        if action == "BUY" and shares == 0.0:
            shares = cash / price
            cash = 0.0
        elif action == "SELL" and shares > 0.0:
            cash = shares * price
            shares = 0.0
        if rf is not None and i > 0:
            cash *= 1.0 + rf[day]
        equity.append(cash + shares * opens.iloc[i + 1])
        decision_log.append({
            "decision_date": day.strftime("%Y-%m-%d"),
            "data_cutoff": cutoff.strftime("%Y-%m-%d"),
            "action": action,
            "error": error,
        })
    out = pd.Series(equity, index=days, name="equity")
    out.attrs.update(
        execution="open",
        information_cutoff="previous session close",
        decision_log=decision_log,
        data_cutoff=decision_log[0]["data_cutoff"],
    )
    return out


def run_strategy(
    strategy,
    ticker: str,
    start: str,
    end: str,
    initial_capital: float = 100_000.0,
) -> pd.Series:
    """A baseline under the decision-before-the-open convention (_simulate_open).

    The in-position signal is computed on the vendor history up to the D-1 of
    the last session (indicator warm-up included, nothing dated D_n or later)
    and the signal at D-1 sets the position taken at Open[D]: True buys,
    False sells. Buy & Hold therefore buys at Open[first session].
    """
    frame, days, opens = _execution_plan(ticker, start, end)
    sig = strategy.signals(frame.loc[: previous_session(ticker, days[-1])]).astype(bool)

    def decide(_day, cutoff):
        return ("BUY" if sig.loc[cutoff] else "SELL"), False

    return _simulate_open(ticker, days, opens, decide, initial_capital)


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

    Decision before the open: for each calendar session D in [start, end],
    decide_fn(prev_date, history_up_to_prev) -> {"BUY", "SELL", "HOLD"} is
    called with prev_date = D-1 and the vendor rows dated <= D-1 (window.attrs
    carries decision_date=D and data_cutoff=D-1). The agent and every tool
    keyed on that date see information up to the close of D-1 only; no row
    dated D or later is passed (the exit session's bar is loaded for its open
    only). Sessions and fail-closed checks: _execution_plan. Fill, open-to-open
    marking and rf accrual (market): _simulate_open.

    A decide_fn that raises or returns anything else falls back to HOLD and
    is counted in equity.attrs["n_decision_errors"]. attrs also carry n_days
    (one decision per session), decision_errors_exceed_limit (E4: error rate
    > MAX_DECISION_ERROR_RATE), execution="open", information_cutoff=
    "previous session close", decision_log (one {"decision_date": D,
    "data_cutoff": D-1, "action", "error"} per session, ISO dates) and
    data_cutoff (D-1 of the first session, the cells.csv data_cutoff).
    """
    frame, days, opens = _execution_plan(ticker, start, end)

    def decide(day, cutoff):
        window = frame.loc[:cutoff].copy()
        window.attrs = {
            "decision_date": day.strftime("%Y-%m-%d"),
            "data_cutoff": cutoff.strftime("%Y-%m-%d"),
        }
        try:
            action = str(decide_fn(cutoff.strftime("%Y-%m-%d"), window)).upper()
            if action not in ("BUY", "SELL", "HOLD"):
                raise ValueError(f"invalid action {action!r}")
            return action, False
        except Exception as e:
            logger.warning("agent decide_fn failed on %s: %s", day, e)
            return "HOLD", True

    out = _simulate_open(ticker, days, opens, decide, initial_capital, market)
    n_days = len(out)
    n_decision_errors = sum(e["error"] for e in out.attrs["decision_log"])
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
