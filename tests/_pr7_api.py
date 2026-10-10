"""PR7 (P4.11, "decision before the open") interface adapter for the tests.

Rule (Rafa, 2026-10-09): for decision day D the agent and every tool see data
up to the D-1 close only; the decision executes at the D open. D-1 is the
previous session of the ticker's exchange (B3 for ``.SA``, NYSE otherwise)
from a fixed calendar.

Every name the PR7 tests assume about the implementation that does not exist
yet at e5ea062 lives in this one module. If PR7 picks different names or
conventions, rewire them here; the tests themselves should not need edits.
Leading underscore: pytest does not collect this file.
"""

from __future__ import annotations

from typing import Callable, Optional

import pandas as pd

# --------------------------------------------------------------------------
# A1. Runner entry points (unchanged public signatures).
# --------------------------------------------------------------------------


def run_agent(
    decide_fn: Callable[..., str],
    ticker: str,
    start: str,
    end: str,
    initial_capital: float = 100_000.0,
    market: Optional[str] = None,
) -> pd.Series:
    from tradingagents.backtest.runner import run_agent_strategy

    return run_agent_strategy(decide_fn, ticker, start, end, initial_capital, market=market)


def run_buy_and_hold(ticker: str, start: str, end: str, initial_capital: float = 100_000.0) -> pd.Series:
    from tradingagents.backtest.runner import run_buy_and_hold as _bh

    return _bh(ticker, start, end, initial_capital)


def run_tradingagents(ticker: str, start: str, end: str, propagate_fn, initial_capital: float = 100_000.0):
    from tradingagents.backtest.agent_integration import run_tradingagents_backtest

    return run_tradingagents_backtest(
        ticker=ticker, start=start, end=end, config={}, initial_capital=initial_capital,
        propagate_fn=propagate_fn,
    )


# --------------------------------------------------------------------------
# A2. What the agent is told. Assumption: the runner keeps calling
# decide_fn(date_str, window) and the date string IS the data cutoff D-1, so
# make_decide_fn -> propagate(ticker, D-1) -> state["trade_date"] = D-1 ->
# prompts ({current_date}) and every tool (curr_date / InjectedState
# trade_date / end_date) are keyed to D-1. A data_cutoff= keyword wins if PR7
# passes it explicitly instead.
# --------------------------------------------------------------------------


def agent_date(args: tuple, kwargs: dict) -> str:
    raw = kwargs.get("data_cutoff", args[0] if args else kwargs.get("curr_date"))
    return pd.Timestamp(raw).strftime("%Y-%m-%d")


def agent_window(args: tuple, kwargs: dict) -> Optional[pd.DataFrame]:
    if "prices" in kwargs:
        return kwargs["prices"]
    if "prices_up_to_date" in kwargs:
        return kwargs["prices_up_to_date"]
    return args[1] if len(args) > 1 else None


# --------------------------------------------------------------------------
# A3. Equity-curve convention. Assumption: the equity series stays indexed by
# decision dates D_0..D_n, and equity[D_i] is the portfolio value at
# open(D_{i+1}) after executing decision D_i at open(D_i). equity[D_n] is
# marked at the open of the session after `end` (the defined exit price).
# Hence the P&L attributed to decision D is open(D+1)/open(D) - 1 when long.
# --------------------------------------------------------------------------


def pnl_by_decision_date(equity: pd.Series, initial_capital: float) -> pd.Series:
    prev = equity.shift(1)
    prev.iloc[0] = initial_capital
    return equity / prev - 1.0


# --------------------------------------------------------------------------
# A4. Fail-closed signal: the runner raises one of these (it must NOT fall
# back to close, forward-fill the open, or slide to an earlier bar).
# --------------------------------------------------------------------------

FAIL_CLOSED_ERRORS: tuple[type[BaseException], ...] = (ValueError, LookupError)


# --------------------------------------------------------------------------
# A5. Per-day decision log (Lingxi #1). Assumption: equity.attrs["decision_log"]
# is a list of dicts with at least decision_date (= D) and data_cutoff (= D-1).
# --------------------------------------------------------------------------

DECISION_LOG_ATTR = "decision_log"
DECISION_DATE_FIELD = "decision_date"
DATA_CUTOFF_FIELD = "data_cutoff"


def decision_log(equity: pd.Series) -> pd.DataFrame:
    log = pd.DataFrame(list(equity.attrs[DECISION_LOG_ATTR]))
    for col in (DECISION_DATE_FIELD, DATA_CUTOFF_FIELD):
        log[col] = pd.to_datetime(log[col]).dt.strftime("%Y-%m-%d")
    return log


# --------------------------------------------------------------------------
# A6. cells.csv (Lingxi #1). Assumption: run_backtest.main(--cells-out) keeps
# start/end as decision dates and adds a `data_cutoff` column holding the
# cutoff of the first decision date (previous session of `start`).
# --------------------------------------------------------------------------

CELLS_DATA_CUTOFF_COLUMN = "data_cutoff"


def run_backtest_module():
    import importlib.util
    from pathlib import Path

    target = Path(__file__).resolve().parents[1] / "run_backtest.py"
    spec = importlib.util.spec_from_file_location("run_backtest_pr7", target)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_cells_grid(cells_path, tickers, arms, seeds, decide_fn_factory, monkeypatch,
                     start: str = "2024-01-02", end: str = "2024-03-28") -> list[dict]:
    """A6b. Produce cells.csv for a (ticker x seed) grid the way H1 is run today:
    one independent ``run_backtest.py --cells-out`` invocation per (ticker, seed)
    with ``--arms`` listing every arm (a shell loop that keeps going after a
    failed process). Returns one outcome per invocation: ``{"ticker", "seed",
    "rc", "exc"}`` where ``exc`` is the exception the invocation raised (None if
    it returned ``rc``). Rewire here if PR7 adds a grid driver.
    """
    module = run_backtest_module()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-dummy")
    monkeypatch.setattr(module, "load_dotenv", lambda *a, **k: None)
    monkeypatch.setattr(module, "make_decide_fn", lambda **_kw: decide_fn_factory())
    outcomes = []
    for seed in seeds:
        for ticker in tickers:
            argv = ["--ticker", ticker, "--start", start, "--end", end,
                    "--arms", ",".join(arms), "--cells-out", str(cells_path), "--seed", str(seed)]
            try:
                outcomes.append({"ticker": ticker, "seed": seed, "rc": module.main(argv), "exc": None})
            except BaseException as exc:  # SystemExit included: a process that died
                outcomes.append({"ticker": ticker, "seed": seed, "rc": None, "exc": exc})
    return outcomes


# --------------------------------------------------------------------------
# A6c. Failed cells.csv row for a BASELINE error (Megabrain 2026-10-09: the
# baseline error is caught, the cell becomes `failed` with a reason, exit != 0).
# The row is pinned to the EXISTING failed-row format that scripts/h1_stats.py
# reads and run_backtest.py already writes for a failed agent arm
# (cells.make_cell_row(status="failed"), tradingagents/backtest/cells.py:110-115):
#   status "failed"; n_days "0"; n_decision_errors "0" (integers: h1_stats
#   _int() rejects empty); sharpe "" (h1_stats reads "" as NaN); sharpe_flat "";
#   rf_source = RF_SOURCE[market]; ticker bare; arm absent/present.
# PR7 adds only the reason, as an extra column (h1_stats ignores extra columns).
# --------------------------------------------------------------------------

FAILURE_REASON_COLUMN = "failure_reason"


def baseline_failed_row(ticker: str, arm: str, seed: int) -> dict[str, str]:
    """Expected h1-schema fields (+ sharpe_flat) of a failed row; reason checked separately."""
    bare = ticker.strip().upper().removesuffix(".SA")
    market = "BR" if ticker.strip().upper().endswith(".SA") else "US"
    return {
        "ticker": bare,
        "market": market,
        "arm": {"baseline": "absent", "macro": "present"}.get(arm, arm),
        "seed": str(seed),
        "status": "failed",
        "n_days": "0",
        "n_decision_errors": "0",
        "sharpe": "",
        "rf_source": {"BR": "BCB-SGS-12", "US": "FRED-DTB3"}[market],
        "sharpe_flat": "",
    }


# --------------------------------------------------------------------------
# A6d. H1 snapshot validation (Lingxi). Assumed PR7 API:
#   tradingagents.backtest.snapshot.validate_snapshot(tickers, start, end)
#   (PR7 rewire: the calendar module is tradingagents.backtest.calendar.sessions)
#   - loads bars through the same vendor path as the runner (yfinance/load_ohlcv),
#   - checks, per ticker on its exchange calendar: every session in [start, end]
#     (bar present and Open finite > 0), the D-1 of the first session (bar
#     present) and the session after the last (bar present and Open finite > 0),
#   - raises an exception in FAIL_CLOSED_ERRORS carrying `.missing`, an iterable
#     of (ticker, date, field) with field "bar" or "Open", listing EVERY defect.
# Calendar sessions: trading_calendar.sessions_between(ticker, start, end).
# --------------------------------------------------------------------------

SNAPSHOT_FIELD_BAR = "bar"
SNAPSHOT_FIELD_OPEN = "Open"


def prereg_tickers_yahoo() -> list[str]:
    from tradingagents.backtest.cells import PREREG_TICKERS

    return [t + ".SA" if m == "BR" else t for t, m in PREREG_TICKERS.items()]


def prereg_window() -> tuple[str, str]:
    from tradingagents.backtest.cells import PREREG_WINDOW_END, PREREG_WINDOW_START

    return PREREG_WINDOW_START, PREREG_WINDOW_END


def validate_snapshot(tickers=None, start=None, end=None):
    from tradingagents.backtest.snapshot import validate_snapshot as _validate

    w_start, w_end = prereg_window()
    return _validate(list(tickers or prereg_tickers_yahoo()), start or w_start, end or w_end)


def snapshot_missing(exc: BaseException) -> set[tuple[str, str, str]]:
    return {(t, pd.Timestamp(d).strftime("%Y-%m-%d"), str(f)) for t, d, f in exc.missing}


def exchange_sessions(ticker: str, start: str, end: str) -> list[str]:
    from tradingagents.backtest.calendar import sessions as sessions_between

    return [pd.Timestamp(d).strftime("%Y-%m-%d") for d in sessions_between(ticker, start, end)]


# --------------------------------------------------------------------------
# A7. Semantics switches (design decisions, Megabrain 2026-10-09).
# --------------------------------------------------------------------------

# Macro daily series (SELIC, PTAX) keep their strict `< trade_date` cap and
# trade_date becomes D-1, so the D-1 observation itself is NOT shown.
MACRO_DAILY_SHOWS_CUTOFF_DAY = False
# Statements / insider keep `available_date < cutoff` with cutoff = D-1, so
# an item whose proxy availability date is D-1 is NOT shown (conservative).
FILINGS_SHOW_ITEMS_AVAILABLE_ON_CUTOFF = False
