"""Offline completeness check of the H1 price snapshot (P4.11).

Same requirements as runner._execution_plan, checked for every ticker at once
and listing every gap instead of stopping at the first. Per the ticker's fixed
exchange calendar (calendar.py):

- each session D in [start, end] needs a bar with a finite positive Open;
- the D-1 of the first session needs a bar (the agent's first cutoff) with a
  finite positive Close (the first decision uses data up to that close);
- the session after ``end`` (the exit) needs a bar with a finite positive Open;
- no vendor date up to the exit session may appear twice (the runner raises on
  a duplicate anywhere in the frame it loads, look-back included).

Vendor bars on non-sessions are ignored, so a holiday on only one exchange is
never flagged for the other. Bars are loaded through the runner's own vendor
path (load_ohlcv, auto_adjust=True, fill_prices=False) unless frames are
passed in. Defaults are the pre-registered grid (cells.PREREG_*). This PR only
ships and tests the function; running it before the pilot is a separate step.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping

import pandas as pd

from tradingagents.dataflows.stockstats_utils import load_ohlcv

from .calendar import exchange_for, next_session, previous_session, sessions
from .cells import PREREG_TICKERS, PREREG_WINDOW_END, PREREG_WINDOW_START
from .runner import DataDefectError

FIELD_BAR = "bar"
FIELD_OPEN = "Open"
FIELD_CLOSE = "Close"
FIELD_DUPLICATE = "duplicate"
_DESCRIBE = {
    FIELD_BAR: "no bar",
    FIELD_OPEN: "Open missing/NaN/<=0",
    FIELD_CLOSE: "Close missing/NaN/<=0",
    FIELD_DUPLICATE: "duplicate vendor bars",
}


class SnapshotIncomplete(DataDefectError):
    """Raised by validate_snapshot.

    ``missing`` lists every defect as ``(ticker, "YYYY-MM-DD", "bar" | "Open" | "Close" | "duplicate")``,
    duplicates first for each ticker (one entry per duplicated date).
    """

    def __init__(self, missing: list[tuple[str, str, str]]):
        self.missing = missing
        lines = [f"{t} {d}: {_DESCRIBE[f]}" for t, d, f in missing]
        super().__init__(f"{len(missing)} missing snapshot item(s):\n" + "\n".join(lines))


def prereg_tickers() -> list[str]:
    """The pre-registered tickers in vendor (Yahoo) form: B3 names get ``.SA``."""
    return [t + ".SA" if m == "BR" else t for t, m in PREREG_TICKERS.items()]


def _frame(ticker: str, exit_day: pd.Timestamp, given: Mapping[str, pd.DataFrame] | None):
    if given is not None:
        frame = given.get(ticker)
    else:
        frame = load_ohlcv(ticker, exit_day.strftime("%Y-%m-%d"), fill_prices=False)
    if frame is None:
        return None
    frame = frame.set_index("Date") if "Date" in frame.columns else frame
    return frame.set_axis(pd.to_datetime(frame.index))


def _close_ok(frame: pd.DataFrame, day: pd.Timestamp) -> bool:
    if "Close" not in frame.columns:
        return False
    value = frame.loc[day, "Close"]
    value = value.iloc[-1] if isinstance(value, pd.Series) else value
    try:
        value = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(value) and value > 0


def _open_ok(frame: pd.DataFrame, day: pd.Timestamp) -> bool:
    if "Open" not in frame.columns:
        return False
    value = frame.loc[day, "Open"]
    value = float(value.iloc[-1]) if isinstance(value, pd.Series) else float(value)
    return math.isfinite(value) and value > 0


def validate_snapshot(
    tickers: Iterable[str] | None = None,
    start: str | None = None,
    end: str | None = None,
    prices_by_ticker: Mapping[str, pd.DataFrame] | None = None,
) -> dict[str, dict]:
    """Return ``{ticker: {exchange, n_sessions, first_d_minus_1, exit_session}}`` or raise.

    Raises SnapshotIncomplete (a runner.DataDefectError) whose ``missing`` names every gap.
    """
    tickers = list(tickers) if tickers is not None else prereg_tickers()
    start = start or PREREG_WINDOW_START
    end = end or PREREG_WINDOW_END
    report: dict[str, dict] = {}
    missing: list[tuple[str, str, str]] = []
    for ticker in tickers:
        window = sessions(ticker, start, end)
        if window.empty:
            raise ValueError(f"{ticker}: no {exchange_for(ticker)} sessions in {start}..{end}")
        first_prev = previous_session(ticker, window[0])
        exit_day = next_session(ticker, window[-1])
        frame = _frame(ticker, exit_day, prices_by_ticker)
        if frame is not None:  # same rule as runner._execution_plan, but list them all
            upto = frame.index[frame.index <= exit_day]
            for day in upto[upto.duplicated()].unique().sort_values():
                missing.append((ticker, day.strftime("%Y-%m-%d"), FIELD_DUPLICATE))
        checks = [(first_prev, False)] + [(d, True) for d in window] + [(exit_day, True)]
        for day, needs_open in checks:
            key = day.strftime("%Y-%m-%d")
            if frame is None or day not in frame.index:
                missing.append((ticker, key, FIELD_BAR))
            elif needs_open and not _open_ok(frame, day):
                missing.append((ticker, key, FIELD_OPEN))
            elif day == first_prev and not _close_ok(frame, day):
                missing.append((ticker, key, FIELD_CLOSE))
        report[ticker] = {
            "exchange": exchange_for(ticker),
            "n_sessions": len(window),
            "first_d_minus_1": first_prev.strftime("%Y-%m-%d"),
            "exit_session": exit_day.strftime("%Y-%m-%d"),
        }
    if missing:
        raise SnapshotIncomplete(missing)
    return report
