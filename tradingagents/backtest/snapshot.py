"""Offline completeness check of the H1 price snapshot (P4.11).

Same requirements as runner._execution_plan, checked for every ticker at once
and listing every gap instead of stopping at the first. Per the ticker's fixed
exchange calendar (calendar.py):

- each session D in [start, end] needs a bar with a finite positive Open;
- the D-1 of the first session needs a bar (the agent's first cutoff);
- the D-1 of every session D needs a finite positive Close (the decision for
  D uses data up to that close); the last session's Close is not required,
  since it is only the D-1 of the exit, where no decision is taken;
- the session after ``end`` (the exit) needs a bar with a finite positive Open;
- no vendor date up to the exit session may appear twice (the runner raises on
  a duplicate anywhere in the frame it loads, look-back included).

A ``prices_by_ticker`` Date column of text is the calendar day written in each
cell. Offsets that span at most 60 minutes (``-05:00`` and ``-04:00`` across a
civil DST change) are read that way. A naive value beside an offset, a wider
span, or two cells that write one day raises ValueError. A datetime64 column
already has one zone, including a named zone whose UTC offset changes over the
year, and keeps that written day.

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


# A calendar day ends in "-DD". That is not an offset. These are: "Z", "+",
# "±HH:MM", "±HHMM", and a short negative offset after a time ("T00:30:00-05").
_OFFSET_MARKER_RE = r"(?:Z|\+|[+-]\d{2}:\d{2}|[+-]\d{4}|T\d{2}:\d{2}:\d{2}-\d{2}$)"

# A civil DST transition shifts the clock by at most one hour (US/Eastern
# -05:00 vs -04:00; Lord Howe by 30 minutes). A wider mix is not that case.
_MAX_DST_OFFSET_SPAN_MINUTES = 60


def _missing_date_cell(value) -> bool:
    if isinstance(value, str):
        return not value.strip()
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return bool(missing)


def _offset_minutes(parsed: pd.Timestamp) -> int | None:
    if getattr(parsed, "tzinfo", None) is None:
        return None
    offset = parsed.utcoffset()
    if offset is None:
        return None
    return int(offset.total_seconds() // 60)


def _rewrite_mixed_offset_dates(frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Read a DST-sized mix of text offsets as the written calendar day.

    Column-wise ``pd.to_datetime`` raises ``Mixed timezones detected`` on
    ``-05:00`` next to ``-04:00``. A span of at most 60 minutes is one civil
    zone, so each cell becomes the day written in it and the column parser
    does not see mixed offsets. A datetime64 index already has one zone and
    is left for the wall-clock conversion. A single offset is left too.

    Raises ValueError when the column mixes naive and aware values, when the
    offsets span more than 60 minutes, or when two cells write one day.
    A value that is not a timestamp is dropped, same as a NaT row.
    """
    index = frame.index
    if isinstance(index, pd.DatetimeIndex) or pd.api.types.is_datetime64_any_dtype(index):
        return frame
    labels = pd.Series(list(index), dtype="string")
    if not labels.str.contains(_OFFSET_MARKER_RE, regex=True, na=False).any():
        return frame

    saw_naive = False
    offsets: set[int] = set()
    parsed_at: list[tuple[int, pd.Timestamp]] = []
    for pos, value in enumerate(index):
        if _missing_date_cell(value):
            continue
        parsed = pd.to_datetime(value, errors="coerce")
        if pd.isna(parsed):
            continue
        offset = _offset_minutes(parsed)
        if offset is None:
            saw_naive = True
        else:
            offsets.add(offset)
            parsed_at.append((pos, parsed))
        if saw_naive and offsets:
            raise ValueError(f"{ticker}: Date mixes timezone-aware and naive timestamps")
    if len(offsets) <= 1:
        return frame
    if max(offsets) - min(offsets) > _MAX_DST_OFFSET_SPAN_MINUTES:
        raise ValueError(f"{ticker}: Date mixes UTC offsets")

    seen: set[pd.Timestamp] = set()
    keep: list[int] = []
    days: list[pd.Timestamp] = []
    for pos, parsed in parsed_at:
        day = parsed.tz_localize(None).normalize()
        if day in seen:
            raise ValueError(f"{ticker}: Date normalizes to duplicate calendar days")
        seen.add(day)
        keep.append(pos)
        days.append(day)
    rewritten = frame.iloc[keep].copy()
    rewritten.index = pd.DatetimeIndex(days, name=index.name)
    return rewritten


def _frame(ticker: str, exit_day: pd.Timestamp, given: Mapping[str, pd.DataFrame] | None):
    if given is not None:
        frame = given.get(ticker)
    else:
        frame = load_ohlcv(ticker, exit_day.strftime("%Y-%m-%d"), fill_prices=False)
    if frame is None:
        return None
    frame = frame.set_index("Date") if "Date" in frame.columns else frame
    # Text ``-05:00``/``-04:00`` is not one dtype. Read each written day when
    # the offsets span at most an hour; fail closed otherwise.
    frame = _rewrite_mixed_offset_dates(frame, ticker)
    # Wall clock, not UTC. ``2024-01-05T00:30+09:00`` is 5 Jan as written
    # (UTC would make it 4 Jan and satisfy the previous session).
    index = pd.DatetimeIndex(pd.to_datetime(frame.index))
    if index.tz is not None:
        index = index.tz_localize(None)
    return frame.set_axis(index.normalize())


def _price_ok(frame: pd.DataFrame, day: pd.Timestamp, column: str) -> bool:
    """A finite positive price; a missing column or a value float() rejects is a defect, not an error."""
    if column not in frame.columns:
        return False
    value = frame.loc[day, column]
    value = value.iloc[-1] if isinstance(value, pd.Series) else value
    try:
        value = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(value) and value > 0


def validate_snapshot(
    tickers: Iterable[str] | None = None,
    start: str | None = None,
    end: str | None = None,
    prices_by_ticker: Mapping[str, pd.DataFrame] | None = None,
) -> dict[str, dict]:
    """Return ``{ticker: {exchange, n_sessions, first_d_minus_1, exit_session}}`` or raise.

    Raises SnapshotIncomplete (a runner.DataDefectError) whose ``missing`` names every gap.
    A Date column of text keeps the calendar day written in each cell. Offsets
    spanning more than 60 minutes, a mix of naive and offset values, or two
    cells that write one calendar day raise ValueError.
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
        close_days = {first_prev, *window[:-1]}
        checks = [(first_prev, False)] + [(d, True) for d in window] + [(exit_day, True)]
        for day, needs_open in checks:
            key = day.strftime("%Y-%m-%d")
            if frame is None or day not in frame.index:
                missing.append((ticker, key, FIELD_BAR))
            else:
                if needs_open and not _price_ok(frame, day, "Open"):
                    missing.append((ticker, key, FIELD_OPEN))
                if day in close_days and not _price_ok(frame, day, "Close"):
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
