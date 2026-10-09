"""Fixed exchange trading calendars (P4.11): D-1 comes from the calendar, not the vendor."""

from __future__ import annotations

from functools import cache

import exchange_calendars as xcals
import pandas as pd

from .risk_free import market_of

EXCHANGE_BY_MARKET = {"BR": "BVMF", "US": "XNYS"}  # B3, NYSE


def exchange_for(ticker: str) -> str:
    return EXCHANGE_BY_MARKET[market_of(ticker)]


@cache
def _calendar(name: str):
    return xcals.get_calendar(name)


def sessions(ticker: str, start, end) -> pd.DatetimeIndex:
    """The ticker's exchange sessions in [start, end] (tz-naive dates)."""
    return _calendar(exchange_for(ticker)).sessions_in_range(pd.Timestamp(start), pd.Timestamp(end))


def previous_session(ticker: str, date) -> pd.Timestamp:
    """The last session strictly before `date` on the ticker's exchange."""
    cal = _calendar(exchange_for(ticker))
    date = pd.Timestamp(date)
    if cal.is_session(date):
        return cal.previous_session(date)
    return cal.date_to_session(date, direction="previous")
