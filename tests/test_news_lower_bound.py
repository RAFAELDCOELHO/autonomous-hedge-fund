"""PR #65 review (Lingxi, mutant M5): the LOWER bound of the yfinance news window.

``_published_in_window`` (tradingagents/dataflows/yfinance_news.py) keeps an
article iff ``start 00:00:00 UTC <= pub < (end + 1 day) 00:00:00 UTC``. The
lower bound is INCLUSIVE and is UTC midnight of a calendar date, for both
tools, each from its own parameter:

- get_news_yfinance(ticker, start_date, end_date): lower = start_date 00:00 UTC;
- get_global_news_yfinance(curr_date, look_back_days, limit): lower =
  (curr_date - look_back_days calendar days) 00:00 UTC.

Before this file, replacing the lower-bound check with ``True`` left the whole
suite green. Here, per tool and item format (nested ``content.pubDate`` /
flat ``providerPublishTime``): an item exactly AT the lower-bound instant is
shown, one 1 second BEFORE it is hidden, a control well inside the window is
shown and an item 30 days before the bound is hidden. All items are before
the upper cutoff (end + 1 day) 00:00 UTC. Process TZ is UTC-3 (as in the
cutoff tests) so a local-midnight bound would not pass by accident.
Offline: yfinance Ticker/Search are faked (tests/_news_fakes.py).
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta

import pytest

from _news_fakes import FakeNews, content_item, epoch, flat_item, set_process_tz

END = "2024-01-17"  # end_date / curr_date (D-1); upper cutoff 2024-01-18T00:00:00Z
CONTROL = "2024-01-16T12:00:00"  # inside every window below
FORMATS = ["content", "flat"]

# ticker: start_date -> lower bound; global: look_back_days -> lower bound
TICKER_STARTS = ["2024-01-10", "2024-01-14"]
GLOBAL_LOOK_BACKS = {7: "2024-01-10", 3: "2024-01-14", 20: "2023-12-28"}  # last one crosses a year


@pytest.fixture(autouse=True)
def _local_tz_not_utc(monkeypatch):
    set_process_tz(monkeypatch, "Etc/GMT+3")
    yield
    monkeypatch.undo()
    time.tzset()


def _shift(iso_utc: str, **delta) -> str:
    return (datetime.fromisoformat(iso_utc) + timedelta(**delta)).isoformat()


def _item(fmt: str, token: str, iso_utc: str) -> dict:
    """iso_utc: 'YYYY-MM-DDTHH:MM:SS' in UTC."""
    if fmt == "content":
        return content_item(token, f"{iso_utc}Z")
    return flat_item(token, epoch(iso_utc))


def _boundary_items(fmt: str, lower_day: str) -> tuple[list, list, list]:
    """(items, tokens that must be shown, tokens that must be hidden)."""
    at = f"{lower_day}T00:00:00"
    spec = {
        "TOK_AT_LOWER": (at, True),
        "TOK_LOWER_MINUS_1S": (_shift(at, seconds=-1), False),
        "TOK_CONTROL": (CONTROL, True),
        "TOK_LOWER_MINUS_30D": (_shift(at, days=-30), False),
    }
    items = [_item(fmt, t, ts) for t, (ts, _) in spec.items()]
    return items, [t for t, (_, s) in spec.items() if s], [t for t, (_, s) in spec.items() if not s]


def _assert(out, shown, hidden):
    assert "Error fetching" not in out, out
    for t in shown:
        assert t in out, f"{t} should be shown:\n{out}"
    for t in hidden:
        assert t not in out, f"{t} should be hidden:\n{out}"


def _ticker(monkeypatch, items, start):
    from tradingagents.dataflows import yfinance_news as yfn

    FakeNews(items).install(monkeypatch)
    return yfn.get_news_yfinance("AAPL", start, END)


def _global(monkeypatch, items, look_back):
    from tradingagents.dataflows import yfinance_news as yfn

    FakeNews(items).install(monkeypatch)
    return yfn.get_global_news_yfinance(END, look_back, 50)


# get_news_yfinance: lower bound = start_date 00:00 UTC, inclusive ------------


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("start", TICKER_STARTS)
def test_ticker_news_lower_bound_is_start_date_midnight_utc_inclusive(monkeypatch, start, fmt):
    items, shown, hidden = _boundary_items(fmt, start)
    _assert(_ticker(monkeypatch, items, start), shown, hidden)


# get_global_news_yfinance: lower bound = (curr_date - look_back_days) 00:00 UTC, inclusive


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("look_back", list(GLOBAL_LOOK_BACKS))
def test_global_news_lower_bound_is_curr_date_minus_look_back_midnight_utc_inclusive(
    monkeypatch, look_back, fmt
):
    items, shown, hidden = _boundary_items(fmt, GLOBAL_LOOK_BACKS[look_back])
    _assert(_global(monkeypatch, items, look_back), shown, hidden)


# The bound is a UTC instant: an aware pubDate in another offset is converted --


@pytest.mark.parametrize("tool", ["ticker", "global"])
def test_lower_bound_compares_utc_instants_not_local_dates(monkeypatch, tool):
    """Lower bound 2024-01-10T00:00:00Z. 2024-01-09T21:00:00-03:00 is exactly
    that instant (shown); 2024-01-09T20:59:59-03:00 is 1 s before (hidden);
    2024-01-10T01:00:00+02:00 is 2024-01-09T23:00Z (hidden, though its local
    date is the start date); a naive stamp is read as UTC."""
    items = [
        content_item("TOK_M0300_AT_LOWER", "2024-01-09T21:00:00-03:00"),
        content_item("TOK_M0300_LOWER_MINUS_1S", "2024-01-09T20:59:59-03:00"),
        content_item("TOK_P0200_IS_PREV_DAY_2300Z", "2024-01-10T01:00:00+02:00"),
        content_item("TOK_NAIVE_AT_LOWER", "2024-01-10T00:00:00"),
        content_item("TOK_NAIVE_LOWER_MINUS_1S", "2024-01-09T23:59:59"),
        content_item("TOK_CONTROL", f"{CONTROL}Z"),
    ]
    out = (_ticker(monkeypatch, items, "2024-01-10") if tool == "ticker"
           else _global(monkeypatch, items, 7))
    _assert(out, ["TOK_M0300_AT_LOWER", "TOK_NAIVE_AT_LOWER", "TOK_CONTROL"],
            ["TOK_M0300_LOWER_MINUS_1S", "TOK_P0200_IS_PREV_DAY_2300Z", "TOK_NAIVE_LOWER_MINUS_1S"])


def test_window_headers_name_the_lower_bound_date(monkeypatch):
    """The returned header states the window each tool actually applied."""
    items = [content_item("TOK_CONTROL", f"{CONTROL}Z")]
    assert "from 2024-01-10 to 2024-01-17" in _ticker(monkeypatch, items, "2024-01-10")
    assert "from 2024-01-14 to 2024-01-17" in _global(monkeypatch, items, 3)
