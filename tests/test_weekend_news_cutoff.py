"""PR #65 review (Lingxi): the yfinance news cutoff across a weekend / holiday gap.

The news tools receive the data cutoff D-1 (the exchange session before the
decision session D) and keep an article only if it was published strictly
before 00:00:00 UTC of the CALENDAR day after D-1
(``_published_in_window`` in tradingagents/dataflows/yfinance_news.py).

tests/test_pr64_yfinance_news_cutoff.py only uses a D that follows D-1 on the
calendar (Wed -> Thu), where "calendar day after D-1" and "next trading day
after D-1" (= D) give the same instant. So the mutant
``cutoff = next trading session after D-1 at 00:00 UTC`` (Monday 00:00 for a
Friday D-1) survived it. Here D is a Monday, the Tuesday after a holiday
Monday (NYSE) or the Ash Wednesday after Carnaval (B3); D-1 is the Friday
before, and anything published from Saturday 00:00 UTC on must be hidden,
while Friday 23:59 UTC stays visible.

D-1 is derived from the ticker's exchange calendar (tradingagents.backtest.
calendar: NYSE for AAPL, B3/BVMF for PETR4.SA), as the runner does, and
checked to be the Friday. Ticker news is fetched for the case's ticker.
Global news takes no ticker: its cutoff is the calendar day after the
curr_date it is given (the graph's trade_date = D-1 of the traded ticker),
so the exchange enters only through D-1; it is run with the same D-1 to pin
that the same instant applies. Offline: yfinance Ticker/Search are faked
(tests/_news_fakes.py).
"""

from __future__ import annotations

import time
from datetime import date, timedelta

import pytest

from _news_fakes import FakeNews, content_item, epoch, flat_item, set_process_tz

TOOLS = ["ticker", "global"]
FORMATS = ["content", "flat"]

# case -> (ticker, exchange, D, expected D-1 = previous session on that exchange,
#          extra hidden instants (UTC) on the closed weekdays between D-1 and D)
CASES = {
    # NYSE. Mon 2024-03-04; D-1 = Fri 2024-03-01; gap = Sat, Sun.
    "monday": ("AAPL", "XNYS", "2024-03-04", "2024-03-01", []),
    # NYSE. Tue 2024-02-20 after Presidents' Day (Mon 2024-02-19 closed);
    # D-1 = Fri 2024-02-16; gap = Sat, Sun and the holiday Monday.
    "holiday_monday": ("AAPL", "XNYS", "2024-02-20", "2024-02-16", ["2024-02-19T12:00:00"]),
    # B3. Ash Wednesday 2024-02-14, first session after Carnaval (Mon 02-12 and
    # Tue 02-13 closed); D-1 = Fri 2024-02-09; gap = Sat, Sun, Carnival Mon, Tue.
    "b3_carnaval": ("PETR4.SA", "BVMF", "2024-02-14", "2024-02-09",
                    ["2024-02-12T12:00:00", "2024-02-13T23:59:59"]),
}


@pytest.fixture(autouse=True)
def _local_tz_not_utc(monkeypatch):
    # Non-UTC process timezone (UTC-3, like the box), as in the PR #64 cutoff tests.
    set_process_tz(monkeypatch, "Etc/GMT+3")
    yield
    monkeypatch.undo()
    time.tzset()


def _d_minus_1(case: str) -> str:
    from tradingagents.backtest.calendar import previous_session

    ticker, _, d, expected_prev, _ = CASES[case]
    prev = previous_session(ticker, d).strftime("%Y-%m-%d")
    assert prev == expected_prev, (case, prev)
    return prev


def _plus_days(iso_day: str, n: int) -> str:
    return (date.fromisoformat(iso_day) + timedelta(days=n)).isoformat()


def _item(fmt: str, token: str, iso_utc: str) -> dict:
    """iso_utc: 'YYYY-MM-DDTHH:MM:SS' in UTC."""
    if fmt == "content":
        return content_item(token, f"{iso_utc}Z")
    return flat_item(token, epoch(iso_utc))


def _fetch(monkeypatch, tool: str, items, case: str, cutoff: str) -> str:
    from tradingagents.dataflows import yfinance_news as yfn

    FakeNews(items).install(monkeypatch)
    if tool == "ticker":
        out = yfn.get_news_yfinance(CASES[case][0], _plus_days(cutoff, -7), cutoff)
    else:
        out = yfn.get_global_news_yfinance(cutoff, 7, 50)
    assert "Error fetching" not in out, out
    return out


@pytest.mark.parametrize("case", list(CASES))
def test_d_minus_1_is_the_friday_before_d(case):
    from tradingagents.backtest.calendar import exchange_for, next_session, sessions

    ticker, exchange, d, _, extra = CASES[case]
    assert exchange_for(ticker) == exchange
    prev = _d_minus_1(case)
    assert date.fromisoformat(prev).weekday() == 4  # Friday
    # Mon / Tue / Wed after 0 / 1 / 2 closed weekdays.
    assert date.fromisoformat(d).weekday() == len(extra)
    # No session strictly between D-1 and D.
    assert list(sessions(ticker, prev, d).strftime("%Y-%m-%d")) == [prev, d]
    # D is the next session after D-1, so "next trading day after D-1" = D,
    # which is NOT the calendar day after D-1 (the gap the tests below cover).
    assert next_session(ticker, prev).strftime("%Y-%m-%d") == d
    assert _plus_days(prev, 1) != d


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("tool", TOOLS)
@pytest.mark.parametrize("case", list(CASES))
def test_saturday_news_excluded_friday_2359_included(monkeypatch, case, tool, fmt):
    """Required case: D-1 = Friday, D after the weekend. Saturday 12:00 UTC is
    after the cutoff (Sat 00:00 UTC) and must be hidden; Friday 23:59 UTC is
    the control."""
    fri = _d_minus_1(case)
    sat = _plus_days(fri, 1)
    items = [
        _item(fmt, "TOK_SAT_1200Z", f"{sat}T12:00:00"),
        _item(fmt, "TOK_FRI_2359Z", f"{fri}T23:59:00"),
    ]
    out = _fetch(monkeypatch, tool, items, case, fri)
    assert "TOK_FRI_2359Z" in out, f"Friday 23:59 UTC control should be shown:\n{out}"
    assert "TOK_SAT_1200Z" not in out, f"Saturday news leaked into D-1 = {fri}:\n{out}"


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("tool", TOOLS)
@pytest.mark.parametrize("case", list(CASES))
def test_whole_non_session_gap_before_d_is_excluded(monkeypatch, case, tool, fmt):
    """Everything from Saturday 00:00:00 UTC (the exact cutoff instant, strict
    '<') through D 00:00 UTC is hidden, including holiday / Carnival weekdays;
    Friday 12:00 and the last second of Friday are shown."""
    _, _, d, _, extra = CASES[case]
    fri = _d_minus_1(case)
    sat, sun = _plus_days(fri, 1), _plus_days(fri, 2)
    hidden = {
        "TOK_SAT_0000Z": f"{sat}T00:00:00",
        "TOK_SAT_1200Z": f"{sat}T12:00:00",
        "TOK_SUN_235959Z": f"{sun}T23:59:59",
        "TOK_D_0000Z": f"{d}T00:00:00",
    }
    for ts in extra:
        hidden[f"TOK_CLOSED_{ts[:10]}_{ts[11:].replace(':', '')}Z"] = ts
    shown = {"TOK_FRI_235959Z": f"{fri}T23:59:59", "TOK_FRI_1200Z": f"{fri}T12:00:00"}
    items = [_item(fmt, t, ts) for t, ts in {**hidden, **shown}.items()]
    out = _fetch(monkeypatch, tool, items, case, fri)
    for t in shown:
        assert t in out, f"{t} should be shown:\n{out}"
    for t in hidden:
        assert t not in out, f"{t} leaked into D-1 = {fri}:\n{out}"
