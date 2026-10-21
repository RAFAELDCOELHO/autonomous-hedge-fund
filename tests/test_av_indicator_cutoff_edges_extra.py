"""Edge cases for the Alpha Vantage indicator date window.

``get_indicator`` keeps a sample when
``(curr_date - look_back_days) <= row date <= curr_date`` (calendar days, both
ends inclusive). The next calendar day stays out, including the Saturday after
a Friday and the Tuesday after MLK. An unparseable ``time`` is dropped, and a
CSV with no ``time`` column fails closed.
"""

from __future__ import annotations

from tradingagents.dataflows import alpha_vantage_indicator as avi


def _install(monkeypatch, csv: str) -> None:
    def _fake(_function_name, _params):
        return csv

    monkeypatch.setattr(avi, "_make_api_request", _fake)


def _lines(out: str) -> list[str]:
    return [
        line
        for line in out.splitlines()
        if len(line) >= 10 and line[:4].isdigit() and line[4] == "-"
    ]


def test_both_bounds_included_next_session_hidden(monkeypatch):
    """Wednesday 2024-01-10, look_back 7 days → 2024-01-03. Thursday stays out."""
    _install(
        monkeypatch,
        "time,RSI\n"
        "2024-01-02,111111\n"
        "\n"
        "2024-01-03,222222\n"
        "2024-01-10,333333\n"
        "2024-01-11,444444\n",
    )

    out = avi.get_indicator("AAPL", "rsi", "2024-01-10", 7)

    assert _lines(out) == ["2024-01-03: 222222", "2024-01-10: 333333"]
    assert "from 2024-01-03 to 2024-01-10" in out
    assert "111111" not in out and "444444" not in out


def test_friday_cutoff_hides_saturday_and_post_holiday_session(monkeypatch):
    """Friday 2024-01-12. Saturday and Tuesday 2024-01-16 (after MLK) stay out.

    ``time`` is not the first column, so a column-0 date read cannot pass.
    """
    _install(
        monkeypatch,
        "RSI,time\n"
        "111111,2024-01-04\n"
        "222222,2024-01-05\n"
        "333333,2024-01-12\n"
        "444444,2024-01-13\n"
        "555555,2024-01-16\n",
    )

    out = avi.get_indicator("AAPL", "rsi", "2024-01-12", 7)

    assert _lines(out) == ["2024-01-05: 222222", "2024-01-12: 333333"]
    assert "111111" not in out and "444444" not in out and "555555" not in out


def test_leap_day_look_back_zero_is_that_day_only(monkeypatch):
    _install(
        monkeypatch,
        "time,RSI\n"
        "2024-02-28,111111\n"
        "2024-02-29,222222\n"
        "2024-03-01,444444\n",
    )

    out = avi.get_indicator("AAPL", "rsi", "2024-02-29", 0)

    assert _lines(out) == ["2024-02-29: 222222"]
    assert "from 2024-02-29 to 2024-02-29" in out
    assert "111111" not in out and "444444" not in out


def test_march_first_look_back_one_includes_leap_day(monkeypatch):
    """2024-03-01 minus one calendar day is 2024-02-29, not 2024-02-28."""
    _install(
        monkeypatch,
        "time,RSI\n"
        "2024-02-28,111111\n"
        "2024-02-29,222222\n"
        "2024-03-01,333333\n"
        "2024-03-02,444444\n",
    )

    out = avi.get_indicator("AAPL", "rsi", "2024-03-01", 1)

    assert _lines(out) == ["2024-02-29: 222222", "2024-03-01: 333333"]
    assert "from 2024-02-29 to 2024-03-01" in out
    assert "111111" not in out and "444444" not in out


def test_look_back_crosses_new_year(monkeypatch):
    """2024-01-02 minus 10 days is 2023-12-23. Dec 22 and Jan 3 stay out."""
    _install(
        monkeypatch,
        "time,RSI\n"
        "2023-12-22,111111\n"
        "2023-12-23,222222\n"
        "2024-01-02,333333\n"
        "2024-01-03,444444\n",
    )

    out = avi.get_indicator("AAPL", "rsi", "2024-01-02", 10)

    assert _lines(out) == ["2023-12-23: 222222", "2024-01-02: 333333"]
    assert "from 2023-12-23 to 2024-01-02" in out
    assert "111111" not in out and "444444" not in out


def test_unparseable_time_is_dropped(monkeypatch):
    _install(
        monkeypatch,
        "time,RSI\n"
        "not-a-date,999999\n"
        "2024-01-12,333333\n",
    )

    out = avi.get_indicator("AAPL", "rsi", "2024-01-12", 7)

    assert _lines(out) == ["2024-01-12: 333333"]
    assert "999999" not in out


def test_missing_time_column_fails_closed(monkeypatch):
    _install(monkeypatch, "value,note\n999999,future-leak\n")

    out = avi.get_indicator("AAPL", "rsi", "2024-01-12", 7)

    assert out.startswith("Error: 'time' column not found")
    assert "999999" not in out and "future-leak" not in out
