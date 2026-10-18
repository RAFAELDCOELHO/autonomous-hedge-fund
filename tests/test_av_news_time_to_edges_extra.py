"""Edge cases for the Alpha Vantage news ``time_to`` ceiling.

Audit §10.1: ``alpha_vantage_news`` sends ``time_to = format_datetime_for_api(date)``
which is ``YYYYMMDDT0000`` (00:00 at the start of that calendar day). The whole
day is excluded. Unlike the yfinance window, this path does not add one day.
"""

from __future__ import annotations

from tradingagents.dataflows import alpha_vantage_news


def _capture(monkeypatch):
    seen = []

    def _fake_api(function_name, params):
        seen.append((function_name, dict(params)))
        return "offline"

    monkeypatch.setattr(alpha_vantage_news, "_make_api_request", _fake_api)
    return seen


def test_friday_end_date_time_to_is_that_midnight_not_the_next_day(monkeypatch):
    """D-1 = 2024-01-12 (Friday before the MLK weekend). time_to stays Friday 00:00.

    A yfinance-style ``end_date + 1 day`` would ask for Saturday 2024-01-13 and
    admit Friday's news. T2359 would admit the whole of Friday.
    """
    seen = _capture(monkeypatch)

    out = alpha_vantage_news.get_news("AAPL", "2024-01-05", "2024-01-12")

    assert out == "offline"
    assert seen == [
        (
            "NEWS_SENTIMENT",
            {
                "tickers": "AAPL",
                "time_from": "20240105T0000",
                "time_to": "20240112T0000",
            },
        )
    ]


def test_later_start_date_does_not_drag_time_to_forward(monkeypatch):
    """An uncapped start_date after end_date must not raise the ceiling."""
    seen = _capture(monkeypatch)

    alpha_vantage_news.get_news("AAPL", "2024-06-01", "2024-01-12")

    assert seen == [
        (
            "NEWS_SENTIMENT",
            {
                "tickers": "AAPL",
                "time_from": "20240601T0000",
                "time_to": "20240112T0000",
            },
        )
    ]


def test_global_look_back_does_not_move_time_to(monkeypatch):
    """look_back_days shifts only time_from, including across a leap-year February."""
    seen = _capture(monkeypatch)

    alpha_vantage_news.get_global_news("2024-03-01", look_back_days=40, limit=3)

    assert seen == [
        (
            "NEWS_SENTIMENT",
            {
                "topics": "financial_markets,economy_macro,economy_monetary",
                "time_from": "20240121T0000",
                "time_to": "20240301T0000",
                "limit": "3",
            },
        )
    ]


def test_global_look_back_zero_on_leap_day_stays_on_that_midnight(monkeypatch):
    """look_back_days=0 is a single midnight, not the next calendar day."""
    seen = _capture(monkeypatch)

    alpha_vantage_news.get_global_news("2024-02-29", look_back_days=0, limit=1)

    assert seen == [
        (
            "NEWS_SENTIMENT",
            {
                "topics": "financial_markets,economy_macro,economy_monetary",
                "time_from": "20240229T0000",
                "time_to": "20240229T0000",
                "limit": "1",
            },
        )
    ]
