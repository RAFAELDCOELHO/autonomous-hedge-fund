"""Edge cases for the Alpha Vantage daily CSV date window.

``get_stock`` keeps a row when ``start_date <= timestamp <= end_date`` and
drops the next calendar day, including the Saturday after a Friday and the
session after a holiday. History before ``start_date`` stays out. The date is
the first column, whatever its name.

An unreadable timestamp is dropped. A timestamp with a timezone keeps the
calendar day written in the cell: the offset is removed and the clock is not
converted to UTC. Kept cells are the original text. A failure that still
escapes the filter returns only the CSV header.
"""

from __future__ import annotations

from tradingagents.dataflows import alpha_vantage_stock as avs

SENTINEL = "987654"


def _load(monkeypatch, csv: str, start: str, end: str) -> str:
    monkeypatch.setattr(avs, "_make_api_request", lambda *_a, **_k: csv)
    return avs.get_stock("AAPL", start, end)


def _dates(out: str) -> list[str]:
    return [line.split(",")[0][:10] for line in out.splitlines()[1:] if line.strip()]


def test_both_bounds_included_next_calendar_day_hidden(monkeypatch):
    """Wednesday 2024-01-10 stays. Tuesday 2024-01-09 stays. Thursday stays out."""
    out = _load(
        monkeypatch,
        "timestamp,open\n"
        "2024-01-08,987654\n"
        "2024-01-09,9\n"
        "2024-01-10,10\n"
        "2024-01-11,987654\n",
        "2024-01-09",
        "2024-01-10",
    )

    assert _dates(out) == ["2024-01-09", "2024-01-10"]
    assert SENTINEL not in out


def test_friday_end_hides_saturday_and_post_holiday_session(monkeypatch):
    """Friday 2024-01-12. Saturday and Tuesday 2024-01-16 (after MLK) stay out."""
    out = _load(
        monkeypatch,
        "timestamp,open\n"
        "2024-01-10,987654\n"
        "2024-01-11,11\n"
        "2024-01-12,12\n"
        "2024-01-13,987654\n"
        "2024-01-15,987654\n"
        "2024-01-16,987654\n",
        "2024-01-11",
        "2024-01-12",
    )

    assert _dates(out) == ["2024-01-11", "2024-01-12"]
    assert SENTINEL not in out


def test_leap_day_end_keeps_that_day_and_hides_march_first(monkeypatch):
    """2024-02-29 is a real session. 2024-03-01 is the next calendar day."""
    out = _load(
        monkeypatch,
        "timestamp,open\n"
        "2024-02-27,987654\n"
        "2024-02-28,28\n"
        "2024-02-29,29\n"
        "2024-03-01,987654\n",
        "2024-02-28",
        "2024-02-29",
    )

    assert _dates(out) == ["2024-02-28", "2024-02-29"]
    assert SENTINEL not in out


def test_window_keeps_prior_year_and_hides_next_session(monkeypatch):
    """2024-01-02 keeps 2023-12-22. 2023-12-21 and 2024-01-03 stay out."""
    out = _load(
        monkeypatch,
        "timestamp,open\n"
        "2023-12-21,987654\n"
        "2023-12-22,22\n"
        "2024-01-02,2\n"
        "2024-01-03,987654\n",
        "2023-12-22",
        "2024-01-02",
    )

    assert _dates(out) == ["2023-12-22", "2024-01-02"]
    assert SENTINEL not in out


def test_rows_strictly_after_end_yield_no_prices(monkeypatch):
    out = _load(
        monkeypatch,
        "timestamp,open\n"
        "2024-01-13,987654\n"
        "2024-01-16,987654\n",
        "2024-01-12",
        "2024-01-12",
    )

    assert _dates(out) == []
    assert SENTINEL not in out


def test_first_column_is_the_date_even_when_not_named_timestamp(monkeypatch):
    """A later column named timestamp must not decide the window."""
    out = _load(
        monkeypatch,
        "Date,timestamp,open\n"
        "2024-01-12,2024-01-12,12\n"
        "2024-01-13,2024-01-12,987654\n",
        "2024-01-12",
        "2024-01-12",
    )

    assert _dates(out) == ["2024-01-12"]
    assert SENTINEL not in out
    assert "2024-01-13" not in out


def test_unparseable_timestamp_does_not_echo_later_prices(monkeypatch):
    """A bad timestamp must not publish prices from after end_date."""
    out = _load(
        monkeypatch,
        "timestamp,open\n"
        "2024-01-12,12\n"
        "not-a-date,1\n"
        "2024-01-13,987654\n",
        "2024-01-12",
        "2024-01-12",
    )

    assert _dates(out) == ["2024-01-12"]
    assert SENTINEL not in out
    assert "2024-01-13" not in out
    assert "not-a-date" not in out


def test_offset_timestamp_keeps_end_date_and_hides_the_next_day(monkeypatch):
    """``2024-01-12T00:00:00-05:00`` is the end date. The next session stays out."""
    out = _load(
        monkeypatch,
        "timestamp,open\n"
        "2024-01-11T00:00:00-05:00,987654\n"
        "2024-01-12T00:00:00-05:00,12\n"
        "2024-01-13T00:00:00-05:00,987654\n",
        "2024-01-12",
        "2024-01-12",
    )

    assert _dates(out) == ["2024-01-12"]
    assert SENTINEL not in out
    assert "2024-01-13" not in out


def test_date_only_and_offset_rows_share_one_window(monkeypatch):
    """A date-only row must not cause the offset row on that day to be dropped."""
    out = _load(
        monkeypatch,
        "timestamp,open\n"
        "2024-01-12,12\n"
        "2024-01-12T00:00:00-05:00,13\n"
        "2024-01-13T00:00:00-05:00,987654\n",
        "2024-01-12",
        "2024-01-12",
    )

    assert _dates(out) == ["2024-01-12", "2024-01-12"]
    assert SENTINEL not in out
    assert "2024-01-13" not in out


def test_positive_offset_uses_the_written_day(monkeypatch):
    """``2024-01-13T00:30+09:00`` is 13 Jan as written. UTC would keep it on 12 Jan."""
    out = _load(
        monkeypatch,
        "timestamp,open\n"
        "2024-01-12T00:00-05:00,12\n"
        "2024-01-13T00:30+09:00,987654\n",
        "2024-01-12",
        "2024-01-12",
    )

    assert _dates(out) == ["2024-01-12"]
    assert SENTINEL not in out
    assert "2024-01-13" not in out


def test_jan_10_to_12_window_keeps_written_day_and_original_text(monkeypatch):
    """Local day over 2024-01-10 .. 2024-01-12. The kept timestamp is not rewritten.

    ``2024-01-13T00:30+09:00`` is 13 Jan and is dropped.
    ``2024-01-13T03:00+05:00`` is 13 Jan and is dropped.
    ``2024-01-12T20:00-05:00`` is 12 Jan and stays, offset included.
    ``2024-01-09T22:00-05:00`` is 9 Jan and is dropped.
    """
    out = _load(
        monkeypatch,
        "timestamp,open\n"
        "2024-01-09T22:00-05:00,987654\n"
        "2024-01-12T20:00-05:00,12\n"
        "2024-01-13T00:30+09:00,987654\n"
        "2024-01-13T03:00+05:00,987654\n",
        "2024-01-10",
        "2024-01-12",
    )

    assert out.splitlines() == ["timestamp,open", "2024-01-12T20:00-05:00,12"]
    assert SENTINEL not in out


def test_offset_row_roundtrips_identical_text(monkeypatch):
    """``2024-01-12T23:00-05:00,1`` comes back unchanged."""
    line = "2024-01-12T23:00-05:00,1"
    out = _load(
        monkeypatch,
        "timestamp,open\n" + line + "\n",
        "2024-01-12",
        "2024-01-12",
    )

    assert out == "timestamp,open\n" + line + "\n"


def test_naive_intraday_stays_on_the_written_day(monkeypatch):
    """``2024-01-12 15:00`` is 12 Jan. Without normalize, 15:00 is after midnight."""
    out = _load(
        monkeypatch,
        "timestamp,open\n"
        "2024-01-12 15:00,1\n"
        "2024-01-13 00:00,987654\n",
        "2024-01-12",
        "2024-01-12",
    )

    assert out.splitlines() == ["timestamp,open", "2024-01-12 15:00,1"]
    assert SENTINEL not in out


def test_generic_filter_exception_returns_header_only(monkeypatch):
    """A crash in the filter must not publish the original prices."""

    def _boom(*_args, **_kwargs):
        raise RuntimeError("synthetic parser failure")

    monkeypatch.setattr(
        "tradingagents.dataflows.alpha_vantage_common.pd.read_csv",
        _boom,
    )
    out = _load(
        monkeypatch,
        "timestamp,open\n"
        "2024-01-12,12\n"
        "2024-01-13,987654\n",
        "2024-01-12",
        "2024-01-12",
    )

    assert out.splitlines() == ["timestamp,open"]
    assert _dates(out) == []
    assert SENTINEL not in out
    assert "2024-01-13" not in out
