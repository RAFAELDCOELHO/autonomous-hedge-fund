"""Date text with more than one UTC offset, passed in and never loaded.

``validate_snapshot`` on a single offset drops the offset and keeps the
calendar day written in the cell. A column of text that mixes offsets, such
as ``-05:00`` and ``-04:00`` across DST, never reaches that step:
``pd.to_datetime`` raises ``Mixed timezones detected``.

The same rule as ``load_ohlcv`` applies here. Each cell is the calendar day
written in it. Offsets that span at most 60 minutes are allowed. A mix of
naive and offset values is rejected. So is a wider span, and so are two cells
that write one day.
"""

from __future__ import annotations

import pandas as pd
import pytest

from tradingagents.backtest.snapshot import SnapshotIncomplete, validate_snapshot

_MIXED_TZ = "Mixed timezones"


def _prices(dates: list[str], opens: list[float] | None = None) -> pd.DataFrame:
    count = len(dates)
    if opens is None:
        opens = [100.0 + i for i in range(count)]
    return pd.DataFrame(
        {
            "Date": dates,
            "Open": opens,
            "Close": [90.0 + i for i in range(count)],
        }
    )


def _validate(frame: pd.DataFrame):
    return validate_snapshot(
        ["AAPL"],
        "2024-01-02",
        "2024-01-03",
        prices_by_ticker={"AAPL": frame},
    )


def _dst_dates() -> list[str]:
    """Four required sessions. The exit stamp is 4 Jan as written and 5 Jan in UTC."""
    return [
        "2024-01-04T22:00:00-04:00",
        "2023-12-29 00:00:00-05:00",
        "2024-01-02 00:00:00-05:00",
        "2024-01-03 00:00:00-05:00",
    ]


def test_dst_offset_text_keeps_the_written_day():
    """``-05:00`` and ``-04:00`` are one zone. The evening bar is the 4 Jan exit.

    ``2024-01-04T22:00:00-04:00`` is 5 Jan in UTC. Converting the column to UTC
    would drop the exit session. The column parser raises before either reading.
    """
    report = _validate(_prices(_dst_dates()))

    assert report["AAPL"]["exit_session"] == "2024-01-04"
    assert report["AAPL"]["first_d_minus_1"] == "2023-12-29"
    assert report["AAPL"]["n_sessions"] == 2


def test_dst_evening_open_is_checked_on_the_written_day():
    """A non-positive Open on the evening stamp is an Open defect on 4 Jan.

    UTC would call that stamp 5 Jan and report a missing 4 Jan bar instead.
    """
    frame = _prices(_dst_dates())
    frame.loc[frame["Date"] == "2024-01-04T22:00:00-04:00", "Open"] = 0.0

    with pytest.raises(SnapshotIncomplete) as info:
        _validate(frame)

    assert info.value.missing == [("AAPL", "2024-01-04", "Open")]


def test_dst_offset_text_on_the_index_keeps_the_written_day():
    report = _validate(_prices(_dst_dates()).set_index("Date"))

    assert report["AAPL"]["exit_session"] == "2024-01-04"


def test_next_written_day_does_not_fill_the_exit():
    """``2024-01-05T00:30:00-04:00`` is 5 Jan as written. It does not fill 4 Jan."""
    dates = [
        "2023-12-29 00:00:00-05:00",
        "2024-01-02 00:00:00-05:00",
        "2024-01-03 00:00:00-05:00",
        "2024-01-05T00:30:00-04:00",
    ]

    with pytest.raises(SnapshotIncomplete) as info:
        _validate(_prices(dates))

    assert info.value.missing == [("AAPL", "2024-01-04", "bar")]


def test_half_hour_offset_span_keeps_the_written_day():
    """A 30-minute shift is inside the one-hour span. Lord Howe is ``+10:30``/``+11:00``.

    ``2024-01-04T00:30:00+11:00`` is 4 Jan as written and 3 Jan in UTC.
    """
    dates = [
        "2024-01-04T00:30:00+11:00",
        "2023-12-29T00:00:00+10:30",
        "2024-01-02T00:00:00+10:30",
        "2024-01-03T00:00:00+11:00",
    ]

    report = _validate(_prices(dates))

    assert report["AAPL"]["exit_session"] == "2024-01-04"


def test_offset_one_minute_past_a_dst_hour_raises():
    """61 minutes is past a one-hour DST shift and stays fail-closed."""
    dates = [
        "2023-12-29T00:00:00-05:00",
        "2024-01-02T00:00:00-05:00",
        "2024-01-03T00:00:00-03:59",
        "2024-01-04T00:00:00-05:00",
    ]

    with pytest.raises(ValueError, match="AAPL: Date mixes UTC offsets") as info:
        _validate(_prices(dates))

    assert _MIXED_TZ not in str(info.value)
    assert not isinstance(info.value, SnapshotIncomplete)


def test_wide_offsets_raise_instead_of_filling_the_exit_from_utc():
    """``2024-01-05T00:30:00+09:00`` is 4 Jan in UTC and 5 Jan as written.

    Forcing one zone with ``utc=True`` would accept it as the 4 Jan exit.
    The span is 14 hours, so the column is rejected.
    """
    dates = [
        "2023-12-29T00:00:00-05:00",
        "2024-01-02T00:00:00-05:00",
        "2024-01-03T00:00:00-05:00",
        "2024-01-05T00:30:00+09:00",
    ]

    with pytest.raises(ValueError, match="AAPL: Date mixes UTC offsets") as info:
        _validate(_prices(dates))

    assert _MIXED_TZ not in str(info.value)


def test_aware_then_naive_dates_raise():
    """A leading offset must not drop the naive row and report it missing."""
    dates = [
        "2024-01-03T00:30:00+09:00",
        "2024-01-02",
        "2023-12-29",
        "2024-01-04",
    ]

    with pytest.raises(ValueError, match="AAPL: Date mixes timezone-aware and naive timestamps") as info:
        _validate(_prices(dates))

    assert not isinstance(info.value, SnapshotIncomplete)


def test_naive_then_aware_dates_raise():
    dates = [
        "2024-01-02",
        "2024-01-03T00:30:00+09:00",
        "2023-12-29",
        "2024-01-04",
    ]

    with pytest.raises(ValueError, match="AAPL: Date mixes timezone-aware and naive timestamps") as info:
        _validate(_prices(dates))

    assert not isinstance(info.value, SnapshotIncomplete)


def test_naive_date_beside_eastern_offsets_raises():
    """A bare date next to ``-05:00`` and ``-04:00`` is not a third written day."""
    dates = [
        "2024-01-02 00:00:00-05:00",
        "2024-07-02 00:00:00-04:00",
        "2024-08-01",
    ]

    with pytest.raises(ValueError, match="AAPL: Date mixes timezone-aware and naive timestamps") as info:
        _validate(_prices(dates))

    assert "Date mixes UTC offsets" not in str(info.value)


def test_dst_offsets_that_share_a_written_day_raise():
    """Two offsets on one written day are two bars for one session.

    ``2024-01-02T22:00:00-05:00`` is 3 Jan in UTC and
    ``2024-01-02T01:00:00-04:00`` is 2 Jan in UTC. The written day is 2 Jan
    for both.
    """
    dates = [
        "2024-01-02T22:00:00-05:00",
        "2024-07-02T00:00:00-04:00",
        "2024-01-02T01:00:00-04:00",
    ]

    with pytest.raises(ValueError, match="AAPL: Date normalizes to duplicate calendar days") as info:
        _validate(_prices(dates))

    assert _MIXED_TZ not in str(info.value)


def test_wide_offsets_on_one_written_day_name_the_mix():
    """A 14-hour mix on one calendar day is still a mix of offsets."""
    dates = [
        "2024-01-02T00:30:00+09:00",
        "2024-01-02T00:30:00-05:00",
    ]

    with pytest.raises(ValueError, match="AAPL: Date mixes UTC offsets") as info:
        _validate(_prices(dates))

    assert "duplicate calendar days" not in str(info.value)


def test_unparseable_date_beside_dst_offsets_is_dropped():
    """A value that is not a timestamp is not a third offset."""
    dates = [
        "not-a-date",
        "2023-12-29 00:00:00-05:00",
        "2024-01-02 00:00:00-05:00",
        "2024-01-03 00:00:00-05:00",
        "2024-01-04 00:00:00-04:00",
    ]

    report = _validate(_prices(dates))

    assert report["AAPL"]["exit_session"] == "2024-01-04"
