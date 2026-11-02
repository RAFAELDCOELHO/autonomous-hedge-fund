"""Timezone-aware OHLCV dates use the calendar day written in the cell.

``load_ohlcv`` compares ``Date`` with a naive ``curr_date``. A tz-aware column
raises ``TypeError``. The offset is dropped without shifting the clock to UTC,
then the day is floored to midnight. ``2024-01-05T00:30+09:00`` is 5 Jan as
written and 4 Jan in UTC. Naive timestamps, including a time that is not
midnight, stay as parsed. A column that mixes naive timestamps with
timezone-aware ones raises ``ValueError``. Offsets that span more than 60
minutes do too. A US/Eastern file whose text offsets are ``-05:00`` and
``-04:00`` keeps each written day. Written days that collide raise
``ValueError``.
"""

from __future__ import annotations

import pandas as pd
import pytest

import tradingagents.dataflows.config as df_config
import tradingagents.dataflows.stockstats_utils as ss
from tradingagents.backtest.runner import DataDefectError, run_buy_and_hold

SENTINEL = 987654.0


def _bar(date: str, open_: float | None) -> dict:
    return {
        "Date": date,
        "Open": open_,
        "High": 2.0,
        "Low": 1.0,
        "Close": 3.0,
        "Volume": 4.0,
    }


def _install_cache(monkeypatch, tmp_path, rows, symbol: str = "AAPL") -> None:
    def _no_network(*_args, **_kwargs):
        raise AssertionError("offline: yfinance download")

    monkeypatch.setattr(ss.yf, "download", _no_network)
    df_config.initialize_config()
    monkeypatch.setitem(df_config._config, "data_cache_dir", str(tmp_path))
    today = pd.Timestamp.today()
    start = (today - pd.DateOffset(years=5)).strftime("%Y-%m-%d")
    end = today.strftime("%Y-%m-%d")
    path = tmp_path / f"{symbol}-YFin-data-{start}-{end}-adj.csv"
    pd.DataFrame(rows).to_csv(path, index=False)


def _load(monkeypatch, tmp_path, rows, curr_date: str, *, fill_prices: bool = True) -> pd.DataFrame:
    _install_cache(monkeypatch, tmp_path, rows)
    return ss.load_ohlcv("AAPL", curr_date, fill_prices=fill_prices)


def _dates(out: pd.DataFrame) -> list[str]:
    return sorted(out["Date"].dt.strftime("%Y-%m-%d"))


def test_positive_offset_keeps_the_written_day_not_the_utc_day(monkeypatch, tmp_path):
    """``2024-01-02T00:30+09:00`` is 2 Jan as written and 1 Jan in UTC.

    ``2024-01-03T00:30+09:00`` is 3 Jan as written and 2 Jan in UTC, so a
    cutoff of 2 Jan must drop it. The comparison must not raise.
    """
    out = _load(
        monkeypatch,
        tmp_path,
        [
            _bar("2024-01-03T00:30:00+09:00", SENTINEL),
            _bar("2024-01-02T00:30:00+09:00", 10.0),
            _bar("2024-01-01T00:30:00+09:00", 9.0),
        ],
        "2024-01-02",
    )

    assert out["Date"].dt.tz is None
    assert _dates(out) == ["2024-01-01", "2024-01-02"]
    assert out.sort_values("Date")["Date"].tolist() == [
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-01-02"),
    ]
    assert not bool(out.isin([SENTINEL]).to_numpy().any())


def test_written_day_after_cutoff_is_not_kept_as_its_utc_day(monkeypatch, tmp_path):
    """``2024-01-05T00:30+09:00`` is 4 Jan in UTC. A 4 Jan cutoff must drop it."""
    out = _load(
        monkeypatch,
        tmp_path,
        [
            _bar("2024-01-05T00:30:00+09:00", SENTINEL),
            _bar("2024-01-04T00:30:00+09:00", 4.0),
        ],
        "2024-01-04",
    )

    assert _dates(out) == ["2024-01-04"]
    assert out["Open"].tolist() == [4.0]
    assert not bool(out.isin([SENTINEL]).to_numpy().any())


def test_negative_offset_on_the_cutoff_evening_stays(monkeypatch, tmp_path):
    """``2024-01-02T22:00-05:00`` is 2 Jan as written and 3 Jan in UTC.

    A 2 Jan cutoff keeps it. The next written day stays out.
    """
    out = _load(
        monkeypatch,
        tmp_path,
        [
            _bar("2024-01-03T00:30:00-05:00", SENTINEL),
            _bar("2024-01-02T22:00:00-05:00", 22.0),
        ],
        "2024-01-02",
    )

    assert out["Date"].dt.tz is None
    assert out["Date"].tolist() == [pd.Timestamp("2024-01-02")]
    assert out["Open"].tolist() == [22.0]
    assert not bool(out.isin([SENTINEL]).to_numpy().any())


def test_midnight_offset_matches_the_naive_session_day(monkeypatch, tmp_path):
    """``2024-01-02 00:00:00-05:00`` is the 2 Jan session, same as a naive date."""
    out = _load(
        monkeypatch,
        tmp_path,
        [
            _bar("2024-01-03 00:00:00-05:00", SENTINEL),
            _bar("2024-01-02 00:00:00-05:00", 2.0),
        ],
        "2024-01-02",
    )

    assert out["Date"].tolist() == [pd.Timestamp("2024-01-02")]
    assert out["Open"].tolist() == [2.0]


def test_tz_aware_gap_is_not_filled_from_the_next_written_day(monkeypatch, tmp_path):
    """A leading NaN must not borrow an Open whose written day is after curr_date.

    ``2024-01-13T00:30+09:00`` is 12 Jan in UTC. Converting to UTC and then
    filling would take the sentinel.
    """
    out = _load(
        monkeypatch,
        tmp_path,
        [
            _bar("2024-01-12T00:30:00+09:00", None),
            _bar("2024-01-13T00:30:00+09:00", SENTINEL),
        ],
        "2024-01-12",
        fill_prices=True,
    )

    assert _dates(out) == ["2024-01-12"]
    assert pd.isna(out["Open"].iloc[0])
    assert not bool(out.isin([SENTINEL]).to_numpy().any())


def test_aware_then_naive_dates_raise(monkeypatch, tmp_path):
    """A leading offset must not drop the naive row.

    ``pd.to_datetime`` keeps ``2024-01-03T00:30+09:00`` and turns ``2024-01-02``
    into NaT. The naive bar would then disappear. Fail closed instead.
    """
    with pytest.raises(ValueError, match="AAPL: Date mixes timezone-aware and naive timestamps"):
        _load(
            monkeypatch,
            tmp_path,
            [
                _bar("2024-01-03T00:30:00+09:00", 3.0),
                _bar("2024-01-02", 2.0),
            ],
            "2024-01-03",
        )


def test_naive_then_aware_dates_raise(monkeypatch, tmp_path):
    """The other order drops the timezone-aware row and keeps the naive one."""
    with pytest.raises(ValueError, match="AAPL: Date mixes timezone-aware and naive timestamps"):
        _load(
            monkeypatch,
            tmp_path,
            [
                _bar("2024-01-02", 2.0),
                _bar("2024-01-03T00:30:00+09:00", 3.0),
            ],
            "2024-01-03",
        )


def test_mixed_offsets_raise(monkeypatch, tmp_path):
    """Two offsets name the problem instead of pandas' mixed-timezone error."""
    with pytest.raises(ValueError, match="AAPL: Date mixes UTC offsets"):
        _load(
            monkeypatch,
            tmp_path,
            [
                _bar("2024-01-02T00:30:00+09:00", 2.0),
                _bar("2024-01-03T00:30:00-05:00", 3.0),
            ],
            "2024-01-03",
        )


def test_us_eastern_dst_offsets_keep_the_written_day(monkeypatch, tmp_path):
    """A US/Eastern CSV stores ``-05:00`` in winter and ``-04:00`` in summer.

    ``2024-07-02T22:00:00-04:00`` is 2 Jul as written and 3 Jul in UTC, so a
    2 Jul cutoff keeps it. ``2024-11-04 00:00:00-05:00`` is the winter offset
    again and is after that cutoff, so it stays out.
    """
    out = _load(
        monkeypatch,
        tmp_path,
        [
            _bar("2024-01-02 00:00:00-05:00", 1.0),
            _bar("2024-03-11 00:00:00-04:00", 11.0),
            _bar("2024-07-02T22:00:00-04:00", 22.0),
            _bar("2024-11-04 00:00:00-05:00", SENTINEL),
        ],
        "2024-07-02",
    )

    assert out["Date"].dt.tz is None
    assert out["Date"].tolist() == [
        pd.Timestamp("2024-01-02"),
        pd.Timestamp("2024-03-11"),
        pd.Timestamp("2024-07-02"),
    ]
    assert out["Open"].tolist() == [1.0, 11.0, 22.0]
    assert not bool(out.isin([SENTINEL]).to_numpy().any())


def test_cached_us_eastern_series_reloads_the_written_days(monkeypatch, tmp_path):
    """A saved America/New_York frame stores ``-05:00`` and ``-04:00`` as text.

    The first load sees one named zone. The cache file is the mixed-offset
    CSV, and the next load must keep the same written days.
    """
    index = pd.DatetimeIndex(
        ["2024-01-02", "2024-03-11", "2024-07-02", "2024-11-04"],
        tz="America/New_York",
        name="Date",
    )
    frame = pd.DataFrame(
        {
            "Open": [1.0, 11.0, 7.0, SENTINEL],
            "High": [1.0, 11.0, 7.0, SENTINEL],
            "Low": [1.0, 11.0, 7.0, SENTINEL],
            "Close": [1.0, 11.0, 7.0, SENTINEL],
            "Volume": [1.0, 1.0, 1.0, 1.0],
        },
        index=index,
    )

    def _download(*_args, **_kwargs):
        return frame.copy()

    monkeypatch.setattr(ss.yf, "download", _download)
    df_config.initialize_config()
    monkeypatch.setitem(df_config._config, "data_cache_dir", str(tmp_path))

    first = ss.load_ohlcv("AAPL", "2024-07-02", fill_prices=False)
    cached = list(tmp_path.glob("*.csv"))
    assert len(cached) == 1
    cached_text = cached[0].read_text()
    assert "-05:00" in cached_text
    assert "-04:00" in cached_text

    def _no_network(*_args, **_kwargs):
        raise AssertionError("offline: yfinance download")

    monkeypatch.setattr(ss.yf, "download", _no_network)
    second = ss.load_ohlcv("AAPL", "2024-07-02", fill_prices=False)

    assert first["Date"].tolist() == second["Date"].tolist() == [
        pd.Timestamp("2024-01-02"),
        pd.Timestamp("2024-03-11"),
        pd.Timestamp("2024-07-02"),
    ]
    assert first["Open"].tolist() == second["Open"].tolist() == [1.0, 11.0, 7.0]
    assert second["Date"].dt.tz is None
    assert not bool(second.isin([SENTINEL]).to_numpy().any())


def test_dst_gap_is_not_filled_from_the_other_offset(monkeypatch, tmp_path):
    """A leading NaN must not borrow an Open on the other side of DST."""
    out = _load(
        monkeypatch,
        tmp_path,
        [
            _bar("2024-01-02 00:00:00-05:00", None),
            _bar("2024-07-02 00:00:00-04:00", SENTINEL),
        ],
        "2024-01-02",
        fill_prices=True,
    )

    assert _dates(out) == ["2024-01-02"]
    assert pd.isna(out["Open"].iloc[0])
    assert not bool(out.isin([SENTINEL]).to_numpy().any())


def test_half_hour_dst_span_keeps_the_written_days(monkeypatch, tmp_path):
    """A 30-minute shift is inside the one-hour span. Lord Howe is ``+10:30``/``+11:00``."""
    out = _load(
        monkeypatch,
        tmp_path,
        [
            _bar("2024-01-02T00:00:00+10:30", 1.0),
            _bar("2024-07-02T00:00:00+11:00", 2.0),
        ],
        "2024-07-02",
    )

    assert out["Date"].tolist() == [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-07-02")]
    assert out["Open"].tolist() == [1.0, 2.0]


def test_offset_one_minute_past_a_dst_hour_raises(monkeypatch, tmp_path):
    """61 minutes is past a one-hour DST shift and stays fail-closed."""
    with pytest.raises(ValueError, match="AAPL: Date mixes UTC offsets"):
        _load(
            monkeypatch,
            tmp_path,
            [
                _bar("2024-01-02T00:00:00-05:00", 1.0),
                _bar("2024-07-02T00:00:00-03:59", 2.0),
            ],
            "2024-07-02",
        )


def test_dst_offsets_that_share_a_written_day_raise(monkeypatch, tmp_path):
    """Two offsets on one written day are two bars for one session.

    ``2024-01-02T22:00:00-05:00`` is 3 Jan in UTC and
    ``2024-01-02T01:00:00-04:00`` is 2 Jan in UTC. The written day is 2 Jan
    for both.
    """
    with pytest.raises(ValueError, match="AAPL: Date normalizes to duplicate calendar days"):
        _load(
            monkeypatch,
            tmp_path,
            [
                _bar("2024-01-02T22:00:00-05:00", 22.0),
                _bar("2024-07-02T00:00:00-04:00", 7.0),
                _bar("2024-01-02T01:00:00-04:00", 1.0),
            ],
            "2024-07-02",
        )


def test_wide_offsets_on_one_written_day_name_the_mix(monkeypatch, tmp_path):
    """A 14-hour mix on one calendar day is still a mix of offsets."""
    with pytest.raises(ValueError, match="AAPL: Date mixes UTC offsets"):
        _load(
            monkeypatch,
            tmp_path,
            [
                _bar("2024-01-02T00:30:00+09:00", 2.0),
                _bar("2024-01-02T00:30:00-05:00", 3.0),
            ],
            "2024-01-02",
        )


def test_naive_date_beside_eastern_offsets_raises(monkeypatch, tmp_path):
    """A bare date next to ``-05:00`` and ``-04:00`` is not a third written day."""
    with pytest.raises(ValueError, match="AAPL: Date mixes timezone-aware and naive timestamps"):
        _load(
            monkeypatch,
            tmp_path,
            [
                _bar("2024-01-02 00:00:00-05:00", 1.0),
                _bar("2024-07-02 00:00:00-04:00", 2.0),
                _bar("2024-08-01", 3.0),
            ],
            "2024-08-01",
        )


def test_unparseable_date_beside_dst_offsets_is_dropped(monkeypatch, tmp_path):
    """A value that is not a timestamp is not a third offset."""
    out = _load(
        monkeypatch,
        tmp_path,
        [
            _bar("not-a-date", SENTINEL),
            _bar("2024-01-02 00:00:00-05:00", 1.0),
            _bar("2024-07-02 00:00:00-04:00", 2.0),
        ],
        "2024-07-02",
    )

    assert out["Date"].tolist() == [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-07-02")]
    assert out["Open"].tolist() == [1.0, 2.0]
    assert not bool(out.isin([SENTINEL]).to_numpy().any())


def test_same_offset_with_and_without_a_colon_keeps_both_days(monkeypatch, tmp_path):
    """``+09:00`` and ``+0900`` are one offset. Both written days stay."""
    out = _load(
        monkeypatch,
        tmp_path,
        [
            _bar("2024-01-02T00:00:00+09:00", 2.0),
            _bar("2024-01-03T00:00:00+0900", 3.0),
        ],
        "2024-01-03",
    )

    assert out["Date"].tolist() == [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]
    assert out["Open"].tolist() == [2.0, 3.0]


def test_unparseable_date_beside_one_offset_is_dropped(monkeypatch, tmp_path):
    """A value that is not a timestamp is not a second zone."""
    out = _load(
        monkeypatch,
        tmp_path,
        [
            _bar("not-a-date", SENTINEL),
            _bar("2024-01-02T00:30:00+09:00", 2.0),
        ],
        "2024-01-02",
    )

    assert out["Date"].tolist() == [pd.Timestamp("2024-01-02")]
    assert out["Open"].tolist() == [2.0]
    assert not bool(out.isin([SENTINEL]).to_numpy().any())


def test_named_zone_across_dst_keeps_the_written_days(monkeypatch, tmp_path):
    """America/New_York is one zone. January is -05:00 and July is -04:00.

    The in-memory vendor frame is datetime64 with that zone. Both sessions
    stay on the calendar day written on the stamp.
    """
    index = pd.DatetimeIndex(["2024-01-02", "2024-07-02"], tz="America/New_York", name="Date")
    frame = pd.DataFrame(
        {
            "Open": [1.0, 2.0],
            "High": [1.0, 2.0],
            "Low": [1.0, 2.0],
            "Close": [1.0, 2.0],
            "Volume": [1.0, 1.0],
        },
        index=index,
    )

    def _download(*_args, **_kwargs):
        return frame.copy()

    monkeypatch.setattr(ss.yf, "download", _download)
    df_config.initialize_config()
    monkeypatch.setitem(df_config._config, "data_cache_dir", str(tmp_path))

    out = ss.load_ohlcv("AAPL", "2024-07-02", fill_prices=False)

    assert out["Date"].dt.tz is None
    assert out["Date"].tolist() == [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-07-02")]
    assert out["Open"].tolist() == [1.0, 2.0]


def test_naive_intraday_stamp_is_not_floored(monkeypatch, tmp_path):
    """A naive time that is not midnight stays as parsed.

    ``2024-01-12 09:30`` is after midnight, so ``Date <= 2024-01-12`` drops it.
    ``2024-01-11 09:30`` is kept with the clock intact.
    """
    out = _load(
        monkeypatch,
        tmp_path,
        [
            _bar("2024-01-11 09:30:00", 11.0),
            _bar("2024-01-12 09:30:00", SENTINEL),
        ],
        "2024-01-12",
    )

    assert out["Date"].tolist() == [pd.Timestamp("2024-01-11 09:30:00")]
    assert out["Open"].tolist() == [11.0]
    assert not bool(out.isin([SENTINEL]).to_numpy().any())


def test_runner_buy_and_hold_uses_the_written_session(monkeypatch, tmp_path):
    """The runner's date slice sees the loader's calendar days, not a TypeError.

    Opens 10, 20, 25, 40 on the written sessions. Buy & Hold at 1,000 buys at
    20 and is marked at 25, then 40.
    """
    _install_cache(
        monkeypatch,
        tmp_path,
        [
            _bar("2023-12-29T00:30:00+09:00", 10.0),
            _bar("2024-01-02T00:30:00+09:00", 20.0),
            _bar("2024-01-03T00:30:00+09:00", 25.0),
            _bar("2024-01-04T00:30:00+09:00", 40.0),
        ],
    )

    equity = run_buy_and_hold("AAPL", "2024-01-02", "2024-01-03", 1_000.0, no_rf=True)

    assert list(equity.index.strftime("%Y-%m-%d")) == ["2023-12-29", "2024-01-02", "2024-01-03"]
    assert equity.tolist() == pytest.approx([1_000.0, 1_250.0, 2_000.0])


def test_runner_does_not_take_the_exit_from_the_utc_day(monkeypatch, tmp_path):
    """``2024-01-05T00:30+09:00`` is 4 Jan in UTC and must not fill the 4 Jan exit."""
    _install_cache(
        monkeypatch,
        tmp_path,
        [
            _bar("2023-12-29T00:30:00+09:00", 10.0),
            _bar("2024-01-02T00:30:00+09:00", 20.0),
            _bar("2024-01-03T00:30:00+09:00", 25.0),
            _bar("2024-01-05T00:30:00+09:00", SENTINEL),
        ],
    )

    with pytest.raises(DataDefectError, match="2024-01-04"):
        run_buy_and_hold("AAPL", "2024-01-02", "2024-01-03", 1_000.0, no_rf=True)
