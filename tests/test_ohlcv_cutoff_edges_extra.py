"""Edge cases for the OHLCV ``curr_date`` ceiling.

``load_ohlcv`` keeps a row when ``Date <= curr_date`` and drops the next
calendar day, including the Saturday after a Friday and the session after a
holiday. History before ``curr_date`` stays. ``fill_prices=False`` (the
runner) does not fill a missing Open from a later bar.

With the default ``fill_prices=True``, gaps are filled only after that
ceiling. A leading NaN Open is filled from a later bar on or before
``curr_date``, and never from a bar after it.
ceiling, forward-fill then backfill. A leading NaN Open is filled from a
later bar on or before ``curr_date``. A NaN on the last day of the window
is filled from the previous kept bar. Neither is taken from a bar after
``curr_date``. An interior gap takes the earlier bar, because forward-fill
runs first.
"""

from __future__ import annotations

import pandas as pd

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


def _load(monkeypatch, tmp_path, rows, curr_date: str, *, fill_prices: bool = True):
    import tradingagents.dataflows.config as df_config
    import tradingagents.dataflows.stockstats_utils as ss

    def _no_network(*_args, **_kwargs):
        raise AssertionError("offline: yfinance download")

    monkeypatch.setattr(ss.yf, "download", _no_network)
    df_config.initialize_config()
    monkeypatch.setitem(df_config._config, "data_cache_dir", str(tmp_path))
    today = pd.Timestamp.today()
    start = (today - pd.DateOffset(years=5)).strftime("%Y-%m-%d")
    end = today.strftime("%Y-%m-%d")
    path = tmp_path / f"AAPL-YFin-data-{start}-{end}-adj.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return ss.load_ohlcv("AAPL", curr_date, fill_prices=fill_prices)


def _dates(out: pd.DataFrame) -> list[str]:
    return sorted(out["Date"].dt.strftime("%Y-%m-%d"))


def _has(out: pd.DataFrame, value: float) -> bool:
    return bool(out.isin([value]).to_numpy().any())


def test_curr_date_kept_and_next_calendar_day_hidden(monkeypatch, tmp_path):
    """Wednesday 2024-01-10 stays. Thursday 2024-01-11 stays out."""
    out = _load(
        monkeypatch,
        tmp_path,
        [
            _bar("2024-01-11", SENTINEL),
            _bar("2024-01-09", 11.0),
            _bar("2024-01-10", 10.0),
        ],
        "2024-01-10",
    )

    assert _dates(out) == ["2024-01-09", "2024-01-10"]
    assert not _has(out, SENTINEL)


def test_friday_cutoff_hides_saturday_and_post_holiday_session(monkeypatch, tmp_path):
    """Friday 2024-01-12. Saturday and Tuesday 2024-01-16 (after MLK) stay out."""
    out = _load(
        monkeypatch,
        tmp_path,
        [
            _bar("2024-01-16", SENTINEL),
            _bar("2024-01-13", SENTINEL),
            _bar("2024-01-11", 11.0),
            _bar("2024-01-12", 12.0),
        ],
        "2024-01-12",
    )

    assert _dates(out) == ["2024-01-11", "2024-01-12"]
    assert not _has(out, SENTINEL)


def test_leap_day_cutoff_keeps_that_day_and_hides_march_first(monkeypatch, tmp_path):
    """2024-02-29 is a real session. 2024-03-01 is the next calendar day."""
    out = _load(
        monkeypatch,
        tmp_path,
        [
            _bar("2024-03-01", SENTINEL),
            _bar("2024-02-28", 28.0),
            _bar("2024-02-29", 29.0),
        ],
        "2024-02-29",
    )

    assert _dates(out) == ["2024-02-28", "2024-02-29"]
    assert not _has(out, SENTINEL)


def test_cutoff_keeps_prior_year_and_hides_next_session(monkeypatch, tmp_path):
    """2024-01-02 keeps 2023-12-22. 2024-01-03 stays out. There is no lookback."""
    out = _load(
        monkeypatch,
        tmp_path,
        [
            _bar("2024-01-03", SENTINEL),
            _bar("2023-12-22", 22.0),
            _bar("2024-01-02", 2.0),
        ],
        "2024-01-02",
    )

    assert _dates(out) == ["2023-12-22", "2024-01-02"]
    assert not _has(out, SENTINEL)


def test_rows_strictly_after_cutoff_yield_an_empty_frame(monkeypatch, tmp_path):
    out = _load(
        monkeypatch,
        tmp_path,
        [
            _bar("2024-01-13", SENTINEL),
            _bar("2024-01-16", SENTINEL),
        ],
        "2024-01-12",
    )

    assert _dates(out) == []
    assert out.empty
    assert not _has(out, SENTINEL)


def test_unparseable_date_is_dropped_and_does_not_echo_its_price(monkeypatch, tmp_path):
    out = _load(
        monkeypatch,
        tmp_path,
        [
            _bar("not-a-date", SENTINEL),
            _bar("2024-01-13", SENTINEL),
            _bar("2024-01-12", 12.0),
        ],
        "2024-01-12",
    )

    assert _dates(out) == ["2024-01-12"]
    assert not _has(out, SENTINEL)


def test_fill_prices_false_does_not_take_open_from_after_cutoff(monkeypatch, tmp_path):
    """The runner's fill_prices=False path leaves a missing Open missing."""
    out = _load(
        monkeypatch,
        tmp_path,
        [
            _bar("2024-01-13", SENTINEL),
            _bar("2024-01-12", None),
        ],
        "2024-01-12",
        fill_prices=False,
    )

    assert _dates(out) == ["2024-01-12"]
    assert pd.isna(out["Open"].iloc[0])
    assert not _has(out, SENTINEL)


def test_leading_nan_open_is_not_filled_from_after_cutoff(monkeypatch, tmp_path):
    """Default fill_prices=True must not borrow an Open from after curr_date."""
    out = _load(
        monkeypatch,
        tmp_path,
        [
            _bar("2024-01-12", None),
            _bar("2024-01-13", SENTINEL),
        ],
        "2024-01-12",
        fill_prices=True,
    )

    assert _dates(out) == ["2024-01-12"]
    assert pd.isna(out["Open"].iloc[0])
    assert not _has(out, SENTINEL)


def test_leading_nan_open_is_filled_from_a_later_bar_on_or_before_cutoff(monkeypatch, tmp_path):
    """A gap still borrows from the next kept bar. The bar after curr_date stays out."""
    out = _load(
        monkeypatch,
        tmp_path,
        [
            _bar("2024-01-11", None),
            _bar("2024-01-12", 12.0),
            _bar("2024-01-13", SENTINEL),
        ],
        "2024-01-12",
        fill_prices=True,
    )

    assert _dates(out) == ["2024-01-11", "2024-01-12"]
    assert out.sort_values("Date")["Open"].tolist() == [12.0, 12.0]
    assert not _has(out, SENTINEL)


def test_gap_on_last_day_and_at_the_start_is_ffill_then_bfill(monkeypatch, tmp_path):
    """Start gap plus a gap on the last day of the window, with an interior gap.

    curr_date is Friday 2024-01-12. Open is missing on the first kept day,
    on the middle day, and on the last day. The bar after the cutoff stays
    out. Forward-fill then backfill fills the start from the next kept bar
    (11), the interior gap from the earlier bar (11, not 15), and the last
    day from the previous kept bar (15). Backfill alone leaves the last day
    empty. Backfill before forward-fill writes 15 into the interior gap.
    """
    out = _load(
        monkeypatch,
        tmp_path,
        [
            _bar("2024-01-08", None),
            _bar("2024-01-09", 11.0),
            _bar("2024-01-10", None),
            _bar("2024-01-11", 15.0),
            _bar("2024-01-12", None),
            _bar("2024-01-16", SENTINEL),
        ],
        "2024-01-12",
        fill_prices=True,
    )

    assert _dates(out) == [
        "2024-01-08",
        "2024-01-09",
        "2024-01-10",
        "2024-01-11",
        "2024-01-12",
    ]
    assert out.sort_values("Date")["Open"].tolist() == [11.0, 11.0, 11.0, 15.0, 15.0]
    assert not _has(out, SENTINEL)
