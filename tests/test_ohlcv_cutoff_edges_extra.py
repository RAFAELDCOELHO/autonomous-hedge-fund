"""Edge cases for the OHLCV ``curr_date`` ceiling.

``load_ohlcv`` keeps a row when ``Date <= curr_date`` and drops the next
calendar day, including the Saturday after a Friday and the session after a
holiday. History before ``curr_date`` stays. ``fill_prices=False`` (the
runner) does not fill a missing Open from a later bar.

With the default ``fill_prices=True``, ``_clean_dataframe`` backfills before
that ceiling, so a leading NaN Open is taken from a bar after ``curr_date``
(audit L7). That case is xfailed; it is not fixed here.
"""

from __future__ import annotations

import pandas as pd
import pytest

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


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason=(
        "L7: bfill runs before the curr_date cutoff, so a leading NaN Open is "
        "filled from a bar after curr_date (docs/TEMPORAL_AUDIT.md)"
    ),
)
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
