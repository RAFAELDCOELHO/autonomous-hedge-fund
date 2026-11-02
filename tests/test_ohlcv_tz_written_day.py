"""Timezone-aware OHLCV dates use the calendar day written in the cell.

``load_ohlcv`` compares ``Date`` with a naive ``curr_date``. A tz-aware column
raises ``TypeError``. The offset is dropped without shifting the clock to UTC,
then the day is floored to midnight. ``2024-01-05T00:30+09:00`` is 5 Jan as
written and 4 Jan in UTC. Naive timestamps, including a time that is not
midnight, stay as parsed.
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
