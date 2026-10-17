"""Extra edge-case coverage for macro trade_date cutoffs (offline-only)."""

import pandas as pd
import pytest

from tradingagents.agents.utils import macro_tools
from tradingagents.agents.utils.macro_tools import (
    get_exchange_rate,
    get_gdp,
    get_inflation,
    get_selic,
)


class _Series:
    def __init__(self, df: pd.DataFrame):
        self._df = df

    def to_dataframe(self) -> pd.DataFrame:
        return self._df


def _rows(markdown: str) -> list[str]:
    if markdown.startswith("No data available before trade_date="):
        return []
    return [line.split("|")[1].strip() for line in markdown.splitlines()[2:]]


@pytest.fixture(autouse=True)
def _offline_brazilfi(monkeypatch):
    class _FakeBacen:
        daily = pd.bdate_range("2024-01-02", "2024-01-12")
        monthly = pd.date_range("2024-01-01", "2024-06-01", freq="MS")

        def selic(self, start=None, end=None):
            return _Series(pd.DataFrame({"value": 1.0}, index=self.daily))

        def dolar(self, start=None, end=None):
            return _Series(pd.DataFrame({"value": 5.0}, index=self.daily))

        def ipca(self, start=None, end=None):
            return _Series(pd.DataFrame({"value": 0.2}, index=self.monthly))

    class _FakeIBGE:
        def pib(self, last=8):
            idx = pd.DatetimeIndex(["2023-01-01", "2023-02-01", "2023-03-01", "2023-04-01", "2024-01-01"])
            return _Series(pd.DataFrame({"value": [1.0, 1.1, 1.2, 1.3, 1.4]}, index=idx))

    monkeypatch.setattr(macro_tools, "Bacen", _FakeBacen)
    monkeypatch.setattr(macro_tools, "IBGE", _FakeIBGE)


@pytest.mark.parametrize("tool", [get_selic, get_exchange_rate])
def test_daily_series_strict_cutoff_at_first_observation(tool):
    # trade_date equals first observation day: strict "< trade_date" should hide it.
    out_same_day = tool.invoke({"trade_date": "2024-01-02", "last_days": 5})
    assert out_same_day == "No data available before trade_date=2024-01-02."

    # One day later, that first observation is now visible.
    rows = _rows(tool.invoke({"trade_date": "2024-01-03", "last_days": 5}))
    assert rows == ["2024-01-02 00:00:00"]


def test_ipca_release_visible_when_published_exactly_on_d_minus_1():
    # Apr/2024 IPCA is treated as published on 2024-05-15.
    # With trade_date 2024-05-16 (D-1 in runner terms), that release is exactly D-1.
    rows = _rows(get_inflation.invoke({"trade_date": "2024-05-16", "last_months": 2}))
    assert "2024-04-01 00:00:00" in rows
    # The next month must not be visible at this boundary.
    assert "2024-05-01 00:00:00" not in rows


def test_gdp_release_visible_when_published_exactly_on_d_minus_1():
    # 2023Q4 (dated 2023-04-01 in this fixture) is known from 2024-03-30.
    # On trade_date 2024-03-31 (D-1 in runner terms), publication is exactly D-1.
    rows = _rows(get_gdp.invoke({"trade_date": "2024-03-31", "last_quarters": 2}))
    assert "2023Q4" in rows
    # The next quarter must not be visible at this boundary.
    assert "2024Q1" not in rows


def test_ipca_day_15_boundary_is_inclusive_and_day_before_is_exclusive():
    # Apr/2024 publication boundary:
    # - day before (2024-05-14): absent
    # - publication day (2024-05-15): visible
    rows_before = _rows(get_inflation.invoke({"trade_date": "2024-05-14", "last_months": 2}))
    assert "2024-04-01 00:00:00" not in rows_before

    rows_on_day = _rows(get_inflation.invoke({"trade_date": "2024-05-15", "last_months": 2}))
    assert "2024-04-01 00:00:00" in rows_on_day


def test_empty_upstream_series_returns_clear_message(monkeypatch):
    empty_df = pd.DataFrame({"value": []}, index=pd.DatetimeIndex([], name="date"))

    class _EmptyBacen:
        def selic(self, start=None, end=None):
            return _Series(empty_df)

        def dolar(self, start=None, end=None):
            return _Series(empty_df)

        def ipca(self, start=None, end=None):
            return _Series(empty_df)

    class _EmptyIBGE:
        def pib(self, last=8):
            # Keep one old quarter so this test focuses on "empty from cutoff"
            # behavior in Bacen-backed tools, not on GDP's separate empty-index path.
            idx = pd.DatetimeIndex(["2020-01-01"])
            return _Series(pd.DataFrame({"value": [1.0]}, index=idx))

    monkeypatch.setattr(macro_tools, "Bacen", _EmptyBacen)
    monkeypatch.setattr(macro_tools, "IBGE", _EmptyIBGE)

    assert get_selic.invoke({"trade_date": "2024-06-12", "last_days": 3}) == (
        "No data available before trade_date=2024-06-12."
    )
    assert get_exchange_rate.invoke({"trade_date": "2024-06-12", "last_days": 3}) == (
        "No data available before trade_date=2024-06-12."
    )
    assert get_inflation.invoke({"trade_date": "2024-06-12", "last_months": 3}) == (
        "No data available before trade_date=2024-06-12."
    )
