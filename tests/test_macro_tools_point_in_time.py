"""Offline: macro tools never return observations published after trade_date."""

import pandas as pd
import pytest
from langchain_core.messages import AIMessage
from langgraph.prebuilt import ToolNode

from tradingagents.agents.utils import macro_tools
from tradingagents.agents.utils.macro_tools import (
    get_exchange_rate,
    get_gdp,
    get_inflation,
    get_selic,
)

TRADE_DATE = "2024-06-12"


class _Series:
    def __init__(self, dates):
        self._df = pd.DataFrame(
            {"value": range(len(dates))},
            index=pd.DatetimeIndex(dates, name="date"),
        )

    def to_dataframe(self):
        return self._df


class _FakeBacen:
    """Ignores start/end, so the tools' own filtering is what gets tested."""

    daily = pd.bdate_range("2024-05-01", "2024-06-30")
    monthly = pd.date_range("2023-01-01", "2024-12-01", freq="MS")

    def selic(self, start=None, end=None):
        return _Series(self.daily)

    def dolar(self, start=None, end=None):
        return _Series(self.daily)

    def ipca(self, start=None, end=None):
        return _Series(self.monthly)


class _FakeIBGE:
    # brazilfi 0.2.1 convention: quarter Q of year Y is dated Y-Q-01.
    def pib(self, last=8):
        return _Series([f"{y}-0{q}-01" for y in (2022, 2023, 2024) for q in (1, 2, 3, 4)])


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    monkeypatch.setattr(macro_tools, "Bacen", _FakeBacen)
    monkeypatch.setattr(macro_tools, "IBGE", _FakeIBGE)


def _rows(markdown):
    return [line.split("|")[1].strip() for line in markdown.splitlines()[2:]]


@pytest.mark.parametrize("tool", [get_selic, get_exchange_rate])
def test_daily_series_stop_strictly_before_trade_date(tool):
    rows = _rows(tool.invoke({"trade_date": TRADE_DATE, "last_days": 5}))
    assert rows == [
        "2024-06-05 00:00:00",
        "2024-06-06 00:00:00",
        "2024-06-07 00:00:00",
        "2024-06-10 00:00:00",
        "2024-06-11 00:00:00",
    ]


def test_ipca_month_known_only_from_the_15th_of_next_month():
    rows = _rows(get_inflation.invoke({"trade_date": "2024-06-14", "last_months": 2}))
    assert rows == ["2024-03-01 00:00:00", "2024-04-01 00:00:00"]
    rows = _rows(get_inflation.invoke({"trade_date": "2024-06-15", "last_months": 2}))
    assert rows == ["2024-04-01 00:00:00", "2024-05-01 00:00:00"]


def test_gdp_quarter_known_only_90_days_after_quarter_end():
    # 2024Q1 ends 2024-03-31 -> known from 2024-06-29.
    rows = _rows(get_gdp.invoke({"trade_date": "2024-06-28", "last_quarters": 2}))
    assert rows == ["2023Q3", "2023Q4"]
    rows = _rows(get_gdp.invoke({"trade_date": "2024-06-29", "last_quarters": 2}))
    assert rows == ["2023Q4", "2024Q1"]


def test_trade_date_is_injected_from_graph_state_and_hidden_from_llm():
    tools = [get_selic, get_inflation, get_gdp, get_exchange_rate]
    for t in tools:
        assert "trade_date" not in t.tool_call_schema.model_json_schema()["properties"]

    call = {"name": "get_selic", "args": {"last_days": 1}, "id": "c1", "type": "tool_call"}
    state = {"messages": [AIMessage(content="", tool_calls=[call])], "trade_date": TRADE_DATE}
    out = ToolNode(tools).invoke(state)
    assert "2024-06-11" in out["messages"][0].content
    assert "2024-06-12" not in out["messages"][0].content


def test_ipca_year_rollover_december_is_visible_only_from_january_15():
    rows = _rows(get_inflation.invoke({"trade_date": "2025-01-14", "last_months": 2}))
    assert rows == ["2024-10-01 00:00:00", "2024-11-01 00:00:00"]
    rows = _rows(get_inflation.invoke({"trade_date": "2025-01-15", "last_months": 2}))
    assert rows == ["2024-11-01 00:00:00", "2024-12-01 00:00:00"]


def test_gdp_year_rollover_q4_is_visible_only_after_90_days():
    rows = _rows(get_gdp.invoke({"trade_date": "2025-03-30", "last_quarters": 2}))
    assert rows == ["2024Q2", "2024Q3"]
    rows = _rows(get_gdp.invoke({"trade_date": "2025-03-31", "last_quarters": 2}))
    assert rows == ["2024Q3", "2024Q4"]


@pytest.mark.parametrize("tool", [get_selic, get_exchange_rate])
def test_daily_series_weekend_trade_date_uses_last_business_day(tool):
    rows = _rows(tool.invoke({"trade_date": "2024-06-16", "last_days": 3}))
    assert rows == [
        "2024-06-12 00:00:00",
        "2024-06-13 00:00:00",
        "2024-06-14 00:00:00",
    ]


@pytest.mark.parametrize("tool", [get_selic, get_exchange_rate])
def test_daily_series_holiday_gap_uses_previous_available_business_day(tool, monkeypatch):
    class _HolidayGapBacen(_FakeBacen):
        daily = _FakeBacen.daily.drop(pd.Timestamp("2024-06-20"))

    monkeypatch.setattr(macro_tools, "Bacen", _HolidayGapBacen)
    rows = _rows(tool.invoke({"trade_date": "2024-06-21", "last_days": 2}))
    assert rows == ["2024-06-18 00:00:00", "2024-06-19 00:00:00"]


@pytest.mark.parametrize(
    ("tool", "args"),
    [
        (get_selic, {"trade_date": "2024-01-01", "last_days": 5}),
        (get_exchange_rate, {"trade_date": "2024-01-01", "last_days": 5}),
        (get_inflation, {"trade_date": "2022-01-01", "last_months": 5}),
        (get_gdp, {"trade_date": "2021-01-01", "last_quarters": 2}),
    ],
)
def test_empty_series_before_cutoff_returns_clear_message(tool, args):
    out = tool.invoke(args)
    assert "No data available" in out


@pytest.mark.parametrize(
    "trade_date",
    ["2024-06-12", "2024-06-15", "2024-06-29", "2025-01-15"],
)
def test_no_output_row_leaks_beyond_trade_date_for_multiple_cutoffs(trade_date):
    cutoff = pd.Timestamp(trade_date)

    for row in _rows(get_selic.invoke({"trade_date": trade_date, "last_days": 30})):
        assert pd.Timestamp(row) < cutoff

    for row in _rows(get_exchange_rate.invoke({"trade_date": trade_date, "last_days": 30})):
        assert pd.Timestamp(row) < cutoff

    for row in _rows(get_inflation.invoke({"trade_date": trade_date, "last_months": 24})):
        month = pd.Timestamp(row)
        publication = month + pd.DateOffset(months=1, days=macro_tools.IPCA_RELEASE_DAY - 1)
        assert publication <= cutoff

    gdp_rows = _rows(get_gdp.invoke({"trade_date": trade_date, "last_quarters": 12}))
    for row in gdp_rows:
        assert row == str(pd.Period(row, freq="Q"))
        quarter = pd.Period(row, freq="Q")
        publication = quarter.end_time.normalize() + pd.Timedelta(days=macro_tools.GDP_LAG_DAYS)
        assert publication <= cutoff
