"""PR7 (P4.11): no tool output may contain information from decision day D.

Each test runs the real runner for a single decision day D with an agent that
behaves like the LLM analysts: it calls the tool with the date it was given
(prompt {current_date} / curr_date / InjectedState trade_date). The fake
vendor carries a sentinel (987654.32) on day D only; it must never show up.
Existing caps (P4.9 get_stock_data, stockstats, macro trade_date, statements /
insider `<` cutoff, news) are unchanged and must simply apply with D-1.

Every test is expected to FAIL on e5ea062 (agent keyed to D) and PASS after PR7.
"""

from __future__ import annotations

import re

import pandas as pd
import pytest

import _pr7_api as api
from _pr7_fakes import (
    CUTOFF_SENTINEL,
    SENTINEL,
    FakeYahoo,
    RecordingAgent,
    install_fake_yahoo,
    make_bars,
)

SENTINEL_DIGITS = "987654"         # survives tabulate's 6-significant-digit 'g' format
CUTOFF_SENTINEL_DIGITS = "876543"


def _shift(d: str, days: int) -> str:
    return (pd.Timestamp(d) + pd.Timedelta(days=days)).strftime("%Y-%m-%d")


def _run_tool_on_day(monkeypatch, tmp_path, yahoo: FakeYahoo, ticker: str, day: str, tool_call):
    install_fake_yahoo(monkeypatch, tmp_path, yahoo)
    outputs: list[str] = []
    agent = RecordingAgent(on_call=lambda date: outputs.append(str(tool_call(date))))
    api.run_agent(agent, ticker, day, day, 100_000.0)
    assert not agent.errors, agent.errors
    assert len(outputs) == 1
    return agent.dates[0], outputs[0]


def _bars_with_sentinel_on(ticker: str, day: str) -> pd.DataFrame:
    bars = make_bars(ticker)
    bars.loc[pd.Timestamp(day), ["Open", "High", "Low", "Close"]] = SENTINEL
    return bars


# --------------------------------------------------------------------------
# Prices: get_stock_data (P4.9 cap) and stockstats indicators
# --------------------------------------------------------------------------


@pytest.mark.parametrize("ticker", ["AAPL", "PETR4.SA"])
def test_get_stock_data_never_returns_day_d_bar(monkeypatch, tmp_path, ticker):
    from tradingagents.agents.utils.core_stock_tools import get_stock_data

    day, cutoff = "2024-05-15", "2024-05-14"
    yahoo = FakeYahoo(bars={ticker: _bars_with_sentinel_on(ticker, day)})

    def call(date):
        # An eager LLM asks past "today"; the P4.9 cap at curr_date must hold.
        return get_stock_data.invoke(
            {"symbol": ticker, "start_date": _shift(date, -10), "end_date": _shift(date, 5), "curr_date": date}
        )

    _, out = _run_tool_on_day(monkeypatch, tmp_path, yahoo, ticker, day, call)

    assert SENTINEL_DIGITS not in out
    rows = [line.split(",")[0] for line in out.splitlines() if line[:4].isdigit()]
    assert day not in rows
    assert rows and rows[-1] == cutoff


@pytest.mark.parametrize("indicator", ["close_10_ema", "boll_ub"])
def test_stockstats_indicators_never_use_day_d_bar(monkeypatch, tmp_path, indicator):
    from tradingagents.agents.utils.technical_indicators_tools import get_indicators

    ticker, day, cutoff = "AAPL", "2024-05-15", "2024-05-14"
    yahoo = FakeYahoo(bars={ticker: _bars_with_sentinel_on(ticker, day)})

    def call(date):
        return get_indicators.invoke(
            {"symbol": ticker, "indicator": indicator, "curr_date": date, "look_back_days": 5}
        )

    _, out = _run_tool_on_day(monkeypatch, tmp_path, yahoo, ticker, day, call)

    dated = dict(re.findall(r"^(\d{4}-\d{2}-\d{2}): (.+)$", out, flags=re.M))
    assert day not in dated
    assert cutoff in dated and dated[cutoff] not in ("", "N/A")
    values = [float(v) for v in dated.values() if re.fullmatch(r"-?\d+(\.\d+)?(e[+-]?\d+)?", v)]
    assert values and max(values) < 10_000, "an indicator value was computed from the day-D sentinel bar"


# --------------------------------------------------------------------------
# Macro tools (InjectedState trade_date)
# --------------------------------------------------------------------------


class _Series:
    def __init__(self, df):
        self._df = df

    def to_dataframe(self):
        return self._df


def _daily_frame(day, cutoff):
    idx = pd.bdate_range("2024-01-02", "2024-06-28", name="date")
    df = pd.DataFrame({"valor": 10.4}, index=idx)
    df.loc[pd.Timestamp(cutoff), "valor"] = CUTOFF_SENTINEL
    df.loc[pd.Timestamp(day), "valor"] = SENTINEL
    return df


@pytest.fixture
def fake_macro(monkeypatch):
    from tradingagents.agents.utils import macro_tools

    state = {}

    class FakeBacen:
        def _window(self, df, start, end):
            if start is not None:
                df = df.loc[pd.Timestamp(start):]
            if end is not None:
                df = df.loc[: pd.Timestamp(end)]
            return _Series(df)

        def selic(self, start=None, end=None):
            return self._window(state["daily"], start, end)

        def dolar(self, start=None, end=None):
            return self._window(state["daily"], start, end)

        def ipca(self, start=None, end=None):
            return self._window(state["monthly"], start, end)

    class FakeIBGE:
        def pib(self, last=8):
            return _Series(state["quarterly"])

    monkeypatch.setattr(macro_tools, "Bacen", FakeBacen)
    monkeypatch.setattr(macro_tools, "IBGE", FakeIBGE)
    return macro_tools, state


@pytest.mark.parametrize("tool_name", ["get_selic", "get_exchange_rate"])
def test_macro_daily_series_exclude_day_d_with_strict_cap_at_cutoff(monkeypatch, tmp_path, fake_macro, tool_name):
    macro_tools, state = fake_macro
    ticker, day, cutoff = "PETR4.SA", "2024-05-15", "2024-05-14"
    state["daily"] = _daily_frame(day, cutoff)
    tool = getattr(macro_tools, tool_name)

    trade_date, out = _run_tool_on_day(
        monkeypatch, tmp_path, FakeYahoo(), ticker, day, lambda date: tool.func(trade_date=date, last_days=5)
    )

    assert trade_date == cutoff
    assert SENTINEL_DIGITS not in out
    # Existing `< trade_date` cap, unchanged, now applied to D-1.
    assert (CUTOFF_SENTINEL_DIGITS in out) is api.MACRO_DAILY_SHOWS_CUTOFF_DAY
    assert "2024-05-13" in out


def test_macro_ipca_published_on_day_d_is_hidden(monkeypatch, tmp_path, fake_macro):
    macro_tools, state = fake_macro
    ticker, day = "PETR4.SA", "2024-05-15"
    monthly = pd.DataFrame({"valor": 0.16}, index=pd.date_range("2023-01-01", "2024-06-01", freq="MS", name="date"))
    monthly.loc[pd.Timestamp("2024-04-01"), "valor"] = SENTINEL  # IPCA Apr/24 "published" 2024-05-15 = D
    monthly.loc[pd.Timestamp("2024-05-01"), "valor"] = SENTINEL
    state["monthly"] = monthly

    _, out = _run_tool_on_day(
        monkeypatch, tmp_path, FakeYahoo(), ticker, day,
        lambda date: macro_tools.get_inflation.func(trade_date=date, last_months=3),
    )

    assert SENTINEL_DIGITS not in out
    assert "2024-03-01" in out


@pytest.mark.parametrize(
    ("day", "hidden_quarter", "visible_quarter"),
    [
        # 2024Q4 end + 90d = 2025-03-31 = D (Mon); D-1 = Fri 2025-03-28.
        ("2025-03-31", "2024-04-01", "2024Q3"),
        # 2023Q2 end + 90d = 2023-09-28 = D (Thu); D-1 = Wed 2023-09-27.
        ("2023-09-28", "2023-02-01", "2023Q1"),
    ],
)
def test_macro_gdp_published_on_day_d_is_hidden(monkeypatch, tmp_path, fake_macro, day, hidden_quarter, visible_quarter):
    macro_tools, state = fake_macro
    # brazilfi 0.2.1 convention: quarter Q of year Y is dated Y-Q-01.
    idx = pd.DatetimeIndex([f"{y}-0{q}-01" for y in (2022, 2023, 2024, 2025) for q in (1, 2, 3, 4)], name="date")
    quarterly = pd.DataFrame({"valor": 1.1}, index=idx)
    quarterly.loc[pd.Timestamp(hidden_quarter):, "valor"] = SENTINEL  # published on/after D
    state["quarterly"] = quarterly

    _, out = _run_tool_on_day(
        monkeypatch, tmp_path, FakeYahoo(), "PETR4.SA", day,
        lambda date: macro_tools.get_gdp.func(trade_date=date, last_quarters=4),
    )

    assert SENTINEL_DIGITS not in out
    assert visible_quarter in out


# --------------------------------------------------------------------------
# Statements and insiders (P4.3 / P4.4 `available_date < cutoff`, cutoff = D-1)
# --------------------------------------------------------------------------

STATEMENT_DAY = "2024-05-15"  # D (Wed); D-1 = 2024-05-14


def _statement_yahoo() -> FakeYahoo:
    # Apple-like 52/53-week calendar, FYE late September.
    quarterly = pd.DataFrame(
        {
            "2024-03-30": [SENTINEL],  # +45d -> available 2024-05-14 = D-1 (on the cutoff)
            "2024-03-31": [SENTINEL],  # +45d -> available 2024-05-15 = D
            "2023-12-30": [1111.0],    # +45d -> available 2024-02-13 (control, visible)
        },
        index=["Total Revenue"],
    )
    annual = pd.DataFrame({"2023-09-30": [2222.0], "2022-09-24": [2221.0]}, index=["Total Revenue"])
    statements = {}
    for q_attr, a_attr in (
        ("quarterly_balance_sheet", "balance_sheet"),
        ("quarterly_cashflow", "cashflow"),
        ("quarterly_income_stmt", "income_stmt"),
    ):
        statements[q_attr] = quarterly
        statements[a_attr] = annual
    return FakeYahoo(statements=statements)


@pytest.mark.parametrize("tool_name", ["get_balance_sheet", "get_cashflow", "get_income_statement"])
def test_statements_filed_on_or_after_cutoff_are_hidden(monkeypatch, tmp_path, tool_name):
    from tradingagents.agents.utils import fundamental_data_tools as fdt

    tool = getattr(fdt, tool_name)
    _, out = _run_tool_on_day(
        monkeypatch, tmp_path, _statement_yahoo(), "AAPL", STATEMENT_DAY,
        lambda date: tool.invoke({"ticker": "AAPL", "curr_date": date, "freq": "quarterly"}),
    )

    assert SENTINEL_DIGITS not in out
    assert ("2024-03-30" in out) is api.FILINGS_SHOW_ITEMS_AVAILABLE_ON_CUTOFF
    assert "2024-03-31" not in out
    assert "2023-12-30" in out


def test_insider_rows_available_on_or_after_cutoff_are_hidden(monkeypatch, tmp_path):
    from tradingagents.agents.utils.news_data_tools import get_insider_transactions

    yahoo = FakeYahoo(
        insider=pd.DataFrame(
            [
                {"Start Date": "2024-05-10", "Value": SENTINEL},  # +2 US BD -> 2024-05-14 = D-1
                {"Start Date": "2024-05-13", "Value": SENTINEL},  # +2 US BD -> 2024-05-15 = D
                {"Start Date": "2024-05-08", "Value": 3333.0},    # +2 US BD -> 2024-05-10 (control)
            ]
        )
    )
    _, out = _run_tool_on_day(
        monkeypatch, tmp_path, yahoo, "AAPL", STATEMENT_DAY,
        lambda date: get_insider_transactions.invoke({"ticker": "AAPL", "curr_date": date}),
    )

    assert SENTINEL_DIGITS not in out
    assert ("2024-05-10" in out) is api.FILINGS_SHOW_ITEMS_AVAILABLE_ON_CUTOFF
    assert "2024-05-13" not in out
    assert "2024-05-08" in out


# --------------------------------------------------------------------------
# News (ticker and global)
# --------------------------------------------------------------------------


def _article(title, pub):
    return {
        "content": {
            "title": title,
            "summary": f"{title} summary",
            "provider": {"displayName": "Wire"},
            "canonicalUrl": {"url": f"https://example.invalid/{title.replace(' ', '-')}"},
            "pubDate": pub,
        }
    }


NEWS = [
    _article("SENTINEL-987654 published during day D", "2024-05-15T14:30:00Z"),
    _article("Control headline before the cutoff", "2024-05-13T15:00:00Z"),
]


def test_ticker_news_published_on_day_d_is_hidden(monkeypatch, tmp_path):
    from tradingagents.agents.utils.news_data_tools import get_news

    _, out = _run_tool_on_day(
        monkeypatch, tmp_path, FakeYahoo(news=NEWS), "AAPL", "2024-05-15",
        lambda date: get_news.invoke({"ticker": "AAPL", "start_date": _shift(date, -7), "end_date": date}),
    )

    assert SENTINEL_DIGITS not in out
    assert "Control headline" in out


def test_global_news_published_on_day_d_is_hidden(monkeypatch, tmp_path):
    from tradingagents.agents.utils.news_data_tools import get_global_news

    _, out = _run_tool_on_day(
        monkeypatch, tmp_path, FakeYahoo(news=NEWS), "AAPL", "2024-05-15",
        lambda date: get_global_news.invoke({"curr_date": date, "look_back_days": 7, "limit": 5}),
    )

    assert SENTINEL_DIGITS not in out
    assert "Control headline" in out
