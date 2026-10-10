"""P4.11: every agent tool caps its dates at the graph's trade_date (D-1). Offline.

Adapted from SWE's P4.6 cross-tool temporal guard package (pitcanary-ref
logs/p46_test_temporal_tool_guard.py, P46_PACKAGE.md): the inventory of graph
tools, the sentinel-after-cutoff fake vendor (FakeTicker, FUTURE sentinel,
ISO-date scan) and the end-to-end calls through route_to_vendor come from
there. What changes here is the date source: SWE's guard checked that tools
obey the curr_date the LLM gives them; this file runs each tool through a
langgraph ToolNode with state trade_date = D-1 while the LLM passes a date
after it (D, D+5), so the injected cap (temporal.cap_date) is what is tested.
"""

from __future__ import annotations

import inspect
import re

import numpy as np
import pandas as pd
import pytest
from langchain_core.messages import AIMessage
from langgraph.prebuilt import ToolNode

from tradingagents.agents.utils import (
    core_stock_tools,
    fingpt_tool,
    fundamental_data_tools,
    kronos_tool,
    news_data_tools,
    technical_indicators_tools,
)
from tradingagents.agents.utils.temporal import cap_date
from tradingagents.dataflows import y_finance
from tradingagents.graph.trading_graph import TradingAgentsGraph

CUTOFF = "2024-02-15"  # D-1 (Thu) of decision D = 2024-02-16
CUTOFF_TS = pd.Timestamp(CUTOFF)
LLM_DATES_AFTER = ["2024-02-16", "2024-02-21"]  # D and D+5
FUTURE = 987654321.0  # carried only by data dated after the cutoff
ISO = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
DATES = pd.bdate_range("2024-01-02", "2024-03-29")


def _value(d) -> float:
    return FUTURE if d > CUTOFF_TS else 1.0


def _dated_text(upto: str) -> str:
    """A vendor reply serving every dated row <= upto, as the real vendors' ceilings do."""
    return "\n".join(
        f"Headline dated {d:%Y-%m-%d} value {_value(d)}" for d in DATES if d <= pd.Timestamp(upto)
    )


def _kronos_frame(upto: str) -> pd.DataFrame:
    days = DATES[DATES <= pd.Timestamp(upto)]
    return pd.DataFrame({"Date": days, "Close": [_value(d) for d in days]})


# name -> (module, patched vendor entry point, position of the date in its args, LLM args for date d)
CAPPED = {
    "get_stock_data": (core_stock_tools, "route_to_vendor", 3,
                       lambda d: {"symbol": "AAPL", "start_date": "2024-01-02", "end_date": d, "curr_date": d}),
    "get_indicators": (technical_indicators_tools, "route_to_vendor", 3,
                       lambda d: {"symbol": "AAPL", "indicator": "rsi", "curr_date": d}),
    "get_kronos_forecast": (kronos_tool, "load_ohlcv", 1, lambda d: {"symbol": "AAPL", "curr_date": d}),
    "get_fundamentals": (fundamental_data_tools, "route_to_vendor", 2,
                         lambda d: {"ticker": "AAPL", "curr_date": d}),
    "get_balance_sheet": (fundamental_data_tools, "route_to_vendor", 3,
                          lambda d: {"ticker": "AAPL", "curr_date": d, "freq": "quarterly"}),
    "get_cashflow": (fundamental_data_tools, "route_to_vendor", 3,
                     lambda d: {"ticker": "AAPL", "curr_date": d, "freq": "quarterly"}),
    "get_income_statement": (fundamental_data_tools, "route_to_vendor", 3,
                             lambda d: {"ticker": "AAPL", "curr_date": d, "freq": "quarterly"}),
    "get_insider_transactions": (news_data_tools, "route_to_vendor", 2,
                                 lambda d: {"ticker": "AAPL", "curr_date": d}),
    "get_news": (news_data_tools, "route_to_vendor", 3,
                 lambda d: {"ticker": "AAPL", "start_date": "2024-01-02", "end_date": d}),
    "get_global_news": (news_data_tools, "route_to_vendor", 1, lambda d: {"curr_date": d}),
    "get_fingpt_sentiment_tool": (fingpt_tool, "route_to_vendor", 3,
                                  lambda d: {"symbol": "AAPL", "curr_date": d}),
}
# No LLM-visible date at all: trade_date is a required InjectedState argument
# (tests/test_macro_tools_point_in_time.py, tests/test_pr7_tool_cutoffs.py).
INJECTED_ONLY = {"get_selic", "get_inflation", "get_gdp", "get_exchange_rate"}
DATE_PARAMS = ("curr_date", "end_date", "start_date", "trade_date")


def _graph_tools():
    nodes = TradingAgentsGraph._create_tool_nodes(None)  # method does not use self
    return {name: t for node in nodes.values() for name, t in node.tools_by_name.items()}


def _invoke(tool, args, trade_date=CUTOFF) -> str:
    call = {"name": tool.name, "args": args, "id": "c1", "type": "tool_call"}
    state = {"messages": [AIMessage(content="", tool_calls=[call])], "trade_date": trade_date}
    out = ToolNode([tool], handle_tool_errors=False).invoke(state)
    return out["messages"][-1].content


def _install_vendor(monkeypatch, name):
    """Fake the tool's vendor boundary; return (vendor args seen, texts passed further down)."""
    module, attr, pos, _args = CAPPED[name]
    seen, downstream = [], []
    if attr == "load_ohlcv":
        def load(symbol, date, *a, **k):
            seen.append((symbol, date))
            return _kronos_frame(date)

        def signal(df, symbol, pred_len=5):
            last = float(df["Close"].iloc[-1])
            return {"pred_len": pred_len, "device": "cpu", "signal": "HOLD", "forecast_return_pct": 0.0,
                    "predicted_close": last, "current_close": last}

        monkeypatch.setattr(module, "load_ohlcv", load)
        monkeypatch.setattr(module, "get_kronos_signal", signal)
        return seen, downstream

    def route(*args):
        seen.append(args)
        return _dated_text(args[pos])

    monkeypatch.setattr(module, attr, route)
    if name == "get_fingpt_sentiment_tool":
        def sentiment(headlines, symbol, curr_date):
            downstream.append("\n".join(headlines) + f"\n{curr_date}")
            return {"scores": [0], "n": 1, "n_failed": 0, "label": "neutral", "score": 0.0}

        monkeypatch.setattr(module, "get_fingpt_sentiment", sentiment)
    return seen, downstream


def _assert_nothing_after_cutoff(text: str):
    late = [d for d in ISO.findall(text) if pd.Timestamp(d) > CUTOFF_TS]
    assert late == [], f"dates after the cutoff: {late}"
    assert "987654321" not in text, "a value dated after the cutoff leaked"


def test_cap_date_is_the_min_and_none_trade_date_keeps_the_llm_date():
    assert cap_date("2024-02-21", CUTOFF) == CUTOFF
    assert cap_date("2024-02-01", CUTOFF) == "2024-02-01"
    assert cap_date("2024-02-21", None) == "2024-02-21"
    assert cap_date(None, CUTOFF) == CUTOFF


@pytest.mark.parametrize("llm_date", LLM_DATES_AFTER)
@pytest.mark.parametrize("name", sorted(CAPPED))
def test_tool_is_capped_at_graph_trade_date(monkeypatch, name, llm_date):
    module, _attr, pos, make_args = CAPPED[name]
    seen, downstream = _install_vendor(monkeypatch, name)

    out = _invoke(getattr(module, name), make_args(llm_date))

    assert seen and all(args[pos] == CUTOFF for args in seen), seen
    for text in (out, *downstream):
        _assert_nothing_after_cutoff(text)


@pytest.mark.parametrize("name", sorted(CAPPED))
def test_earlier_llm_date_is_kept(monkeypatch, name):
    module, _attr, pos, make_args = CAPPED[name]
    seen, _ = _install_vendor(monkeypatch, name)
    _invoke(getattr(module, name), make_args("2024-02-01"))
    assert seen and [args[pos] for args in seen] == ["2024-02-01"] * len(seen)


@pytest.mark.parametrize("name", sorted(CAPPED))
def test_trade_date_is_hidden_from_the_llm_schema(name):
    module = CAPPED[name][0]
    assert "trade_date" not in getattr(module, name).tool_call_schema.model_json_schema()["properties"]


def test_every_dated_graph_tool_is_capped_and_exercised():
    """A new graph tool taking a date must be added to CAPPED (and so be tested) to pass."""
    tools = _graph_tools()
    dated = {n for n, t in tools.items() if set(DATE_PARAMS) & set(inspect.signature(t.func).parameters)}
    assert dated - set(CAPPED) - INJECTED_ONLY == set(), "dated tool without the trade_date cap test"
    assert set(CAPPED) | INJECTED_ONLY <= set(tools), "listed tool no longer bound in the graph"
    for name in dated:
        assert "trade_date" in inspect.signature(tools[name].func).parameters, f"{name}: no injected trade_date"
        assert "trade_date" not in tools[name].tool_call_schema.model_json_schema()["properties"], name


# ---------------------------------------------------------------------------
# End to end through route_to_vendor and the default yfinance vendor (SWE P4.6 fakes).
# ---------------------------------------------------------------------------


class FakeTicker:
    def __init__(self, symbol):
        self.symbol = symbol

    def history(self, start=None, end=None, **kwargs):  # end exclusive, like yfinance
        idx = pd.bdate_range("2023-06-01", "2024-06-28")
        close = 20.0 + np.arange(len(idx)) * 0.01
        close[idx > CUTOFF_TS] = FUTURE
        df = pd.DataFrame({"Open": close, "High": close, "Low": close, "Close": close, "Volume": 1000},
                          index=idx.tz_localize("America/New_York"))
        naive = df.index.tz_localize(None)
        return df[(naive >= pd.Timestamp(start)) & (naive < pd.Timestamp(end))]

    @property
    def insider_transactions(self):
        return pd.DataFrame({
            "Insider": ["OLD", "FRESH", "SAMEDAY", "FUTURE"],
            "Shares": [1, int(FUTURE), int(FUTURE), int(FUTURE)],
            # FRESH: 2024-02-14 + 2 US business days = 02-16 = D, public after the cutoff.
            "Start Date": pd.to_datetime(["2024-01-05", "2024-02-14", CUTOFF, "2024-03-01"]),
        })


@pytest.fixture
def fake_yfinance(monkeypatch):
    monkeypatch.setattr(y_finance, "yf_retry", lambda fn, *a, **k: fn())
    monkeypatch.setattr(y_finance.yf, "Ticker", FakeTicker)


@pytest.mark.parametrize("llm_date", LLM_DATES_AFTER)
def test_get_stock_data_end_to_end_last_row_is_the_cutoff(fake_yfinance, llm_date):
    out = _invoke(core_stock_tools.get_stock_data,
                  {"symbol": "AAPL", "start_date": "2024-01-02", "end_date": llm_date, "curr_date": llm_date})
    rows = [ln.split(",")[0][:10] for ln in out.splitlines() if ln[:4].isdigit()]
    assert rows and max(rows) == CUTOFF
    _assert_nothing_after_cutoff(out)


@pytest.mark.parametrize("llm_date", LLM_DATES_AFTER)
def test_get_insider_transactions_end_to_end_hides_rows_public_after_cutoff(fake_yfinance, llm_date):
    out = _invoke(news_data_tools.get_insider_transactions, {"ticker": "AAPL", "curr_date": llm_date})
    assert "OLD" in out
    for name in ("FRESH", "SAMEDAY", "FUTURE"):
        assert name not in out
    _assert_nothing_after_cutoff(out)
