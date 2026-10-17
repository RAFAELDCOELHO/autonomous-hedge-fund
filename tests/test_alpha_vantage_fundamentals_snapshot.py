from __future__ import annotations

from pathlib import Path

from langchain_core.messages import AIMessage
from langgraph.prebuilt import ToolNode

from tradingagents.agents.utils import fundamental_data_tools
from tradingagents.dataflows import alpha_vantage_fundamentals, interface


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "alpha_vantage_get_fundamentals_overview.json"


def _invoke_get_fundamentals(curr_date: str, trade_date: str) -> str:
    call = {
        "name": "get_fundamentals",
        "id": "av-fundamentals",
        "type": "tool_call",
        "args": {"ticker": "AAPL", "curr_date": curr_date},
    }
    state = {"messages": [AIMessage(content="", tool_calls=[call])], "trade_date": trade_date}
    out = ToolNode([fundamental_data_tools.get_fundamentals], handle_tool_errors=False).invoke(state)
    return out["messages"][-1].content


def test_alpha_vantage_get_fundamentals_matches_recorded_snapshot(monkeypatch):
    recorded = FIXTURE_PATH.read_text(encoding="utf-8")

    calls = []

    def _fake_api(function_name, params):
        calls.append((function_name, params))
        return recorded

    monkeypatch.setattr(alpha_vantage_fundamentals, "_make_api_request", _fake_api)

    out = alpha_vantage_fundamentals.get_fundamentals("AAPL", "2026-10-17")

    assert out == recorded
    assert calls == [("OVERVIEW", {"symbol": "AAPL"})]


def test_alpha_vantage_get_fundamentals_uses_capped_trade_date(monkeypatch):
    seen = []

    def _fake_alpha_vantage_vendor(ticker: str, curr_date: str | None = None):
        seen.append((ticker, curr_date))
        return f"{ticker}:{curr_date}"

    monkeypatch.setitem(interface.VENDOR_METHODS["get_fundamentals"], "alpha_vantage", _fake_alpha_vantage_vendor)
    monkeypatch.setattr(interface, "get_vendor", lambda _category, _method=None: "alpha_vantage")

    out = _invoke_get_fundamentals(curr_date="2026-10-17", trade_date="2026-10-15")

    assert seen == [("AAPL", "2026-10-15")]
    assert out == "AAPL:2026-10-15"


def test_alpha_vantage_get_fundamentals_keeps_earlier_model_date(monkeypatch):
    seen = []

    def _fake_alpha_vantage_vendor(ticker: str, curr_date: str | None = None):
        seen.append((ticker, curr_date))
        return f"{ticker}:{curr_date}"

    monkeypatch.setitem(interface.VENDOR_METHODS["get_fundamentals"], "alpha_vantage", _fake_alpha_vantage_vendor)
    monkeypatch.setattr(interface, "get_vendor", lambda _category, _method=None: "alpha_vantage")

    out = _invoke_get_fundamentals(curr_date="2026-10-14", trade_date="2026-10-15")

    assert seen == [("AAPL", "2026-10-14")]
    assert out == "AAPL:2026-10-14"
