from __future__ import annotations

from pathlib import Path

from tradingagents.agents.utils import fundamental_data_tools
from tradingagents.dataflows import alpha_vantage_fundamentals, interface


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "alpha_vantage_get_fundamentals_overview.json"


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

    out = fundamental_data_tools.get_fundamentals.func("AAPL", "2026-10-17", "2026-10-15")

    assert seen == [("AAPL", "2026-10-15")]
    assert out == "AAPL:2026-10-15"


def test_alpha_vantage_get_fundamentals_keeps_earlier_model_date(monkeypatch):
    seen = []

    def _fake_alpha_vantage_vendor(ticker: str, curr_date: str | None = None):
        seen.append((ticker, curr_date))
        return f"{ticker}:{curr_date}"

    monkeypatch.setitem(interface.VENDOR_METHODS["get_fundamentals"], "alpha_vantage", _fake_alpha_vantage_vendor)
    monkeypatch.setattr(interface, "get_vendor", lambda _category, _method=None: "alpha_vantage")

    out = fundamental_data_tools.get_fundamentals.func("AAPL", "2026-10-14", "2026-10-15")

    assert seen == [("AAPL", "2026-10-14")]
    assert out == "AAPL:2026-10-14"
