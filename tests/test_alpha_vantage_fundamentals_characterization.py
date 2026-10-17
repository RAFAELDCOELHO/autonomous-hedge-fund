"""Characterization tests for Alpha Vantage fundamentals lookahead behavior.

These tests intentionally lock today's behavior of
`alpha_vantage_fundamentals.get_fundamentals`: it ignores `curr_date` and calls
Alpha Vantage OVERVIEW with only `{symbol}`, which is not point-in-time and
therefore a known lookahead leak (known residual, see #63).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tradingagents.agents.utils import fundamental_data_tools
from tradingagents.dataflows import alpha_vantage_fundamentals, interface


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "alpha_vantage_overview_synthetic.json"


def _strip_provenance(payload_text: str) -> dict:
    payload = json.loads(payload_text)
    payload.pop("_provenance", None)
    return payload


def test_characterization_av_get_fundamentals_matches_synthetic_snapshot(monkeypatch):
    synthetic = FIXTURE_PATH.read_text(encoding="utf-8")
    calls = []

    def _fake_api(function_name, params):
        calls.append((function_name, params))
        return synthetic

    monkeypatch.setattr(alpha_vantage_fundamentals, "_make_api_request", _fake_api)

    out = alpha_vantage_fundamentals.get_fundamentals("AAPL", "2026-10-17")

    assert _strip_provenance(out) == _strip_provenance(synthetic)
    assert calls == [("OVERVIEW", {"symbol": "AAPL"})]


def test_characterization_tool_forwards_capped_trade_date_to_vendor(monkeypatch):
    """Shows only tool-level forwarding of the capped date; AV OVERVIEW discards it."""
    seen = []

    def _fake_alpha_vantage_vendor(ticker: str, curr_date: str | None = None):
        seen.append((ticker, curr_date))
        return f"{ticker}:{curr_date}"

    monkeypatch.setitem(interface.VENDOR_METHODS["get_fundamentals"], "alpha_vantage", _fake_alpha_vantage_vendor)
    monkeypatch.setattr(interface, "get_vendor", lambda _category, _method=None: "alpha_vantage")

    out = fundamental_data_tools.get_fundamentals.func("AAPL", "2026-10-17", "2026-10-15")

    assert seen == [("AAPL", "2026-10-15")]
    assert out == "AAPL:2026-10-15"


def test_characterization_tool_forwards_earlier_model_date_to_vendor(monkeypatch):
    """Shows only tool-level forwarding of earlier date; AV OVERVIEW still discards it."""
    seen = []

    def _fake_alpha_vantage_vendor(ticker: str, curr_date: str | None = None):
        seen.append((ticker, curr_date))
        return f"{ticker}:{curr_date}"

    monkeypatch.setitem(interface.VENDOR_METHODS["get_fundamentals"], "alpha_vantage", _fake_alpha_vantage_vendor)
    monkeypatch.setattr(interface, "get_vendor", lambda _category, _method=None: "alpha_vantage")

    out = fundamental_data_tools.get_fundamentals.func("AAPL", "2026-10-14", "2026-10-15")

    assert seen == [("AAPL", "2026-10-14")]
    assert out == "AAPL:2026-10-14"


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="AV OVERVIEW is not point-in-time; known residual, see #63",
)
def test_characterization_expected_date_aware_vendor_would_change_output(monkeypatch):
    """Secondary alarm: snapshot is primary; this xfail tracks AV date-awareness only.

    It does not flip to XPASS just because #63 merges. It flips only when the
    Alpha Vantage path itself becomes point-in-time.
    """
    forwarded_non_symbol_params = []

    def _date_aware_vendor(function_name: str, params: dict) -> str:
        # Record every forwarded request param except the required symbol key.
        forwarded_non_symbol_params.append(
            {key: value for key, value in params.items() if key != "symbol"}
        )
        # A point-in-time OVERVIEW path would need a date/as-of input and vary by it.
        return json.dumps(
            {"function": function_name, "symbol": params.get("symbol"), "as_of": params.get("curr_date")},
            sort_keys=True,
        )

    monkeypatch.setattr(alpha_vantage_fundamentals, "_make_api_request", _date_aware_vendor)

    out_early = alpha_vantage_fundamentals.get_fundamentals("AAPL", "2026-10-14")
    out_late = alpha_vantage_fundamentals.get_fundamentals("AAPL", "2026-10-17")

    forwarded_keys = {key for payload in forwarded_non_symbol_params for key in payload}
    assert out_early != out_late or bool(forwarded_keys), (
        "Expected date-aware behavior (different output by curr_date) or forwarded non-symbol params; "
        f"forwarded keys were: {sorted(forwarded_keys)}"
    )
