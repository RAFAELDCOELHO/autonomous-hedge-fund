"""Alpha Vantage news: client-side time_published ceiling (D-1 rule).

time_to is sent as YYYYMMDDT0000 of the end date; these tests assume AV ignores
it and check that items published at/after that instant never reach the agent.
Fail closed: an item with a missing/unparseable time_published is dropped.
"""

from __future__ import annotations

import json

import pytest

from tradingagents.dataflows import alpha_vantage_news


def _item(title, ts):
    item = {"title": title, "url": f"https://x/{title}", "summary": "s"}
    if ts is not ...:
        item["time_published"] = ts
    return item


FEED = [
    _item("old", "20240110T153000"),
    _item("edge_before", "20240111T235959"),
    _item("at_cutoff", "20240112T000000"),
    _item("after", "20240112T153000"),
    _item("future", "20240301T090000"),
    _item("missing", ...),
    _item("garbage", "yesterday"),
    _item("none", None),
    _item("short_ok", "20240111T1200"),
]
KEPT = ["old", "edge_before", "short_ok"]


def _payload():
    return {"items": str(len(FEED)), "sentiment_score_definition": "d", "feed": [dict(i) for i in FEED]}


def _patch(monkeypatch, response):
    seen = []

    def _fake(function_name, params):
        seen.append(dict(params))
        return response() if callable(response) else response

    monkeypatch.setattr(alpha_vantage_news, "_make_api_request", _fake)
    return seen


def _titles(out):
    data = json.loads(out) if isinstance(out, str) else out
    return [i["title"] for i in data["feed"]]


@pytest.mark.parametrize("as_text", [False, True])
def test_get_news_drops_items_at_or_after_time_to(monkeypatch, as_text):
    _patch(monkeypatch, (lambda: json.dumps(_payload())) if as_text else _payload)
    out = alpha_vantage_news.get_news("AAPL", "2024-01-05", "2024-01-12")
    assert isinstance(out, str) is as_text
    assert _titles(out) == KEPT


@pytest.mark.parametrize("as_text", [False, True])
def test_get_global_news_drops_items_at_or_after_curr_date(monkeypatch, as_text):
    _patch(monkeypatch, (lambda: json.dumps(_payload())) if as_text else _payload)
    out = alpha_vantage_news.get_global_news("2024-01-12", look_back_days=7, limit=50)
    assert _titles(out) == KEPT


def test_other_fields_preserved_and_items_count_updated(monkeypatch):
    _patch(monkeypatch, lambda: json.dumps(_payload(), indent=2))
    out = json.loads(alpha_vantage_news.get_news("AAPL", "2024-01-05", "2024-01-12"))
    assert out["sentiment_score_definition"] == "d"
    assert out["items"] == str(len(KEPT))
    assert out["feed"][0] == FEED[0]


def test_request_params_unchanged(monkeypatch):
    seen = _patch(monkeypatch, _payload)
    alpha_vantage_news.get_news("AAPL", "2024-01-05", "2024-01-12")
    assert seen == [{"tickers": "AAPL", "time_from": "20240105T0000", "time_to": "20240112T0000"}]


@pytest.mark.parametrize(
    "bad_feed",
    ["20240301T090000 future text", {"title": "future", "time_published": "20240301T090000"}, 7, None],
)
def test_non_list_feed_never_echoed(monkeypatch, bad_feed):
    _patch(monkeypatch, lambda: {"feed": bad_feed})
    out = alpha_vantage_news.get_news("AAPL", "2024-01-05", "2024-01-12")
    assert "20240301" not in json.dumps(out)
    assert out["feed"] == []


def test_non_dict_items_dropped(monkeypatch):
    _patch(monkeypatch, lambda: {"feed": ["20240301T090000", 3, _item("ok", "20240102T000000")]})
    out = alpha_vantage_news.get_news("AAPL", "2024-01-05", "2024-01-12")
    assert _titles(out) == ["ok"]


def test_payload_without_feed_passes_through(monkeypatch):
    """Error/info JSON from AV has no items: returned unchanged."""
    note = {"Information": "Invalid inputs"}
    _patch(monkeypatch, lambda: dict(note))
    assert alpha_vantage_news.get_news("AAPL", "2024-01-05", "2024-01-12") == note
    _patch(monkeypatch, lambda: json.dumps(note))
    assert json.loads(alpha_vantage_news.get_global_news("2024-01-12")) == note


def test_non_json_text_passes_through(monkeypatch):
    _patch(monkeypatch, "offline")
    assert alpha_vantage_news.get_news("AAPL", "2024-01-05", "2024-01-12") == "offline"


def test_top_level_list_never_echoed(monkeypatch):
    _patch(monkeypatch, lambda: json.dumps([_item("future", "20240301T090000")]))
    out = alpha_vantage_news.get_news("AAPL", "2024-01-05", "2024-01-12")
    assert "20240301" not in out
    assert json.loads(out) == {"feed": []}


def test_non_string_time_published_dropped_not_crash(monkeypatch):
    _patch(monkeypatch, lambda: {"feed": [_item("int", 20240102), _item("ok", "20240102T000000")]})
    out = alpha_vantage_news.get_news("AAPL", "2024-01-05", "2024-01-12")
    assert _titles(out) == ["ok"]
