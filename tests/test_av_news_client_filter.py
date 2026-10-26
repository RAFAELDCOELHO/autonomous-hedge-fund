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


# --- dict payloads without a "feed" key -------------------------------------


def _no_feed_payload():
    return {
        "Information": "renamed key",
        "articles": [_item("old", "20240110T153000"), _item("future", "20240301T090000"), _item("missing", ...)],
        "meta": {"nested": [_item("nested_future", "20240112T000000"), _item("nested_old", "20240102T000000")]},
        "tags": ["a", "b"],
    }


@pytest.mark.parametrize("as_text", [False, True])
@pytest.mark.parametrize("fn", ["news", "global"])
def test_no_feed_dict_lists_with_time_published_are_filtered(monkeypatch, as_text, fn):
    _patch(monkeypatch, (lambda: json.dumps(_no_feed_payload())) if as_text else _no_feed_payload)
    if fn == "news":
        out = alpha_vantage_news.get_news("AAPL", "2024-01-05", "2024-01-12")
    else:
        out = alpha_vantage_news.get_global_news("2024-01-12")
    data = json.loads(out) if as_text else out
    assert "20240301" not in json.dumps(data) and "20240112T" not in json.dumps(data)
    assert [i["title"] for i in data["articles"]] == ["old"]
    assert [i["title"] for i in data["meta"]["nested"]] == ["nested_old"]
    assert data["tags"] == ["a", "b"] and data["Information"] == "renamed key"


def test_no_feed_list_mixing_stamped_and_unstamped_items_drops_unstamped(monkeypatch):
    _patch(monkeypatch, lambda: {"data": [{"title": "x"}, _item("future", "20240301T090000"), 5]})
    out = alpha_vantage_news.get_news("AAPL", "2024-01-05", "2024-01-12")
    assert out == {"data": []}


@pytest.mark.parametrize(
    "note",
    [
        {"Information": "Invalid inputs"},
        {"Error Message": "bad", "detail": ["x", {"k": 1}]},
        {"Note": "rate", "nested": {"list": [1, 2]}},
    ],
)
def test_pure_error_dicts_pass_through_unchanged(monkeypatch, note):
    text = json.dumps(note)
    _patch(monkeypatch, text)
    assert alpha_vantage_news.get_news("AAPL", "2024-01-05", "2024-01-12") == text
    _patch(monkeypatch, lambda: json.loads(text))
    assert alpha_vantage_news.get_global_news("2024-01-12") == note


def test_no_feed_single_stamped_dict_after_cutoff_is_removed(monkeypatch):
    _patch(monkeypatch, lambda: {"top": _item("future", "20240301T090000"), "keep": _item("old", "20240102T0000")})
    out = alpha_vantage_news.get_news("AAPL", "2024-01-05", "2024-01-12")
    assert out == {"keep": _item("old", "20240102T0000")}


@pytest.mark.parametrize("ts", ["20240301T090000", "bad", None])
def test_no_feed_payload_that_is_itself_a_late_item_is_not_echoed(monkeypatch, ts):
    _patch(monkeypatch, lambda: json.dumps(_item("future", ts)))
    out = alpha_vantage_news.get_news("AAPL", "2024-01-05", "2024-01-12")
    assert json.loads(out) == {"feed": []}


def test_no_feed_items_nested_under_unstamped_list_are_filtered(monkeypatch):
    payload = {"groups": [{"name": "g", "items": [_item("future", "20240301T090000"), _item("old", "20240102T0000")]}]}
    _patch(monkeypatch, lambda: payload)
    out = alpha_vantage_news.get_news("AAPL", "2024-01-05", "2024-01-12")
    assert out == {"groups": [{"name": "g", "items": [_item("old", "20240102T0000")]}]}


# --- the time_to sent to AV (the client filter compares against it) --------


def test_time_to_sent_to_av_is_midnight_of_end_date_without_tz_shift(monkeypatch):
    seen = _patch(monkeypatch, _payload)
    alpha_vantage_news.get_news("AAPL", "2023-12-29", "2024-03-10")  # US DST starts 2024-03-10
    alpha_vantage_news.get_global_news("2024-07-04", look_back_days=1)
    assert seen[0]["time_to"] == "20240310T0000"
    assert seen[1]["time_to"] == "20240704T0000"
    for params in seen:
        assert len(params["time_to"]) == 13 and params["time_to"][8] == "T"
