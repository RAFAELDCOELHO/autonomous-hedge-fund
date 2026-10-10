"""P4.11 news look-ahead guard (yfinance ticker and global paths). Offline.

Decision D = 2024-01-04, so the tools receive D-1 = 2024-01-03 and the cutoff is
strictly before D 00:00:00 UTC. Both item formats are covered: ``content`` items
(ISO ``pubDate``, any UTC offset) and flat items (``providerPublishTime`` epoch).
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from tradingagents.dataflows import yfinance_news

D_MINUS_1 = "2024-01-03"


def _epoch(iso: str) -> int:
    return int(datetime.fromisoformat(iso).replace(tzinfo=UTC).timestamp())


def content(title, pub=None):
    body = {"title": title, "summary": "", "provider": {"displayName": "X"},
            "canonicalUrl": {"url": "https://example.test/" + title}}
    if pub is not None:
        body["pubDate"] = pub
    return {"content": body}


def flat(title, epoch=None):
    item = {"title": title, "publisher": "X", "link": "https://example.test/" + title}
    if epoch is not None:
        item["providerPublishTime"] = epoch
    return item


ITEMS_EXCLUDED = [
    content("C_AFTER_D", "2024-01-05T10:00:00Z"),
    flat("F_AFTER_D", _epoch("2024-01-05T10:00:00")),
    content("C_AT_CUTOFF", "2024-01-04T00:00:00Z"),
    flat("F_AT_CUTOFF", _epoch("2024-01-04T00:00:00")),
    content("C_UNDATED"),
    flat("F_UNDATED"),
    content("C_UNPARSEABLE", "not a date"),
    flat("F_UNPARSEABLE", "not a number"),
    content("C_NAIVE_AT_CUTOFF", "2024-01-04T00:00:00"),  # naive stamps are read as UTC
    content("C_OFFSET_AFTER", "2024-01-03T20:30:00-05:00"),  # = 2024-01-04 01:30 UTC
    flat("F_2026", _epoch("2026-10-07T12:00:00")),  # Lingxi's live repro on main
]
ITEMS_INCLUDED = [
    content("C_LAST_SECOND", "2024-01-03T23:59:59Z"),
    flat("F_LAST_SECOND", _epoch("2024-01-03T23:59:59")),
    content("C_OFFSET_BEFORE", "2024-01-04T01:00:00+03:00"),  # = 2024-01-03 22:00 UTC
    content("C_NAIVE_BEFORE", "2024-01-03T23:00:00"),  # naive, read as UTC
]


def _ticker_news(monkeypatch, items):
    class FakeTicker:
        def __init__(self, *_a, **_k):
            pass

        def get_news(self, count=20):
            return list(items)

    monkeypatch.setattr(yfinance_news.yf, "Ticker", FakeTicker)
    return yfinance_news.get_news_yfinance("AAPL", "2023-12-27", D_MINUS_1)


def _global_news(monkeypatch, items, limit=50):
    class FakeSearch:
        def __init__(self, *_a, **_k):
            self.news = list(items)

    monkeypatch.setattr(yfinance_news.yf, "Search", FakeSearch)
    return yfinance_news.get_global_news_yfinance(D_MINUS_1, 7, limit)


@pytest.mark.parametrize("path", [_ticker_news, _global_news], ids=["ticker", "global"])
def test_after_d_undated_and_cutoff_items_never_appear_in_either_format(monkeypatch, path):
    text = path(monkeypatch, ITEMS_EXCLUDED + ITEMS_INCLUDED)
    for item in ITEMS_EXCLUDED:
        title = item.get("title") or item["content"]["title"]
        assert f"### {title} " not in text, title
    for item in ITEMS_INCLUDED:
        title = item.get("title") or item["content"]["title"]
        assert f"### {title} " in text, title


@pytest.mark.parametrize("path", [_ticker_news, _global_news], ids=["ticker", "global"])
def test_only_excluded_items_give_no_news(monkeypatch, path):
    text = path(monkeypatch, ITEMS_EXCLUDED)
    assert "###" not in text
    assert text.startswith("No ")


@pytest.mark.parametrize("path", [_ticker_news, _global_news], ids=["ticker", "global"])
@pytest.mark.parametrize(
    ("item", "visible"),
    [
        (content("B", "2024-01-04T00:00:00Z"), False),
        (content("B", "2024-01-03T23:59:59Z"), True),
        (flat("B", _epoch("2024-01-04T00:00:00")), False),
        (flat("B", _epoch("2024-01-03T23:59:59")), True),
    ],
    ids=["content_D_0000", "content_Dm1_235959", "flat_D_0000", "flat_Dm1_235959"],
)
def test_cutoff_is_strict_at_d_midnight_utc(monkeypatch, path, item, visible):
    assert ("### B " in path(monkeypatch, [item])) is visible


def test_global_limit_is_applied_after_the_date_filter(monkeypatch):
    future = [content(f"FUT{i}", "2024-01-06T12:00:00Z") for i in range(3)]
    valid = [content(f"OK{i}", "2024-01-02T12:00:00Z") for i in range(3)]
    text = _global_news(monkeypatch, future + valid, limit=2)
    assert "FUT" not in text
    assert text.count("### OK") == 2


def test_global_repro_flat_2026_headlines_are_not_shown_for_2024(monkeypatch):
    items = [flat(f"H{i}", _epoch("2026-10-07T12:00:00")) for i in range(5)]
    text = _global_news(monkeypatch, items, limit=5)
    assert "###" not in text
