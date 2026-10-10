"""Offline fakes for the yfinance news tools (PR #64 news cutoff tests).

`content_item` mimics yfinance's nested news format (``content.pubDate`` ISO
string); `flat_item` mimics the legacy flat format (``providerPublishTime``
epoch seconds, no ``content`` wrapper). Every item's title carries a unique
token so tests can assert presence/absence on the returned string.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

MISSING = object()  # leave the date field out entirely


def content_item(token: str, pub_date=MISSING) -> dict:
    content = {
        "title": f"{token} headline",
        "summary": f"{token} summary",
        "provider": {"displayName": "Wire"},
        "canonicalUrl": {"url": f"https://example.invalid/{token}"},
    }
    if pub_date is not MISSING:
        content["pubDate"] = pub_date
    return {"id": token, "content": content}


def flat_item(token: str, publish_time=MISSING) -> dict:
    item = {
        "uuid": token,
        "title": f"{token} headline",
        "publisher": "Wire",
        "link": f"https://example.invalid/{token}",
        "type": "STORY",
    }
    if publish_time is not MISSING:
        item["providerPublishTime"] = publish_time
    return item


def epoch(iso_utc: str) -> int:
    """'2024-01-04T00:00:00' (UTC) -> epoch seconds."""
    return int(datetime.fromisoformat(iso_utc).replace(tzinfo=timezone.utc).timestamp())


class FakeNews:
    """Routes yfinance.Ticker(...).get_news and yfinance.Search(...).news to
    one item list. The fake returns its whole list regardless of `count` /
    `news_count` (see PACKAGE.md: real yfinance truncates to news_count)."""

    def __init__(self, items):
        self.items = list(items)
        self.search_calls = []

    def install(self, monkeypatch):
        import yfinance

        fake = self

        class _Ticker:
            def __init__(self, symbol):
                self.symbol = symbol

            def get_news(self, count=10, tab="news", **kwargs):
                return list(fake.items)

        class _Search:
            def __init__(self, query=None, news_count=8, **kwargs):
                fake.search_calls.append({"query": query, "news_count": news_count})
                self.news = list(fake.items)

        monkeypatch.setattr(yfinance, "Ticker", _Ticker)
        monkeypatch.setattr(yfinance, "Search", _Search)
        return self


def set_process_tz(monkeypatch, tzname: str) -> None:
    """Force the process-local timezone (affects naive datetime.fromtimestamp /
    astimezone). monkeypatch restores TZ; the caller's teardown re-runs tzset."""
    monkeypatch.setenv("TZ", tzname)
    time.tzset()
