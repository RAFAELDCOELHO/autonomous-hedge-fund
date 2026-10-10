"""OPTIONAL, NOT in tests.patch. Finding F1: real yfinance.Search truncates to
news_count, and get_global_news_yfinance asks for news_count=limit and stops
collecting at len(all_news) >= limit BEFORE the date filter. With a fake that
honours news_count, the fix as described (slice after filter) still returns a
header with zero items when the first `limit` results are post-cutoff."""

import time
import pytest
from _news_fakes import FakeNews, content_item, set_process_tz


def test_global_limit_after_filter_with_realistic_news_count(monkeypatch):
    import yfinance
    from tradingagents.dataflows import yfinance_news as yfn

    items = [content_item(f"TOK_RF{i}", f"2026-10-0{i + 1}T12:00:00Z") for i in range(5)]
    items += [content_item(f"TOK_RO{i}", f"2024-01-0{i + 1}T10:00:00Z") for i in range(3)]

    class _Search:
        def __init__(self, query=None, news_count=8, **kw):
            self.news = items[:news_count]

    monkeypatch.setattr(yfinance, "Search", _Search)
    out = yfn.get_global_news_yfinance("2024-01-03", 7, 3)
    assert sum(line.startswith("### ") for line in out.splitlines()) == 3, out
