from __future__ import annotations

from datetime import datetime, timezone
import importlib
import sys
import types
import unittest
from unittest import mock

MODULE_NAME = "tradingagents.dataflows.yfinance_news"


def _load_extract_article_data():
    """Import helper with scoped dependency stubs to keep tests offline."""
    fake_yfinance = types.ModuleType("yfinance")
    fake_exceptions = types.ModuleType("yfinance.exceptions")

    class _FakeYFRateLimitError(Exception):
        pass

    fake_exceptions.YFRateLimitError = _FakeYFRateLimitError
    fake_yfinance.exceptions = fake_exceptions

    fake_stockstats_utils = types.ModuleType("tradingagents.dataflows.stockstats_utils")
    fake_stockstats_utils.yf_retry = lambda fn: fn()

    fake_dateutil = types.ModuleType("dateutil")
    fake_relativedelta_mod = types.ModuleType("dateutil.relativedelta")
    fake_relativedelta_mod.relativedelta = lambda **kwargs: None
    fake_dateutil.relativedelta = fake_relativedelta_mod

    with mock.patch.dict(
        sys.modules,
        {
            "yfinance": fake_yfinance,
            "yfinance.exceptions": fake_exceptions,
            "tradingagents.dataflows.stockstats_utils": fake_stockstats_utils,
            "dateutil": fake_dateutil,
            "dateutil.relativedelta": fake_relativedelta_mod,
        },
    ):
        sys.modules.pop(MODULE_NAME, None)
        mod = importlib.import_module(MODULE_NAME)
        helper = mod._extract_article_data
        sys.modules.pop(MODULE_NAME, None)
        return helper


class YFinanceNewsHelperTests(unittest.TestCase):
    def test_extract_article_data_from_nested_content(self):
        extract_article_data = _load_extract_article_data()
        article = {
            "content": {
                "title": "Macro outlook improves",
                "summary": "Analysts expect softer inflation.",
                "provider": {"displayName": "Reuters"},
                "canonicalUrl": {"url": "https://example.com/story"},
                "pubDate": "2026-01-02T12:30:00Z",
            }
        }

        extracted = extract_article_data(article)

        self.assertEqual(extracted["title"], "Macro outlook improves")
        self.assertEqual(extracted["summary"], "Analysts expect softer inflation.")
        self.assertEqual(extracted["publisher"], "Reuters")
        self.assertEqual(extracted["link"], "https://example.com/story")
        self.assertEqual(extracted["pub_date"], datetime(2026, 1, 2, 12, 30, tzinfo=timezone.utc))

    def test_extract_article_data_uses_clickthrough_and_tolerates_bad_date(self):
        extract_article_data = _load_extract_article_data()
        article = {
            "content": {
                "title": "No canonical URL",
                "summary": "",
                "provider": {},
                "clickThroughUrl": {"url": "https://example.com/alt"},
                "pubDate": "not-a-date",
            }
        }

        extracted = extract_article_data(article)

        self.assertEqual(extracted["title"], "No canonical URL")
        self.assertEqual(extracted["summary"], "")
        self.assertEqual(extracted["publisher"], "Unknown")
        self.assertEqual(extracted["link"], "https://example.com/alt")
        self.assertIsNone(extracted["pub_date"])

    def test_extract_article_data_from_flat_payload(self):
        extract_article_data = _load_extract_article_data()
        article = {
            "title": "Flat payload",
            "summary": "Fallback shape",
            "publisher": "AP",
            "link": "https://example.com/flat",
        }

        extracted = extract_article_data(article)

        self.assertEqual(
            extracted,
            {
                "title": "Flat payload",
                "summary": "Fallback shape",
                "publisher": "AP",
                "link": "https://example.com/flat",
                "pub_date": None,
            },
        )


if __name__ == "__main__":
    unittest.main()
