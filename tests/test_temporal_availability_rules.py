"""Tests for approximate public-availability lag rules (P4.3/P4.4)."""

from __future__ import annotations

import copy
import unittest
from unittest.mock import Mock, patch

import pandas as pd

from tradingagents.agents.utils import news_data_tools
from tradingagents.dataflows import alpha_vantage_fundamentals, alpha_vantage_news, y_finance


class StatementAvailabilityTests(unittest.TestCase):
    def test_alpha_vantage_filters_use_strict_boundaries(self):
        result = {
            "annualReports": [{"fiscalDateEnding": "2023-12-31", "tag": "annual"}],
            "quarterlyReports": [{"fiscalDateEnding": "2024-03-31", "tag": "q1"}],
        }

        out_deadline = alpha_vantage_fundamentals._filter_reports_by_date(
            copy.deepcopy(result), "2024-03-31"
        )
        out_after = alpha_vantage_fundamentals._filter_reports_by_date(
            copy.deepcopy(result), "2024-04-01"
        )

        self.assertEqual(out_deadline["annualReports"], [])
        self.assertEqual(len(out_after["annualReports"]), 1)

    def test_alpha_vantage_q4_quarter_uses_annual_lag(self):
        result = {
            "annualReports": [{"fiscalDateEnding": "2024-09-30"}],
            "quarterlyReports": [
                {"fiscalDateEnding": "2024-06-30", "tag": "q3"},
                {"fiscalDateEnding": "2024-09-30", "tag": "q4"},
            ],
        }

        out_deadline = alpha_vantage_fundamentals._filter_reports_by_date(
            copy.deepcopy(result), "2024-12-30"
        )
        out_after = alpha_vantage_fundamentals._filter_reports_by_date(
            copy.deepcopy(result), "2024-12-31"
        )

        self.assertEqual([r["tag"] for r in out_deadline["quarterlyReports"]], ["q3"])
        self.assertEqual([r["tag"] for r in out_after["quarterlyReports"]], ["q3", "q4"])

    def test_yfinance_balance_sheet_applies_quarterly_lag(self):
        quarterly = pd.DataFrame([[1.0]], columns=["2024-03-31"])
        annual = pd.DataFrame([[1.0]], columns=["2023-12-31"])

        ticker_obj = Mock()
        ticker_obj.quarterly_balance_sheet = quarterly
        ticker_obj.balance_sheet = annual

        with patch.object(y_finance.yf, "Ticker", return_value=ticker_obj):
            out_deadline = y_finance.get_balance_sheet("AAPL", "quarterly", "2024-05-15")
            out_after = y_finance.get_balance_sheet("AAPL", "quarterly", "2024-05-16")

        self.assertIn("No balance sheet data found", out_deadline)
        self.assertIn("2024-03-31", out_after)

    def test_yfinance_quarterly_balance_sheet_tolerates_annual_fetch_failure(self):
        quarterly = pd.DataFrame([[1.0]], columns=["2024-03-31"])

        class TickerWithFailingAnnual:
            @property
            def quarterly_balance_sheet(self):
                return quarterly

            @property
            def balance_sheet(self):
                raise RuntimeError("annual fetch failed")

        ticker_obj = TickerWithFailingAnnual()

        with patch.object(y_finance.yf, "Ticker", return_value=ticker_obj):
            out_deadline = y_finance.get_balance_sheet("AAPL", "quarterly", "2024-05-15")
            out_after = y_finance.get_balance_sheet("AAPL", "quarterly", "2024-05-16")

        self.assertIn("No balance sheet data found", out_deadline)
        self.assertIn("2024-03-31", out_after)


class InsiderAvailabilityTests(unittest.TestCase):
    def test_insider_tool_schema_requires_curr_date(self):
        schema = news_data_tools.get_insider_transactions.args_schema.model_json_schema()
        self.assertIn("required", schema)
        self.assertIn("ticker", schema["required"])
        self.assertIn("curr_date", schema["required"])

    def test_insider_tool_wrapper_passes_curr_date_to_route(self):
        with patch(
            "tradingagents.agents.utils.news_data_tools.route_to_vendor",
            return_value="ok",
        ) as mock_route:
            output = news_data_tools.get_insider_transactions.func("AAPL", "2024-07-09")

        self.assertEqual(output, "ok")
        mock_route.assert_called_once_with("get_insider_transactions", "AAPL", "2024-07-09")

    def test_yfinance_insider_uses_us_business_day_lag_with_strict_boundary(self):
        insider_data = pd.DataFrame(
            [
                {"Start Date": "2024-07-03", "Shares": 10},
                {"Start Date": "2024-07-05", "Shares": 20},
            ]
        )
        ticker_obj = Mock()
        ticker_obj.insider_transactions = insider_data

        with patch.object(y_finance.yf, "Ticker", return_value=ticker_obj):
            out_deadline = y_finance.get_insider_transactions("AAPL", "2024-07-08")
            out_after = y_finance.get_insider_transactions("AAPL", "2024-07-09")

        self.assertIn("No insider transactions data found", out_deadline)
        self.assertIn("2024-07-03", out_after)
        self.assertNotIn("2024-07-05", out_after)

    def test_alpha_vantage_insider_uses_us_business_day_lag_with_strict_boundary(self):
        payload = {
            "data": [
                {"transaction_date": "2024-07-03", "shares": "10"},
                {"transaction_date": "2024-07-05", "shares": "20"},
            ]
        }
        with patch.object(alpha_vantage_news, "_make_api_request", return_value=dict(payload)):
            out_deadline = alpha_vantage_news.get_insider_transactions("AAPL", "2024-07-08")
        with patch.object(alpha_vantage_news, "_make_api_request", return_value=dict(payload)):
            out_after = alpha_vantage_news.get_insider_transactions("AAPL", "2024-07-09")

        self.assertEqual(out_deadline["data"], [])
        self.assertEqual(len(out_after["data"]), 1)
        self.assertEqual(out_after["data"][0]["transaction_date"], "2024-07-03")


class ReviewFollowUpTests(unittest.TestCase):
    def test_alpha_vantage_statements_filter_raw_json_text(self):
        import json
        payload = {
            "annualReports": [{"fiscalDateEnding": "2023-12-31"}],
            "quarterlyReports": [{"fiscalDateEnding": "2024-03-31"}],
        }
        with patch.object(
            alpha_vantage_fundamentals, "_make_api_request", return_value=json.dumps(payload)
        ):
            out = alpha_vantage_fundamentals.get_balance_sheet("AAPL", "quarterly", "2024-03-31")
        parsed = json.loads(out)
        self.assertEqual(parsed["annualReports"], [])
        self.assertEqual(parsed["quarterlyReports"], [])

    def test_alpha_vantage_insider_filters_raw_json_text(self):
        import json
        payload = {"data": [{"transaction_date": "2024-07-03"}, {"transaction_date": "2024-07-05"}]}
        with patch.object(alpha_vantage_news, "_make_api_request", return_value=json.dumps(payload)):
            out = alpha_vantage_news.get_insider_transactions("AAPL", "2024-07-09")
        self.assertEqual([r["transaction_date"] for r in json.loads(out)["data"]], ["2024-07-03"])

    def test_yfinance_non_quarterly_freq_alias_uses_annual_lag(self):
        ticker_obj = Mock()
        ticker_obj.income_stmt = pd.DataFrame([[1.0]], columns=["2024-09-30"])
        with patch.object(y_finance.yf, "Ticker", return_value=ticker_obj):
            out_deadline = y_finance.get_income_statement("AAPL", "yearly", "2024-12-30")
            out_after = y_finance.get_income_statement("AAPL", "yearly", "2024-12-31")
        self.assertIn("No income statement data found", out_deadline)
        self.assertIn("2024-09-30", out_after)

    def test_q4_detection_tolerates_leap_year_fiscal_year_end(self):
        from tradingagents.dataflows.stockstats_utils import filter_financials_by_date

        data = pd.DataFrame([[1.0]], columns=["2024-02-29"])
        kw = dict(freq="quarterly", annual_period_ends=["2024-02-29", "2025-02-28"])
        self.assertEqual(list(filter_financials_by_date(data, "2024-05-29", **kw).columns), [])
        self.assertEqual(
            list(filter_financials_by_date(data, "2024-05-30", **kw).columns), ["2024-02-29"]
        )

    def test_insider_filter_fails_closed_without_known_date_column(self):
        from tradingagents.dataflows.stockstats_utils import filter_insider_transactions_by_date

        data = pd.DataFrame([{"Filing Date": "2030-01-01", "Shares": 1}])
        self.assertEqual(len(filter_insider_transactions_by_date(data, "2024-01-01")), 0)


if __name__ == "__main__":
    unittest.main()
