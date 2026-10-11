"""Tests for approximate public-availability lag rules (P4.3/P4.4)."""

from __future__ import annotations

import copy
import unittest
import warnings
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


class StatementToolSchemaTests(unittest.TestCase):
    def test_statement_tools_require_curr_date(self):
        from tradingagents.agents.utils import fundamental_data_tools as fdt

        for tool in (fdt.get_balance_sheet, fdt.get_cashflow, fdt.get_income_statement):
            schema = tool.args_schema.model_json_schema()
            self.assertIn("curr_date", schema.get("required", []), tool.name)


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


class LingxiFollowUpTests(unittest.TestCase):
    def test_yfinance_cashflow_passes_freq_and_annual_period_ends_to_filter(self):
        quarterly = pd.DataFrame([[1.0]], columns=["2024-06-30"])
        annual = pd.DataFrame([[1.0, 2.0]], columns=["2022-06-30", "2023-09-30"])
        ticker_obj = Mock()
        ticker_obj.quarterly_cashflow = quarterly
        ticker_obj.cashflow = annual

        with patch.object(y_finance.yf, "Ticker", return_value=ticker_obj), patch.object(
            y_finance,
            "filter_financials_by_date",
            side_effect=lambda *args, **kwargs: args[0],
        ) as mock_filter:
            out = y_finance.get_cashflow("AAPL", "quarterly", "2024-12-31")

        self.assertIn("2024-06-30", out)
        mock_filter.assert_called_once()
        _, kwargs = mock_filter.call_args
        self.assertEqual(kwargs.get("freq"), "quarterly")
        self.assertEqual(list(kwargs.get("annual_period_ends")), list(annual.columns))

    def test_yfinance_income_statement_passes_freq_and_annual_period_ends_to_filter(self):
        quarterly = pd.DataFrame([[1.0]], columns=["2024-06-30"])
        annual = pd.DataFrame([[1.0, 2.0]], columns=["2022-06-30", "2023-09-30"])
        ticker_obj = Mock()
        ticker_obj.quarterly_income_stmt = quarterly
        ticker_obj.income_stmt = annual

        with patch.object(y_finance.yf, "Ticker", return_value=ticker_obj), patch.object(
            y_finance,
            "filter_financials_by_date",
            side_effect=lambda *args, **kwargs: args[0],
        ) as mock_filter:
            out = y_finance.get_income_statement("AAPL", "quarterly", "2024-12-31")

        self.assertIn("2024-06-30", out)
        mock_filter.assert_called_once()
        _, kwargs = mock_filter.call_args
        self.assertEqual(kwargs.get("freq"), "quarterly")
        self.assertEqual(list(kwargs.get("annual_period_ends")), list(annual.columns))

    def test_yfinance_cashflow_applies_temporal_filter(self):
        quarterly = pd.DataFrame([[1.0]], columns=["2024-03-31"])
        annual = pd.DataFrame([[1.0]], columns=["2023-12-31"])
        ticker_obj = Mock()
        ticker_obj.quarterly_cashflow = quarterly
        ticker_obj.cashflow = annual

        with patch.object(y_finance.yf, "Ticker", return_value=ticker_obj):
            out_deadline = y_finance.get_cashflow("AAPL", "quarterly", "2024-05-15")
            out_after = y_finance.get_cashflow("AAPL", "quarterly", "2024-05-16")

        self.assertIn("No cash flow data found", out_deadline)
        self.assertIn("2024-03-31", out_after)

    def test_yfinance_uses_most_recent_annual_period_for_fye_inference(self):
        quarterly = pd.DataFrame([[1.0]], columns=["2024-09-30"])
        annual = pd.DataFrame([[1.0, 2.0]], columns=["2022-06-30", "2023-09-30"])
        ticker_obj = Mock()
        ticker_obj.quarterly_income_stmt = quarterly
        ticker_obj.income_stmt = annual

        with patch.object(y_finance.yf, "Ticker", return_value=ticker_obj):
            out_before_q4_publication = y_finance.get_income_statement("AAPL", "quarterly", "2024-11-20")
            out_after_q4_publication = y_finance.get_income_statement("AAPL", "quarterly", "2024-12-31")

        self.assertIn("No income statement data found", out_before_q4_publication)
        self.assertIn("2024-09-30", out_after_q4_publication)

    def test_non_december_fye_q4_and_regular_quarter_lags_hold_for_all_statements(self):
        annual = pd.DataFrame([[1.0, 2.0]], columns=["2023-09-30", "2022-09-24"])
        quarterly = pd.DataFrame([[1.0, 2.0]], columns=["2024-06-30", "2024-09-30"])

        cases = [
            ("balance_sheet", "quarterly_balance_sheet", "balance_sheet", "No balance sheet data found"),
            ("cashflow", "quarterly_cashflow", "cashflow", "No cash flow data found"),
            ("income_statement", "quarterly_income_stmt", "income_stmt", "No income statement data found"),
        ]

        for tool_name, quarterly_attr, annual_attr, not_found in cases:
            ticker_obj = Mock()
            setattr(ticker_obj, quarterly_attr, quarterly)
            setattr(ticker_obj, annual_attr, annual)
            fn = getattr(y_finance, f"get_{tool_name}")

            with self.subTest(tool=tool_name), patch.object(y_finance.yf, "Ticker", return_value=ticker_obj):
                out_regular_deadline = fn("AAPL", "quarterly", "2024-08-14")
                out_regular_after = fn("AAPL", "quarterly", "2024-08-15")
                out_q4_deadline = fn("AAPL", "quarterly", "2024-12-30")
                out_q4_after = fn("AAPL", "quarterly", "2024-12-31")

            self.assertIn(not_found, out_regular_deadline)
            self.assertIn("2024-06-30", out_regular_after)
            self.assertNotIn("2024-09-30", out_regular_after)
            self.assertIn("2024-06-30", out_q4_deadline)
            self.assertNotIn("2024-09-30", out_q4_deadline)
            self.assertIn("2024-09-30", out_q4_after)

    def test_alpha_vantage_filters_run_through_make_api_request_raw_text_path(self):
        import json
        import os

        statement_payload = {
            "annualReports": [{"fiscalDateEnding": "2023-12-31", "tag": "annual"}],
            "quarterlyReports": [{"fiscalDateEnding": "2024-03-31", "tag": "q1"}],
        }
        insider_payload = {
            "data": [
                {"transaction_date": "2024-07-03", "shares": "1"},
                {"transaction_date": "2024-07-05", "shares": "2"},
            ]
        }

        def fake_get(_url, params):
            payload = (
                statement_payload
                if params.get("function") == "BALANCE_SHEET"
                else insider_payload
            )
            response = Mock()
            response.text = json.dumps(payload)
            response.raise_for_status = Mock()
            return response

        with patch.dict(os.environ, {"ALPHA_VANTAGE_API_KEY": "test-key"}, clear=False), patch(
            "tradingagents.dataflows.alpha_vantage_common.requests.get",
            side_effect=fake_get,
        ) as mock_get:
            statement_out = alpha_vantage_fundamentals.get_balance_sheet("AAPL", "quarterly", "2024-03-31")
            insider_out = alpha_vantage_news.get_insider_transactions("AAPL", "2024-07-09")

        statement = json.loads(statement_out)
        insider = json.loads(insider_out)
        self.assertEqual(statement["annualReports"], [])
        self.assertEqual(statement["quarterlyReports"], [])
        self.assertEqual([row["transaction_date"] for row in insider["data"]], ["2024-07-03"])
        self.assertEqual(mock_get.call_count, 2)


class MutationHardeningFollowUpTests(unittest.TestCase):
    """Kill mutants that survived the #62 suite (sort removal, unparseable insider dates)."""

    def setUp(self):
        # A garbage first row makes pandas fall back to per-element dateutil parsing
        # and emit "Could not infer format"; that fallback is exactly what we exercise.
        self._warnings = warnings.catch_warnings()
        self._warnings.__enter__()
        warnings.filterwarnings("ignore", message="Could not infer format", category=UserWarning)

    def tearDown(self):
        self._warnings.__exit__(None, None, None)

    # Fiscal-year change (Sep-30 FYE -> Dec-31 FYE). The most recent annual period
    # (2023-12-31) must win regardless of the order the vendor sends columns in.
    _FYE_CHANGE_ORDERS = {
        "oldest_to_newest": ["2021-09-30", "2022-09-30", "2023-12-31"],
        "newest_to_oldest": ["2023-12-31", "2022-09-30", "2021-09-30"],
        "unordered": ["2022-09-30", "2023-12-31", "2021-09-30"],
    }

    def test_fye_inference_is_order_invariant_across_fiscal_year_change(self):
        from tradingagents.dataflows.stockstats_utils import get_fiscal_year_end_month_day

        for label, annual_ends in self._FYE_CHANGE_ORDERS.items():
            with self.subTest(order=label):
                self.assertEqual(get_fiscal_year_end_month_day(annual_ends), (12, 31))
                self.assertEqual(
                    get_fiscal_year_end_month_day(pd.to_datetime(annual_ends)), (12, 31)
                )

    def test_q4_rule_follows_most_recent_fye_regardless_of_vendor_order(self):
        from tradingagents.dataflows.stockstats_utils import filter_financials_by_date

        # 2023-12-31 is Q4 under the current (Dec-31) FYE -> +3 months -> 2024-03-31.
        # Under the stale Sep-30 FYE it would be a regular quarter -> +45d -> 2024-02-14.
        data = pd.DataFrame([[1.0]], columns=["2023-12-31"])
        for label, annual_ends in self._FYE_CHANGE_ORDERS.items():
            kw = dict(freq="quarterly", annual_period_ends=annual_ends)
            with self.subTest(order=label):
                self.assertEqual(list(filter_financials_by_date(data, "2024-02-20", **kw).columns), [])
                self.assertEqual(list(filter_financials_by_date(data, "2024-03-31", **kw).columns), [])
                self.assertEqual(
                    list(filter_financials_by_date(data, "2024-04-01", **kw).columns), ["2023-12-31"]
                )

    def test_yfinance_statement_q4_rule_with_newest_first_annual_columns_after_fye_change(self):
        # yfinance returns annual columns newest-first.
        quarterly = pd.DataFrame([[1.0]], columns=["2023-12-31"])
        annual = pd.DataFrame([[1.0, 2.0, 3.0]], columns=["2023-12-31", "2022-09-30", "2021-09-30"])
        ticker_obj = Mock()
        ticker_obj.quarterly_income_stmt = quarterly
        ticker_obj.income_stmt = annual

        with patch.object(y_finance.yf, "Ticker", return_value=ticker_obj):
            out_before = y_finance.get_income_statement("AAPL", "quarterly", "2024-02-20")
            out_after = y_finance.get_income_statement("AAPL", "quarterly", "2024-04-01")

        self.assertIn("No income statement data found", out_before)
        self.assertIn("2023-12-31", out_after)

    def test_yfinance_insider_rows_with_unparseable_date_are_dropped(self):
        insider_data = pd.DataFrame(
            [
                {"Start Date": "not-a-date", "Shares": 111},
                {"Start Date": "2024-07-03", "Shares": 10},
                {"Start Date": "", "Shares": 222},
                {"Start Date": None, "Shares": 333},
            ]
        )
        ticker_obj = Mock()
        ticker_obj.insider_transactions = insider_data

        with patch.object(y_finance.yf, "Ticker", return_value=ticker_obj):
            out = y_finance.get_insider_transactions("AAPL", "2024-07-09")

        self.assertIn("2024-07-03", out)
        self.assertNotIn("not-a-date", out)
        for garbage_shares in ("111", "222", "333"):
            self.assertNotIn(garbage_shares, out)

    def test_insider_filter_drops_unparseable_dates_directly(self):
        from tradingagents.dataflows.stockstats_utils import filter_insider_transactions_by_date

        data = pd.DataFrame(
            [
                {"Start Date": "garbage", "Shares": 1},
                {"Start Date": "2024-07-03", "Shares": 2},
            ]
        )
        out = filter_insider_transactions_by_date(data, "2024-07-09")
        self.assertEqual(list(out["Shares"]), [2])

        only_garbage = pd.DataFrame([{"Start Date": "garbage", "Shares": 1}])
        self.assertEqual(len(filter_insider_transactions_by_date(only_garbage, "2099-01-01")), 0)

    def test_alpha_vantage_insider_rows_with_unparseable_date_are_dropped(self):
        import json

        payload = {
            "data": [
                {"transaction_date": "not-a-date", "shares": "111"},
                {"transaction_date": "2024-07-03", "shares": "10"},
                {"transaction_date": "", "shares": "222"},
                {"transaction_date": None, "shares": "333"},
                {"shares": "444"},
            ]
        }
        with patch.object(alpha_vantage_news, "_make_api_request", return_value=copy.deepcopy(payload)):
            out_dict = alpha_vantage_news.get_insider_transactions("AAPL", "2099-01-01")
        with patch.object(alpha_vantage_news, "_make_api_request", return_value=json.dumps(payload)):
            out_text = json.loads(alpha_vantage_news.get_insider_transactions("AAPL", "2099-01-01"))

        for out in (out_dict, out_text):
            self.assertEqual([row["shares"] for row in out["data"]], ["10"])


if __name__ == "__main__":
    unittest.main()
