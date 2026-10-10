"""Offline tests for stockstats dataframe normalization helpers."""

from __future__ import annotations

import unittest

import pandas as pd

from tradingagents.dataflows.stockstats_utils import (
    _clean_dataframe,
    filter_financials_by_date,
    filter_insider_transactions_by_date,
    get_fiscal_year_end_month_day,
)


def test_price_cache_key_records_auto_adjust(monkeypatch, tmp_path):
    """The cache file name carries -adj, so an old unadjusted cache is never reused."""
    from _pr7_fakes import install_fake_yahoo

    from tradingagents.dataflows.stockstats_utils import load_ohlcv

    install_fake_yahoo(monkeypatch, tmp_path)
    cache = tmp_path / "ohlcv-cache"
    cache.mkdir()
    today = pd.Timestamp.today()
    span = f"{(today - pd.DateOffset(years=5)):%Y-%m-%d}-{today:%Y-%m-%d}"
    pd.DataFrame({"Date": ["2024-05-14"], "Open": [987654.0], "High": [987654.0], "Low": [987654.0],
                  "Close": [987654.0], "Volume": [1.0]}).to_csv(cache / f"AAPL-YFin-data-{span}.csv", index=False)

    data = load_ohlcv("AAPL", "2024-05-15")

    assert (cache / f"AAPL-YFin-data-{span}-adj.csv").exists()
    assert not (data[["Open", "Close"]] == 987654.0).any().any()
    assert len(data) > 1


class CleanDataFrameTests(unittest.TestCase):
    def test_clean_dataframe_parses_dates_drops_bad_rows_and_fills_prices(self):
        raw = pd.DataFrame(
            {
                "Date": ["2024-01-01", "bad-date", "2024-01-03", "2024-01-04"],
                "Open": ["10", "11", None, "13"],
                "High": ["11", "12", "14", "15"],
                "Low": ["9", "10", "12", "13"],
                "Close": ["10.5", "11.5", "oops", "13.5"],
                "Volume": ["1000", "1100", "1200", None],
            }
        )

        cleaned = _clean_dataframe(raw.copy())

        # bad Date row and non-numeric Close row are removed
        self.assertEqual(len(cleaned), 2)
        self.assertEqual(
            cleaned["Date"].dt.strftime("%Y-%m-%d").tolist(),
            ["2024-01-01", "2024-01-04"],
        )
        self.assertEqual(cleaned["Close"].tolist(), [10.5, 13.5])
        # Volume gap in surviving rows is filled from neighboring rows.
        self.assertEqual(cleaned["Volume"].tolist(), [1000.0, 1000.0])


class FilterFinancialsByDateTests(unittest.TestCase):
    def test_quarterly_boundary_is_strict_and_uses_45_day_lag(self):
        data = pd.DataFrame(
            [[1.0, 2.0, 3.0]],
            columns=["2023-12-31", "2024-03-31", "2024-06-30"],
        )

        out_deadline = filter_financials_by_date(
            data,
            "2024-05-15",
            freq="quarterly",
            annual_period_ends=["2023-12-31"],
        )
        out_after_deadline = filter_financials_by_date(
            data,
            "2024-05-16",
            freq="quarterly",
            annual_period_ends=["2023-12-31"],
        )

        self.assertEqual(list(out_deadline.columns), ["2023-12-31"])
        self.assertEqual(list(out_after_deadline.columns), ["2023-12-31", "2024-03-31"])

    def test_annual_boundary_is_strict_and_uses_three_calendar_months(self):
        data = pd.DataFrame(
            [[1.0, 2.0]],
            columns=["2023-12-31", "2024-12-31"],
        )

        out_deadline = filter_financials_by_date(data, "2024-03-31", freq="annual")
        out_after_deadline = filter_financials_by_date(data, "2024-04-01", freq="annual")

        self.assertEqual(list(out_deadline.columns), [])
        self.assertEqual(list(out_after_deadline.columns), ["2023-12-31"])

    def test_q4_in_quarterly_series_uses_annual_lag(self):
        data = pd.DataFrame(
            [[10.0, 20.0]],
            columns=["2024-06-30", "2024-09-30"],
        )

        out_deadline = filter_financials_by_date(
            data,
            "2024-12-30",
            freq="quarterly",
            annual_period_ends=["2024-09-30"],
        )
        out_after_deadline = filter_financials_by_date(
            data,
            "2024-12-31",
            freq="quarterly",
            annual_period_ends=["2024-09-30"],
        )

        self.assertEqual(list(out_deadline.columns), ["2024-06-30"])
        self.assertEqual(list(out_after_deadline.columns), ["2024-06-30", "2024-09-30"])

    def test_fiscal_year_end_falls_back_to_december_31_when_annual_unavailable(self):
        self.assertEqual(get_fiscal_year_end_month_day([]), (12, 31))
        self.assertEqual(get_fiscal_year_end_month_day(None), (12, 31))


class FilterInsiderTransactionsByDateTests(unittest.TestCase):
    def test_insider_filter_uses_two_us_business_days_with_strict_boundary(self):
        # 2024-07-03 + 2 US business days (Jul-04 holiday) => 2024-07-08
        data = pd.DataFrame(
            [
                {"Start Date": "2024-07-03", "Shares": 10},
                {"Start Date": "2024-07-05", "Shares": 20},  # +2bd => 2024-07-09
            ]
        )

        out_deadline = filter_insider_transactions_by_date(data, "2024-07-08")
        out_after_deadline = filter_insider_transactions_by_date(data, "2024-07-09")

        self.assertEqual(len(out_deadline), 0)
        self.assertEqual(len(out_after_deadline), 1)
        self.assertEqual(out_after_deadline.iloc[0]["Shares"], 10)


if __name__ == "__main__":
    unittest.main()
