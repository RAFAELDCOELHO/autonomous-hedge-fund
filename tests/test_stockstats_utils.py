"""Offline tests for stockstats dataframe normalization helpers."""

from __future__ import annotations

import unittest

import pandas as pd

from tradingagents.dataflows.stockstats_utils import (
    _clean_dataframe,
    filter_financials_by_date,
)


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
    def test_filter_financials_by_date_keeps_only_columns_up_to_cutoff(self):
        data = pd.DataFrame(
            [[1.0, 2.0, 3.0]],
            columns=["2023-12-31", "2024-03-31", "2024-06-30"],
        )

        out = filter_financials_by_date(data, "2024-04-01")

        self.assertEqual(list(out.columns), ["2023-12-31", "2024-03-31"])
        self.assertEqual(out.iloc[0].tolist(), [1.0, 2.0])


if __name__ == "__main__":
    unittest.main()
