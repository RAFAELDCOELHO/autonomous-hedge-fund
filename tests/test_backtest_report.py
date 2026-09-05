"""Offline unit tests for backtest report formatting helpers."""

import unittest

import pandas as pd

from tradingagents.backtest.report import _num, _pct, format_table_markdown


class BacktestReportHelperTests(unittest.TestCase):
    def test_pct_formats_fraction_and_handles_none(self):
        self.assertEqual(_pct(None), "—")
        self.assertEqual(_pct(0.1234), "12.34")

    def test_num_formats_numeric_special_cases(self):
        self.assertEqual(_num(None), "—")
        self.assertEqual(_num(float("inf")), "inf")
        self.assertEqual(_num(1.23456), "1.235")

    def test_format_table_markdown_outputs_expected_grid(self):
        df = pd.DataFrame(
            [
                {"Strategy": "A", "CR (%)": "10.00"},
                {"Strategy": "B", "CR (%)": "12.50"},
            ]
        )
        table = format_table_markdown(df)
        self.assertEqual(
            table,
            "\n".join(
                [
                    "| Strategy | CR (%) |",
                    "| --- | --- |",
                    "| A | 10.00 |",
                    "| B | 12.50 |",
                ]
            ),
        )


if __name__ == "__main__":
    unittest.main()
