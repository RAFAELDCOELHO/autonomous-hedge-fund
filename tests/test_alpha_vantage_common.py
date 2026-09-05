import unittest
from unittest.mock import patch

from tradingagents.dataflows.alpha_vantage_common import (
    _filter_csv_by_date_range,
    format_datetime_for_api,
)


class AlphaVantageCommonTests(unittest.TestCase):
    def test_format_datetime_for_api_normalizes_supported_string_inputs(self):
        self.assertEqual(format_datetime_for_api("2026-01-31"), "20260131T0000")
        self.assertEqual(format_datetime_for_api("2026-01-31 14:45"), "20260131T1445")
        self.assertEqual(format_datetime_for_api("20260131T1445"), "20260131T1445")

    def test_filter_csv_by_date_range_keeps_only_inclusive_window(self):
        csv_data = (
            "timestamp,open,close\n"
            "2026-01-01,10,11\n"
            "2026-01-02,11,12\n"
            "2026-01-03,12,13\n"
        )

        filtered = _filter_csv_by_date_range(csv_data, "2026-01-02", "2026-01-03")

        self.assertEqual(
            filtered.splitlines(),
            ["timestamp,open,close", "2026-01-02,11,12", "2026-01-03,12,13"],
        )

    def test_filter_csv_by_date_range_returns_original_when_parsing_fails(self):
        csv_data = "timestamp,open\n2026-01-01,10\n"
        with patch(
            "tradingagents.dataflows.alpha_vantage_common.pd.read_csv",
            side_effect=Exception("synthetic parser failure"),
        ):
            self.assertEqual(
                _filter_csv_by_date_range(csv_data, "2026-01-01", "2026-01-31"),
                csv_data,
            )


if __name__ == "__main__":
    unittest.main()
