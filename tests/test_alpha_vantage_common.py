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

    def test_filter_csv_by_date_range_fails_closed_when_parsing_fails(self):
        csv_data = "timestamp,open\n2026-01-01,10\n"
        with patch(
            "tradingagents.dataflows.alpha_vantage_common.pd.read_csv",
            side_effect=Exception("synthetic parser failure"),
        ):
            filtered = _filter_csv_by_date_range(
                csv_data, "2026-01-01", "2026-01-31"
            )

        self.assertEqual(filtered, "timestamp,open\n")
        self.assertNotIn("2026-01-01,10", filtered)

    def test_filter_csv_by_date_range_preserves_cell_text(self):
        """Kept cells are the original bytes.

        ``1.50`` is not shortened to ``1.5``. ``NA`` is not blanked.
        """
        csv_data = (
            "timestamp,open,volume\n"
            "2026-01-02,1.50,NA\n"
            "2026-01-03,2.00,4\n"
        )

        filtered = _filter_csv_by_date_range(csv_data, "2026-01-02", "2026-01-02")

        self.assertEqual(filtered, "timestamp,open,volume\n2026-01-02,1.50,NA\n")

    def test_filter_csv_by_date_range_returns_crlf_input_as_lf(self):
        """A CRLF body comes back with LF separators. Cell text is not rewritten."""
        csv_data = (
            "timestamp,open,volume\r\n"
            "2026-01-02,1.50,NA\r\n"
            "2026-01-03,2.00,4\r\n"
        )

        filtered = _filter_csv_by_date_range(csv_data, "2026-01-02", "2026-01-02")

        self.assertEqual(filtered, "timestamp,open,volume\n2026-01-02,1.50,NA\n")
        self.assertNotIn("\r", filtered)


if __name__ == "__main__":
    unittest.main()
