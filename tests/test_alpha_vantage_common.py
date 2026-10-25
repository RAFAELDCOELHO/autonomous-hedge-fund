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

    def test_filter_csv_by_date_range_ends_with_trailing_newline(self):
        """A parsed body ends with ``\\n`` with or without a final input newline.

        LF input that ends with a newline, LF input that does not, and CRLF
        input that does not all return the same LF text, including the final
        newline.
        """
        expected = "timestamp,open,volume\n2026-01-02,1.50,NA\n"
        lf_with_final = (
            "timestamp,open,volume\n"
            "2026-01-02,1.50,NA\n"
            "2026-01-03,2.00,4\n"
        )
        lf_without_final = (
            "timestamp,open,volume\n"
            "2026-01-02,1.50,NA\n"
            "2026-01-03,2.00,4"
        )
        crlf_without_final = (
            "timestamp,open,volume\r\n"
            "2026-01-02,1.50,NA\r\n"
            "2026-01-03,2.00,4"
        )

        cases = (
            ("lf_with_final", lf_with_final),
            ("lf_without_final", lf_without_final),
            ("crlf_without_final", crlf_without_final),
        )
        for label, csv_data in cases:
            with self.subTest(input=label):
                filtered = _filter_csv_by_date_range(
                    csv_data, "2026-01-02", "2026-01-02"
                )
                self.assertEqual(filtered, expected)

    def test_filter_csv_by_date_range_preserves_break_inside_quoted_cell(self):
        """Record separators become LF. A break inside a quoted cell stays.

        CRLF between records is rewritten to LF. A CRLF or LF inside quotes
        is copied through. LF-separated records that hold a CRLF inside
        quotes and have no final newline still end with a trailing LF.
        """
        embedded_crlf = (
            "timestamp,note\r\n"
            '2026-01-02,"hello\r\nworld"\r\n'
            "2026-01-03,plain\r\n"
        )
        embedded_lf = (
            "timestamp,note\n"
            '2026-01-02,"hello\nworld"\n'
            "2026-01-03,plain\n"
        )
        lf_records_embedded_crlf_no_final = (
            "timestamp,note\n"
            '2026-01-02,"hello\r\nworld"\n'
            "2026-01-03,plain"
        )

        filtered_crlf = _filter_csv_by_date_range(
            embedded_crlf, "2026-01-02", "2026-01-02"
        )
        filtered_lf = _filter_csv_by_date_range(
            embedded_lf, "2026-01-02", "2026-01-02"
        )
        filtered_lf_records_embedded_crlf_no_final = _filter_csv_by_date_range(
            lf_records_embedded_crlf_no_final, "2026-01-02", "2026-01-02"
        )

        self.assertEqual(
            filtered_crlf,
            'timestamp,note\n2026-01-02,"hello\r\nworld"\n',
        )
        self.assertEqual(
            filtered_lf,
            'timestamp,note\n2026-01-02,"hello\nworld"\n',
        )
        self.assertEqual(
            filtered_lf_records_embedded_crlf_no_final,
            'timestamp,note\n2026-01-02,"hello\r\nworld"\n',
        )


if __name__ == "__main__":
    unittest.main()
