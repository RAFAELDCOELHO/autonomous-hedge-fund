"""Offline tests for date helper utilities in tradingagents.dataflows.utils."""

from datetime import datetime
import unittest

from tradingagents.dataflows.utils import get_next_weekday


class GetNextWeekdayTests(unittest.TestCase):
    def test_weekday_string_returns_same_calendar_date(self):
        monday = get_next_weekday("2026-09-07")
        self.assertEqual(monday, datetime(2026, 9, 7))

    def test_saturday_rolls_forward_to_monday(self):
        saturday_rollover = get_next_weekday("2026-09-05")
        self.assertEqual(saturday_rollover, datetime(2026, 9, 7))

    def test_sunday_rolls_forward_to_monday(self):
        sunday_rollover = get_next_weekday("2026-09-06")
        self.assertEqual(sunday_rollover, datetime(2026, 9, 7))

    def test_datetime_input_is_supported(self):
        friday = datetime(2026, 9, 4)
        self.assertEqual(get_next_weekday(friday), friday)

    def test_invalid_string_format_raises_value_error(self):
        with self.assertRaises(ValueError):
            get_next_weekday("2026/09/07")


if __name__ == "__main__":
    unittest.main()
