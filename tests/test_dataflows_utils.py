"""Offline tests for small helpers in tradingagents.dataflows.utils."""

from __future__ import annotations

import importlib.util
import unittest
from datetime import datetime
import sys
import types
from pathlib import Path
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
UTILS = REPO / "tradingagents" / "dataflows" / "utils.py"


def _load_get_next_weekday():
    pandas_stub = types.ModuleType("pandas")
    pandas_stub.DataFrame = object
    with patch.dict(sys.modules, {"pandas": pandas_stub}):
        spec = importlib.util.spec_from_file_location("dataflows_utils", UTILS)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    return mod.get_next_weekday


class DataflowsUtilsTests(unittest.TestCase):
    def setUp(self):
        self.get_next_weekday = _load_get_next_weekday()

    def test_get_next_weekday_keeps_weekday_string_unchanged(self):
        out = self.get_next_weekday("2026-09-04")  # Friday
        self.assertEqual(out, datetime(2026, 9, 4))

    def test_get_next_weekday_rolls_saturday_to_monday(self):
        out = self.get_next_weekday("2026-09-05")  # Saturday
        self.assertEqual(out, datetime(2026, 9, 7))

    def test_get_next_weekday_rolls_sunday_datetime_to_monday(self):
        out = self.get_next_weekday(datetime(2026, 9, 6))  # Sunday
        self.assertEqual(out, datetime(2026, 9, 7))


if __name__ == "__main__":
    unittest.main()
