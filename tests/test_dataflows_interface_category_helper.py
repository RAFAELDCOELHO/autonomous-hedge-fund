"""Offline unit tests for dataflow interface category helper."""

import importlib
import sys
import types
import unittest
from unittest.mock import patch


def _noop(*args, **kwargs):
    return None


def _stub_module(name: str, **attrs) -> types.ModuleType:
    mod = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(mod, key, value)
    return mod


class InterfaceCategoryHelperTests(unittest.TestCase):
    def _import_interface_module(self):
        av_rate_limit_error = type("AlphaVantageRateLimitError", (Exception,), {})
        stubs = {
            "tradingagents.dataflows.y_finance": _stub_module(
                "tradingagents.dataflows.y_finance",
                get_YFin_data_online=_noop,
                get_stock_stats_indicators_window=_noop,
                get_fundamentals=_noop,
                get_balance_sheet=_noop,
                get_cashflow=_noop,
                get_income_statement=_noop,
                get_insider_transactions=_noop,
            ),
            "tradingagents.dataflows.yfinance_news": _stub_module(
                "tradingagents.dataflows.yfinance_news",
                get_news_yfinance=_noop,
                get_global_news_yfinance=_noop,
            ),
            "tradingagents.dataflows.alpha_vantage": _stub_module(
                "tradingagents.dataflows.alpha_vantage",
                get_stock=_noop,
                get_indicator=_noop,
                get_fundamentals=_noop,
                get_balance_sheet=_noop,
                get_cashflow=_noop,
                get_income_statement=_noop,
                get_insider_transactions=_noop,
                get_news=_noop,
                get_global_news=_noop,
            ),
            "tradingagents.dataflows.alpha_vantage_common": _stub_module(
                "tradingagents.dataflows.alpha_vantage_common",
                AlphaVantageRateLimitError=av_rate_limit_error,
            ),
            "tradingagents.dataflows.config": _stub_module(
                "tradingagents.dataflows.config",
                get_config=lambda: {},
            ),
        }

        with patch.dict(sys.modules, stubs):
            sys.modules.pop("tradingagents.dataflows.interface", None)
            return importlib.import_module("tradingagents.dataflows.interface")

    def test_returns_expected_category_for_known_methods(self):
        mod = self._import_interface_module()
        self.assertEqual(mod.get_category_for_method("get_stock_data"), "core_stock_apis")
        self.assertEqual(mod.get_category_for_method("get_indicators"), "technical_indicators")
        self.assertEqual(mod.get_category_for_method("get_fundamentals"), "fundamental_data")
        self.assertEqual(mod.get_category_for_method("get_news"), "news_data")

    def test_all_declared_tools_map_back_to_their_category(self):
        mod = self._import_interface_module()
        for category, info in mod.TOOLS_CATEGORIES.items():
            for method in info["tools"]:
                with self.subTest(category=category, method=method):
                    self.assertEqual(mod.get_category_for_method(method), category)

    def test_raises_value_error_for_unknown_method(self):
        mod = self._import_interface_module()
        with self.assertRaises(ValueError) as ctx:
            mod.get_category_for_method("totally_unknown_helper")
        self.assertIn("totally_unknown_helper", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
