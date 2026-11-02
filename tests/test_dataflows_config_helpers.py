import unittest
from copy import deepcopy

from tradingagents.dataflows import config as config_module
from tradingagents.default_config import DEFAULT_CONFIG


class DataflowsConfigHelpersTests(unittest.TestCase):
    def setUp(self):
        self._original_config = deepcopy(config_module._config)

    def tearDown(self):
        config_module._config = deepcopy(self._original_config)

    def test_get_config_returns_defensive_copy(self):
        first = config_module.get_config()
        first["llm_provider"] = "mutated"
        second = config_module.get_config()
        self.assertNotEqual(first["llm_provider"], second["llm_provider"])
        self.assertEqual(second["llm_provider"], self._original_config["llm_provider"])

    def test_get_config_nested_vendor_maps_are_not_live(self):
        """Editing the returned vendor maps must not change later lookups.

        A shallow copy shares ``data_vendors`` and ``tool_vendors`` with the
        live config (and with DEFAULT_CONFIG). route_to_vendor reads those
        maps on every call, so one in-place edit would retarget the process.
        """
        original_vendor = DEFAULT_CONFIG["data_vendors"]["core_stock_apis"]
        original_tools = dict(DEFAULT_CONFIG["tool_vendors"])
        try:
            first = config_module.get_config()
            first["data_vendors"]["core_stock_apis"] = "alpha_vantage"
            first["tool_vendors"]["get_stock_data"] = "alpha_vantage"

            second = config_module.get_config()
            self.assertEqual(second["data_vendors"]["core_stock_apis"], original_vendor)
            self.assertNotIn("get_stock_data", second["tool_vendors"])
            self.assertEqual(
                DEFAULT_CONFIG["data_vendors"]["core_stock_apis"], original_vendor
            )
            self.assertEqual(DEFAULT_CONFIG["tool_vendors"], original_tools)
            self.assertIsNot(first["data_vendors"], second["data_vendors"])
            self.assertIsNot(first["tool_vendors"], second["tool_vendors"])
        finally:
            DEFAULT_CONFIG["data_vendors"]["core_stock_apis"] = original_vendor
            DEFAULT_CONFIG["tool_vendors"].clear()
            DEFAULT_CONFIG["tool_vendors"].update(original_tools)

    def test_set_config_updates_only_selected_keys(self):
        before = config_module.get_config()
        config_module.set_config({"llm_provider": "openai"})
        after = config_module.get_config()
        self.assertEqual(after["llm_provider"], "openai")
        self.assertEqual(after["deep_think_llm"], before["deep_think_llm"])

    def test_get_config_initializes_when_internal_state_is_none(self):
        config_module._config = None
        loaded = config_module.get_config()
        self.assertIn("llm_provider", loaded)
        self.assertIn("data_vendors", loaded)

    def test_initialize_config_nested_maps_are_independent_of_default(self):
        """Mutating DEFAULT_CONFIG or _config must not change the other.

        initialize_config used to store DEFAULT_CONFIG.copy(). That shares
        data_vendors and tool_vendors, so an in-place edit of either dict
        retargets vendor lookups for the rest of the process.
        """
        vendors, tools = _vendor_snapshot()
        config_module._config = None
        try:
            config_module.initialize_config()
            loaded = config_module.get_config()
            self.assertEqual(loaded["data_vendors"], vendors)
            self.assertEqual(loaded["tool_vendors"], tools)
            self.assertEqual(loaded["llm_provider"], DEFAULT_CONFIG["llm_provider"])
            self.assertIsNot(
                config_module._config["data_vendors"], DEFAULT_CONFIG["data_vendors"]
            )
            self.assertIsNot(
                config_module._config["tool_vendors"], DEFAULT_CONFIG["tool_vendors"]
            )

            config_module._config["data_vendors"]["core_stock_apis"] = "local"
            config_module._config["tool_vendors"]["get_stock_data"] = "local"
            self.assertEqual(DEFAULT_CONFIG["data_vendors"], vendors)
            self.assertEqual(DEFAULT_CONFIG["tool_vendors"], tools)

            DEFAULT_CONFIG["data_vendors"]["technical_indicators"] = "alpha_vantage"
            DEFAULT_CONFIG["tool_vendors"]["get_news"] = "alpha_vantage"
            after = config_module.get_config()
            self.assertEqual(after["data_vendors"]["core_stock_apis"], "local")
            self.assertEqual(after["data_vendors"]["technical_indicators"], vendors["technical_indicators"])
            self.assertEqual(after["tool_vendors"], {"get_stock_data": "local"})
            self.assertEqual(after["llm_provider"], DEFAULT_CONFIG["llm_provider"])
        finally:
            _restore_vendor_snapshot(vendors, tools)

    def test_set_config_nested_maps_are_independent_of_caller(self):
        """The dict passed to set_config must not stay wired to _config.

        set_config used to _config.update(config), which stores the caller's
        data_vendors and tool_vendors objects. Editing either side then
        changes the other.
        """
        caller = {
            "llm_provider": "openai",
            "data_vendors": {
                "core_stock_apis": "yfinance",
                "technical_indicators": "yfinance",
                "fundamental_data": "alpha_vantage",
                "news_data": "yfinance",
            },
            "tool_vendors": {"get_stock_data": "alpha_vantage"},
        }
        before = config_module.get_config()
        config_module.set_config(caller)

        stored = config_module.get_config()
        self.assertEqual(stored["llm_provider"], "openai")
        self.assertEqual(stored["deep_think_llm"], before["deep_think_llm"])
        self.assertEqual(stored["data_vendors"], caller["data_vendors"])
        self.assertEqual(stored["tool_vendors"], {"get_stock_data": "alpha_vantage"})
        self.assertIsNot(config_module._config["data_vendors"], caller["data_vendors"])
        self.assertIsNot(config_module._config["tool_vendors"], caller["tool_vendors"])

        caller["data_vendors"]["core_stock_apis"] = "mutated"
        caller["tool_vendors"]["get_news"] = "mutated"
        after_caller_edit = config_module.get_config()
        self.assertEqual(after_caller_edit["data_vendors"]["core_stock_apis"], "yfinance")
        self.assertEqual(after_caller_edit["data_vendors"]["fundamental_data"], "alpha_vantage")
        self.assertEqual(after_caller_edit["tool_vendors"], {"get_stock_data": "alpha_vantage"})
        self.assertEqual(after_caller_edit["llm_provider"], "openai")

        config_module._config["data_vendors"]["news_data"] = "local"
        config_module._config["tool_vendors"]["get_stock_data"] = "local"
        self.assertEqual(caller["data_vendors"]["news_data"], "yfinance")
        self.assertEqual(caller["tool_vendors"]["get_stock_data"], "alpha_vantage")
        self.assertEqual(caller["tool_vendors"]["get_news"], "mutated")

    def test_set_config_shallow_copy_of_default_does_not_alias(self):
        """A shallow copy of DEFAULT_CONFIG still shares its nested maps.

        Callers such as TradingAgentsGraph build that shallow copy and pass
        it to set_config. The stored maps must be copies, and the scalar
        overrides on the caller must be the values later reads return.
        """
        vendors, tools = _vendor_snapshot()
        caller = DEFAULT_CONFIG.copy()
        caller["llm_provider"] = "openai"
        try:
            config_module.set_config(caller)
            stored = config_module.get_config()
            self.assertEqual(stored["llm_provider"], "openai")
            self.assertEqual(stored["deep_think_llm"], DEFAULT_CONFIG["deep_think_llm"])
            self.assertEqual(stored["data_vendors"], vendors)
            self.assertEqual(stored["tool_vendors"], tools)
            self.assertIsNot(config_module._config["data_vendors"], caller["data_vendors"])
            self.assertIsNot(config_module._config["data_vendors"], DEFAULT_CONFIG["data_vendors"])
            self.assertIsNot(config_module._config["tool_vendors"], caller["tool_vendors"])
            self.assertIsNot(config_module._config["tool_vendors"], DEFAULT_CONFIG["tool_vendors"])

            caller["data_vendors"]["core_stock_apis"] = "from-caller"
            DEFAULT_CONFIG["data_vendors"]["technical_indicators"] = "from-default"
            DEFAULT_CONFIG["tool_vendors"]["get_news"] = "from-default"
            after = config_module.get_config()
            self.assertEqual(after["data_vendors"], vendors)
            self.assertEqual(after["tool_vendors"], tools)
            self.assertEqual(after["llm_provider"], "openai")

            config_module._config["data_vendors"]["news_data"] = "local"
            config_module._config["tool_vendors"]["get_stock_data"] = "local"
            self.assertEqual(caller["data_vendors"]["news_data"], vendors["news_data"])
            self.assertEqual(DEFAULT_CONFIG["data_vendors"]["news_data"], vendors["news_data"])
            self.assertNotIn("get_stock_data", caller["tool_vendors"])
            self.assertNotIn("get_stock_data", DEFAULT_CONFIG["tool_vendors"])
        finally:
            _restore_vendor_snapshot(vendors, tools)

    def test_set_config_from_none_does_not_share_default_nested_maps(self):
        """set_config's own init path must not shallow-copy DEFAULT_CONFIG.

        A partial update leaves data_vendors and tool_vendors in place. Those
        leftover maps used to be DEFAULT_CONFIG's objects.
        """
        vendors, tools = _vendor_snapshot()
        config_module._config = None
        try:
            config_module.set_config({"llm_provider": "openai"})
            stored = config_module.get_config()
            self.assertEqual(stored["llm_provider"], "openai")
            self.assertEqual(stored["deep_think_llm"], DEFAULT_CONFIG["deep_think_llm"])
            self.assertEqual(stored["data_vendors"], vendors)
            self.assertEqual(stored["tool_vendors"], tools)
            self.assertIsNot(
                config_module._config["data_vendors"], DEFAULT_CONFIG["data_vendors"]
            )
            self.assertIsNot(
                config_module._config["tool_vendors"], DEFAULT_CONFIG["tool_vendors"]
            )

            DEFAULT_CONFIG["data_vendors"]["core_stock_apis"] = "alpha_vantage"
            DEFAULT_CONFIG["tool_vendors"]["get_stock_data"] = "alpha_vantage"
            after_default_edit = config_module.get_config()
            self.assertEqual(after_default_edit["data_vendors"], vendors)
            self.assertEqual(after_default_edit["tool_vendors"], tools)
            self.assertEqual(after_default_edit["llm_provider"], "openai")

            config_module._config["data_vendors"]["technical_indicators"] = "local"
            config_module._config["tool_vendors"]["get_news"] = "local"
            self.assertEqual(DEFAULT_CONFIG["data_vendors"]["technical_indicators"], vendors["technical_indicators"])
            self.assertEqual(DEFAULT_CONFIG["tool_vendors"], {"get_stock_data": "alpha_vantage"})
        finally:
            _restore_vendor_snapshot(vendors, tools)


def _vendor_snapshot():
    return dict(DEFAULT_CONFIG["data_vendors"]), dict(DEFAULT_CONFIG["tool_vendors"])


def _restore_vendor_snapshot(vendors, tools):
    DEFAULT_CONFIG["data_vendors"].clear()
    DEFAULT_CONFIG["data_vendors"].update(vendors)
    DEFAULT_CONFIG["tool_vendors"].clear()
    DEFAULT_CONFIG["tool_vendors"].update(tools)


if __name__ == "__main__":
    unittest.main()
