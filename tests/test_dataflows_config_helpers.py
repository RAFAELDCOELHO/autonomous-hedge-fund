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


if __name__ == "__main__":
    unittest.main()
