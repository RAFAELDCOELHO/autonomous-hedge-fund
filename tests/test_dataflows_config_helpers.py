import unittest
from copy import deepcopy

from tradingagents.dataflows import config as config_module


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
