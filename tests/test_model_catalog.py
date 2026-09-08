import importlib.util
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MODULE = REPO / "tradingagents" / "llm_clients" / "model_catalog.py"


def _load_model_catalog():
    spec = importlib.util.spec_from_file_location("model_catalog", MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ModelCatalogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = _load_model_catalog()

    def test_get_model_options_is_case_insensitive(self):
        lower = self.mod.get_model_options("openai", "quick")
        mixed = self.mod.get_model_options("OpenAI", "quick")
        self.assertEqual(mixed, lower)

    def test_custom_option_exists_for_configurable_providers(self):
        deepseek_quick = self.mod.get_model_options("deepseek", "quick")
        qwen_deep = self.mod.get_model_options("qwen", "deep")

        self.assertIn(("Custom model ID", "custom"), deepseek_quick)
        self.assertIn(("Custom model ID", "custom"), qwen_deep)

    def test_get_known_models_returns_unique_sorted_values(self):
        known = self.mod.get_known_models()
        self.assertIn("openai", known)
        openai_models = known["openai"]

        self.assertTrue(openai_models)
        self.assertEqual(openai_models, sorted(openai_models))
        self.assertEqual(len(openai_models), len(set(openai_models)))


if __name__ == "__main__":
    unittest.main()
