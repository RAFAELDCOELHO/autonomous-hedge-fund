"""Unit tests for ``create_llm_client`` provider routing.

Offline by design: every provider client class is patched, so no SDK is imported
for real and no network call is made. The optional ``langchain_*`` packages are
stubbed so the factory module imports even when they are not installed.
"""

import sys
import types
import unittest
from unittest.mock import patch

_SDK_STUBS = {
    "langchain_openai": ("ChatOpenAI", "AzureChatOpenAI"),
    "langchain_anthropic": ("ChatAnthropic",),
    "langchain_google_genai": ("ChatGoogleGenerativeAI",),
}
_ORIGINAL_MODULES = {name: sys.modules.get(name) for name in _SDK_STUBS}

for _name, _classes in _SDK_STUBS.items():
    if _name not in sys.modules:
        _stub = types.ModuleType(_name)
        for _cls in _classes:
            setattr(_stub, _cls, type(_cls, (), {}))
        sys.modules[_name] = _stub

from tradingagents.llm_clients.factory import _OPENAI_COMPATIBLE, create_llm_client


def tearDownModule():
    for module_name, original in _ORIGINAL_MODULES.items():
        if original is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = original


class CreateLLMClientTests(unittest.TestCase):
    """Each provider name must reach exactly one client class with args passed through."""

    def _patch_client(self, class_name):
        patcher = patch(f"tradingagents.llm_clients.factory.{class_name}")
        client_cls = patcher.start()
        self.addCleanup(patcher.stop)
        client_cls.return_value = object()
        return client_cls

    def test_openai_compatible_provider_forwards_args_and_normalized_provider(self):
        openai_client = self._patch_client("OpenAIClient")

        out = create_llm_client(
            provider="DeepSeek",
            model="deepseek-chat",
            base_url="https://api.deepseek.com",
            timeout=30,
        )

        self.assertIs(out, openai_client.return_value)
        openai_client.assert_called_once_with(
            "deepseek-chat",
            "https://api.deepseek.com",
            provider="deepseek",
            timeout=30,
        )

    def test_every_openai_compatible_provider_routes_to_openai_client(self):
        openai_client = self._patch_client("OpenAIClient")

        for provider in _OPENAI_COMPATIBLE:
            with self.subTest(provider=provider):
                openai_client.reset_mock()
                create_llm_client(provider, "m")
                openai_client.assert_called_once_with("m", None, provider=provider)

    def test_anthropic_provider_routes_to_anthropic_client(self):
        anthropic_client = self._patch_client("AnthropicClient")

        out = create_llm_client("anthropic", "claude-sonnet-4-6", max_tokens=4000)

        self.assertIs(out, anthropic_client.return_value)
        anthropic_client.assert_called_once_with("claude-sonnet-4-6", None, max_tokens=4000)

    def test_google_provider_routes_to_google_client(self):
        google_client = self._patch_client("GoogleClient")

        out = create_llm_client("google", "gemini-3-flash")

        self.assertIs(out, google_client.return_value)
        google_client.assert_called_once_with("gemini-3-flash", None)

    def test_azure_provider_routes_to_azure_client(self):
        azure_client = self._patch_client("AzureOpenAIClient")

        out = create_llm_client("AZURE", "gpt-5")

        self.assertIs(out, azure_client.return_value)
        azure_client.assert_called_once_with("gpt-5", None)

    def test_unsupported_provider_raises_value_error_naming_original_input(self):
        with self.assertRaises(ValueError) as ctx:
            create_llm_client("Not-A-Provider", "model")

        self.assertEqual(str(ctx.exception), "Unsupported LLM provider: Not-A-Provider")


if __name__ == "__main__":
    unittest.main()
