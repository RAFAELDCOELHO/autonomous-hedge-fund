import sys
import types
import unittest
from unittest.mock import patch

# Stub optional provider SDK imports so this test stays offline and dependency-light.
_ORIGINAL_MODULES = {
    name: sys.modules.get(name)
    for name in ("langchain_openai", "langchain_anthropic", "langchain_google_genai")
}

if "langchain_openai" not in sys.modules:
    langchain_openai = types.ModuleType("langchain_openai")
    langchain_openai.ChatOpenAI = type("ChatOpenAI", (), {})
    langchain_openai.AzureChatOpenAI = type("AzureChatOpenAI", (), {})
    sys.modules["langchain_openai"] = langchain_openai

if "langchain_anthropic" not in sys.modules:
    langchain_anthropic = types.ModuleType("langchain_anthropic")
    langchain_anthropic.ChatAnthropic = type("ChatAnthropic", (), {})
    sys.modules["langchain_anthropic"] = langchain_anthropic

if "langchain_google_genai" not in sys.modules:
    langchain_google = types.ModuleType("langchain_google_genai")
    langchain_google.ChatGoogleGenerativeAI = type("ChatGoogleGenerativeAI", (), {})
    sys.modules["langchain_google_genai"] = langchain_google

from tradingagents.llm_clients.factory import create_llm_client


def tearDownModule():
    for module_name, original in _ORIGINAL_MODULES.items():
        if original is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = original


class CreateLLMClientTests(unittest.TestCase):
    def test_openai_compatible_provider_routes_to_openai_client(self):
        with patch("tradingagents.llm_clients.factory.OpenAIClient") as openai_client:
            sentinel = object()
            openai_client.return_value = sentinel

            out = create_llm_client(
                provider="DeepSeek",
                model="deepseek-chat",
                base_url="https://api.deepseek.com",
                timeout=30,
            )

        self.assertIs(out, sentinel)
        openai_client.assert_called_once_with(
            "deepseek-chat",
            "https://api.deepseek.com",
            provider="deepseek",
            timeout=30,
        )

    def test_anthropic_provider_routes_to_anthropic_client(self):
        with patch("tradingagents.llm_clients.factory.AnthropicClient") as anthropic_client:
            sentinel = object()
            anthropic_client.return_value = sentinel

            out = create_llm_client("anthropic", "claude-sonnet-4-6", max_tokens=4000)

        self.assertIs(out, sentinel)
        anthropic_client.assert_called_once_with(
            "claude-sonnet-4-6",
            None,
            max_tokens=4000,
        )

    def test_google_provider_routes_to_google_client(self):
        with patch("tradingagents.llm_clients.factory.GoogleClient") as google_client:
            sentinel = object()
            google_client.return_value = sentinel

            out = create_llm_client("google", "gemini-3-flash")

        self.assertIs(out, sentinel)
        google_client.assert_called_once_with("gemini-3-flash", None)

    def test_azure_provider_routes_to_azure_client(self):
        with patch("tradingagents.llm_clients.factory.AzureOpenAIClient") as azure_client:
            sentinel = object()
            azure_client.return_value = sentinel

            out = create_llm_client("azure", "gpt-5")

        self.assertIs(out, sentinel)
        azure_client.assert_called_once_with("gpt-5", None)

    def test_unsupported_provider_raises_value_error(self):
        with self.assertRaises(ValueError) as ctx:
            create_llm_client("not-a-provider", "model")

        self.assertIn("Unsupported LLM provider: not-a-provider", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
