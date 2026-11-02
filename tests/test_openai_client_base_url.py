import unittest
from unittest.mock import patch

from tradingagents.llm_clients.openai_client import OpenAIClient


class OpenAIClientBaseUrlTests(unittest.TestCase):
    """config backend_url is passed as base_url and must reach ChatOpenAI.

    For providers in _PROVIDER_CONFIG, get_llm used to replace that URL with
    the built-in default. The CLI stores a different host for Qwen
    (dashscope.aliyuncs.com) and GLM (open.bigmodel.cn) than those defaults
    (dashscope-intl, api.z.ai), so the selected endpoint never left the client.

    An empty string is not a host. It falls back the same way None does:
    a known provider keeps its _PROVIDER_CONFIG default, and native OpenAI
    omits base_url. A check that only rejects None forwards "" as the host.
    """

    @patch("tradingagents.llm_clients.openai_client.NormalizedChatOpenAI")
    def test_explicit_base_url_overrides_provider_default(self, mock_chat):
        cases = [
            (
                "qwen",
                "qwen-plus",
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
                None,
            ),
            (
                "glm",
                "glm-4.7",
                "https://open.bigmodel.cn/api/paas/v4/",
                None,
            ),
            (
                "ollama",
                "qwen3:latest",
                "http://127.0.0.1:11434/v1",
                "ollama",
            ),
        ]
        for provider, model, base_url, api_key in cases:
            with self.subTest(provider=provider):
                mock_chat.reset_mock()
                OpenAIClient(model, base_url, provider=provider).get_llm()
                kwargs = mock_chat.call_args.kwargs
                self.assertEqual(kwargs["base_url"], base_url)
                self.assertEqual(kwargs["model"], model)
                if api_key is not None:
                    self.assertEqual(kwargs["api_key"], api_key)

    @patch("tradingagents.llm_clients.openai_client.NormalizedChatOpenAI")
    def test_missing_base_url_keeps_provider_default(self, mock_chat):
        OpenAIClient("qwen-plus", None, provider="qwen").get_llm()
        self.assertEqual(
            mock_chat.call_args.kwargs["base_url"],
            "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        )

    @patch("tradingagents.llm_clients.openai_client.NormalizedChatOpenAI")
    def test_empty_base_url_falls_back_like_none(self, mock_chat):
        """Empty string uses the provider default, the same host None does.

        Known providers use ``self.base_url or default_base_url``. Native
        OpenAI uses ``elif self.base_url``. Replacing either with
        ``is not None`` forwards "" and fails this test.
        """
        qwen_default = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
        for base_url in ("", None):
            with self.subTest(provider="qwen", base_url=base_url):
                mock_chat.reset_mock()
                OpenAIClient("qwen-plus", base_url, provider="qwen").get_llm()
                self.assertEqual(
                    mock_chat.call_args.kwargs.get("base_url"),
                    qwen_default,
                    msg=f"base_url={base_url!r} must fall back to the provider default",
                )

            with self.subTest(provider="openai", base_url=base_url):
                mock_chat.reset_mock()
                OpenAIClient("gpt-4.1", base_url, provider="openai").get_llm()
                self.assertNotIn(
                    "base_url",
                    mock_chat.call_args.kwargs,
                    msg=f"base_url={base_url!r} must not be forwarded as the host",
                )


if __name__ == "__main__":
    unittest.main()
