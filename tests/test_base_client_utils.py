import unittest

from tradingagents.llm_clients.base_client import BaseLLMClient, normalize_content


class _DummyResponse:
    def __init__(self, content):
        self.content = content


class DummyClient(BaseLLMClient):
    def get_llm(self):
        return object()

    def validate_model(self) -> bool:
        return True


class TestBaseClientUtils(unittest.TestCase):
    def test_normalize_content_extracts_text_blocks_and_strings(self):
        response = _DummyResponse(
            [
                {"type": "reasoning", "summary": "hidden"},
                {"type": "text", "text": "first line"},
                "second line",
                {"type": "tool_use", "name": "ignored"},
                {"type": "text", "text": "third line"},
            ]
        )

        result = normalize_content(response)

        self.assertIs(result, response)
        self.assertEqual(response.content, "first line\nsecond line\nthird line")

    def test_normalize_content_keeps_plain_string_content(self):
        response = _DummyResponse("already normalized")

        result = normalize_content(response)

        self.assertIs(result, response)
        self.assertEqual(response.content, "already normalized")

    def test_get_provider_name_prefers_provider_attribute(self):
        client = DummyClient("gpt-5.4")
        client.provider = "xai"

        self.assertEqual(client.get_provider_name(), "xai")

    def test_get_provider_name_falls_back_to_class_name(self):
        client = DummyClient("gpt-5.4")

        self.assertEqual(client.get_provider_name(), "dummy")


if __name__ == "__main__":
    unittest.main()
