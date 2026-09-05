import unittest
from types import SimpleNamespace

from langchain_core.messages import AIMessage

from cli.stats_handler import StatsCallbackHandler


class StatsCallbackHandlerTests(unittest.TestCase):
    def test_llm_and_chat_start_increment_llm_calls(self):
        handler = StatsCallbackHandler()

        handler.on_llm_start(serialized={}, prompts=["hello"])
        handler.on_chat_model_start(serialized={}, messages=[[]])

        self.assertEqual(handler.get_stats()["llm_calls"], 2)

    def test_tool_start_increments_tool_calls(self):
        handler = StatsCallbackHandler()

        handler.on_tool_start(serialized={}, input_str="input")
        handler.on_tool_start(serialized={}, input_str="next")

        self.assertEqual(handler.get_stats()["tool_calls"], 2)

    def test_on_llm_end_accumulates_token_usage_from_ai_message(self):
        handler = StatsCallbackHandler()
        message = AIMessage(
            content="ok",
            usage_metadata={"input_tokens": 7, "output_tokens": 11, "total_tokens": 18},
        )
        generation = SimpleNamespace(message=message)
        response = SimpleNamespace(generations=[[generation]])

        handler.on_llm_end(response)

        stats = handler.get_stats()
        self.assertEqual(stats["tokens_in"], 7)
        self.assertEqual(stats["tokens_out"], 11)

    def test_on_llm_end_ignores_missing_or_malformed_generations(self):
        handler = StatsCallbackHandler()

        handler.on_llm_end(SimpleNamespace(generations=[]))
        handler.on_llm_end(SimpleNamespace(generations=[[]]))
        handler.on_llm_end(SimpleNamespace(generations=None))

        stats = handler.get_stats()
        self.assertEqual(stats["tokens_in"], 0)
        self.assertEqual(stats["tokens_out"], 0)

    def test_on_llm_end_ignores_non_ai_message_usage(self):
        handler = StatsCallbackHandler()
        generation = SimpleNamespace(message=SimpleNamespace(usage_metadata={"input_tokens": 5}))
        response = SimpleNamespace(generations=[[generation]])

        handler.on_llm_end(response)

        stats = handler.get_stats()
        self.assertEqual(stats["tokens_in"], 0)
        self.assertEqual(stats["tokens_out"], 0)


if __name__ == "__main__":
    unittest.main()
