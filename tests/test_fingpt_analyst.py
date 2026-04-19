"""Tests for FinGPT-inspired sentiment classifier.

The anthropic SDK is mocked in every test; no real API calls.
"""

import os
import unittest
from unittest.mock import MagicMock, patch


def _fake_response(text: str):
    block = MagicMock()
    block.text = text
    resp = MagicMock()
    resp.content = [block]
    return resp


def _fake_client(responses):
    """Build a mock anthropic client whose messages.create returns the
    given sequence of response texts."""
    client = MagicMock()
    client.messages.create.side_effect = [_fake_response(r) for r in responses]
    return client


class FingptSentimentTests(unittest.TestCase):
    def setUp(self):
        from tradingagents.dataflows import fingpt_analyst
        self.mod = fingpt_analyst
        os.environ["ANTHROPIC_API_KEY"] = "test-key"

    def tearDown(self):
        os.environ.pop("ANTHROPIC_API_KEY", None)

    def test_returns_none_when_api_key_missing(self):
        os.environ.pop("ANTHROPIC_API_KEY", None)
        out = self.mod.get_fingpt_sentiment(["some news"], "AAPL", "2024-05-10")
        self.assertIsNone(out)

    def test_returns_none_when_anthropic_import_fails(self):
        with patch.object(self.mod, "_get_anthropic_client", return_value=None):
            out = self.mod.get_fingpt_sentiment(["n1"], "AAPL", "2024-05-10")
        self.assertIsNone(out)

    def test_returns_none_when_texts_empty(self):
        out = self.mod.get_fingpt_sentiment([], "AAPL", "2024-05-10")
        self.assertIsNone(out)

    def test_parses_positive_negative_neutral(self):
        client = _fake_client(["Positive", "negative", "neutral"])
        with patch.object(self.mod, "_get_anthropic_client", return_value=client):
            out = self.mod.get_fingpt_sentiment(
                ["a", "b", "c"], "AAPL", "2024-05-10"
            )
        self.assertEqual(out["scores"], [1, -1, 0])
        self.assertEqual(out["n"], 3)
        self.assertEqual(out["n_failed"], 0)
        self.assertAlmostEqual(out["score"], 0.0)
        self.assertEqual(out["label"], "neutral")

    def test_aggregate_label_positive(self):
        client = _fake_client(["positive"] * 3)
        with patch.object(self.mod, "_get_anthropic_client", return_value=client):
            out = self.mod.get_fingpt_sentiment(["a", "b", "c"], "AAPL", "2024-05-10")
        self.assertEqual(out["label"], "positive")
        self.assertEqual(out["score"], 1.0)

    def test_aggregate_label_negative(self):
        client = _fake_client(["negative", "negative", "neutral"])
        with patch.object(self.mod, "_get_anthropic_client", return_value=client):
            out = self.mod.get_fingpt_sentiment(["a", "b", "c"], "AAPL", "2024-05-10")
        self.assertEqual(out["label"], "negative")
        self.assertLess(out["score"], -0.15)

    def test_label_threshold_neutral(self):
        # 1 positive, 9 neutrals → avg = 0.1, below +0.15
        responses = ["positive"] + ["neutral"] * 9
        client = _fake_client(responses)
        with patch.object(self.mod, "_get_anthropic_client", return_value=client):
            out = self.mod.get_fingpt_sentiment(["x"] * 10, "AAPL", "2024-05-10")
        self.assertEqual(out["label"], "neutral")
        self.assertAlmostEqual(out["score"], 0.1, places=4)

    def test_ambiguous_response_is_skipped(self):
        client = _fake_client(["positive", "I cannot determine", "negative"])
        with patch.object(self.mod, "_get_anthropic_client", return_value=client):
            out = self.mod.get_fingpt_sentiment(["a", "b", "c"], "AAPL", "2024-05-10")
        self.assertEqual(out["n"], 2)
        self.assertEqual(out["n_failed"], 1)
        self.assertEqual(out["scores"], [1, -1])

    def test_all_responses_fail_returns_none(self):
        client = _fake_client(["???", "no idea", "maybe"])
        with patch.object(self.mod, "_get_anthropic_client", return_value=client):
            out = self.mod.get_fingpt_sentiment(["a", "b", "c"], "AAPL", "2024-05-10")
        self.assertIsNone(out)

    def test_response_parsing_case_insensitive(self):
        client = _fake_client(["POSITIVE", "Positive", "positive"])
        with patch.object(self.mod, "_get_anthropic_client", return_value=client):
            out = self.mod.get_fingpt_sentiment(["a", "b", "c"], "AAPL", "2024-05-10")
        self.assertEqual(out["scores"], [1, 1, 1])

    def test_prompt_matches_fingpt_v3_template(self):
        client = _fake_client(["positive"])
        with patch.object(self.mod, "_get_anthropic_client", return_value=client):
            self.mod.get_fingpt_sentiment(["breaking news"], "AAPL", "2024-05-10")
        _, kwargs = client.messages.create.call_args
        content = kwargs["messages"][0]["content"]
        self.assertIn("Instruction: What is the sentiment of this news?", content)
        self.assertIn("{negative/neutral/positive}", content)
        self.assertIn("Input: breaking news", content)
        self.assertTrue(content.rstrip().endswith("Answer:"))

    def test_uses_haiku_model(self):
        client = _fake_client(["neutral"])
        with patch.object(self.mod, "_get_anthropic_client", return_value=client):
            out = self.mod.get_fingpt_sentiment(["x"], "AAPL", "2024-05-10")
        self.assertEqual(out["model"], "claude-haiku-4-5-20251001")
        _, kwargs = client.messages.create.call_args
        self.assertEqual(kwargs["model"], "claude-haiku-4-5-20251001")

    def test_api_exception_returns_none_when_all_fail(self):
        client = MagicMock()
        client.messages.create.side_effect = RuntimeError("network down")
        with patch.object(self.mod, "_get_anthropic_client", return_value=client):
            out = self.mod.get_fingpt_sentiment(["a", "b"], "AAPL", "2024-05-10")
        self.assertIsNone(out)


if __name__ == "__main__":
    unittest.main()
