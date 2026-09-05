"""Offline unit tests for FinGPT headline extraction helper."""

import unittest

from tradingagents.agents.utils.fingpt_tool import _extract_headlines


class ExtractHeadlinesTests(unittest.TestCase):
    def test_empty_or_missing_text_returns_empty_list(self):
        self.assertEqual(_extract_headlines(""), [])
        self.assertEqual(_extract_headlines(None), [])

    def test_filters_metadata_headers_and_short_lines(self):
        blob = """
# Heading that should be ignored
Date: 2025-06-01
Source: Example Wire
URL: https://example.com/story
Short title
This is a sufficiently long market headline for PETR4 today.
"""
        self.assertEqual(
            _extract_headlines(blob),
            ["This is a sufficiently long market headline for PETR4 today."],
        )

    def test_extracts_and_strips_lines_in_original_order(self):
        blob = """

  First valid line about macro inflation surprises in Brazil.  
Second valid line discussing overnight global risk sentiment shifts.

Third valid line with enough characters to be kept as a headline.
"""
        self.assertEqual(
            _extract_headlines(blob),
            [
                "First valid line about macro inflation surprises in Brazil.",
                "Second valid line discussing overnight global risk sentiment shifts.",
                "Third valid line with enough characters to be kept as a headline.",
            ],
        )

    def test_limits_to_maximum_number_of_headlines(self):
        many = "\n".join(
            [f"Headline number {i:02d} with enough content to pass minimum length." for i in range(40)]
        )
        out = _extract_headlines(many)
        self.assertEqual(len(out), 30)
        self.assertEqual(out[0], "Headline number 00 with enough content to pass minimum length.")
        self.assertEqual(out[-1], "Headline number 29 with enough content to pass minimum length.")


if __name__ == "__main__":
    unittest.main()
