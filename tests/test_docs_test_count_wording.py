"""Guard docs against brittle hard-coded offline test-count claims.

These docs should point to a live count command instead of embedding a number
that drifts as tests are added over time.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TARGET_DOCS = [
    REPO / "ROADMAP.md",
    REPO / "docs" / "ARCHITECTURE.md",
]


class DocsTestCountWordingTests(unittest.TestCase):
    def test_no_numeric_test_count_on_guarded_lines(self):
        any_number = re.compile(r"\d")
        for path in TARGET_DOCS:
            for line in path.read_text(encoding="utf-8").splitlines():
                if "Test suite green" in line or "offline unit tests" in line:
                    self.assertIsNone(
                        any_number.search(line),
                        f"{path.name} reintroduced a hard-coded test count: {line}",
                    )

    def test_collect_only_command_is_present(self):
        expected = "python -m pytest --collect-only -q"
        for path in TARGET_DOCS:
            text = path.read_text(encoding="utf-8")
            self.assertIn(expected, text, path.name)


if __name__ == "__main__":
    unittest.main()
