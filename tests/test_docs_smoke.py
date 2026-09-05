"""Offline smoke checks for core documentation artifact paths."""

from __future__ import annotations

import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


class DocsSmokeTests(unittest.TestCase):
    def test_required_artifact_files_exist(self):
        required = [
            "README.md",
            "docs/brazilbench.tex",
            "docs/NEURIPS_CHECKLIST.md",
            "CITATION.cff",
            "LICENSE",
        ]
        for rel in required:
            self.assertTrue((REPO / rel).is_file(), f"missing required file: {rel}")

    def test_readme_links_key_artifacts_and_reproduce(self):
        readme = (REPO / "README.md").read_text(encoding="utf-8")
        self.assertIn("(docs/brazilbench.tex)", readme)
        self.assertIn("(docs/NEURIPS_CHECKLIST.md)", readme)
        self.assertIn("make reproduce", readme)


if __name__ == "__main__":
    unittest.main()
