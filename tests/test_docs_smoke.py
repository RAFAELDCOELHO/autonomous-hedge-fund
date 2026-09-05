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
        missing = [rel for rel in required if not (REPO / rel).is_file()]
        self.assertEqual(missing, [], f"missing required files: {missing}")

    def test_readme_links_key_artifacts_and_reproduce(self):
        readme = (REPO / "README.md").read_text(encoding="utf-8")
        self.assertIn("[`docs/brazilbench.tex`](docs/brazilbench.tex)", readme)
        self.assertIn("[`docs/NEURIPS_CHECKLIST.md`](docs/NEURIPS_CHECKLIST.md)", readme)
        self.assertIn("make reproduce", readme)
        self.assertIn("Reproduce the paper tables (offline, no API key)", readme)

    def test_neurips_checklist_rows_have_expected_status_values(self):
        checklist = (REPO / "docs/NEURIPS_CHECKLIST.md").read_text(encoding="utf-8")
        rows = [
            line
            for line in checklist.splitlines()
            if line.startswith("| ")
            and not line.startswith("|---")
            and "Yes/No/NA" not in line
            and "Repo-grounded justification" not in line
        ]
        self.assertGreaterEqual(len(rows), 10, "expected at least 10 checklist rows")
        for row in rows:
            cells = [c.strip() for c in row.strip("|").split("|")]
            self.assertGreaterEqual(len(cells), 3, f"malformed checklist row: {row}")
            self.assertIn(cells[1], {"Yes", "No", "NA"}, f"invalid checklist status in row: {row}")
            self.assertNotIn("TODO", row)
            self.assertNotIn("TBD", row)


if __name__ == "__main__":
    unittest.main()
