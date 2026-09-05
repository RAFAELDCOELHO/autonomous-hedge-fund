"""Offline contract for `scripts/headline_arena_arms.py` ($0 path).

The script must list the two arena arms without importing paid/live deps.
"""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "headline_arena_arms.py"


class HeadlineArenaArmsTests(unittest.TestCase):
    def test_arm_mapping_invariants_hold(self):
        ns: dict[str, object] = {}
        code = SCRIPT.read_text(encoding="utf-8")
        exec(compile(code, str(SCRIPT), "exec"), ns)
        arms = ns["ARMS"]
        self.assertEqual(set(arms), {"macro", "no_macro"})
        macro = set(arms["macro"]["selected_analysts"])
        no_macro = set(arms["no_macro"]["selected_analysts"])
        self.assertEqual(macro - no_macro, {"macro"})
        self.assertTrue(no_macro <= macro)
        self.assertNotEqual(arms["macro"]["arena_agent"], arms["no_macro"]["arena_agent"])

    def test_default_cli_lists_arms_without_key(self):
        completed = subprocess.run(
            [sys.executable, str(SCRIPT)],
            cwd=REPO,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        out = completed.stdout
        self.assertIn("macro", out)
        self.assertIn("no_macro", out)
        self.assertIn("ahf-tradingagents-macro", out)
        self.assertIn("ahf-tradingagents-no-macro", out)


if __name__ == "__main__":
    unittest.main()
