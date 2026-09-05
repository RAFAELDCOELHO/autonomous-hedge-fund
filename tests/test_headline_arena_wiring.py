"""P2.1 offline contract: dual-agent config + dry-run (no network)."""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EXAMPLE = REPO / "config" / "headline_arena.example.yaml"
SCRIPT = REPO / "scripts" / "headline_arena_dry_run.py"
RUNBOOK = REPO / "docs" / "HEADLINE_ARENA.md"
OUT = REPO / "benchmark" / "results" / "headline_arena" / "dry_run.json"


class HeadlineArenaWiringTests(unittest.TestCase):
    def test_example_and_runbook_exist(self):
        self.assertTrue(EXAMPLE.is_file())
        self.assertTrue(RUNBOOK.is_file())
        text = EXAMPLE.read_text(encoding="utf-8")
        self.assertIn("ahf-tradingagents-macro", text)
        self.assertIn("ahf-tradingagents-no-macro", text)
        self.assertNotRegex(text, r"sk-[a-zA-Z0-9]{10,}")
        self.assertNotIn("client_secret:", text.lower().replace("client_secret_env", ""))

    def test_dry_run_writes_payload(self):
        proc = subprocess.run(
            [sys.executable, str(SCRIPT)],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertTrue(OUT.is_file())
        data = json.loads(OUT.read_text(encoding="utf-8"))
        self.assertEqual(data["mode"], "dry_run")
        self.assertFalse(data["network"])
        self.assertEqual(set(data["arms"]), {"macro", "no_macro"})
        a = data["arms"]["macro"]["arena_agent"]
        b = data["arms"]["no_macro"]["arena_agent"]
        self.assertNotEqual(a, b)
        self.assertNotEqual(
            data["arms"]["macro"]["credential_slots"]["credentials_file"],
            data["arms"]["no_macro"]["credential_slots"]["credentials_file"],
        )

    def test_live_without_env_exits_2(self):
        env = {k: v for k, v in __import__("os").environ.items() if not k.startswith("HEADLINE_ARENA_")}
        proc = subprocess.run(
            [sys.executable, str(SCRIPT), "--live"],
            cwd=REPO,
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )
        self.assertEqual(proc.returncode, 2, proc.stdout + proc.stderr)


if __name__ == "__main__":
    unittest.main()
