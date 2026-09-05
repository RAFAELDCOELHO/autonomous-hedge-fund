"""P2.1 offline contract: dual-agent config + dry-run (no network)."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EXAMPLE = REPO / "config" / "headline_arena.example.yaml"
SCRIPT = REPO / "scripts" / "headline_arena_dry_run.py"
RUNBOOK = REPO / "docs" / "HEADLINE_ARENA.md"
OUT = REPO / "benchmark" / "results" / "headline_arena" / "dry_run.json"
ARMS_PY = REPO / "scripts" / "headline_arena_arms.py"

SECRETISH = re.compile(
    r"(?:sk-[A-Za-z0-9]{20,}|(?<!client_secret_env)(?<!agent_id_env)"
    r"\bclient_secret\s*[:=]\s*['\"]?[A-Za-z0-9+/=_-]{24,})",
    re.IGNORECASE,
)


class HeadlineArenaWiringTests(unittest.TestCase):
    def test_example_and_runbook_exist(self):
        self.assertTrue(EXAMPLE.is_file())
        self.assertTrue(RUNBOOK.is_file())
        text = EXAMPLE.read_text(encoding="utf-8")
        self.assertIn("ahf-tradingagents-macro", text)
        self.assertIn("ahf-tradingagents-no-macro", text)
        self.assertIn("HEADLINE_ARENA_MACRO", text)
        self.assertIn("HEADLINE_ARENA_NO_MACRO", text)
        self.assertNotRegex(text, r"sk-[a-zA-Z0-9]{10,}")
        # No literal client_secret values — only *_env / credentials_file keys.
        self.assertNotRegex(text, r"^\s*client_secret\s*:", re.M)

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
        self.assertEqual(a, "ahf-tradingagents-macro")
        self.assertEqual(b, "ahf-tradingagents-no-macro")
        self.assertNotEqual(
            data["arms"]["macro"]["credential_slots"]["credentials_file"],
            data["arms"]["no_macro"]["credential_slots"]["credentials_file"],
        )
        self.assertNotEqual(
            data["arms"]["macro"]["credential_slots"]["client_secret_env"],
            data["arms"]["no_macro"]["credential_slots"]["client_secret_env"],
        )
        for arm in data["arms"].values():
            self.assertIn("forecast", arm)
            self.assertIn("scorecard", arm)

    def test_arms_py_names_match_config(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location("headline_arena_arms", ARMS_PY)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        text = EXAMPLE.read_text(encoding="utf-8")
        for key, arm in mod.ARMS.items():
            self.assertIn(arm["arena_agent"], text)

    def test_live_without_env_exits_2(self):
        env = {k: v for k, v in os.environ.items() if not k.startswith("HEADLINE_ARENA_")}
        proc = subprocess.run(
            [sys.executable, str(SCRIPT), "--live"],
            cwd=REPO,
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )
        self.assertEqual(proc.returncode, 2, proc.stdout + proc.stderr)

    def test_committed_wiring_has_no_secretish_strings(self):
        for path in (EXAMPLE, RUNBOOK, SCRIPT, ARMS_PY, OUT):
            text = path.read_text(encoding="utf-8")
            self.assertIsNone(
                SECRETISH.search(text),
                f"possible secret-like token in {path.relative_to(REPO)}",
            )
            self.assertNotIn("sk-ant-", text)


if __name__ == "__main__":
    unittest.main()
