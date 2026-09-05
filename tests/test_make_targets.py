"""Makefile contract checks for the zero-cost D&B path.

These tests guard the user-facing commands documented in README:
`make bench`, `make reproduce`, `make reliability`, and `make docker-bench`.
They do not execute make; they validate target recipes stay aligned with
the documented offline workflow.
"""

from __future__ import annotations

import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MAKEFILE = REPO / "Makefile"


def _targets() -> dict[str, list[str]]:
    targets: dict[str, list[str]] = {}
    current: str | None = None

    for raw in MAKEFILE.read_text(encoding="utf-8").splitlines():
        if raw.startswith("\t"):
            if current is not None:
                targets[current].append(raw.strip())
            continue

        if not raw or raw.startswith("#") or raw.startswith(".PHONY:"):
            current = None
            continue

        if ":" in raw:
            name = raw.split(":", 1)[0].strip()
            if " " in name:
                current = None
                continue
            current = name
            targets.setdefault(current, [])
            continue

        current = None

    return targets


class MakefileContractTests(unittest.TestCase):
    def setUp(self):
        self.targets = _targets()

    def test_zero_cost_targets_exist(self):
        for target in ("bench", "reproduce", "reliability", "docker-bench"):
            self.assertIn(target, self.targets, target)

    def test_reproduce_recipe_pins_expected_scripts(self):
        recipe = self.targets["reproduce"]
        self.assertEqual(
            recipe,
            [
                "$(PY) scripts/run_random_n100.py",
                "$(PY) scripts/run_ew_portfolio.py",
                "$(PY) scripts/run_brazilbench.py --write",
            ],
        )

    def test_bench_and_reliability_stay_single_script_entrypoints(self):
        self.assertEqual(self.targets["bench"], ["$(PY) scripts/run_brazilbench.py"])
        self.assertEqual(self.targets["reliability"], ["$(PY) scripts/reliability_diagram.py"])

    def test_docker_bench_writes_outputs_back_to_repo(self):
        recipe = self.targets["docker-bench"]
        self.assertEqual(recipe[0], "docker build -f Dockerfile.bench -t brazilbench .")
        self.assertIn("-v \"$$PWD/benchmark/results:/app/benchmark/results\"", recipe[1])
        self.assertIn("-v \"$$PWD/docs:/app/docs\"", recipe[1])
        self.assertIn("brazilbench", recipe[1])


if __name__ == "__main__":
    unittest.main()
