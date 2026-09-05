"""P0.2: reproducibility manifest exists and names required $0 artifacts.

Offline: no network, no LLM. Asserts paths that are already on main exist,
and that open-PR-only result dirs are documented as not-yet-on-main.
"""

from __future__ import annotations

import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "docs" / "REPRODUCIBILITY.md"

# Committed on main — must exist and be named in the manifest.
REQUIRED_ON_MAIN = [
    "docs/REPRODUCIBILITY.md",
    "benchmark/results/random_n100/summary.csv",
    "benchmark/results/random_n100/per_seed_returns.csv",
    "benchmark/results/baselines/ew_returns.csv",
    "benchmark/results/baselines/per_cell_returns.csv",
    "benchmark/results/brazilbench/matrix.csv",
    "benchmark/results/reliability/SCHEMA.md",
    "benchmark/results/reliability/decisions.jsonl",
    "benchmark/results/reliability/bins.csv",
    "benchmark/results/reliability/reliability.svg",
    "determinism_results.json",
    "brazilbench_mistral_results.json",
    "docs/paper_random_n100.tex",
    "docs/paper_ew_portfolio_baselines.tex",
    "docs/brazilbench_baselines.md",
    "scripts/regime_lib.py",
    "scripts/run_random_n100.py",
    "scripts/run_ew_portfolio.py",
    "scripts/run_brazilbench.py",
    "scripts/reliability_diagram.py",
    "Makefile",
]

# Present on open feature PRs only — must be named in the manifest, must NOT
# be required to exist on this checkout of main.
OPEN_PR_ONLY = [
    "benchmark/results/qwen_n10/",
    "benchmark/results/chronos/",
    "benchmark/results/hmm_regimes/",
    "benchmark/results/multi_asset/",
    "benchmark/results/survivorship/",
]


class ReproManifestTests(unittest.TestCase):
    def test_manifest_exists(self):
        self.assertTrue(MANIFEST.is_file(), f"missing {MANIFEST}")

    def test_required_on_main_paths_exist_and_are_listed(self):
        text = MANIFEST.read_text(encoding="utf-8")
        for rel in REQUIRED_ON_MAIN:
            path = REPO / rel
            self.assertTrue(path.exists(), f"required main artifact missing: {rel}")
            self.assertIn(rel, text, f"manifest must list {rel}")

    def test_open_pr_artifacts_are_documented_not_required_on_main(self):
        text = MANIFEST.read_text(encoding="utf-8")
        lowered = text.lower()
        self.assertIn("not on `main`", lowered)
        self.assertIn("pending", lowered)
        self.assertIn("claude", lowered)
        for rel in OPEN_PR_ONLY:
            self.assertIn(rel, text, f"manifest must name open-PR path {rel}")
            # Still open: directory should be absent on main checkouts.
            self.assertFalse(
                (REPO / rel.rstrip("/")).is_dir(),
                f"{rel} unexpectedly present on this tree; update manifest if merged",
            )

    def test_manifest_covers_seeds_and_models(self):
        text = MANIFEST.read_text(encoding="utf-8")
        for needle in (
            "seed=42",
            "0..99",
            "mistral:7b",
            "qwen2.5:7b",
            "amazon/chronos-t5-tiny",
            "Ollama",
            "SCHEMA.md",
            "make reproduce",
            "PENDING",
        ):
            self.assertIn(needle, text, needle)


if __name__ == "__main__":
    unittest.main()
