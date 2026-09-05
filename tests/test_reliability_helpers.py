from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "reliability_diagram.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("reliability_diagram_helpers", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec for {SCRIPT}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class ReliabilityHelperTests(unittest.TestCase):
    def test_bins_csv_emits_expected_header_and_rows(self):
        rd = _load_module()
        bins = [
            {
                "bin_lo": 0.0,
                "bin_hi": 0.1,
                "n": 3,
                "n_unique_prompts": 2,
                "mean_confidence": 0.07,
                "win_rate": 0.33,
            },
            {
                "bin_lo": 0.9,
                "bin_hi": 1.0,
                "n": 1,
                "n_unique_prompts": 1,
                "mean_confidence": 0.95,
                "win_rate": 1.0,
            },
        ]
        out = rd.bins_csv(bins)
        lines = out.splitlines()
        self.assertEqual(lines[0], ",".join(rd.BIN_FIELDS))
        self.assertEqual(len(lines), 3)
        self.assertEqual(lines[1], "0.0,0.1,3,2,0.07,0.33")
        self.assertEqual(lines[2], "0.9,1.0,1,1,0.95,1.0")

    def test_svg_contains_diagonal_and_bin_markers(self):
        rd = _load_module()
        bins = [
            {
                "bin_lo": 0.4,
                "bin_hi": 0.5,
                "n": 4,
                "n_unique_prompts": 3,
                "mean_confidence": 0.45,
                "win_rate": 0.5,
            }
        ]
        out = rd.svg(bins)
        self.assertIn("<svg", out)
        self.assertIn("stroke-dasharray=\"4\"", out)
        self.assertIn("<circle", out)
        self.assertIn("n=4 (unique=3)", out)


if __name__ == "__main__":
    unittest.main()
