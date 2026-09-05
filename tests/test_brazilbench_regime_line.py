"""Offline tests for the BrazilBench regime-line formatting helper."""

import unittest

from tradingagents.backtest import brazilbench


class BrazilBenchRegimeLineTests(unittest.TestCase):
    def test_regime_line_contains_all_declared_regimes_in_order(self):
        line = brazilbench._regime_line()

        self.assertTrue(line.startswith("Regimes: "))
        body = line.removeprefix("Regimes: ")
        parts = body.split("; ")

        self.assertEqual(len(parts), len(brazilbench.REGIMES))
        expected_parts = [
            f"{name} {start} .. {end}"
            for name, (start, end) in brazilbench.REGIMES.items()
        ]
        self.assertEqual(parts, expected_parts)

    def test_regime_line_has_stable_separator_count(self):
        line = brazilbench._regime_line()
        self.assertEqual(line.count("; "), max(len(brazilbench.REGIMES) - 1, 0))


if __name__ == "__main__":
    unittest.main()
