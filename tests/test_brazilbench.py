"""Offline BrazilBench contract: frozen splits, classical baselines, no LLM.

Uses synthetic Close series and/or committed fixtures. No yfinance, no Ollama,
no Gradio, no ANTHROPIC_API_KEY.
"""

from __future__ import annotations

import ast
import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
BRAZILBENCH_PY = REPO / "tradingagents" / "backtest" / "brazilbench.py"
REGIME_LIB_PY = REPO / "scripts" / "regime_lib.py"
PRICE_DIR = REPO / "benchmark" / "prices"

README_TICKERS = ["ITUB4", "BPAC11", "PETR4", "VALE3", "WEGE3", "RADL3"]
PAPER_FIVE = ["PETR4", "VALE3", "ITUB4", "BBDC4", "^BVSP"]
FROZEN_REGIMES = {
    "bull_2019": ("2019-01-02", "2019-12-31"),
    "crisis_2020": ("2020-02-03", "2020-05-29"),
    "recovery_2021": ("2021-01-04", "2021-06-30"),
    "high_rates_2022": ("2022-01-03", "2022-12-30"),
}


def _load_regime_lib():
    spec = importlib.util.spec_from_file_location("regime_lib_for_dates", REGIME_LIB_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_linear_universe(price_dir: Path) -> None:
    """Close-only CSVs covering FETCH_START..FETCH_END for every README ticker."""
    price_dir.mkdir(parents=True, exist_ok=True)
    idx = pd.bdate_range("2018-01-01", "2023-01-15")
    for i, ticker in enumerate(README_TICKERS):
        close = 10.0 + i + np.arange(len(idx), dtype=float) * 0.01
        pd.DataFrame({"Date": idx, "Close": close}).to_csv(
            price_dir / f"{ticker}.csv", index=False
        )


class BrazilBenchUniverseTests(unittest.TestCase):
    def test_tickers_are_readme_six_not_paper_five(self):
        from tradingagents.backtest import brazilbench as bb

        self.assertEqual(list(bb.TICKERS), README_TICKERS)
        self.assertNotEqual(list(bb.TICKERS), PAPER_FIVE)
        self.assertNotIn("BBDC4", bb.TICKERS)
        self.assertNotIn("^BVSP", bb.TICKERS)
        self.assertNotIn("Momentum", [s.name for s in bb.STRATEGIES])

    def test_frozen_regime_windows(self):
        from tradingagents.backtest import brazilbench as bb

        self.assertEqual(dict(bb.REGIMES), FROZEN_REGIMES)
        self.assertEqual(list(bb.REGIME_ORDER), list(FROZEN_REGIMES))

    def test_regime_dates_match_regime_lib(self):
        from tradingagents.backtest import brazilbench as bb

        rl = _load_regime_lib()
        alias = {"high_rates_2022": "hi_rates_2022"}
        for name, window in bb.REGIMES.items():
            rl_name = alias.get(name, name)
            self.assertEqual(window, rl.REGIMES[rl_name], msg=name)

    def test_baselines_are_buy_hold_macd_sma_only(self):
        from tradingagents.backtest import brazilbench as bb
        from tradingagents.backtest.baselines import (
            BuyAndHold,
            MACDStrategy,
            SMACrossStrategy,
        )

        names = [type(s).__name__ for s in bb.STRATEGIES]
        self.assertEqual(names, ["BuyAndHold", "MACDStrategy", "SMACrossStrategy"])
        self.assertIsInstance(bb.STRATEGIES[0], BuyAndHold)
        self.assertIsInstance(bb.STRATEGIES[1], MACDStrategy)
        self.assertIsInstance(bb.STRATEGIES[2], SMACrossStrategy)


class BrazilBenchMetricsTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.price_dir = Path(self._tmp.name)
        _write_linear_universe(self.price_dir)

    def tearDown(self):
        self._tmp.cleanup()

    def test_run_cell_returns_cr_sharpe_mdd(self):
        from tradingagents.backtest import brazilbench as bb
        from tradingagents.backtest.baselines import BuyAndHold
        from tradingagents.backtest.metrics import ExtendedMetricsCalculator

        prices = bb.load_close("ITUB4", price_dir=self.price_dir)
        row = bb.run_cell(BuyAndHold(), prices, "bull_2019")
        self.assertEqual(set(row), {"cr", "sharpe", "mdd", "n_days"})
        win = prices.loc["2019-01-02":"2019-12-31"]
        expected = ExtendedMetricsCalculator().compute(
            BuyAndHold().run(win, bb.INITIAL_CAPITAL)
        )
        self.assertAlmostEqual(row["cr"], expected["cr"], places=10)
        self.assertAlmostEqual(row["sharpe"], expected["sharpe"], places=10)
        self.assertAlmostEqual(row["mdd"], expected["mdd"], places=10)
        self.assertEqual(row["n_days"], expected["n_days"])

    def test_buy_and_hold_cr_uses_regime_window(self):
        from tradingagents.backtest import brazilbench as bb
        from tradingagents.backtest.baselines import BuyAndHold

        prices = bb.load_close("PETR4", price_dir=self.price_dir)
        row = bb.run_cell(BuyAndHold(), prices, "crisis_2020")
        win = prices.loc["2020-02-03":"2020-05-29"]
        expected_cr = float(win["Close"].iloc[-1] / win["Close"].iloc[0] - 1.0)
        self.assertAlmostEqual(row["cr"], expected_cr, places=10)

    def test_run_matrix_covers_every_cell(self):
        from tradingagents.backtest import brazilbench as bb

        rows = bb.run_matrix(price_dir=self.price_dir)
        self.assertEqual(len(rows), 3 * 6 * 4)
        keys = {(r["strategy"], r["ticker"], r["regime"]) for r in rows}
        for strat in ("Buy & Hold", "MACD(12,26,9)", "SMA(50/200)"):
            for ticker in README_TICKERS:
                for regime in FROZEN_REGIMES:
                    self.assertIn((strat, ticker, regime), keys)
        for r in rows:
            self.assertIsInstance(r["cr"], float)
            self.assertIsInstance(r["sharpe"], float)
            self.assertIsInstance(r["mdd"], float)

    def test_render_table_includes_metric_headers(self):
        from tradingagents.backtest import brazilbench as bb

        text = bb.render_table(bb.run_matrix(price_dir=self.price_dir))
        self.assertIn("CR (%)", text)
        self.assertIn("Sharpe", text)
        self.assertIn("MDD (%)", text)
        self.assertIn("ITUB4", text)
        self.assertIn("high_rates_2022", text)


class BrazilBenchIsolationTests(unittest.TestCase):
    def test_source_never_mentions_graph_key_or_network(self):
        self.assertTrue(BRAZILBENCH_PY.is_file())
        src = BRAZILBENCH_PY.read_text(encoding="utf-8")
        tree = ast.parse(src)
        imported = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module.split(".")[0])
        self.assertNotIn("yfinance", imported)
        self.assertNotIn("dotenv", imported)
        self.assertNotIn("gradio", imported)
        lowered = src.lower()
        self.assertNotIn("tradingagentsgraph", lowered)
        self.assertNotIn("anthropic", lowered)
        self.assertNotIn("load_dotenv", lowered)
        self.assertNotIn("ollama", lowered)
        self.assertNotIn("gradio", lowered)

    def test_run_matrix_subprocess_never_imports_graph_or_yfinance(self):
        with tempfile.TemporaryDirectory() as tmp:
            price_dir = Path(tmp)
            _write_linear_universe(price_dir)
            env = os.environ.copy()
            env.pop("ANTHROPIC_API_KEY", None)
            env["PYTHONPATH"] = str(REPO)
            code = (
                "import os, sys\n"
                "from pathlib import Path\n"
                "os.environ.pop('ANTHROPIC_API_KEY', None)\n"
                "orig_get = os.environ.get\n"
                "orig_getitem = os.environ.__getitem__\n"
                "def _get(key, default=None):\n"
                "    if key == 'ANTHROPIC_API_KEY':\n"
                "        raise AssertionError('read ANTHROPIC_API_KEY')\n"
                "    return orig_get(key, default)\n"
                "def _getitem(key):\n"
                "    if key == 'ANTHROPIC_API_KEY':\n"
                "        raise AssertionError('read ANTHROPIC_API_KEY')\n"
                "    return orig_getitem(key)\n"
                "os.environ.get = _get\n"
                "os.environ.__getitem__ = _getitem\n"
                "from tradingagents.backtest.brazilbench import run_matrix\n"
                f"rows = run_matrix(price_dir=Path({str(price_dir)!r}))\n"
                "assert len(rows) == 72\n"
                "assert 'tradingagents.graph.trading_graph' not in sys.modules\n"
                "assert 'yfinance' not in sys.modules\n"
                "assert 'dotenv' not in sys.modules\n"
                "print('ok')\n"
            )
            proc = subprocess.run(
                [sys.executable, "-c", code],
                cwd=REPO,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, proc.stdout + "\n" + proc.stderr)
            self.assertIn("ok", proc.stdout)


class BrazilBenchFixtureTests(unittest.TestCase):
    def test_committed_close_fixtures_cover_frozen_windows(self):
        from tradingagents.backtest import brazilbench as bb

        self.assertTrue(PRICE_DIR.is_dir(), "commit Close fixtures under benchmark/prices/")
        for ticker in README_TICKERS:
            path = PRICE_DIR / f"{ticker}.csv"
            self.assertTrue(path.is_file(), f"missing {path}")
            df = bb.load_close(ticker, price_dir=PRICE_DIR)
            self.assertIn("Close", df.columns)
            for regime, (start, end) in FROZEN_REGIMES.items():
                win = df.loc[start:end]
                self.assertGreater(len(win), 20, f"{ticker} {regime} too short")
            # SMA(200) warmup needs history before the first regime.
            pre = df.loc[: "2019-01-01"]
            self.assertGreaterEqual(len(pre), 200, f"{ticker} missing SMA warmup")

    def test_committed_matrix_stays_on_metrics_path(self):
        from tradingagents.backtest import brazilbench as bb

        rows = bb.run_matrix(price_dir=PRICE_DIR)
        self.assertEqual(len(rows), 72)
        petr_bh = next(
            r for r in rows
            if r["strategy"] == "Buy & Hold"
            and r["ticker"] == "PETR4"
            and r["regime"] == "bull_2019"
        )
        self.assertIsNotNone(petr_bh["cr"])
        self.assertIsNotNone(petr_bh["sharpe"])
        self.assertIsNotNone(petr_bh["mdd"])
        # Buy & Hold CR is the Close ratio on the frozen window, independent of warmup.
        prices = bb.load_close("PETR4", price_dir=PRICE_DIR)
        win = prices.loc["2019-01-02":"2019-12-31"]
        expected_cr = float(win["Close"].iloc[-1] / win["Close"].iloc[0] - 1.0)
        self.assertAlmostEqual(petr_bh["cr"], expected_cr, places=10)


if __name__ == "__main__":
    unittest.main()
