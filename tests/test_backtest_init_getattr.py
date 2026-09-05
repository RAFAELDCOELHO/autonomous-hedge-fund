"""Offline tests for tradingagents.backtest.__getattr__ lazy exports."""

import sys
import types
import unittest
from unittest.mock import patch


class BacktestInitGetattrTests(unittest.TestCase):
    def _fake_runner_module(self):
        fake = types.ModuleType("tradingagents.backtest.runner")

        def run_strategy(*args, **kwargs):
            return ("run_strategy", args, kwargs)

        def run_buy_and_hold(*args, **kwargs):
            return ("run_buy_and_hold", args, kwargs)

        def run_agent_strategy(*args, **kwargs):
            return ("run_agent_strategy", args, kwargs)

        fake.run_strategy = run_strategy
        fake.run_buy_and_hold = run_buy_and_hold
        fake.run_agent_strategy = run_agent_strategy
        return fake

    def test_known_lazy_exports_resolve_from_runner_module(self):
        import tradingagents.backtest as backtest

        fake_runner = self._fake_runner_module()
        with patch.dict(sys.modules, {"tradingagents.backtest.runner": fake_runner}):
            self.assertIs(backtest.run_strategy, fake_runner.run_strategy)
            self.assertIs(backtest.run_buy_and_hold, fake_runner.run_buy_and_hold)
            self.assertIs(backtest.run_agent_strategy, fake_runner.run_agent_strategy)

    def test_unknown_attribute_raises_attribute_error(self):
        import tradingagents.backtest as backtest

        with self.assertRaises(AttributeError) as ctx:
            _ = backtest.not_a_real_export
        self.assertIn("tradingagents.backtest", str(ctx.exception))
        self.assertIn("not_a_real_export", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
