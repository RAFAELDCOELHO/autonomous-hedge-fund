"""Offline tests for tradingagents.backtest.__getattr__ lazy exports."""

import sys
import types
import unittest


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
        fake_runner = self._fake_runner_module()
        original_runner = sys.modules.get("tradingagents.backtest.runner")
        sys.modules["tradingagents.backtest.runner"] = fake_runner
        try:
            import tradingagents.backtest as backtest

            # Force module-level lookup path to execute __getattr__.
            backtest.__dict__.pop("runner", None)
            for name in ("run_strategy", "run_buy_and_hold", "run_agent_strategy"):
                backtest.__dict__.pop(name, None)
                self.assertIs(getattr(backtest, name), getattr(fake_runner, name))
        finally:
            if original_runner is None:
                sys.modules.pop("tradingagents.backtest.runner", None)
            else:
                sys.modules["tradingagents.backtest.runner"] = original_runner

    def test_unknown_attribute_raises_attribute_error(self):
        import tradingagents.backtest as backtest

        with self.assertRaises(AttributeError) as ctx:
            _ = backtest.not_a_real_export
        self.assertIn("tradingagents.backtest", str(ctx.exception))
        self.assertIn("not_a_real_export", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
