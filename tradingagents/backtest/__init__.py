"""Backtesting module: academic metrics, baselines, runner, report."""

from .metrics import ExtendedMetricsCalculator
from .baselines import BuyAndHold, MACDStrategy, SMACrossStrategy, MomentumStrategy
from .report import build_comparison_table, format_table_markdown, print_comparison

__all__ = [
    "ExtendedMetricsCalculator",
    "BuyAndHold",
    "MACDStrategy",
    "SMACrossStrategy",
    "MomentumStrategy",
    "run_strategy",
    "run_buy_and_hold",
    "run_agent_strategy",
    "build_comparison_table",
    "format_table_markdown",
    "print_comparison",
]


def __getattr__(name: str):
    # Keep runner (yfinance) off the BrazilBench import path.
    if name in {"run_strategy", "run_buy_and_hold", "run_agent_strategy"}:
        from . import runner as _runner
        return getattr(_runner, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
