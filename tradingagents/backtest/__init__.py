"""Backtesting module: academic metrics, baselines, runner, report."""

from .metrics import ExtendedMetricsCalculator
from .baselines import BuyAndHold, MACDStrategy, SMACrossStrategy
from .runner import run_strategy, run_buy_and_hold, run_agent_strategy
from .report import build_comparison_table, format_table_markdown, print_comparison

__all__ = [
    "ExtendedMetricsCalculator",
    "BuyAndHold",
    "MACDStrategy",
    "SMACrossStrategy",
    "run_strategy",
    "run_buy_and_hold",
    "run_agent_strategy",
    "build_comparison_table",
    "format_table_markdown",
    "print_comparison",
]
