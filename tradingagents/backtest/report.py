"""Report layer for backtests.

Converts metric fractions to percentage strings for presentation.
"""

from __future__ import annotations

from typing import Dict

import pandas as pd

from .metrics import ExtendedMetricsCalculator


def build_comparison_table(equity_curves: Dict[str, pd.Series]) -> pd.DataFrame:
    """Build a DataFrame with columns: Strategy, CR (%), AR (%), Sharpe, MDD (%).

    Values are formatted strings ready for display.
    """
    calc = ExtendedMetricsCalculator()
    rows = []
    for name, eq in equity_curves.items():
        m = calc.compute(eq)
        rows.append({
            "Strategy": name,
            "CR (%)": _pct(m["cr"]),
            "AR (%)": _pct(m["ar"]),
            "Sharpe": _num(m["sharpe"]),
            "MDD (%)": _pct(m["mdd"]),
        })
    return pd.DataFrame(rows, columns=["Strategy", "CR (%)", "AR (%)", "Sharpe", "MDD (%)"])


def _pct(x) -> str:
    if x is None:
        return "—"
    return f"{x * 100:.2f}"


def _num(x) -> str:
    if x is None:
        return "—"
    if x == float("inf"):
        return "inf"
    return f"{x:.3f}"


def format_table_markdown(df: pd.DataFrame) -> str:
    header = "| " + " | ".join(df.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(df.columns)) + " |"
    lines = [header, sep]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row.values) + " |")
    return "\n".join(lines)


def print_comparison(equity_curves: Dict[str, pd.Series]) -> pd.DataFrame:
    df = build_comparison_table(equity_curves)
    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(title="Backtest Results", show_lines=False)
        for col in df.columns:
            table.add_column(col)
        for _, row in df.iterrows():
            table.add_row(*[str(v) for v in row.values])
        console.print(table)
    except Exception:
        print(format_table_markdown(df))
    return df
