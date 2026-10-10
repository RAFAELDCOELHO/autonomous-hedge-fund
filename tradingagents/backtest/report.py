"""Report layer for backtests.

Converts metric fractions to percentage strings for presentation.
"""

from __future__ import annotations

from typing import Dict

import pandas as pd

from .metrics import FLAT_RF_SENSITIVITY, flat_rf_metrics, h1_cell_metrics

SHARPE_FLAT_COL = f"Sharpe @ flat {FLAT_RF_SENSITIVITY:.2%} (exploratory)"
H1_SHARPE_COL = "H1 Sharpe (excess over daily rf)"
_TABLE_COLUMNS = ["Strategy", "CR (%)", "AR (%)", SHARPE_FLAT_COL, H1_SHARPE_COL, "MDD (%)"]


def build_comparison_table(
    equity_curves: Dict[str, pd.Series],
    market: str | None = None,
) -> pd.DataFrame:
    """Build a DataFrame of display strings for the comparison table.

    SHARPE_FLAT_COL is the flat exploratory rf (FLAT_RF_SENSITIVITY) for every
    strategy. H1_SHARPE_COL is the excess Sharpe from h1_cell_metrics over that
    market's daily rf, for the agent arms and for every curve whose cash earned
    that same daily rf (attrs["rf_market"] == market: every runner curve of a
    run_backtest.py run without --no-rf). Other curves, e.g. all of a --no-rf
    run, keep an em dash: their cash did not earn the daily rf.

    The table's ``attrs["cash_rf"]`` (cash_rf_label) names the rf each curve's
    cash earned; print_comparison prints it under the table.
    """
    rows = []
    for name, eq in equity_curves.items():
        m = flat_rf_metrics(eq)
        attrs = getattr(eq, "attrs", {})
        # A curve that records its rf (runner attrs) gets the H1 Sharpe only if its cash
        # earned this market's rf: --no-rf agent arms show an em dash like the baselines.
        # Agent curves without that record (built by hand) keep the old name rule.
        earned_rf = attrs.get("rf_market") == market if "rf_market" in attrs else name.startswith("TradingAgents (")
        if market is not None and earned_rf:
            h1_sharpe = _num(h1_cell_metrics(eq, market)["sharpe"])
        else:
            h1_sharpe = "—"
        rows.append({
            "Strategy": name,
            "CR (%)": _pct(m["cr"]),
            "AR (%)": _pct(m["ar"]),
            SHARPE_FLAT_COL: _num(m["sharpe"]),
            H1_SHARPE_COL: h1_sharpe,
            "MDD (%)": _pct(m["mdd"]),
        })
    df = pd.DataFrame(rows, columns=_TABLE_COLUMNS)
    df.attrs["cash_rf"] = cash_rf_label(equity_curves)
    return df


def cash_rf_label(equity_curves: Dict[str, pd.Series]) -> str:
    """'Cash rf: <source>' (one source for all curves) or per strategy, from attrs["rf_source"]."""
    sources = {name: getattr(eq, "attrs", {}).get("rf_source") for name, eq in equity_curves.items()}
    sources = {name: src for name, src in sources.items() if src}
    if not sources:
        return ""
    if len(set(sources.values())) == 1:
        return f"Cash rf: {next(iter(sources.values()))} (all strategies)"
    return "Cash rf: " + "; ".join(f"{name}: {src}" for name, src in sources.items())


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


def print_comparison(
    equity_curves: Dict[str, pd.Series],
    market: str | None = None,
) -> pd.DataFrame:
    df = build_comparison_table(equity_curves, market=market)
    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(title="Backtest Results", caption=df.attrs["cash_rf"] or None, show_lines=False)
        for col in df.columns:
            table.add_column(col)
        for _, row in df.iterrows():
            table.add_row(*[str(v) for v in row.values])
        console.print(table)
    except Exception:
        print(format_table_markdown(df))
        if df.attrs["cash_rf"]:
            print(df.attrs["cash_rf"])
    return df
