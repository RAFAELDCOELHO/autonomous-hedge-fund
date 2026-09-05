"""Offline BrazilBench baselines on frozen B3 regime splits.

Universe: README Brazilian equities (ITUB4, BPAC11, PETR4, VALE3, WEGE3, RADL3).
Strategies: Buy & Hold, MACD(12,26,9), SMA(50/200). No LLM. No API key.
Prices: committed Close CSVs under benchmark/prices/ (Yahoo .SA, auto_adjusted,
FETCH_START 2018-01-01 through FETCH_END 2023-01-15). Never downloads.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from tradingagents.backtest.baselines import (
    BuyAndHold,
    MACDStrategy,
    SMACrossStrategy,
    _simulate,
)
from tradingagents.backtest.metrics import ExtendedMetricsCalculator

REPO = Path(__file__).resolve().parents[2]
PRICE_DIR = REPO / "benchmark" / "prices"

# README Brazilian equity set. Not the paper 9-asset table, not regime_lib's five names.
TICKERS: list[str] = ["ITUB4", "BPAC11", "PETR4", "VALE3", "WEGE3", "RADL3"]

# Dates copied from docs/brazilbench.tex tab:regimes / scripts/regime_lib.REGIMES.
# Public label is high_rates_2022 (tex); regime_lib stores the same window as hi_rates_2022.
REGIMES: dict[str, tuple[str, str]] = {
    "bull_2019": ("2019-01-02", "2019-12-31"),
    "crisis_2020": ("2020-02-03", "2020-05-29"),
    "recovery_2021": ("2021-01-04", "2021-06-30"),
    "high_rates_2022": ("2022-01-03", "2022-12-30"),
}
REGIME_ORDER: list[str] = list(REGIMES)

STRATEGIES = [BuyAndHold(), MACDStrategy(), SMACrossStrategy()]
INITIAL_CAPITAL = 100_000.0

# `make reproduce` artifacts (scripts/run_brazilbench.py --write).
MATRIX_CSV = REPO / "benchmark" / "results" / "brazilbench" / "matrix.csv"
MATRIX_MD = REPO / "docs" / "brazilbench_baselines.md"


def load_close(ticker: str, *, price_dir: Path | None = None) -> pd.DataFrame:
    """Load a committed Date,Close fixture. Never downloads."""
    root = Path(price_dir) if price_dir is not None else PRICE_DIR
    path = root / f"{ticker}.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing Close fixture {path}. BrazilBench is offline; "
            "commit the CSV under benchmark/prices/."
        )
    df = pd.read_csv(path, parse_dates=["Date"]).set_index("Date").sort_index()
    if "Close" not in df.columns:
        raise ValueError(f"{path} has no Close column")
    out = df[["Close"]].astype(float).dropna()
    out.index.name = "Date"
    return out


def warm_window(df: pd.DataFrame, regime: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """History through window end (indicator warmup) and the regime slice itself."""
    start, end = REGIMES[regime]
    warmed = df.loc[:end]
    return warmed, warmed.loc[start:end]


def run_cell(strategy, prices: pd.DataFrame, regime: str,
             capital: float = INITIAL_CAPITAL) -> dict:
    """CR / Sharpe / MDD on one (strategy, ticker, regime) cell.

    Indicators see pre-window history. The equity curve starts at `capital`
    on the first bar of the frozen window (same semantics as regime_lib).
    """
    warmed, win = warm_window(prices, regime)
    if win.empty:
        raise ValueError(f"No Close bars for regime {regime}")
    sig = strategy.signals(warmed).loc[win.index]
    equity = _simulate(win, sig, capital)
    m = ExtendedMetricsCalculator().compute(equity)
    return {
        "cr": m["cr"],
        "sharpe": m["sharpe"],
        "mdd": m["mdd"],
        "n_days": m["n_days"],
    }


def run_matrix(*, price_dir: Path | None = None) -> list[dict]:
    """All baseline cells: 3 strategies x 6 tickers x 4 regimes."""
    root = Path(price_dir) if price_dir is not None else PRICE_DIR
    frames = {ticker: load_close(ticker, price_dir=root) for ticker in TICKERS}
    rows: list[dict] = []
    for strategy in STRATEGIES:
        for ticker in TICKERS:
            for regime in REGIME_ORDER:
                metrics = run_cell(strategy, frames[ticker], regime)
                rows.append({
                    "strategy": strategy.name,
                    "ticker": ticker,
                    "regime": regime,
                    **metrics,
                })
    return rows


def _display_frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    return pd.DataFrame({
        "strategy": df["strategy"],
        "ticker": df["ticker"],
        "regime": df["regime"],
        "CR (%)": df["cr"].map(lambda x: f"{100.0 * x:.2f}"),
        "Sharpe": df["sharpe"].map(lambda x: f"{x:.3f}"),
        "MDD (%)": df["mdd"].map(lambda x: f"{100.0 * x:.2f}"),
    })


def render_table(rows: list[dict]) -> str:
    return _display_frame(rows).to_string(index=False)


def _regime_line() -> str:
    return "Regimes: " + "; ".join(
        f"{name} {start} .. {end}" for name, (start, end) in REGIMES.items()
    )


def write_outputs(rows: list[dict], csv_path: Path = MATRIX_CSV,
                  md_path: Path = MATRIX_MD) -> None:
    """Persist the matrix: 6-dp CSV plus a markdown table for docs/."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).round(6).to_csv(csv_path, index=False)

    shown = _display_frame(rows)
    cols = list(shown.columns)
    lines = [
        "<!-- Auto-generated by `make reproduce` "
        "(scripts/run_brazilbench.py --write). Do not edit by hand. -->",
        "",
        "# BrazilBench offline baselines",
        "",
        "Buy & Hold, MACD(12,26,9), SMA(50/200) on committed Close fixtures "
        "(`benchmark/prices/`), initial capital 100,000, frictionless, "
        "long-only. No API key. No LLM.",
        "",
        "Tickers: " + ", ".join(TICKERS),
        "",
        _regime_line(),
        "",
        "| " + " | ".join(cols) + " |",
        "|" + "|".join("---" for _ in cols) + "|",
        *("| " + " | ".join(str(v) for v in r) + " |"
          for r in shown.itertuples(index=False)),
        "",
    ]
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true",
                        help=f"also write {MATRIX_CSV} and {MATRIX_MD}")
    args = parser.parse_args(argv)
    rows = run_matrix()
    print("BrazilBench offline baselines (Buy & Hold, MACD, SMA).")
    print("Tickers: " + ", ".join(TICKERS))
    print(_regime_line())
    print("No API key. No LLM.")
    print()
    print(render_table(rows))
    if args.write:
        write_outputs(rows)
        print(f"\nWrote:\n  {MATRIX_CSV}\n  {MATRIX_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
