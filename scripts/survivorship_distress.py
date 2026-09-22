#!/usr/bin/env python3
"""P1.9 - Survivorship bracket on distressed B3 names ($0, offline).

Runs the five classical agents (buy_and_hold, macd, sma_crossover, momentum,
random N=100 mean) on the distressed set OIBR3 / MGLU3 / AMER3 over the four
hand regimes, and side by side on the liquid paper-five (PETR4, VALE3, ITUB4,
BBDC4, ^BVSP). The gap "liquid minus distressed" mean CR brackets how much a
liquid-only universe flatters the baselines (cf. Lesmond 1999, 1-3pp/yr).

AMER3 (Americanas) substitutes GOLL4: Yahoo Finance raises YFTzMissingError
for GOLL4.SA (delisted, no timezone), so no GOLL4 prices exist here and none
are invented.

Same semantics as the paper tables: long-only, binary, frictionless, warm
indicators, no look-ahead; prices are the committed fixtures only.

Outputs
    benchmark/results/survivorship/per_cell.csv  (universe, agent, regime, ticker)
    benchmark/results/survivorship/summary.csv   (agent, regime, liquid vs distressed mean CR)

Usage
    uv run python scripts/survivorship_distress.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import regime_lib as rl  # noqa: E402
from tradingagents.backtest.baselines import (  # noqa: E402
    BuyAndHold, MACDStrategy, SMACrossStrategy, MomentumStrategy,
)

OUT_DIR = rl.REPO / "benchmark" / "results" / "survivorship"
PERCELL_CSV = OUT_DIR / "per_cell.csv"
SUMMARY_CSV = OUT_DIR / "summary.csv"

DETERMINISTIC = {
    "buy_and_hold": BuyAndHold(),
    "macd": MACDStrategy(),
    "sma_crossover": SMACrossStrategy(),
    "momentum": MomentumStrategy(),
}
AGENT_ORDER = [*DETERMINISTIC, "random"]
UNIVERSES = {"liquid": rl.TICKERS, "distressed": rl.DISTRESSED}


def load_fixture(ticker: str) -> pd.DataFrame:
    """Committed fixture only: never falls through to a download."""
    path = rl._cache_path(ticker)
    if not path.exists():
        raise SystemExit(f"Missing fixture {path}; P1.9 does not download.")
    return rl.load_prices(ticker)


def cell_return_pct(agent: str, df: pd.DataFrame, regime: str) -> float:
    if agent == "random":
        r = np.mean([rl.run_random_cell(df, regime, s) for s in range(rl.N_SEEDS)])
    else:
        r = rl.run_strategy_cell(DETERMINISTIC[agent], df, regime)
    return float(r) * 100.0


def compute() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for universe, tickers in UNIVERSES.items():
        price = {t: load_fixture(t) for t in tickers}
        for agent in AGENT_ORDER:
            for regime in rl.REGIME_ORDER:
                for t in tickers:
                    rows.append({
                        "universe": universe, "agent": agent, "regime": regime,
                        "ticker": t, "n_days": len(rl.regime_window(price[t], regime)),
                        "total_return_pct": round(cell_return_pct(agent, price[t], regime), 4),
                    })
    percell = pd.DataFrame(rows)

    mean = (percell.groupby(["agent", "regime", "universe"])["total_return_pct"]
            .mean().unstack("universe"))
    idx = pd.MultiIndex.from_product([AGENT_ORDER, rl.REGIME_ORDER], names=["agent", "regime"])
    summary = mean.reindex(idx).reset_index().rename(columns={
        "liquid": "liquid_mean_cr_pct", "distressed": "distressed_mean_cr_pct"})
    summary = summary[["agent", "regime", "liquid_mean_cr_pct", "distressed_mean_cr_pct"]]
    summary["gap_pp"] = summary["liquid_mean_cr_pct"] - summary["distressed_mean_cr_pct"]
    return percell, summary.round(4)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    percell, summary = compute()
    percell.to_csv(PERCELL_CSV, index=False)
    summary.to_csv(SUMMARY_CSV, index=False)

    print("Mean CR (%) liquid paper-five vs distressed OIBR3/MGLU3/AMER3; "
          "gap_pp = liquid - distressed\n")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:8.2f}"))
    print(f"\nWrote:\n  {PERCELL_CSV}\n  {SUMMARY_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
