#!/usr/bin/env python3
"""P1.10 — Chronos-t5-tiny as a comparator trading agent on paper fixtures.

Long-only, binary, frictionless close semantics via ``regime_lib`` (same as
the classical baselines). Prices come from committed paper fixtures through
``regime_lib.load_prices``; this script never downloads.

Optional extra: ``uv run --extra chronos python scripts/chronos_comparator.py``
(or ``make chronos``). If the ``chronos`` package is missing or
``CHRONOS_SKIP=1``, prints a skip line and exits 0 so CI without the extra
stays green.

Design choices (documented in the artifact README):
  * Model: amazon/chronos-t5-tiny on CPU.
  * Tickers: PETR4, ^BVSP. Regimes: all four from ``regime_lib``.
  * Context: last CONTEXT bars of Close ending at the decision bar.
  * Horizon: 1; long iff mean sample forecast of next Close > current Close.
  * Decisions: every trading bar (Chronos-t5-tiny is fast enough on CPU;
    DECISION_STRIDE can be raised if a heavier model is substituted).
  * Comparators in the same cells: buy_and_hold and Momentum(60).

Outputs under ``benchmark/results/chronos/``:
  per_cell.csv, summary.csv, README.md
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import regime_lib as rl  # noqa: E402
from tradingagents.backtest.baselines import BuyAndHold, MomentumStrategy  # noqa: E402

OUT_DIR = rl.REPO / "benchmark" / "results" / "chronos"
PER_CELL = OUT_DIR / "per_cell.csv"
SUMMARY = OUT_DIR / "summary.csv"
README = OUT_DIR / "README.md"

TICKERS = ["PETR4", "^BVSP"]
MODEL_ID = "amazon/chronos-t5-tiny"
CONTEXT = 128
HORIZON = 1
NUM_SAMPLES = 20
BATCH_SIZE = 32
# Every bar. Raise (e.g. 5) only if a slower Chronos variant is used.
DECISION_STRIDE = 1


def _skip(msg: str) -> int:
    print(f"chronos_comparator: skip — {msg}")
    return 0


def _try_load_pipeline():
    """Return ChronosPipeline or None when the optional extra is absent."""
    if os.environ.get("CHRONOS_SKIP", "").strip() == "1":
        return None
    try:
        import torch
        from chronos import ChronosPipeline
    except ImportError:
        return None
    pipe = ChronosPipeline.from_pretrained(
        MODEL_ID,
        device_map="cpu",
        dtype=torch.float32,
    )
    return pipe


def chronos_signals(
    pipe,
    closes: pd.Series,
    *,
    context: int = CONTEXT,
    stride: int = DECISION_STRIDE,
) -> pd.Series:
    """Boolean in-position series aligned to ``closes``.

    For each decision index ``i`` (every ``stride`` bars, carrying the last
    decision forward between updates), take the trailing ``context`` Closes
    ending at ``i``, forecast the next Close, and go long iff the mean of
    the sample paths exceeds ``closes[i]``. Bars with fewer than 2 history
    points stay cash.
    """
    import torch

    c = closes.astype(float)
    n = len(c)
    values = c.to_numpy()
    in_pos = np.zeros(n, dtype=bool)

    decision_idxs = list(range(0, n, stride))
    # Build contexts for bars that have at least 2 points.
    usable: list[int] = []
    contexts: list = []
    for i in decision_idxs:
        start = max(0, i + 1 - context)
        hist = values[start : i + 1]
        if len(hist) < 2:
            continue
        usable.append(i)
        contexts.append(torch.tensor(hist, dtype=torch.float32))

    preds: dict[int, float] = {}
    for b0 in range(0, len(contexts), BATCH_SIZE):
        batch = contexts[b0 : b0 + BATCH_SIZE]
        idxs = usable[b0 : b0 + BATCH_SIZE]
        samples = pipe.predict(batch, prediction_length=HORIZON, num_samples=NUM_SAMPLES)
        # samples: (batch, num_samples, horizon)
        means = samples[:, :, 0].mean(dim=1).detach().cpu().numpy()
        for i, m in zip(idxs, means):
            preds[i] = float(m)

    last = False
    next_decision = 0
    for i in range(n):
        if i in preds:
            last = preds[i] > values[i]
            next_decision = i
        elif stride > 1 and i > next_decision:
            # carry last decision between stride updates
            pass
        in_pos[i] = last
    return pd.Series(in_pos, index=closes.index, name="chronos")


def run_chronos_cell(pipe, df: pd.DataFrame, regime: str) -> float:
    warmed, win = rl.warm_window(df, regime)
    # Signals on the full warmed history so context reaches before the window;
    # simulate only on the regime slice (same pattern as run_strategy_cell).
    sig_full = chronos_signals(pipe, warmed["Close"])
    sig = sig_full.loc[win.index]
    return rl.total_return(rl._simulate_signal(win["Close"], sig, rl.INITIAL_CAPITAL))


def write_readme(per_cell: pd.DataFrame, summary: pd.DataFrame) -> None:
    lines = [
        "# Chronos-t5-tiny comparator (P1.10)",
        "",
        "Produced by `make chronos` (`scripts/chronos_comparator.py`).",
        "",
        "## What ran",
        "",
        f"- Model: `{MODEL_ID}` on CPU (`torch.float32`).",
        f"- Tickers: {', '.join(TICKERS)} from committed paper fixtures",
        "  (`regime_lib.load_prices` → `benchmark/prices/paper/`).",
        "- Regimes: all four in `regime_lib.REGIMES`.",
        f"- Context window: last {CONTEXT} Close bars at each decision.",
        f"- Forecast horizon: {HORIZON}; long iff mean of {NUM_SAMPLES} sample",
        "  paths for next Close exceeds current Close; else cash.",
        f"- Decision frequency: every {DECISION_STRIDE} trading bar(s)",
        "  (full bar coverage; Chronos-t5-tiny is cheap on CPU).",
        "- Position sizing: long-only binary, frictionless, execute at Close",
        "  (same harness as `regime_lib` / classical baselines).",
        "- Same cells also report `buy_and_hold` and `momentum` (Momentum(60))",
        "  via `regime_lib.run_strategy_cell`.",
        "",
        "## Honest limits",
        "",
        "- This is a **forecast→sign** trading rule, not an LLM agent and not",
        "  a claim of Chronos alpha. Zero costs understates real turnover.",
        "- Chronos was pretrained on broad public series; these B3 windows",
        "  may overlap pretraining corpora (no leakage probe here).",
        "- Tiny checkpoint only; results are a comparator cell, not an",
        "  exhaustive foundation-model study. Kronos remains future work.",
        "- Optional dependency: `[project.optional-dependencies] chronos`.",
        "  `CHRONOS_SKIP=1` or a missing install exits 0 without rewriting",
        "  committed CSVs.",
        "",
        "## `per_cell.csv`",
        "",
        "Columns: `agent`, `regime`, `ticker`, `total_return_pct`.",
        "Agents: `chronos`, `buy_and_hold`, `momentum`.",
        "",
        "## `summary.csv`",
        "",
        "Per (agent, ticker): mean / min / max CR (%) across the four regimes,",
        "plus mean CR gap vs buy_and_hold on matching cells (`gap_vs_bh_pct`).",
        "",
        "## Snapshot (this commit)",
        "",
        "```",
        summary.to_string(index=False),
        "```",
        "",
    ]
    README.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    pipe = _try_load_pipeline()
    if pipe is None:
        reason = (
            "CHRONOS_SKIP=1"
            if os.environ.get("CHRONOS_SKIP", "").strip() == "1"
            else "chronos package not installed (uv sync --extra chronos)"
        )
        return _skip(reason)

    price = {t: rl.load_prices(t) for t in TICKERS}
    bh = BuyAndHold()
    mom = MomentumStrategy()

    rows: list[dict] = []
    for ticker in TICKERS:
        df = price[ticker]
        for regime in rl.REGIME_ORDER:
            cr_c = run_chronos_cell(pipe, df, regime)
            cr_bh = rl.run_strategy_cell(bh, df, regime)
            cr_m = rl.run_strategy_cell(mom, df, regime)
            for agent, cr in (
                ("chronos", cr_c),
                ("buy_and_hold", cr_bh),
                ("momentum", cr_m),
            ):
                rows.append(
                    {
                        "agent": agent,
                        "regime": regime,
                        "ticker": ticker,
                        "total_return_pct": round(cr * 100.0, 4),
                    }
                )
            print(
                f"{ticker:6} {regime:14} chronos={cr_c*100:+7.2f}%  "
                f"bh={cr_bh*100:+7.2f}%  mom={cr_m*100:+7.2f}%"
            )

    per_cell = pd.DataFrame(rows)
    # Stable order: agent, regime, ticker
    agent_order = ["chronos", "buy_and_hold", "momentum"]
    per_cell["agent"] = pd.Categorical(per_cell["agent"], agent_order, ordered=True)
    per_cell["regime"] = pd.Categorical(per_cell["regime"], rl.REGIME_ORDER, ordered=True)
    per_cell = per_cell.sort_values(["agent", "regime", "ticker"]).reset_index(drop=True)
    per_cell["agent"] = per_cell["agent"].astype(str)
    per_cell["regime"] = per_cell["regime"].astype(str)

    bh_map = {
        (r.regime, r.ticker): r.total_return_pct
        for r in per_cell[per_cell["agent"] == "buy_and_hold"].itertuples()
    }
    summary_rows: list[dict] = []
    for agent in agent_order:
        sub = per_cell[per_cell["agent"] == agent]
        for ticker in TICKERS:
            s = sub[sub["ticker"] == ticker]["total_return_pct"]
            gaps = [
                float(sub[(sub["ticker"] == ticker) & (sub["regime"] == reg)]["total_return_pct"].iloc[0]
                      - bh_map[(reg, ticker)])
                for reg in rl.REGIME_ORDER
            ]
            summary_rows.append(
                {
                    "agent": agent,
                    "ticker": ticker,
                    "mean_cr_pct": round(float(s.mean()), 4),
                    "min_cr_pct": round(float(s.min()), 4),
                    "max_cr_pct": round(float(s.max()), 4),
                    "gap_vs_bh_pct": round(float(np.mean(gaps)), 4),
                }
            )
    summary = pd.DataFrame(summary_rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    per_cell.to_csv(PER_CELL, index=False)
    summary.to_csv(SUMMARY, index=False)
    write_readme(per_cell, summary)
    print(f"wrote {PER_CELL.relative_to(rl.REPO)}")
    print(f"wrote {SUMMARY.relative_to(rl.REPO)}")
    print(f"wrote {README.relative_to(rl.REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
