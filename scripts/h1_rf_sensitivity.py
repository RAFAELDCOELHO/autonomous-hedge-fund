"""EXPLORATORY rf sensitivity of the H1 contrast (docs/PREREGISTRATION.md §7).

Not confirmatory, descriptive only: no p-values, permutations or significance
claims. Loads rows with scripts/h1_stats.py (unchanged), replaces each row's
``sharpe`` with its ``sharpe_flat`` (the flat FLAT_RF_SENSITIVITY rate instead of
the pre-registered daily rf), applies h1_stats' exclusions and reports
per-ticker deltas and D = mean delta(BR sensitive) - mean delta(US).

Usage::

    uv run python scripts/h1_rf_sensitivity.py cells.csv [--out result.json]
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np

from tradingagents.backtest.cells import SHARPE_FLAT_FIELD
from tradingagents.backtest.report import SHARPE_FLAT_COL

_spec = importlib.util.spec_from_file_location("h1_stats", Path(__file__).with_name("h1_stats.py"))
h1_stats = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(h1_stats)

BANNER = (
    f"EXPLORATORY (PREREGISTRATION §7, not confirmatory; no significance claims): "
    f"H1 contrast with sharpe = {SHARPE_FLAT_COL}"
)


def load_flat_rows(path: Path) -> list[dict]:
    """h1_stats.load_cells rows with ``sharpe`` replaced by ``sharpe_flat``."""
    rows = h1_stats.load_cells(path)
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if SHARPE_FLAT_FIELD not in (reader.fieldnames or []):
            raise ValueError(f"missing column {SHARPE_FLAT_FIELD!r}")
        raw = list(reader)
    for line, (row, r) in enumerate(zip(rows, raw), start=2):
        text = (r[SHARPE_FLAT_FIELD] or "").strip()
        try:
            row["sharpe"] = float(text) if text else math.nan
        except ValueError:
            raise ValueError(f"line {line}: {SHARPE_FLAT_FIELD}={text!r} is not a number") from None
    return rows


def describe(rows: list[dict]) -> dict:
    """Exclusions, per-ticker deltas and D via h1_stats' pure helpers (no permutations)."""
    kept, excluded = h1_stats.apply_exclusions(rows)
    deltas = {t: h1_stats._delta(a) for t, a in kept.items()}
    per_ticker = [
        {
            "ticker": t,
            "market": h1_stats.TICKERS[t],
            "role": "control" if t in h1_stats.BR_CONTROL else "primary",
            "n_absent": len(kept[t]["absent"]),
            "n_present": len(kept[t]["present"]),
            "mean_sharpe_absent": float(np.mean([x["sharpe"] for x in kept[t]["absent"]])),
            "mean_sharpe_present": float(np.mean([x["sharpe"] for x in kept[t]["present"]])),
            "delta_sharpe": deltas[t],
        }
        for t in h1_stats.TICKERS if t in kept
    ]
    counts = {}
    for e in excluded:
        counts[e["reason"]] = counts.get(e["reason"], 0) + 1
    br = [deltas[t] for t in h1_stats.BR_SENSITIVE if t in deltas]
    us = [deltas[t] for t in h1_stats.US_TICKERS if t in deltas]
    return {
        "exploratory": BANNER,
        "n_rows": len(rows),
        "n_valid_runs": sum(len(v) for a in kept.values() for v in a.values()),
        "exclusion_counts": counts,
        "excluded": excluded,
        "per_ticker": per_ticker,
        **({"D": float(np.mean(br) - np.mean(us))} if br and us
           else {"not_evaluable": "no BR-sensitive or no US ticker left"}),
    }


def report(result: dict) -> str:
    lines = [
        BANNER,
        f"rows={result['n_rows']} valid_runs={result['n_valid_runs']} exclusions={result['exclusion_counts'] or 'none'}",
        "",
        f"{'ticker':<8}{'mkt':<4}{'n_abs':>6}{'n_pre':>6}{'sharpe_abs':>12}{'sharpe_pre':>12}{'delta':>10}",
    ]
    for r in result["per_ticker"]:
        tag = " (control)" if r["role"] == "control" else ""
        lines.append(
            f"{r['ticker']:<8}{r['market']:<4}{r['n_absent']:>6}{r['n_present']:>6}"
            f"{r['mean_sharpe_absent']:>12.4f}{r['mean_sharpe_present']:>12.4f}{r['delta_sharpe']:>+10.4f}{tag}"
        )
    lines.append("")
    if "D" in result:
        lines.append(f"D (flat rf, exploratory) = mean dSharpe(BR sensitive) - mean dSharpe(US) = {result['D']:+.4f}")
    else:
        lines.append(f"D (flat rf, exploratory) not evaluable: {result['not_evaluable']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=BANNER)
    ap.add_argument("cells", type=Path, help="cells.csv with a sharpe_flat column")
    ap.add_argument("--out", type=Path, help="also write this descriptive result as JSON here")
    args = ap.parse_args(argv)
    try:
        rows = load_flat_rows(args.cells)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    result = describe(rows)
    print(report(result))
    if args.out:
        args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
