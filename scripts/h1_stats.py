"""Pre-registered H1 test (docs/PREREGISTRATION.md). Offline, numpy + stdlib.

Usage::

    uv run python scripts/h1_stats.py cells.csv [--out h1_result.json]

Input: ``cells.csv``, one row per backtest run of the 2x2 factorial
(ticker x macro arm x seed). Required columns (extra columns are ignored):

    ticker             one of TICKERS below (B3 names without ``.SA``)
    market             ``US`` or ``BR``; must match the ticker
    arm                ``absent`` (4-analyst baseline) or ``present`` (+ Macro Agent)
    seed               replicate index, integer >= 0
    status             ``ok`` or ``failed`` (run aborted / did not finish)
    n_days             trading days in the equity curve, integer >= 0
    n_decision_errors  decisions that fell back to HOLD on error, integer >= 0
    sharpe             annualized excess-return Sharpe (may be empty if failed)
    rf_source          ``BCB-SGS-12`` for BR rows, ``FRED-DTB3`` for US rows

(ticker, arm, seed) must be unique. A schema violation raises ``ValueError``
(exit code 2 from the CLI); nothing is silently coerced.

Exclusion rules, applied in this order and all reported:
    1. status == failed                              -> ``failed``
    2. sharpe missing / non-finite                   -> ``missing_sharpe``
    3. n_days below the ticker's max n_days          -> ``truncated``
    4. n_decision_errors / n_days > MAX_ERROR_RATE   -> ``decision_errors``
    5. a ticker with < MIN_VALID_SEEDS valid runs in either arm is dropped
       from every analysis (both arms)               -> ``ticker_dropped``

Tests (per-ticker delta_t = mean Sharpe(present) - mean Sharpe(absent)):
    primary    D = mean(delta, BR macro-sensitive) - mean(delta, US), one-sided,
               exact permutation of market labels across tickers.
    secondary  S1: mean(delta, BR macro-sensitive) > 0, one-sided;
               S2: mean(delta, US) != 0, two-sided; both by within-ticker
               permutation of arm labels (Monte Carlo, fixed seed), Holm-adjusted.
RADL3 (control) enters only the descriptive per-ticker table.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np

ALPHA = 0.05
MIN_VALID_SEEDS = 3
MAX_ERROR_RATE = 0.05
N_RESAMPLES = 10_000
RNG_SEED = 20260925
EPS = 1e-12

US_TICKERS = ("AAPL", "GOOGL", "AMZN")
BR_SENSITIVE = ("ITUB4", "BPAC11", "PETR4", "VALE3", "WEGE3")
BR_CONTROL = ("RADL3",)
TICKERS = {t: "US" for t in US_TICKERS} | {t: "BR" for t in BR_SENSITIVE + BR_CONTROL}
RF_SOURCE = {"BR": "BCB-SGS-12", "US": "FRED-DTB3"}
ARMS = ("absent", "present")
STATUSES = ("ok", "failed")
COLUMNS = (
    "ticker", "market", "arm", "seed", "status",
    "n_days", "n_decision_errors", "sharpe", "rf_source",
)


def _int(value: str, field: str, line: int) -> int:
    try:
        out = int(value)
    except ValueError:
        raise ValueError(f"line {line}: {field}={value!r} is not an integer") from None
    if out < 0:
        raise ValueError(f"line {line}: {field}={value!r} must be >= 0")
    return out


def load_cells(path: Path) -> list[dict]:
    """Read and validate cells.csv; return typed rows."""
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        missing = [c for c in COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"missing required columns: {missing}")
        raw = list(reader)
    if not raw:
        raise ValueError("cells.csv has no data rows")

    rows, seen = [], set()
    for line, r in enumerate(raw, start=2):
        ticker, market, arm, status = r["ticker"], r["market"], r["arm"], r["status"]
        if ticker not in TICKERS:
            raise ValueError(f"line {line}: ticker {ticker!r} is not pre-registered")
        if market != TICKERS[ticker]:
            raise ValueError(f"line {line}: {ticker} must have market={TICKERS[ticker]}, got {market!r}")
        if arm not in ARMS:
            raise ValueError(f"line {line}: arm must be one of {ARMS}, got {arm!r}")
        if status not in STATUSES:
            raise ValueError(f"line {line}: status must be one of {STATUSES}, got {status!r}")
        if r["rf_source"] != RF_SOURCE[market]:
            raise ValueError(f"line {line}: {market} rows need rf_source={RF_SOURCE[market]}, got {r['rf_source']!r}")
        seed = _int(r["seed"], "seed", line)
        key = (ticker, arm, seed)
        if key in seen:
            raise ValueError(f"line {line}: duplicate (ticker, arm, seed) {key}")
        seen.add(key)
        sharpe_txt = (r["sharpe"] or "").strip()
        try:
            sharpe = float(sharpe_txt) if sharpe_txt else math.nan
        except ValueError:
            raise ValueError(f"line {line}: sharpe={sharpe_txt!r} is not a number") from None
        rows.append({
            "ticker": ticker, "market": market, "arm": arm, "seed": seed, "status": status,
            "n_days": _int(r["n_days"], "n_days", line),
            "n_decision_errors": _int(r["n_decision_errors"], "n_decision_errors", line),
            "sharpe": sharpe,
        })
    return rows


def apply_exclusions(rows: list[dict]) -> tuple[dict, list[dict]]:
    """Return ({ticker: {arm: [sharpe, ...]}}, excluded rows with reasons)."""
    excluded = []
    max_days = {}
    for r in rows:
        if r["status"] == "ok":
            max_days[r["ticker"]] = max(max_days.get(r["ticker"], 0), r["n_days"])

    valid = {}
    for r in rows:
        if r["status"] == "failed":
            reason = "failed"
        elif not math.isfinite(r["sharpe"]):
            reason = "missing_sharpe"
        elif r["n_days"] < max_days[r["ticker"]]:
            reason = "truncated"
        elif r["n_days"] == 0 or r["n_decision_errors"] / r["n_days"] > MAX_ERROR_RATE:
            reason = "decision_errors"
        else:
            valid.setdefault(r["ticker"], {a: [] for a in ARMS})[r["arm"]].append(r["sharpe"])
            continue
        excluded.append({"ticker": r["ticker"], "arm": r["arm"], "seed": r["seed"], "reason": reason})

    kept = {}
    for ticker in TICKERS:
        arms = valid.get(ticker, {a: [] for a in ARMS})
        if min(len(arms[a]) for a in ARMS) < MIN_VALID_SEEDS:
            excluded.append({"ticker": ticker, "arm": None, "seed": None, "reason": "ticker_dropped"})
        else:
            kept[ticker] = arms
    return kept, excluded


def _delta(arms: dict) -> float:
    return float(np.mean(arms["present"]) - np.mean(arms["absent"]))


def primary_test(deltas: dict) -> dict:
    """Exact one-sided permutation of market labels over BR-sensitive + US tickers."""
    br = [t for t in BR_SENSITIVE if t in deltas]
    us = [t for t in US_TICKERS if t in deltas]
    if not br or not us:
        return {"evaluable": False, "reason": "no BR-sensitive or no US ticker left", "reject_h0": False}
    pool = np.array([deltas[t] for t in br + us])
    n_us = len(us)
    total = pool.sum()

    def stat(us_idx: tuple[int, ...]) -> float:
        us_sum = pool[list(us_idx)].sum()
        return float((total - us_sum) / (len(pool) - n_us) - us_sum / n_us)

    observed = stat(tuple(range(len(br), len(pool))))
    perms = [stat(c) for c in itertools.combinations(range(len(pool)), n_us)]
    p = sum(s >= observed - EPS for s in perms) / len(perms)
    return {
        "evaluable": True,
        "statistic": observed,
        "p_value": p,
        "n_permutations": len(perms),
        "min_attainable_p": 1 / len(perms),
        "br_tickers": br,
        "us_tickers": us,
        "reject_h0": p <= ALPHA,
    }


def _within_ticker_perm(kept: dict, tickers: list[str], rng, two_sided: bool) -> dict:
    """Mean delta over ``tickers``; arm labels shuffled within each ticker."""
    observed = float(np.mean([_delta(kept[t]) for t in tickers]))
    pools = [(np.array(kept[t]["absent"] + kept[t]["present"]), len(kept[t]["present"])) for t in tickers]
    null = np.empty(N_RESAMPLES)
    for b in range(N_RESAMPLES):
        ds = []
        for pool, n_present in pools:
            perm = rng.permutation(pool)
            ds.append(perm[:n_present].mean() - perm[n_present:].mean())
        null[b] = np.mean(ds)
    hits = (np.abs(null) >= abs(observed) - EPS) if two_sided else (null >= observed - EPS)
    return {"statistic": observed, "p_value": float((1 + hits.sum()) / (N_RESAMPLES + 1))}


def holm(p_values: list[float]) -> list[float]:
    """Holm step-down adjusted p-values (same order as input)."""
    order = sorted(range(len(p_values)), key=lambda i: p_values[i])
    adjusted, running = [0.0] * len(p_values), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p_values) - rank) * p_values[i]))
        adjusted[i] = running
    return adjusted


def analyze(rows: list[dict]) -> dict:
    kept, excluded = apply_exclusions(rows)
    deltas = {t: _delta(a) for t, a in kept.items()}
    per_ticker = [
        {
            "ticker": t,
            "market": TICKERS[t],
            "role": "control" if t in BR_CONTROL else "primary",
            "n_absent": len(kept[t]["absent"]),
            "n_present": len(kept[t]["present"]),
            "mean_sharpe_absent": float(np.mean(kept[t]["absent"])),
            "mean_sharpe_present": float(np.mean(kept[t]["present"])),
            "delta_sharpe": deltas[t],
        }
        for t in TICKERS if t in kept
    ]

    rng = np.random.default_rng(RNG_SEED)
    secondary = []
    for name, tickers, two_sided in (
        ("S1_br_sensitive_delta_gt_0", [t for t in BR_SENSITIVE if t in kept], False),
        ("S2_us_delta_ne_0", [t for t in US_TICKERS if t in kept], True),
    ):
        if tickers:
            secondary.append({"name": name, "tickers": tickers, **_within_ticker_perm(kept, tickers, rng, two_sided)})
        else:
            secondary.append({"name": name, "tickers": [], "statistic": None, "p_value": None})
    testable = [s for s in secondary if s["p_value"] is not None]
    for s, p_adj in zip(testable, holm([s["p_value"] for s in testable])):
        s["p_holm"] = p_adj
        s["reject_h0"] = p_adj <= ALPHA

    reasons = {}
    for e in excluded:
        reasons[e["reason"]] = reasons.get(e["reason"], 0) + 1
    return {
        "preregistration": "docs/PREREGISTRATION.md",
        "alpha": ALPHA,
        "n_rows": len(rows),
        "n_valid_runs": sum(len(v) for a in kept.values() for v in a.values()),
        "exclusion_counts": reasons,
        "excluded": excluded,
        "per_ticker": per_ticker,
        "primary": primary_test(deltas),
        "secondary": secondary,
        "rng_seed": RNG_SEED,
        "n_resamples": N_RESAMPLES,
    }


def report(result: dict) -> str:
    lines = [
        f"H1 pre-registered analysis ({result['preregistration']}), alpha={result['alpha']}",
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
    p = result["primary"]
    lines.append("")
    if p["evaluable"]:
        lines.append(
            f"PRIMARY D = mean dSharpe(BR sensitive) - mean dSharpe(US) = {p['statistic']:+.4f}; "
            f"exact one-sided permutation p = {p['p_value']:.4f} "
            f"({p['n_permutations']} relabelings, min attainable p = {p['min_attainable_p']:.4f}) -> "
            f"{'REJECT H0' if p['reject_h0'] else 'do not reject H0'}"
        )
    else:
        lines.append(f"PRIMARY not evaluable: {p['reason']} -> do not reject H0")
    for s in result["secondary"]:
        if s["p_value"] is None:
            lines.append(f"{s['name']}: not evaluable (no tickers left)")
        else:
            lines.append(
                f"{s['name']}: stat = {s['statistic']:+.4f}, p = {s['p_value']:.4f}, "
                f"Holm p = {s['p_holm']:.4f} -> {'reject' if s['reject_h0'] else 'do not reject'}"
            )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Pre-registered H1 test on a factorial cells.csv.")
    ap.add_argument("cells", type=Path, help="cells.csv (schema in this module's docstring)")
    ap.add_argument("--out", type=Path, help="also write the full result as JSON here")
    args = ap.parse_args(argv)
    try:
        result = analyze(load_cells(args.cells))
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(report(result))
    if args.out:
        args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
