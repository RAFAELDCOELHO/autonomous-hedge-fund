#!/usr/bin/env python3
"""P1.6 - Reliability diagram from EXISTING local decision logs ($0, offline).

Normalises the two committed mistral:7b logs into one JSONL, joins each
decision with the next trading day's Close from the committed PETR4 fixture,
bins by stated confidence and writes mean-confidence vs realised win rate.

No LLM call, no download, no key. Stdlib only; the plot is a hand-written SVG.
Outcomes are computed, never invented: rows whose next-day Close is not in the
fixture are dropped and counted; HOLD rows get win=null (no directional bet).

Inputs (committed)
    determinism_results.json           6 runs x 1 prompt, with confidence
    brazilbench_mistral_results.json   6 runs x 80 days, signal only
    benchmark/prices/PETR4.csv         Date,Close (auto-adjusted)

Outputs (benchmark/results/reliability/, schema in SCHEMA.md there)
    decisions.jsonl   one row per logged decision
    bins.csv          one row per non-empty confidence bin
    reliability.svg   the diagram

Usage
    make reliability            (= uv run python scripts/reliability_diagram.py)
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DETERMINISM_JSON = REPO / "determinism_results.json"
MISTRAL_JSON = REPO / "brazilbench_mistral_results.json"
PRICES_CSV = REPO / "benchmark" / "prices" / "PETR4.csv"
OUT_DIR = REPO / "benchmark" / "results" / "reliability"
JSONL_PATH = OUT_DIR / "decisions.jsonl"
BINS_CSV = OUT_DIR / "bins.csv"
SVG_PATH = OUT_DIR / "reliability.svg"

# Hard-coded in the scripts that produced the logs (determinism_test.py,
# brazilbench_mistral_test.py); neither log stores them.
TICKER = "PETR4"
DETERMINISM_DATE = "2020-03-06"
MISTRAL_REGIME_START = "2020-02-03"  # first trading day >= this, 80 days
SIGNALS = ("BUY", "SELL", "HOLD")
N_BINS = 10
FIELDS = ("source", "run", "model", "ticker", "date", "date_inferred", "signal",
          "confidence", "next_date", "next_ret", "win")
BIN_FIELDS = ("bin_lo", "bin_hi", "n", "n_unique_prompts", "mean_confidence", "win_rate")


def load_prices() -> tuple[list[str], dict[str, float]]:
    with PRICES_CSV.open(newline="") as f:
        rows = [(r["Date"], float(r["Close"])) for r in csv.DictReader(f)]
    return [d for d, _ in rows], dict(rows)


def _runs(obj: dict, key: str):
    for i, run in enumerate(obj[key], 1):
        yield f"{key.split('_')[0]}_{i}", run


def raw_decisions() -> list[dict]:
    """Decisions before pricing; `date` is an ISO day or an offset (date_inferred)."""
    out = []
    for rec in json.loads(DETERMINISM_JSON.read_text()):
        for key in ("warm_outputs", "cold_outputs"):
            for run, text in _runs(rec, key):
                try:
                    d = json.loads(text)
                    sig, conf = d.get("signal"), float(d["confidence"])
                except (ValueError, KeyError, TypeError):
                    sig, conf = None, None
                out.append(dict(source=DETERMINISM_JSON.name, run=run, model=rec["model"],
                                date=DETERMINISM_DATE, date_inferred=False,
                                signal=sig, confidence=conf))
    m = json.loads(MISTRAL_JSON.read_text())
    for key in ("warm_runs", "cold_runs"):
        for run, sigs in _runs(m, key):
            for offset, sig in enumerate(sigs):
                out.append(dict(source=MISTRAL_JSON.name, run=run, model=m["model"],
                                date=offset, date_inferred=True, signal=sig, confidence=None))
    return out


def normalise() -> tuple[list[dict], dict[str, int]]:
    dates, close = load_prices()
    start = next(i for i, d in enumerate(dates) if d >= MISTRAL_REGIME_START)
    dropped = {"bad_signal": 0, "no_next_close": 0}
    rows = []
    for r in raw_decisions():
        if r["signal"] not in SIGNALS:
            dropped["bad_signal"] += 1
            continue
        date = dates[start + r["date"]] if r["date_inferred"] else r["date"]
        i = dates.index(date) if date in close else -1
        if i < 0 or i + 1 >= len(dates):
            dropped["no_next_close"] += 1
            continue
        nxt = dates[i + 1]
        ret = round(close[nxt] / close[date] - 1, 6)
        win = None if r["signal"] == "HOLD" else (ret > 0 if r["signal"] == "BUY" else ret < 0)
        rows.append({**r, "ticker": TICKER, "date": date, "next_date": nxt, "next_ret": ret, "win": win})
    rows.sort(key=lambda r: (r["source"], r["date"], r["run"]))
    return [{k: r[k] for k in FIELDS} for r in rows], dropped


def bin_stats(rows: list[dict]) -> list[dict]:
    scored = [r for r in rows if r["confidence"] is not None and r["win"] is not None]
    bins = []
    for b in range(N_BINS):
        lo, hi = b / N_BINS, (b + 1) / N_BINS
        last = b == N_BINS - 1  # closed on the right so confidence=1.0 lands in [0.9, 1.0]
        members = [r for r in scored if lo <= r["confidence"] < hi or (last and r["confidence"] == 1.0)]
        if not members:
            continue
        bins.append(dict(
            bin_lo=lo, bin_hi=hi, n=len(members),
            n_unique_prompts=len({(r["ticker"], r["date"]) for r in members}),
            mean_confidence=round(sum(r["confidence"] for r in members) / len(members), 4),
            win_rate=round(sum(r["win"] for r in members) / len(members), 4),
        ))
    return bins


def ece(bins: list[dict]) -> float | None:
    n = sum(b["n"] for b in bins)
    return None if not n else round(sum(b["n"] / n * abs(b["mean_confidence"] - b["win_rate"]) for b in bins), 4)


def svg(bins: list[dict]) -> str:
    # ponytail: hand-rolled SVG, matplotlib is not in uv.lock; swap when it is
    s, m = 300, 40

    def x(v):
        return m + v * s

    def y(v):
        return m + s - v * s

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{s + 2 * m}" height="{s + 2 * m}" font-family="sans-serif" font-size="11">',
             f'<rect x="{m}" y="{m}" width="{s}" height="{s}" fill="none" stroke="#999"/>',
             f'<line x1="{x(0)}" y1="{y(0)}" x2="{x(1)}" y2="{y(1)}" stroke="#bbb" stroke-dasharray="4"/>']
    for t in (0, 0.5, 1):
        parts.append(f'<text x="{x(t)}" y="{y(0) + 14}" text-anchor="middle">{t}</text>')
        parts.append(f'<text x="{x(0) - 4}" y="{y(t) + 4}" text-anchor="end">{t}</text>')
    for b in bins:
        cx, cy = x(b["mean_confidence"]), y(b["win_rate"])
        parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="5" fill="#1f77b4"/>')
        parts.append(f'<text x="{cx + 8:.1f}" y="{cy + 4:.1f}">n={b["n"]} (unique={b["n_unique_prompts"]})</text>')
    parts.append(f'<text x="{x(0.5)}" y="{s + 2 * m - 6}" text-anchor="middle">mean stated confidence</text>')
    parts.append(f'<text x="12" y="{y(0.5)}" transform="rotate(-90 12 {y(0.5)})" text-anchor="middle">realised next-day win rate</text>')
    return "\n".join(parts) + "\n</svg>\n"


def bins_csv(bins: list[dict]) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=BIN_FIELDS, lineterminator="\n")
    w.writeheader()
    w.writerows(bins)
    return buf.getvalue()


def main() -> int:
    rows, dropped = normalise()
    bins = bin_stats(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    JSONL_PATH.write_text("".join(json.dumps(r) + "\n" for r in rows))
    BINS_CSV.write_text(bins_csv(bins))
    SVG_PATH.write_text(svg(bins))

    scored = [r for r in rows if r["confidence"] is not None]
    unscored = [r for r in rows if r["confidence"] is None and r["win"] is not None]
    print(f"rows: {len(rows)}  with_confidence: {len(scored)}  "
          f"unique_prompts_with_confidence: {len({(r['ticker'], r['date']) for r in scored})}  dropped: {dropped}")
    print(f"{'bin':>11} {'n':>4} {'uniq':>4} {'mean_conf':>9} {'win_rate':>8}")
    for b in bins:
        print(f"[{b['bin_lo']:.1f},{b['bin_hi']:.1f}) {b['n']:>4} {b['n_unique_prompts']:>4} {b['mean_confidence']:>9.3f} {b['win_rate']:>8.3f}")
    print(f"ECE: {ece(bins)}  (weighted |mean_conf - win_rate|; n counts replicate rows, not independent decisions)")
    if unscored:
        buys = [r for r in unscored if r["signal"] == "BUY"]
        print(f"reference, signal-only rows (no confidence, not binned): BUY next-day win rate "
              f"{sum(r['win'] for r in buys)}/{len(buys)} = {sum(r['win'] for r in buys) / len(buys):.3f}")
    print(f"wrote {JSONL_PATH}, {BINS_CSV}, {SVG_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
