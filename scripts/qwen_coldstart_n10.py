#!/usr/bin/env python3
"""P1.5 - Qwen 2.5-7B, N>=10 independent COLD-START sessions, mean +/- std ($0).

Protocol (same as determinism_test.py, phase 2): between every run the local
Ollama server is killed and restarted with OLLAMA_KEEP_ALIVE=0, so each call
loads the model from disk into a fresh process (no KV cache, no runner reuse).
We assert cold-ness before each call via /api/ps and record load_duration.

Prompt: BrazilBench PETR4/crisis_2020 decision prompt built from the committed
fixture benchmark/prices/PETR4.csv (Date, auto-adjusted Close). No download.
The fixture has no volume, so the "volume z-score" line of the original prompt
is omitted; macro constants are the ones hard-coded in brazilbench_mistral_test.py.

Outputs (benchmark/results/qwen_n10/)
    runs.jsonl     one row per cold-start run
    summary.csv    one row per prompt date: n, agreement, confidence mean/std
    summary.json   same + run metadata (model digest, prompt hashes, options)

Usage
    make qwen-n10                       (= uv run python scripts/qwen_coldstart_n10.py)
    python scripts/qwen_coldstart_n10.py --n 10 --dates 2020-03-06 2020-03-18

Needs: `ollama` on PATH with qwen2.5:7b pulled. No key, no paid API.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import statistics
import subprocess
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PRICES_CSV = REPO / "benchmark" / "prices" / "PETR4.csv"
OUT_DIR = REPO / "benchmark" / "results" / "qwen_n10"
OLLAMA = "http://localhost:11434"

MODEL = "qwen2.5:7b"
TICKER = "PETR4"
# Critical crisis_2020 cells: the determinism_test.py date, the crash low, the rebound.
DEFAULT_DATES = ("2020-03-06", "2020-03-18", "2020-04-30")
MACRO = {"selic": 4.25, "ipca": 0.42, "usdbrl": 4.87}  # as in brazilbench_mistral_test.py
OPTIONS = {"temperature": 0}
SIGNALS = ("BUY", "SELL", "HOLD")

SYSTEM_PROMPT = """You are a quantitative trading analyst evaluating Brazilian and US equities on behalf of an institutional portfolio.

For each market state, output EXACTLY this JSON and nothing else (no preamble, no markdown fences):
{"signal": "BUY"|"SELL"|"HOLD", "confidence": 0.0-1.0, "reasoning": "<= 80 words>"}

=== DECISION RULES ===
- BUY only when you expect positive risk-adjusted return over ~20 trading days relative to SELIC.
- SELL when: (R1) unrealized return < -8% from entry price, OR (R2) position is up > +15% AND 20d return has turned negative.
- HOLD is the default. Prefer HOLD under high uncertainty.
- For BR tickers: SELIC > 12% historically pressures equities; IPCA > 6% signals inflation regime; USD/BRL > 5.50 adds FX risk.
- Never base decisions on news outside the provided data.
- Output ONLY the JSON object."""

RUN_FIELDS = ("date", "run", "model", "cold", "signal", "confidence", "reasoning",
              "response_hash", "load_duration_s", "total_duration_s", "raw")
SUMMARY_FIELDS = ("date", "n", "n_parsed", "n_unique_responses", "majority_signal", "agreement",
                  "n_buy", "n_sell", "n_hold", "confidence_mean", "confidence_std",
                  "load_duration_mean_s", "total_duration_mean_s")


# ── prompt from fixture ──────────────────────────────────────────────────────
def load_prices() -> tuple[list[str], list[float]]:
    with PRICES_CSV.open(newline="") as f:
        rows = [(r["Date"], float(r["Close"])) for r in csv.DictReader(f)]
    return [d for d, _ in rows], [c for _, c in rows]


def build_prompt(dates: list[str], closes: list[float], date: str) -> str:
    """BrazilBench user prompt for `date` (brazilbench_mistral_test.build_prompt minus volume)."""
    idx = dates.index(date)
    lk20, lk60 = closes[idx - 20:idx], closes[idx - 60:idx]
    close = closes[idx]
    dr = [b / a - 1 for a, b in zip(lk20[:-1], lk20[1:])]
    vol_20d = statistics.pstdev(dr) * 252 ** 0.5 * 100
    return f"""TICKER: {TICKER} (Brazil/B3) DATE: {date}

RECENT PRICES (last 20 trading days, oldest to newest):
{", ".join(f"{p:.2f}" for p in lk20)}

PRICE STATISTICS:
- close: {close:.2f}
- 20d return: {(close / lk20[0] - 1) * 100:.1f}% | 60d return: {(close / lk60[0] - 1) * 100:.1f}%
- 20d realized vol (annualized): {vol_20d:.1f}%
- 20d high/low: {max(lk20):.2f} / {min(lk20):.2f}

POSITION: None (flat)

MACRO CONTEXT:
- SELIC: {MACRO['selic']}%
- IPCA (monthly, last): {MACRO['ipca']}%
- USD/BRL: {MACRO['usdbrl']}

Output the JSON decision now."""


# ── ollama plumbing ──────────────────────────────────────────────────────────
def _get(path: str, timeout: float = 2) -> dict:
    with urllib.request.urlopen(f"{OLLAMA}{path}", timeout=timeout) as r:
        return json.loads(r.read())


def ollama_up() -> bool:
    try:
        _get("/api/tags")
        return True
    except Exception:
        return False


def model_available(model: str = MODEL) -> bool:
    return ollama_up() and any(m["name"] == model for m in _get("/api/tags")["models"])


def loaded_models() -> list[str]:
    return [m["name"] for m in _get("/api/ps")["models"]]


def cold_restart() -> None:
    """Kill every ollama process, start a fresh server with KEEP_ALIVE=0, wait for it."""
    subprocess.run(["pkill", "ollama"], capture_output=True)
    time.sleep(3)
    # If the desktop app respawns its own server, this bind fails; either way the
    # runner is a new process (asserted via /api/ps before each call).
    subprocess.Popen(["ollama", "serve"], env={**os.environ, "OLLAMA_KEEP_ALIVE": "0"},
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(30):
        if ollama_up():
            return
        time.sleep(1)
    raise RuntimeError("ollama did not come back after restart")


def call_ollama(user_prompt: str, model: str = MODEL) -> dict:
    payload = json.dumps({"model": model, "system": SYSTEM_PROMPT, "prompt": user_prompt,
                          "options": OPTIONS, "stream": False}).encode()
    req = urllib.request.Request(f"{OLLAMA}/api/generate", data=payload,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())


def parse_decision(text: str) -> dict:
    """{'signal','confidence','reasoning'} or all-None if not the expected JSON."""
    t = text.strip()
    if t.startswith("```"):  # tolerate fenced output; the prompt forbids it but log it anyway
        t = t.strip("`").removeprefix("json").strip()
    try:
        d = json.loads(t)
        sig = d.get("signal")
        conf = float(d["confidence"])
        if sig not in SIGNALS or not 0.0 <= conf <= 1.0:
            raise ValueError
        return {"signal": sig, "confidence": conf, "reasoning": d.get("reasoning")}
    except (ValueError, KeyError, TypeError, AttributeError):
        return {"signal": None, "confidence": None, "reasoning": None}


# ── stats ────────────────────────────────────────────────────────────────────
def _mean_std(xs: list[float]) -> tuple[float | None, float | None]:
    if not xs:
        return None, None
    return round(statistics.mean(xs), 4), (round(statistics.stdev(xs), 4) if len(xs) > 1 else None)


def summarise(date: str, rows: list[dict]) -> dict:
    parsed = [r for r in rows if r["signal"] is not None]
    counts = {s: sum(r["signal"] == s for r in parsed) for s in SIGNALS}
    majority = max(counts, key=counts.get) if parsed else None
    c_mean, c_std = _mean_std([r["confidence"] for r in parsed])
    return dict(
        date=date, n=len(rows), n_parsed=len(parsed),
        n_unique_responses=len({r["response_hash"] for r in rows}),
        majority_signal=majority,
        agreement=round(counts[majority] / len(rows), 4) if parsed else None,
        n_buy=counts["BUY"], n_sell=counts["SELL"], n_hold=counts["HOLD"],
        confidence_mean=c_mean, confidence_std=c_std,
        load_duration_mean_s=_mean_std([r["load_duration_s"] for r in rows])[0],
        total_duration_mean_s=_mean_std([r["total_duration_s"] for r in rows])[0],
    )


def to_csv(rows: list[dict], fields: tuple[str, ...]) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=fields, lineterminator="\n")
    w.writeheader()
    w.writerows({k: r[k] for k in fields} for r in rows)
    return buf.getvalue()


# ── main ─────────────────────────────────────────────────────────────────────
def run(n: int, dates: list[str], out_dir: Path = OUT_DIR) -> list[dict]:
    if not model_available():
        raise SystemExit(f"{MODEL} not available at {OLLAMA}; run `ollama pull {MODEL}`")
    digest = next(m["digest"] for m in _get("/api/tags")["models"] if m["name"] == MODEL)
    all_dates, closes = load_prices()
    prompts = {d: build_prompt(all_dates, closes, d) for d in dates}

    rows: list[dict] = []
    for date in dates:
        print(f"\n{date}  close={closes[all_dates.index(date)]:.2f}  N={n} cold starts")
        for i in range(1, n + 1):
            cold_restart()
            still = loaded_models()
            if still:  # never silently report a warm run as cold
                raise RuntimeError(f"model still loaded after restart: {still}")
            resp = call_ollama(prompts[date])
            text = resp.get("response", "")
            dec = parse_decision(text)
            row = dict(date=date, run=i, model=MODEL, cold=True, **dec,
                       response_hash=hashlib.sha256(text.encode()).hexdigest()[:12],
                       load_duration_s=round(resp.get("load_duration", 0) / 1e9, 3),
                       total_duration_s=round(resp.get("total_duration", 0) / 1e9, 3),
                       raw=text)
            rows.append(row)
            print(f"  run {i:>2}/{n}  load={row['load_duration_s']:5.1f}s  total={row['total_duration_s']:5.1f}s  "
                  f"{dec['signal']} conf={dec['confidence']}  hash={row['response_hash']}")

    summary = [summarise(d, [r for r in rows if r["date"] == d]) for d in dates]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "runs.jsonl").write_text("".join(json.dumps({k: r[k] for k in RUN_FIELDS}) + "\n" for r in rows))
    (out_dir / "summary.csv").write_text(to_csv(summary, SUMMARY_FIELDS))
    (out_dir / "summary.json").write_text(json.dumps({
        "model": MODEL, "model_digest": digest, "options": OPTIONS, "protocol": "cold-start",
        "cold_start": "pkill ollama; ollama serve (OLLAMA_KEEP_ALIVE=0); assert /api/ps empty; one call",
        "ticker": TICKER, "regime": "crisis_2020", "n_per_date": n, "macro": MACRO,
        "fixture": str(PRICES_CSV.relative_to(REPO)), "fixture_note": "auto-adjusted Close; no volume line in prompt",
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        "user_prompt_sha256": {d: hashlib.sha256(p.encode()).hexdigest() for d, p in prompts.items()},
        "std": "sample std (n-1); null when n_parsed < 2",
        "per_date": summary,
    }, indent=2) + "\n")

    print(f"\n{'date':<10} {'n':>3} {'parsed':>6} {'uniq':>4} {'majority':>8} {'agree':>6} {'conf mean':>9} {'conf std':>8}")
    for s in summary:
        print(f"{s['date']:<10} {s['n']:>3} {s['n_parsed']:>6} {s['n_unique_responses']:>4} {str(s['majority_signal']):>8} "
              f"{s['agreement']!s:>6} {s['confidence_mean']!s:>9} {s['confidence_std']!s:>8}")
    print(f"wrote {out_dir}/runs.jsonl, summary.csv, summary.json")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=10, help="cold-start sessions per date (default 10)")
    ap.add_argument("--dates", nargs="+", default=list(DEFAULT_DATES), help="PETR4 fixture dates in crisis_2020")
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    a = ap.parse_args()
    run(a.n, a.dates, a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
