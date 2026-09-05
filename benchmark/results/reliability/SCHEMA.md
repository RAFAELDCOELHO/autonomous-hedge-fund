# Reliability-diagram inputs (P1.6, $0 path)

Produced by `make reliability` (`scripts/reliability_diagram.py`) from the two
committed mistral:7b logs at the repo root and `benchmark/prices/PETR4.csv`.
Offline: no LLM call, no download, no key. Regenerate and `git status` must be clean.

Related contract (when present on the checked-out branch):
`benchmark/results/SCHEMA.md` documents repository-wide `benchmark/results/`
artifacts. This file stays reliability-specific.

## `decisions.jsonl` — one row per logged decision

| field | type | meaning |
|---|---|---|
| `source` | str | `determinism_results.json` or `brazilbench_mistral_results.json` |
| `run` | str | `warm_1..3` / `cold_1..3` (server persistent / restarted between runs) |
| `model` | str | `mistral:7b`, temperature 0 |
| `ticker` | str | `PETR4` (the only ticker in either log) |
| `date` | ISO day | decision day t; the prompt contained the Close of t |
| `date_inferred` | bool | `false`: date is in the producing script. `true`: the 80-day log stored no dates; day = fixture trading calendar at offset from first day >= 2020-02-03 |
| `signal` | `BUY`/`SELL`/`HOLD` | as emitted |
| `confidence` | float or null | as emitted; null when the log has no confidence field |
| `next_date` | ISO day | trading day t+1 in the fixture |
| `next_ret` | float | `Close[t+1] / Close[t] - 1` from the fixture (auto-adjusted) |
| `win` | bool or null | BUY: `next_ret > 0`; SELL: `next_ret < 0`; HOLD: null |

### Row-acceptance contract (`decisions.jsonl`)

Rows are accepted only when all of the following hold:

1. `signal` is exactly one of `BUY` / `SELL` / `HOLD`.
2. `date` resolves to a trading day present in `benchmark/prices/PETR4.csv`.
3. A next trading day (`next_date`) exists in that fixture for the same row.

Rows failing any of (1), (2), or (3) are dropped and reported on stdout by
`scripts/reliability_diagram.py` (`bad_signal` or `no_next_close` buckets).
The exact counts are input-dependent because they follow the currently
committed logs; treat script stdout as the source of truth for accepted vs
dropped rows.

### Derived-field caveats

- `next_ret` is computed from fixture closes as `Close[t+1] / Close[t] - 1`,
  never copied from logs.
- `win` is directional and intentionally asymmetric:
  - BUY: `next_ret > 0`
  - SELL: `next_ret < 0`
  - HOLD: always `null` (excluded from calibration bins)
- `date_inferred=true` means the log lacked explicit dates and alignment used
  the fixture calendar offset from the first trading day >= `2020-02-03`.

## `bins.csv` — one row per non-empty confidence bin

Only rows with non-null `confidence` and non-null `win` are binned.

Bin construction contract:

- 10 confidence bins, width `0.1`.
- Last bin is right-closed at `1.0` (so confidence `1.0` is in-range).
- Empty bins are omitted from `bins.csv` (file is sparse by design).

Interpretation contract:

- `n` counts rows and includes deterministic repeats of the same prompt.
- `n_unique_prompts` counts distinct `(ticker, date)` prompts.
- Calibration interpretation should be based on `n_unique_prompts`, not raw `n`.
- Low `n_unique_prompts` means the diagram is a scaffold, not evidence.

## What is actually here

Only `determinism_results.json` carries confidence: 6 replicate outputs of one
prompt (PETR4, 2020-03-06, BUY, 0.9). The 480 rows of the 80-day log are
signal-only and appear as a reference base rate on stdout, not in the diagram.
This is the honest $0 state; a Claude calibration needs paid logged runs.

## Operator checklist (zero-cost reruns)

When rerunning `make reliability`, verify:

1. `decisions.jsonl` exists and every `signal` is in `{BUY, SELL, HOLD}`.
2. stdout reports dropped-row reasons (if any); use that output as the
   canonical row-accounting record.
3. `bins.csv` is non-empty only if at least one row has both `confidence` and
   directional `win`.
4. `reliability.svg` reflects sparse-bin reality (no fabricated dense curve).
