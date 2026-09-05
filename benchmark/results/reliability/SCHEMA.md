# Reliability-diagram inputs (P1.6, $0 path)

Produced by `make reliability` (`scripts/reliability_diagram.py`) from the two
committed mistral:7b logs at the repo root and `benchmark/prices/PETR4.csv`.
Offline: no LLM call, no download, no key. Regenerate and `git status` must be clean.

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

Rows are dropped, and counted on stdout, only if the signal is not one of the
three or t+1 is missing from the fixture. Current run: 486 rows, 0 dropped.

## `bins.csv` — one row per non-empty confidence bin

Only rows with non-null `confidence` and non-null `win` are binned (10 bins of
width 0.1, last bin closed at 1.0). `n` counts rows, including temperature-0
replicates of the same prompt; `n_unique_prompts` counts distinct
`(ticker, date)`. Judge the diagram by `n_unique_prompts`, not `n`.

## What is actually here

Only `determinism_results.json` carries confidence: 6 replicate outputs of one
prompt (PETR4, 2020-03-06, BUY, 0.9). The 480 rows of the 80-day log are
signal-only and appear as a reference base rate on stdout, not in the diagram.
This is the honest $0 state; a Claude calibration needs paid logged runs.
