# Offline artifact schema (`make reproduce`, `make reliability`)

This folder stores the committed outputs for the zero-cost reproducibility path.
All files here are generated from committed fixtures and scripts; no API key, no
LLM call, and no price download are required.

## Provenance by target

- `make reproduce` regenerates:
  - `benchmark/results/random_n100/summary.csv`
  - `benchmark/results/random_n100/per_seed_returns.csv`
  - `benchmark/results/baselines/ew_returns.csv`
  - `benchmark/results/baselines/per_cell_returns.csv`
  - `benchmark/results/brazilbench/matrix.csv`
- `make reliability` regenerates:
  - `benchmark/results/reliability/decisions.jsonl`
  - `benchmark/results/reliability/bins.csv`
  - `benchmark/results/reliability/reliability.svg`

`tests/test_reproduce.py` enforces byte-level stability for the `make reproduce`
artifacts. `tests/test_reliability.py` validates the reliability outputs.

## CSV contracts (`make reproduce`)

### `random_n100/summary.csv`

Per-regime aggregated random-window statistics for one ticker.

| column | type | meaning |
|---|---|---|
| `regime` | string | Regime id (for example `bull_2019`) |
| `ticker` | string | Ticker symbol |
| `n_days` | int | Trading days in the regime slice |
| `n_seeds` | int | Number of sampled seeds |
| `mean_return_pct` | float | Mean return (%) over seeds |
| `std_return_pct` | float | Return std-dev (%) over seeds |
| `p5_return_pct` | float | 5th percentile return (%) |
| `p95_return_pct` | float | 95th percentile return (%) |
| `min_return_pct` | float | Minimum sampled return (%) |
| `max_return_pct` | float | Maximum sampled return (%) |
| `seed42_return_pct` | float | Return (%) for seed 42 |
| `seed42_percentile` | float | Percentile rank for seed 42 |

### `random_n100/per_seed_returns.csv`

Per-seed return rows backing `summary.csv`.

| column | type | meaning |
|---|---|---|
| `regime` | string | Regime id |
| `ticker` | string | Ticker symbol |
| `seed` | int | RNG seed |
| `total_return_pct` | float | Total return (%) for that seed |

### `baselines/ew_returns.csv`

Equal-weight baseline return by strategy-like agent label and regime.

| column | type | meaning |
|---|---|---|
| `agent` | string | Agent label (for example `buy_and_hold`) |
| `regime` | string | Regime id |
| `ew_return_pct` | float | Equal-weight return (%) |
| `n_tickers` | int | Number of tickers in the equal-weight basket |

### `baselines/per_cell_returns.csv`

Per-agent, per-regime, per-ticker returns behind the equal-weight rollup.

| column | type | meaning |
|---|---|---|
| `agent` | string | Agent label |
| `regime` | string | Regime id |
| `ticker` | string | Ticker symbol |
| `total_return_pct` | float | Total return (%) |

### `brazilbench/matrix.csv`

BrazilBench matrix used by `make bench` and docs mirror output.

| column | type | meaning |
|---|---|---|
| `strategy` | string | Display strategy name |
| `ticker` | string | B3 ticker |
| `regime` | string | Regime id |
| `cr` | float | Cumulative return (fraction) |
| `sharpe` | float | Sharpe ratio |
| `mdd` | float | Maximum drawdown (fraction) |
| `n_days` | int | Trading days in that regime |

## Reliability contract

`benchmark/results/reliability/` has its own detailed schema and caveats:
[`benchmark/results/reliability/SCHEMA.md`](reliability/SCHEMA.md).
