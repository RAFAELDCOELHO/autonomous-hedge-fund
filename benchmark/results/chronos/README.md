# Chronos-t5-tiny comparator (P1.10)

Produced by `make chronos` (`scripts/chronos_comparator.py`).

## What ran

- Model: `amazon/chronos-t5-tiny` on CPU (`torch.float32`).
- Tickers: PETR4, ^BVSP from committed paper fixtures
  (`regime_lib.load_prices` → `benchmark/prices/paper/`).
- Regimes: all four in `regime_lib.REGIMES`.
- Context window: last 128 Close bars at each decision.
- Forecast horizon: 1; long iff mean of 20 sample
  paths for next Close exceeds current Close; else cash.
- Decision frequency: every 1 trading bar(s)
  (full bar coverage; Chronos-t5-tiny is cheap on CPU).
- Position sizing: long-only binary, frictionless, execute at Close
  (same harness as `regime_lib` / classical baselines).
- Same cells also report `buy_and_hold` and `momentum` (Momentum(60))
  via `regime_lib.run_strategy_cell`.

## Honest limits

- This is a **forecast→sign** trading rule, not an LLM agent and not
  a claim of Chronos alpha. Zero costs understates real turnover.
- Chronos was pretrained on broad public series; these B3 windows
  may overlap pretraining corpora (no leakage probe here).
- Tiny checkpoint only; results are a comparator cell, not an
  exhaustive foundation-model study. Kronos remains future work.
- Optional dependency: `[project.optional-dependencies] chronos`.
  `CHRONOS_SKIP=1` or a missing install exits 0 without rewriting
  committed CSVs.

## `per_cell.csv`

Columns: `agent`, `regime`, `ticker`, `total_return_pct`.
Agents: `chronos`, `buy_and_hold`, `momentum`.

## `summary.csv`

Per (agent, ticker): mean / min / max CR (%) across the four regimes,
plus mean CR gap vs buy_and_hold on matching cells (`gap_vs_bh_pct`).

## Snapshot (this commit)

```
       agent ticker  mean_cr_pct  min_cr_pct  max_cr_pct  gap_vs_bh_pct
     chronos  PETR4       1.6806    -37.2777     28.1330        -8.7965
     chronos  ^BVSP      -1.8656    -15.7814      9.5571        -5.9898
buy_and_hold  PETR4      10.4771    -27.8212     34.9021         0.0000
buy_and_hold  ^BVSP       4.1242    -23.7514     27.4162         0.0000
    momentum  PETR4       4.6340    -14.4493     26.0319        -5.8432
    momentum  ^BVSP       2.0058     -7.7738     18.2767        -2.1184
```

