# PR queue — one squash-merge per day, each one moving the paper

Open PRs #17–#42 were generated as "residual polish" (small offline tests for upstream helpers). They stay
as the daily vehicle, but each one is upgraded on its day with a substantive Track B ($0, offline) or
Phase 2/3 deliverable listed here. Rules: one PR per calendar day (America/Fortaleza), squash merge, the
branch is pushed only on its day. Upgrades are prepared locally on `up/<N>` branches beforehand.

| Day | PR | Keeps (original) | Adds (upgrade) | Roadmap item |
|---|---|---|---|---|
| 2026-09-09 | #16 | CI runs `make reproduce` and fails on drift | — | P2.4 |
| 2026-09-09 | #15 | NeurIPS D&B checklist, docs smoke | README fix (REPRODUCIBILITY.md exists), this queue | P2.2 |
| 2026-09-10 | #17 | Makefile contract tests | `cost_bps` in `_simulate`, `make bench-costs`, `cost_sensitivity.csv` (0/10/25/50 bps), trade counts | Track B: transaction costs (PAPER §8) |
| 2026-09-11 | #18 | `benchmark/results/SCHEMA.md` | `docs/PREREGISTRATION.md` (H1 criterion, exact permutation DiD, α=0.05), `scripts/h1_stats.py`, factorial cells schema | Phase 2: statistical criterion |
| 2026-09-12 | #19 | runner fallback test | `backtest/seeds.py`: multi-seed runs, mean±std, cells rows | Phase 2: seeds (PAPER §8) |
| 2026-09-13 | #20 | Headline Arena arms test | frozen fixtures for the 9 factorial tickers (2023-01..2024-03), `backtest/factorial.py`, `make factorial-classical` | Phase 2: grid runnable offline |
| 2026-09-14 | #21 | `make help` | `macro_tools` config + `select_macro_tools`; prompt built from selected tools; offline graph tests | Phase 3: ablation switch |
| 2026-09-15 | #22 | test-count docs | remove hard-coded test counts + guard test; ROADMAP Track B section; WEEKLY_LOG Sept entry | Docs truth |
| 2026-09-16 | #23 | yfinance news helper test | SELIC-based risk-free rate per regime (committed SELIC fixture) → Sharpe sensitivity table | Track B: rf assumption (PAPER §8) |
| 2026-09-17 | #24 | weekday helper test | block-bootstrap 95% CIs for baseline CR/Sharpe per cell | Track B: uncertainty |
| 2026-09-18 | #25 | Alpha Vantage helper test | deflated Sharpe + Holm correction across the 24-cell grid | Track B: multiplicity |
| 2026-09-19 | #26 | base LLM helper test | rolling 60-day Sharpe across regime boundaries (CSV + SVG) | Track B: regime robustness |
| 2026-09-20 | #27 | CLI announcements test | BRL- vs USD-denominated returns (committed BRL/USD fixture) | Track B: cross-market comparability |
| 2026-09-21 | #28 | reliability SCHEMA.md | recorded brazilfi fixtures + offline tests for the four macro tools' output format | Macro agent contract |
| 2026-09-22 | #29 | model catalog test | factorial smoke runner with a mock agent (`--dry-run`) writing `cells.csv` skeleton | Phase 2: smoke test |
| 2026-09-23 | #30 | reliability `_runs` test | `scripts/factorial_table.py`: cells.csv → PAPER §7 2×2 table + per-ticker ΔSharpe | Phase 3: results table |
| 2026-09-24 | #31 | stockstats helper tests | pytest markers `offline`/`network`/`llm`; CI selects by marker instead of `--ignore` | Test hygiene |
| 2026-09-25 | #32 | dataflows config test | regime-definition consistency test (brazilbench ↔ regime_lib ↔ tex table) | Track B: consistency |
| 2026-09-26 | #33 | `map_signal` tests | parse the macro report's indicator table into a dict for per-decision logging | Macro agent observability |
| 2026-09-27 | #34 | ConditionalLogic tests | per-decision JSONL log (date,ticker,macro,seed,signal,confidence) writer + schema | Phase 2: logging → reliability |
| 2026-09-28 | #35 | backtest `__getattr__` test | Momentum(60) added to BrazilBench as separate artifact `momentum.csv` | Track B: 4th baseline |
| 2026-09-29 | #36 | regime percentile test | baselines re-run on HMM regimes vs manual regimes (sensitivity) | Track B: regime sensitivity |
| 2026-09-30 | #37 | StatsCallbackHandler tests | per-run token/cost accounting JSON so Phase 2 reports cost per cell | Phase 2: cost reporting |
| 2026-10-01 | #38 | Reflector helper tests | PAPER.md §4/§8 text update consolidating costs, seeds, pre-registration, ablation switch | Paper draft |
| 2026-10-02 | #39 | `_simulate` helper test | `--curves` export of equity curves per cell for figures | Track B: figures |
| 2026-10-03 | #40 | reliability helpers test | reliability diagram accepts the factorial decision log (from #34) | Phase 3: calibration |
| 2026-10-04 | #41 | interface category test | `make check` (bench + reproduce + offline tests) as the single pre-merge gate | DX |
| 2026-10-05 | #42 | LLM client factory tests/docs | drop its hard-coded test count edits; RESEARCH_LOG Track B summary | Docs truth |

Conflicts with `main` (README, Makefile `.PHONY`, ROADMAP) are expected and resolved locally on the day.
Every upgrade is offline, needs no API key, and never changes bytes of artifacts pinned by `tests/test_reproduce.py`.
