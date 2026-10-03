# Roadmap

Research plan for testing H1 — that explicit macroeconomic reasoning matters more for LLM trading agents in emerging markets than in developed ones. Hypothesis and experimental design are specified in [PAPER.md](PAPER.md); system design in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md); progress detail in [RESEARCH_LOG.md](RESEARCH_LOG.md).

## Phase 1: Infrastructure ✅ (complete)

Everything needed to run the experiment exists and is validated. See PAPER.md §3–4 for the full treatment.

- [x] Macro Economist Agent — fifth analyst, brazilfi-backed tools, opt-in via `selected_analysts`
- [x] Backtest runner with look-ahead prevention by construction (`prices.iloc[:i+1]`)
- [x] Agent integration layer — 5→3 signal mapping, dependency-injected `decide_fn` factory
- [x] Test suite green at phase close (current offline count intentionally not hard-coded; check live inventory with `python -m pytest --collect-only -q` in the project env)
- [x] Neutrality validation — mock always-BUY agent numerically identical to BuyAndHold (PAPER.md Table 1)

## Phase 2: H1 Experiment 🔄 (in progress)

Execute the 2×2 factorial (Market: US vs. Brazil × Macro Agent: absent vs. present) over Jan–Mar 2024. Nine tickers total: AAPL, GOOGL, AMZN (US) and ITUB4, BPAC11, PETR4, VALE3, WEGE3, RADL3 (B3).

- [ ] Smoke test: 1 ticker × 1 week — end-to-end run with real LLM calls before committing to the full grid
- [ ] Full baseline: 9 tickers × Jan–Mar 2024, four-analyst pipeline (no Macro Agent)
- [ ] Full H1: 9 tickers × Jan–Mar 2024, with Macro Agent
- [ ] Results table: 2×2 factorial Sharpe / CR / MDD per cell, with per-ticker ΔSharpe and the RADL3 control comparison
- [ ] Statistical significance: pre-registered test on ΔSharpe (macro present − absent), specified before execution per PAPER.md §8. Signed protocol in [docs/PREREGISTRATION.md](docs/PREREGISTRATION.md) (`scripts/h1_stats.py`)
- [x] P3.7 (PR #51): per-market daily risk-free rate (BR: BCB SGS 12 CDI; US: FRED DTB3; both dated t−1 = previous trading day, DTB3 forward-filled, never back-filled) from the SHA-256-verified `data/rf/` snapshots; `run_agent_strategy(..., market=)` lets cash (not the stock position) accrue `rf_t`; `ExtendedMetricsCalculator.compute(equity, rf=...)` computes Sharpe on excess return, √252 for both markets, and raises when rf does not cover every return date (flat 4.34% kept only as exploratory sensitivity); each run carries `n_days` and `n_decision_errors` (`propagate()` exceptions + unparseable signals) and is flagged invalid above 5% errors (exclusion rule E4); `h1_cell_metrics` returns the per-run cells.csv metric fields. Wired into the real CLI path: `run_backtest.py`'s TradingAgents arm passes `market=market_of(ticker)`, so its cash earns the daily rf (the window must lie inside the rf snapshot coverage). CLI decision-error counting goes through `make_decide_fn` (P3.8, PR #50).
- [x] P3.9 (PR #52): `run_backtest.py --cells-out PATH --seed N` appends one `cells.csv` row per TradingAgents arm. The writer reuses `h1_cell_metrics` and `market_of` and follows PREREGISTRATION §4 / `scripts/h1_stats.py`: ticker (B3 names without `.SA`), market, arm (`baseline`→`absent`, `macro`→`present`), seed, status, n_days, n_decision_errors, sharpe, rf_source. The comparison table prints that arm's H1 Sharpe (excess over the daily rf) beside `Sharpe @ flat 4.34% (exploratory)`; Buy & Hold, MACD, and SMA stay on the flat column only.
- [x] P3.11 (PR #54): `--cells-out` now preflights ticker/market, fixed prereg window (2024-01-02..2024-03-28), and existing-header validity before any API key checks, downloads, or strategy/LLM work; duplicates are refused early; per-arm exceptions still write `status=failed` rows and continue, but any failed arm returns non-zero with a failed-arm summary; `start`/`end` are appended via atomic file replacement.
- [ ] P3.15: `--cells-out` adds optional `sharpe_flat` (same `flat_rf_metrics` as the table's flat-rf Sharpe; old files migrated atomically, failed rows empty); `scripts/h1_rf_sensitivity.py` reruns h1_stats on it as the exploratory §7 flat-rf contrast.

## Phase 3: Paper Submission 📝 (planned)

Turn the working draft into a submittable paper once Phase 2 produces data.

- [ ] Complete PAPER.md results — replace the **[PENDING]** sections (§7) with factorial results and analysis
- [ ] Ablation: which macro indicator contributes most (SELIC vs. IPCA vs. GDP vs. BRL/USD, tool-by-tool removal)
- [ ] arXiv submission (cs.AI or q-fin.TR)
- [ ] Workshop submission: AAAI FinAI or NeurIPS Finance

## Phase 4: Extensions 🔮 (future)

Generalization beyond the single-country, single-window design — each item maps to an architecture extension in [docs/ARCHITECTURE.md §5](docs/ARCHITECTURE.md#5-future-architecture).

- [ ] ANBIMA yield curve as a Macro Agent tool — rate *expectations*, not just the spot SELIC series
- [ ] Multi-country replication (Mexico, India) via the country data-layer interface
- [ ] Portuguese financial text fine-tuning — native handling of B3 news and CVM filings
