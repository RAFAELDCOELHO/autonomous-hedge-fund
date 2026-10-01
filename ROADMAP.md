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
- [ ] Statistical significance: pre-registered test on ΔSharpe (macro present − absent), specified before execution per PAPER.md §8. Draft in [docs/PREREGISTRATION.md](docs/PREREGISTRATION.md) (`scripts/h1_stats.py`), awaiting author sign-off
- [ ] P3.7 (partial, PR #51): per-market daily risk-free rate (BR: BCB SGS 12 CDI; US: FRED DTB3; both dated t−1 = previous trading day, DTB3 forward-filled, never back-filled) from the SHA-256-verified `data/rf/` snapshots; `run_agent_strategy(..., market=)` lets cash (not the stock position) accrue `rf_t`; `ExtendedMetricsCalculator.compute(equity, rf=...)` computes Sharpe on excess return, √252 for both markets, and raises when rf does not cover every return date (flat 4.34% kept only as exploratory sensitivity); each run carries `n_days` and `n_decision_errors` (`propagate()` exceptions + unparseable signals) and is flagged invalid above 5% errors (exclusion rule E4); `h1_cell_metrics` returns the per-run cells.csv fields. Wired into the real CLI path: `run_backtest.py`'s TradingAgents arm passes `market=market_of(ticker)`, so its cash earns the daily rf (the window must lie inside the rf snapshot coverage). Still pending: the `cells.csv` writer (→ P3.9); `print_comparison` still shows every arm's Sharpe at the flat 4.34%, and the CLI agent arm still maps unparseable output to HOLD without counting it (until PR #50's `make_decide_fn` path lands)

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
