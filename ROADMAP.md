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
- [x] P3.12: wire `macro_report` only into Bull/Bear prompts (not Research Manager/Trader/Risk), preserving byte-identical Bull/Bear prompt text when macro is absent; add real-graph prompt-routing regression tests for absent vs present arms (PR #55).
- [x] P3.13 (PR #56): tests and docs for P3.11 review nits: separate start-only / end-only `--cells-out` window rejection tests, a test that a rejected preflight (exit 2) never creates or modifies the cells file, and a `run_backtest.py` module docstring that describes the P3.11 preflight and failed-arm behavior.
- [x] P3.14: `--cells-out` now logs per-run model/debate config (`deep_think_llm`, `quick_think_llm`, `temperature`, `max_debate_rounds`, `max_risk_discuss_rounds`) alongside `start`/`end`; legacy files are migrated atomically via `_ensure_cells_extra_columns` and preflight rejects only when existing logged config differs from the current run (legacy rows and unrelated extra columns remain accepted). (PR #57)
- [x] P3.15: `--cells-out` adds optional `sharpe_flat` (same `flat_rf_metrics` as the table's flat-rf Sharpe; old files migrated atomically, failed rows empty); `scripts/h1_rf_sensitivity.py` reports descriptive per-ticker ΔSharpe and D on it with h1_stats' exclusions (no permutations, no p-values; exploratory §7). (PR #58)
- [x] P3.16 (PR #59): update PAPER.md to reflect signed pre-registration status, per-market daily rf (CDI/DTB3, t−1, SHA-256-verified `data/rf/` snapshots) as the primary H1 metric with flat 4.34% only exploratory sensitivity, and the Bull/Bear → Research Manager plan → Trader flow; add a regression assert that `curr_situation` memory lookup excludes `macro_report` text.

## F4: Temporal integrity

- [x] P4.1 (PR #60): temporal look-ahead audit doc with verified file:line citations on main 80d2f80 (leaks + clean paths); no runtime fixes.
- [ ] P4.2: aberto, #63 pausado. #71 (`1296fd68c6e065b1a6574ef4126dc01858ee527d`, GitHub title "tests: snapshot Alpha Vantage get_fundamentals") characterizes the current Alpha Vantage `get_fundamentals` path and does not add point-in-time snapshots.
- [x] P4.3 (PR #62): filter financial statements by approximate public availability (quarterly +45 days, annual +3 calendar months, Q4-in-quarterly treated as annual when the period end is within ±7 days of the fiscal-year-end month/day; strict `<` boundary with Dec-31 fallback when annual fiscal-year-end inference is unavailable). Hardening tests: #65 (`9621c2e3df03e5322f3c0cef1de7bfd444089ad7`), #73 (`d696c34be9e0744a28891a642ede0156f54407b8`, GitHub title "tests: financial statements date filter edges").
- [x] P4.4 (PR #62): filter insider transactions by approximate public availability (2 US business days Form 4 lag with strict `<` boundary; B3 yfinance insider rows unavailable in spot-check so CVM month-end+10 rule deferred pending vendor data availability). Hardening tests: #65 (`9621c2e3df03e5322f3c0cef1de7bfd444089ad7`), #72 (`6fb49a7acbaafd861eaf50a940b7fd340ae72858`, GitHub title "tests: insider date-column fail-closed edge").
- [ ] P4.5: News temporal filtering hardening. aguardando decisão.
- [ ] P4.6: Cross-tool temporal guardrail injection. No PR in #64-#85 mapped to this item.
- [ ] P4.7: Amendment / prereg alignment updates. aguardando decisão.
- [x] P4.8 (PR #61): remove wall-clock retrieval timestamp lines from yfinance tool outputs while preserving remaining output format. Hardening tests: #65 (`9621c2e3df03e5322f3c0cef1de7bfd444089ad7`).
- [x] P4.9 (PR #61): cap `get_stock_data` end date at `curr_date` (inclusive of `curr_date`, excluding later rows).
- [ ] P4.10: As-of pricing policy clarification and enforcement. aguardando decisão.
- [x] P4.11: backtest runner decides before the open of D (cutoff D-1, fill at Open[D]) in #64 (`c6d990d05ac7cb6284138f715eab4ecf61c9919c`). Follow-ups: #65 (`9621c2e3df03e5322f3c0cef1de7bfd444089ad7`; news-cutoff docs, AR note, duplicate dates in `validate_snapshot`, and the #61/#62 hardening tests above); #66 (`9e1b7148df6eb0864668bc4e6b237559a0ab46ef`; news cutoff across a weekend / holiday gap, NYSE + B3); #67 (`cc439d22cfe30d72889857f5a8cfdd0fb2bbc716`; lower bound of the yfinance news window); #68 (`ea9ce7767f242fffd924ce37423135dcc01f3ff1`; every strategy's cash earns the daily rf, fail closed, `--no-rf` opt-out); #69 (`bb172cdafd889d33bebda0ae531901132cd46d31`; `_simulate_open` requires the cash rate, fail closed); #70 (`c0f6dfcb9264af1744738291a776783241cc9506`; audit note that `_simulate_open` also enforces the cash rate); #75 (`def5b340e79121366747c2ef97c300550016f2e4`; GitHub title "tests: validate_snapshot edge cases").
- [ ] P4.12: GDP vintage-safe macro data path. aguardando decisão.
- [x] P4.13 (PR #61): pass `(ticker, start_date, end_date)` to FinGPT news routing (`start_date = curr_date - look_back_days`, `end_date = curr_date`).

não mapeado — merged PRs #64-#85 whose git-log subject and GitHub title do not name a §8 item:

- #74 `2a0660402f9cd707baf0cab641fc60a48f437c47` — tests: cells.csv row edge cases
- #76 `252d9f43d1b619d318af1b8dc0ff29838b24eb90` — tests: macro tools cutoff edges
- #77 `a48c6bee1bdedef0cb15090569414abcd7908ec1` — tests: map_signal edge cases
- #78 `b04fd9783dc3c93654b56268362583976e7a79fc` — tests: AV news time_to cutoff edges
- #79 `b08346ee63688ed17c7b35f1786277ef6e90314f` — tests: AV indicator cutoff edges
- #80 `b7e9611155d420dfe83bc32883139dfdfa223d37` — tests: ohlcv cutoff edges
- #81 `e500cbd09cb7b6760f28b1b590c0d458e920f1b9` — tests: AV stock CSV cutoff edges
- #82 `1a72c6c7f2335cee7349d9b63295af2fb493dbc7` — fix: fail closed on Alpha Vantage CSV date filter
- #83 `d2c8128cd974d5d9d0c8f6ba8a825b988e3ec9c6` — tests: pin Alpha Vantage CSV cell text and CRLF endings
- #84 `6b9ced49374f6fde3b130883e3cacaecc1608c61` — tests: pin AV CSV trailing newline and quoted-cell breaks
- #85 `cbe8a81c44c5132fc06a845a481e12fb7219f8da` — tests: drop redundant AV CSV newline assertions

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
