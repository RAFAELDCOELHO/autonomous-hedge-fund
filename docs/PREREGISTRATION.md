# H1 pre-registration — statistical criterion

| | |
|---|---|
| Date | 2026-09-25 |
| Status | **draft, awaiting author sign-off** |
| Scope | Phase 2 factorial experiment (ROADMAP.md), PAPER.md §5 and §8 |
| Analysis code | [`scripts/h1_stats.py`](../scripts/h1_stats.py), tests in [`tests/test_h1_stats.py`](../tests/test_h1_stats.py) |

This document fixes how H1 will be tested **before** any factorial run. PAPER.md §8 says the word "significantly" in H1 has no pre-registered criterion yet. This document supplies one. Nothing here is a result. None of the choices below is approved until the author signs off. Every choice that still needs the author is listed in [Decisões em aberto para o autor](#decisões-em-aberto-para-o-autor). Once the author signs off, record the sign-off commit hash below. Any later change must be logged in the amendment log, with the reason and date, before data are seen.

Sign-off commit: _not signed_.

## 1. Hypotheses

The unit of analysis is the per-ticker Sharpe delta:

  Δ_t = mean over valid seeds of Sharpe(t, present) − mean over valid seeds of Sharpe(t, absent)

- **present**: the four-analyst TradingAgents pipeline plus the Macro Economist Agent (`selected_analysts` includes `macro`).
- **absent (baseline arm)**: the original four-analyst TradingAgents pipeline (Market, Social, News, Fundamentals). It has the same models, prompts, debate rounds, window and harness, and differs from the present arm only in `selected_analysts` (PAPER.md §3.1, §5). The classical baselines (Buy & Hold, MACD, SMA) are **not** the comparison arm for H1.

Ticker groups (PAPER.md §5, Table 2):

| Group | Tickers | Role |
|---|---|---|
| US | AAPL, GOOGL, AMZN | developed market |
| BR macro-sensitive | ITUB4, BPAC11, PETR4, VALE3, WEGE3 | emerging market (primary) |
| BR control | RADL3 | low-macro-sensitivity control, excluded from the primary test (PAPER.md §5, §7) |

Primary contrast: D = mean(Δ_t, BR macro-sensitive) − mean(Δ_t, US).

- **H0 (null):** the ticker-level Sharpe deltas do not depend on market. Formally, the eight Δ_t values are exchangeable across the US / BR-sensitive labels.
- **H1 (alternative, one-sided):** the Macro Agent's Sharpe uplift is larger on the BR macro-sensitive tickers than on the US tickers (D > 0).

H1 in the paper also mentions cumulative return. CR is **not** part of the confirmatory test; it is exploratory (§7). The confirmatory claim is about Sharpe only.

## 2. Design constants

| Item | Value | Source / rationale |
|---|---|---|
| Factorial cells | 4 = Market {US, BR} × Macro Agent {absent, present} | PAPER.md §5 |
| Tickers | 9 (3 US + 5 BR sensitive + 1 BR control) | PAPER.md §5 |
| Ticker × arm conditions | 18 | 9 × 2 |
| Seeds (replicates) per ticker × arm | **5** (proposed) | PAPER.md §8 asks for multiple runs; see open decisions |
| Total backtest runs | 90 = 9 × 2 × 5 | |
| Evaluation window | Jan–Mar 2024, first to last trading day of each market (proposed 2024-01-02 to 2024-03-28) | PAPER.md §5 |
| Decision frequency | one `propagate()` per trading day, as in `run_agent_strategy` | PAPER.md §4 |
| Minimum valid seeds per ticker × arm | 3 | exclusion rule E5 |
| α (primary) | 0.05, one-sided | |
| α (secondary family) | 0.05 family-wise, Holm | |

"Seed" is a **replicate index**. The Anthropic API exposes no sampling seed, so replicate *k* means the *k*-th independent run of the same configuration. Model ID, temperature and debate rounds are logged per run and must be identical across arms and seeds.

Cost order of magnitude: the paper estimates 17–25 LLM calls per decision (PAPER.md §3). At about 60 trading days × 90 runs, that is roughly 5,400 decisions, or about 92k–135k LLM calls. No dollar figure is claimed here.

## 3. Sharpe ratio and risk-free rate

Each run is scored with the harness Sharpe definition (`tradingagents/backtest/metrics.py`). The flat 4.34% rate is replaced by a market-specific daily risk-free series. PAPER.md §8 criticizes applying one US-level rate to both arms.

  S = √252 · mean(r_t − rf_t) / sd(r_t − rf_t)  (sd with ddof = 1; S = 0 when sd = 0, as in the harness)

- r_t = V_t / V_{t−1} − 1 is the daily portfolio return from the equity curve.
- **Brazil:** CDI daily rate, Banco Central do Brasil SGS series **12** ("Taxa de juros – CDI", % per business day). rf_t is the CDI rate dated on the previous B3 trading day t−1. It is compounded over every CDI business day in [t−1, t) if the calendars ever diverge. CDI is quoted on a 252-business-day basis, so no further conversion is needed. `rf_source = BCB-SGS-12`.
- **United States:** 3-month Treasury bill, secondary market, FRED series **DTB3** (annualized %, discount basis). rf_t = (1 + DTB3_{t−1}/100)^(1/252) − 1, where DTB3_{t−1} is the last value published on or before t−1 (forward-filled over bond-market holidays). The discount-basis vs bond-equivalent difference is ignored. `rf_source = FRED-DTB3`.
- **Alignment and no look-ahead:** rf for the return ending on day t uses only rates dated t−1 or earlier.
- **Annualization:** √252 for both markets. The B3 calendar has slightly fewer sessions per year, but 252 is kept as the harness convention for both. This only rescales Sharpe within a market and cannot change the sign of any Δ_t.
- **Cash earns the risk-free rate.** On days the strategy holds cash, the portfolio return is rf_t, so its excess return is exactly 0. Without this rule, an all-cash run under a time-varying SELIC-level rf would get a large negative Sharpe driven by rf noise alone.

**Implementation prerequisites (not done by this document).** Today the harness takes a scalar `annual_rf_rate` (`ExtendedMetricsCalculator`), and cash earns 0 in `run_agent_strategy`. Both must change before execution, and the rf series must be committed as fixtures. `scripts/h1_stats.py` consumes precomputed Sharpe values. It enforces the rf choice only through the `rf_source` column.

## 4. Data contract: `cells.csv`

One row per backtest run. Extra columns (e.g. `cr`, `mdd`, `model`, `temperature`) are allowed and ignored by the test. The authoritative definition is the docstring of `scripts/h1_stats.py`.

| Column | Type | Allowed values |
|---|---|---|
| `ticker` | str | the 9 tickers above (B3 names without `.SA`) |
| `market` | str | `US` or `BR`, must match the ticker |
| `arm` | str | `absent` or `present` |
| `seed` | int ≥ 0 | replicate index |
| `status` | str | `ok` or `failed` (run aborted or did not finish) |
| `n_days` | int ≥ 0 | trading days in the equity curve |
| `n_decision_errors` | int ≥ 0 | decisions that fell back to HOLD because `propagate()` raised (API error after retries, timeout) or the output could not be parsed by `map_signal` |
| `sharpe` | float | per §3; may be empty when `status = failed` |
| `rf_source` | str | `BCB-SGS-12` for BR rows, `FRED-DTB3` for US rows |

(ticker, arm, seed) must be unique. Any violation (unknown ticker, wrong market, wrong rf source, bad type, duplicate, missing column) aborts the analysis with exit code 2. Nothing is coerced.

The runner currently logs decision errors but does not count them (`runner.py`, the `except` branch → HOLD). Producing `n_decision_errors` is an implementation prerequisite.

## 5. Exclusion rules

The rules are applied in this order. Every exclusion is counted and reported by reason (`exclusion_counts` and `excluded` in the JSON output), and reported in the paper.

| # | Rule | Reason code |
|---|---|---|
| E1 | `status = failed` | `failed` |
| E2 | `sharpe` missing or non-finite on an `ok` run | `missing_sharpe` |
| E3 | `n_days` below the maximum `n_days` observed for that ticker (a truncated curve from missing price bars or an early stop) | `truncated` |
| E4 | `n_days == 0` **or** `n_decision_errors / n_days > 0.05` (more than 5% of decisions were silent HOLD fallbacks) | `decision_errors` |
| E5 | fewer than 3 valid runs in **either** arm of a ticker: the ticker is dropped from all analyses, both arms | `ticker_dropped` |

- **Re-runs.** A run excluded under E1 or E4 because of infrastructure failure (API outage, rate limit, network) may be re-run **once** under a new, previously unused seed index. The failed row stays in `cells.csv`. Runs that completed and passed E1–E4 are never re-run or replaced.
- **Missing market data** (a ticker's price series is unavailable for the window) removes the ticker from both arms through E5 and is reported.
- **Reduced primary test.** If E5 drops tickers, the primary test runs on the remaining tickers with the same procedure. The number of relabelings and the minimum attainable p are reported. With 5 BR + 2 US tickers the minimum p is 1/21 ≈ 0.048, so the test can still reject. With one US ticker left it cannot reject at α = 0.05, and that outcome is reported as **"não rejeitado (sem poder)"**.
- No other data-dependent exclusion (e.g. outlier Sharpe) is permitted.

## 6. Statistical tests

### 6.1 Primary test (confirmatory)

An **exact one-sided permutation test of D over market labels**, with tickers as the unit of analysis.

1. Compute Δ_t for the 8 primary tickers (5 BR-sensitive, 3 US) from the valid runs.
2. Enumerate all C(8, 3) = 56 ways to assign the "US" label to 3 of the 8 tickers, and compute D for each assignment.
3. p = #{assignments with D ≥ D_obs} / 56 (ties counted, tolerance 1e−12).
4. Reject H0 if p ≤ 0.05.

The enumeration is exact, so there is no random resampling and no seed.

**Why this test:**

- **The claim is about markets, so tickers are the replication unit.** All seeds of a ticker share one 2024 price path, so seeds are not independent draws of the market. Pooling seeds × tickers as n = 45 or 90 independent observations, as in a naive paired t-test, would be pseudoreplication and would overstate significance. Seeds are averaged within ticker. They reduce the LLM-decision noise in Δ_t but are not counted as extra evidence about markets.
- **No distributional assumptions.** A Sharpe ratio from about 60 daily returns is neither normal nor homoscedastic across tickers, and 8 units are far too few for asymptotic tests or reliable bootstrap intervals. The permutation test is exact in finite samples under the exchangeability null.
- **Jobson–Korkie–Memmel was considered and rejected.** It compares the Sharpe ratios of two return series on one asset using an asymptotic normal approximation under iid returns. It answers "does the macro arm beat the baseline on ticker t?". H1 asks "is the uplift larger in BR than in the US?", a between-market contrast of deltas. Combining JKM statistics across tickers would still need an across-ticker model and still rely on asymptotics over about 60 correlated daily returns.
- **Block or stationary bootstrap was considered and rejected for the primary test.** Resampling days within the 60-day window addresses serial dependence in returns, but it does not address the between-ticker variability that H1 is about. With 8 tickers, a ticker-level bootstrap is unreliable.
- **One-sided.** H1 is directional (a larger uplift in the emerging market). An effect in the opposite direction is reported descriptively but is not evidence for H1.
- **Known limitation (power).** The minimum attainable p is 1/56 ≈ 0.018. Rejecting at α = 0.05 requires the observed labeling to rank in the top 2 of 56, which in practice means near-complete separation of BR-sensitive deltas above US deltas. The test is conservative by design. A non-rejection with a positive D is reported as "not significant", not as evidence of no effect.

### 6.2 Secondary tests (confirmatory, Holm-corrected)

These tests condition on the tickers studied, so they support claims about *these tickers in this window*, not about markets. Each uses a within-ticker permutation of arm labels: in every ticker of the group, valid Sharpe values are first ordered by `seed`, then pooled across arms with arm sizes preserved, and the statistic is the mean Δ_t over the group. The procedure uses 10,000 Monte Carlo resamples and one independent RNG per test: `numpy.random.default_rng([20260925, k])`, where `k=0` for S1 and `k=1` for S2.

| ID | Statistic | Alternative | Question |
|---|---|---|---|
| S1 | mean Δ_t over BR macro-sensitive tickers | one-sided, > 0 | Does the Macro Agent improve Sharpe on BR macro-sensitive names at all? |
| S2 | mean Δ_t over US tickers | two-sided, ≠ 0 | Does the Brazil-specific macro context help or hurt US names? (PAPER.md §5 predicts ≈ 0). Computed exactly as \(p_{S2}=(1+\#\{|T^*|\ge|T_{obs}|-10^{-12}\})/(10{,}000+1)\). |

Holm's step-down correction is applied across the evaluable subset of {S1, S2} at family-wise α = 0.05, and adjusted p-values are reported. If only one secondary test is evaluable, Holm uses \(m=1\), so \(p_{Holm}=p_{raw}\) for that test. The primary test is not part of this family: it is tested alone at α = 0.05. The paper's §7 patterns map to these tests as follows. Pattern (a) is primary rejected and S1 rejected. Pattern (b) is S1 and S2 both positive with the primary not rejected. Pattern (c) is S1 not rejected.

### 6.3 Reporting

The paper reports, for every analysis: the per-ticker table (n per arm, mean Sharpe per arm, Δ_t, RADL3 marked as control), D_obs, the exact p, the number of relabelings, S1/S2 raw and Holm p-values, and all exclusions by reason. This is exactly what `scripts/h1_stats.py` prints and writes with `--out`.

## 7. Exploratory analyses (not confirmatory, no significance claims)

These are reported descriptively and labeled exploratory. They cannot turn a non-rejection of the primary test into support for H1.

- **RADL3 control:** Δ_RADL3 compared with the BR-sensitive deltas (PAPER.md §7 falsification probe). A single-ticker contrast has at most 6 relabelings, so no p-value is claimed.
- Per-ticker Δ_t with block-bootstrap intervals over days.
- Cumulative return, annualized return and maximum drawdown deltas, which appear in H1's wording but are not confirmatory.
- The primary contrast recomputed with the harness's legacy flat 4.34% rf (sensitivity to the rf choice).
- Transaction-cost sensitivity, signal-class distribution per arm, and decision agreement across seeds.
- The macro-tool ablation (ROADMAP Phase 3).

## 8. Running the analysis

```bash
uv run python scripts/h1_stats.py path/to/cells.csv --out path/to/h1_result.json
```

It is offline and deterministic: repeated runs on the same `cells.csv` produce byte-identical JSON. The constants above (α, 3 minimum seeds, 5% error threshold, 10,000 resamples, seed 20260925, ticker groups, rf sources) are module constants with no CLI flag, so the analysis cannot be tuned after the data are in.

## Decisões em aberto para o autor

None of these has been approved. Each needs an explicit decision by the author (Rafa) before sign-off. The values in this draft are proposals.

1. **Primary test.** Should the primary test be the ticker-level exact permutation (§6.1: conservative, min p ≈ 0.018, inference about markets)? The alternative is a pooled run-level test that conditions on the tickers (more power, narrower claim). The ROADMAP previously named "t-test on ΔSharpe", which this draft replaces.
2. **Primary contrast.** D uses BR macro-sensitive tickers (RADL3 excluded) vs US, following PAPER.md §5. Alternative: all 6 BR tickers vs US.
3. **Direction.** One-sided primary (D > 0). Should it be two-sided instead?
4. **α.** 0.05 for the primary test and 0.05 family-wise (Holm) for S1/S2.
5. **Secondary family.** Should S1/S2 be exactly these two tests, and should S2 be two-sided?
6. **Seeds.** 5 replicates per ticker × arm (90 runs, about 5,400 decisions, about 92k–135k LLM calls). This depends on budget and Claude rate limits.
7. **Model pin and sampling settings.** Which Claude model IDs (deep- and quick-think), temperature, and number of debate / risk rounds, fixed across all runs.
8. **Window endpoints.** Exact first and last trading day for Jan–Mar 2024 in each market (proposed 2024-01-02 to 2024-03-28; B3 Carnival and Good Friday closures follow each exchange's calendar).
9. **Brazil rf series.** CDI (SGS 12) vs SELIC over (SGS 11). They differ by a few basis points; CDI is proposed because it is the standard Brazilian cash benchmark.
10. **US rf series and conversion.** DTB3 with the (1 + y)^(1/252) conversion, ignoring discount vs bond-equivalent basis.
11. **Cash earns rf.** A harness change is required. The alternative is to keep cash at 0 and accept rf-driven Sharpe noise.
12. **Exclusion thresholds.** The 5% decision-error threshold (E4), minimum 3 valid seeds (E5), and one re-run per infrastructure failure.
13. **Annualization.** √252 for both markets.
14. **Sign-off.** Once the items above are settled, set the status to "signed off", record the commit hash, and commit before the first factorial run.

## Amendment log

| Date | Change | Reason |
|---|---|---|
| 2026-09-25 | Initial draft | P3.3 |
