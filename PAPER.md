# Macro Context Asymmetry in Multi-Agent LLM Trading: Evidence from Emerging vs. Developed Markets

**Working draft: research in progress.** Sections marked **[PENDING]** require data from the factorial experiment, which has not yet been run. No results are reported beyond infrastructure validation and classical baselines.

## Abstract

Multi-agent LLM trading frameworks coordinate specialized analyst agents (technical, sentiment, news, and fundamentals), yet none of the analyst roles in the reference architecture, TradingAgents (Xiao et al., 2024), consumes explicit macroeconomic indicators as structured inputs. This omission is defensible in developed markets, where policy rates move slowly, but is a potential blind spot in emerging markets: over the past decade the Brazilian SELIC rate ranged from 2% to 14.25%, with multiple 300–500 basis-point swings inside single 12-month windows, against a 0%–5.5% range for the US Federal Funds Rate. This work introduces a Macro Economist Agent: a fifth analyst that consumes structured Brazilian macroeconomic data (SELIC, IPCA inflation, GDP growth, BRL/USD) through brazilfi, a typed Python SDK for Brazilian official data sources. A 2×2 factorial design (Market: US vs. Brazil × Macro Agent: absent vs. present) tests whether macro context contributes asymmetrically across market types. The evaluation infrastructure is implemented end to end: classical baselines, a no-lookahead backtest runner, and an agent–runner integration layer validated for numerical neutrality against Buy & Hold. Infrastructure and baselines are complete; factorial experiment results are pending.

## 1. Introduction

Multi-agent LLM systems are an active design pattern for financial trading. TradingAgents (Xiao et al., 2024) showed that decomposing the trading decision into specialized analysts, adversarial researcher debate, and layered risk management can outperform classical rule-based baselines on US equities. Subsequent work pushes specialization further, with foundation models purpose-built for financial time series (Shi et al., 2025) and financial text (Yang et al., 2023).

This evidence, however, comes from a single macroeconomic regime: large-capitalization US equities, where the policy rate moved within a 0%–5.5% band over a decade. No existing work tests these frameworks in markets where macroeconomic conditions are first-order drivers of equity prices. Brazil is a sharp instance: the SELIC policy rate ranged from 2% to 14.25% over the same decade, with multiple 300–500 basis-point swings inside 12-month windows. Moves of this magnitude propagate to equities through several channels at once: credit costs, fixed-income competition, and currency effects on exporters. A system with no structured view of these variables may miss the information that dominates price formation.

The TradingAgents analyst layer contains four roles: Market (technical), Social (sentiment), News (general events), and Fundamentals (financial statements). None consumes interest rates, inflation, GDP growth, or exchange rates as structured inputs. This gap motivates the central hypothesis:

> **H1.** Adding a Macro Economist Agent that consumes structured macroeconomic data (interest rates, inflation, GDP growth, exchange rates) to the TradingAgents pipeline improves trading performance, measured by Sharpe ratio and cumulative return, significantly more in emerging markets than in developed markets, due to differences in macroeconomic volatility.

This work makes three contributions. First, a Macro Economist Agent: a fifth analyst that retrieves Brazilian macroeconomic series through typed tools and writes a structured macro report into the shared agent state before the debate stage. Second, a 2×2 factorial experimental design (Market × Macro Agent) with a sector-diversified Brazilian ticker set that includes a low-macro-sensitivity control. Third, the brazilfi integration, which reduces the agent's data layer to four typed wrapper functions over official Brazilian sources, together with a validated backtest harness that connects the multi-agent graph to the same evaluation loop as the classical baselines.

This is a research-in-progress report: infrastructure, agent, and validation are complete; the factorial experiment has not yet been executed. The design is documented to be auditable before results exist, and result-dependent sections are explicitly marked as pending.

## 2. Related work

**TradingAgents.** Xiao et al. (2024) propose a multi-agent LLM framework in which analyst agents produce reports, Bull and Bear researchers debate them, and a manager–trader–risk pipeline converts the debate into a trading decision. The framework is model-agnostic via a LangChain wrapper and was evaluated on US equities (AAPL, GOOGL, AMZN) using cumulative return, Sharpe ratio, and maximum drawdown. Two properties matter for this work: the analyst layer is extensible by construction, and no analyst consumes structured macroeconomic data.

**Kronos.** Shi et al. (2025) introduce a foundation model trained directly on financial time series ("the language of financial markets"). Here, Kronos backs the Market Analyst's forecasting in place of generic LLM reasoning over price tables, on the premise that a model trained on quantitative market structure suits numerical forecasting better than a general-purpose language model.

**FinGPT.** Yang et al. (2023) develop open-source financial LLMs with an emphasis on sentiment tasks. The Social Analyst follows the FinGPT methodology rather than a generic backbone, for the same reason: financial sentiment benefits from domain-specific modeling.

**FINCH.** Dong et al. (2025) present a benchmark for finance and accounting agents across spreadsheet-centric enterprise workflows. Its task domain differs from trading, but it belongs to the same shift toward evaluating LLM agents on realistic, domain-grounded financial tasks rather than synthetic ones.

**Emerging-market macro context.** This work does not attempt a survey of emerging-market macro-finance; the motivation is framed empirically through the SELIC-versus-Fed-Funds volatility contrast quantified in Section 1. The hypothesis is that this difference in macroeconomic volatility changes the marginal value of explicit macro reasoning inside a multi-agent trading system.

## 3. System architecture

The system extends the TradingAgents pipeline, orchestrated as a LangGraph `StateGraph`. A full trading-day decision proceeds through three layers and costs approximately 17–25 LLM calls, depending on the number of selected analysts and debate rounds.

**Analyst layer.** Five analysts each write a report into the shared agent state. They are conceptually independent (none reads another's report) and execute sequentially in graph order: Market (technical analysis, Kronos-backed forecasting), Social (sentiment, FinGPT methodology), News (event analysis), Fundamentals (financial statements), and the Macro Economist Agent introduced in this work. Market data comes from Yahoo Finance and OpenBB; macroeconomic data comes from brazilfi.

**Debate layer.** Bull and Bear researchers argue over the analyst reports across multiple rounds; a Research Manager adjudicates and produces an investment plan. Reasoning and debate run on Claude (Anthropic) models.

**Decision layer.** A Trader agent converts the plan into a proposed trade; a three-agent risk debate (aggressive, conservative, neutral) stress-tests it; a Portfolio Manager makes the final call; and a Signal Processor reduces the output to a discrete trading signal.

### 3.1 Macro Economist Agent (original contribution)

The Macro Economist Agent follows the same factory pattern as the four existing analysts: a creation function (`create_macro_economist`) binds an LLM to a toolset and a domain-specific system prompt, returning a node closure used by the graph orchestrator. The agent consumes four LangChain tools backed by brazilfi (Listing 1):

```python
from brazilfi import Bacen, IBGE

# Tools exposed to the agent
def get_selic(last_days: int = 90) -> pd.DataFrame:
    return Bacen().selic(last=last_days).to_dataframe()

def get_inflation(last_months: int = 12) -> pd.DataFrame:
    return Bacen().ipca(last=last_months).to_dataframe()

def get_gdp(last_quarters: int = 8) -> pd.DataFrame:
    return IBGE().pib(last=last_quarters).to_dataframe()

def get_exchange_rate(last_days: int = 90) -> pd.DataFrame:
    return Bacen().dolar(last=last_days).to_dataframe()
```

**Listing 1.** The agent's complete data layer. Without brazilfi, each function would require custom parsers for heterogeneous government APIs (Bacen's SGS endpoints, IBGE's SIDRA nested JSON); with it, the tool layer is four typed, tested wrapper functions. brazilfi (v0.3.0), developed independently by the same author and distributed on PyPI, unifies Bacen (the central bank), IBGE (the statistics institute), Tesouro Direto, and B3 (the exchange) behind Pydantic-typed, CI-validated models.

The system prompt encodes an analysis procedure rather than free-form instructions: first establish the monetary policy environment (SELIC and inflation), then assess currency pressure, optionally pulling GDP for cycle context; classify the macro regime (tightening, easing, stagflation risk, or stable); and derive sector implications. Banks benefit from high SELIC via interest margins, commodity and industrial exporters from a weaker BRL, and defensive consumer names are comparatively insensitive. The output is a structured macro report written into a new `macro_report` state field, consumed by the Bull/Bear researchers alongside the other four reports.

Integration required three changes to the upstream framework: a creation block in the graph setup, the `macro_report` field in `AgentState`, and the factory export. The agent is opt-in via the `selected_analysts` configuration parameter: deselected, the pipeline is the original four-analyst TradingAgents baseline. This preserves the reference configuration's reproducibility and gives the factorial design its "absent" condition without code changes.

## 4. Backtest infrastructure

The original TradingAgents repository exposes a one-day `propagate()` entry point but no backtest harness; the evaluation loop is left to the user. The infrastructure built for this work lives in `tradingagents/backtest/` as four single-responsibility modules.

**`baselines.py`** implements three reference strategies as subclasses of a common abstract base: Buy & Hold, a MACD(12, 26, 9) crossover (long when the MACD line exceeds the signal line), and an SMA(50/200) golden cross. All are long-only with binary positioning (100% equity or 100% cash, no shorting). The simulation is frictionless (no transaction costs, slippage, or taxes) and decisions execute at the close of the bar they observe. Both simplifications apply identically to baselines and agents, so cross-condition comparisons are unaffected even though absolute levels are optimistic.

**`metrics.py`** computes Cumulative Return, Annualized Return, Sharpe Ratio, and Maximum Drawdown from the equity curve: CR = (V_end − V_start)/V_start; AR = (1 + CR)^(252/n) − 1; Sharpe = √252 · mean(excess)/std(excess) with the harness default risk-free rate of 4.34% annually (revisited in Section 8); MDD = min over t of (V_t − cummax(V))/cummax(V).

**`runner.py`** is the execution engine. `run_agent_strategy(decide_fn, ...)` walks the price series day by day; at step *i* the decision function receives only `prices.iloc[:i+1]`, i.e., data up to and including the current day. This slicing is the structural guarantee against look-ahead: no future bar is ever visible to a decision, by construction rather than by convention. A failed decision degrades to HOLD rather than aborting the run.

**`agent_integration.py`** bridges the multi-agent graph to the runner through three pieces. `map_signal` normalizes the Signal Processor's five output classes (BUY, OVERWEIGHT, HOLD, UNDERWEIGHT, SELL) to the runner's three actions, collapsing OVERWEIGHT to BUY and UNDERWEIGHT to SELL, with HOLD as the defensive fallback for malformed output. `make_decide_fn` is a factory returning a closure with the runner's decision signature; it closes over the graph's `propagate` entry point and the ticker, and accepts an injected `propagate_fn`, which keeps the orchestrator swappable and easy to mock. `run_tradingagents_backtest` is the high-level wrapper that stitches graph construction, decision-function creation, and the runner call.

**Validation.** Before any hypothesis test, the integration layer must be shown not to bias outcomes. The test: an agent that always answers BUY must be numerically identical to Buy & Hold. Using a mock `propagate_fn` returning BUY on every call, on AAPL over January 2024:

| Strategy | CR | Sharpe | MDD |
|---|---|---|---|
| Buy & Hold | −0.67% | −0.502 | −5.52% |
| Agent (mock BUY) | −0.67% | −0.502 | −5.52% |

**Table 1.** Neutrality validation. Identical values across all three metrics confirm that `map_signal` routes the BUY class correctly and that the runner treats the agent's decision stream exactly like a baseline's: no hidden costs, off-by-one errors, or state leakage. (The negative sign is coincidental; what matters is row equality.) This validation is a precondition for H1: any Sharpe difference in the factorial experiment must come from the agents, not the plumbing. The check exercises only the BUY path over a single month; SELL/HOLD-path checks and a B3-ticker run are planned before the experiment.

## 5. Experimental design

H1 is tested with a 2×2 factorial design.

**Factor A — Market.** Developed (US): AAPL, GOOGL, AMZN, matching the original paper's evaluation set. Emerging (Brazil): six native B3 listings chosen to span sectors with distinct macro exposure profiles:

| Ticker | Sector | Macro exposure rationale |
|---|---|---|
| ITUB4 (Itaú Unibanco) | Banking | High SELIC sensitivity (interest margins) |
| BPAC11 (BTG Pactual) | Banking | High SELIC sensitivity |
| PETR4 (Petrobras) | Energy | Dual FX and commodity-price sensitivity |
| VALE3 (Vale) | Mining | FX sensitivity plus China demand exposure |
| WEGE3 (WEG) | Industrial exporter | FX sensitivity, less commodity-cycle dependence |
| RADL3 (Raia Drogasil) | Defensive consumer | Low macro sensitivity — **control** |

**Table 2.** Brazilian ticker set. Sector coverage across banking, energy, mining, industrial exports, and defensive consumer ensures the result cannot be driven by a single sector's behavior.

**Factor B — Macro Agent.** Absent: the baseline four-analyst TradingAgents pipeline. Present: the same pipeline plus the Macro Economist Agent, toggled purely through `selected_analysts`. The agent's toolset is Brazil-specific by design, so the US-market/macro-present cell injects macro context expected to be largely irrelevant to the ticker. This cell tests whether the pipeline ignores the irrelevant context or is degraded by it; H1 predicts a small or neutral effect for developed markets.

**Metrics.** Following the original paper: Cumulative Return, Annualized Return, Sharpe Ratio, and Maximum Drawdown, computed by the harness of Section 4.

**Evaluation period.** January–March 2024 for both markets. This matches the original paper for the US arm (direct comparability) and is held identical for the Brazilian arm (within-study consistency).

**Expected outcome under H1.** The Macro Agent's contribution, defined as the delta in Sharpe ratio between the present and absent conditions, should be small or neutral on US tickers and substantially positive on Brazilian tickers, *excluding* the defensive control RADL3.

## 6. Baseline results (AAPL 2023)

Classical baselines on AAPL over full-year 2023 validate the harness end to end:

| Strategy | CR (%) | Sharpe | MDD (%) |
|---|---|---|---|
| Buy & Hold | 54.80 | 2.100 | −14.93 |
| MACD(12,26,9) | 36.04 | 2.047 | −6.40 |
| SMA(50/200) | 9.64 | 0.722 | −5.09 |

**Table 3.** Reference strategies, AAPL, Jan–Dec 2023. These figures validate the infrastructure (they exercise data loading, signal generation, simulation, and metric computation on a long window); they are not a test of H1, which compares agent configurations on the 2024 factorial grid. The ordering (Buy & Hold dominating in a bull year, trend-following strategies trading return for drawdown protection) is consistent with AAPL's strong 2023.

## 7. Preliminary analysis and expected results **[PENDING]**

*Predictions only: no factorial results exist yet; nothing here is evidence.*

**Predicted mechanism.** The macro volatility channel predicts that explicit SELIC, inflation, and FX context changes decisions most where those variables move most. During tightening or easing cycles, banking names (ITUB4, BPAC11) reprice with rate expectations; commodity exporters (PETR4, VALE3) carry FX sensitivity that a macro-blind pipeline can only infer indirectly from news text. Under H1, the macro-present condition should produce its largest Sharpe deltas in these names.

**Internal validity check.** RADL3 is the falsification probe inside the Brazilian arm. It was selected as a candidate low-macro-sensitivity name on the basis of defensive domestic demand and low FX exposure (a characterization to be verified empirically, not a measured beta). Its predicted macro-agent uplift is small even if H1 holds. If RADL3 shows an uplift comparable to the macro-sensitive names, the effect is more plausibly a generic "more analysts help" artifact than a macro-information channel.

**Result patterns.** Three outcomes are distinguishable: (a) ΔSharpe ≫ 0 on macro-sensitive Brazilian tickers and ≈ 0 on US tickers and RADL3, which supports H1; (b) a uniform uplift across both markets, which refutes the asymmetry mechanism and suggests macro context matters everywhere; (c) no uplift anywhere, or degradation, which refutes H1 and indicates the macro report either adds noise to the debate or is ignored downstream. Pattern (b) would still be a positive architectural finding, but not the hypothesized one.

Results tables for the four factorial cells, per-ticker Sharpe deltas, and the RADL3 control comparison will replace this section once the experiment is executed. **[PENDING]**

## 8. Limitations and future work

**Single-country emerging-market test.** Brazil is one point in the emerging-market distribution. Even a clean H1 result would not establish generality. Extension to markets with different macro profiles (India, Mexico, South Africa) is the natural next step; it requires only a country-specific data layer equivalent to brazilfi.

**Short evaluation window.** Three months is short and results may be regime-specific: an easing-cycle window may reward macro reasoning differently than a tightening shock or stable regime. Windows spanning multiple SELIC cycles are future work.

**Data coverage.** brazilfi currently unifies four sources (Bacen, IBGE, Tesouro Direto, B3). Yield-curve data (ANBIMA, the capital-markets association) and regulatory filings (CVM, the securities regulator) are absent; both could materially improve the Macro Economist Agent's view, particularly for rate-expectation reasoning that the spot SELIC series cannot capture.

**Single-run LLM stochasticity and statistical criteria.** Multi-agent LLM pipelines are stochastic; robust conclusions will require multiple runs per cell and seeds/temperature reporting, which the current plan does not yet fix. Relatedly, H1's "significantly" is not yet backed by a pre-registered statistical criterion; the test must be specified before execution.

**Confidence calibration [PENDING — requires paid Claude API runs].** The Signal Processor emits a confidence score alongside each discrete trading signal, but whether those confidences are *calibrated* — i.e., whether higher-confidence decisions are more often correct on the following day — is untested. The intended analysis is a reliability diagram: bin decisions by stated confidence, plot bin mean-confidence against realised win rate, and compare to the diagonal, across multiple seeds. This requires logged Claude decisions with per-decision confidence over the ticker/regime grid, which incurs paid API cost and has not been executed. The analysis is deferred until those logs exist; no calibration result is fabricated in the interim. (The zero-cost classical baselines — a RandomAgent null distribution over $N=100$ seeds and equal-weight rule-based portfolios across four regime windows — are reported separately as `docs/paper_random_n100.tex` and `docs/paper_ew_portfolio_baselines.tex`.)

**Metric assumptions.** The harness applies one default risk-free rate (4.34% annually) to both arms, which is defensible for the US but questionable for Brazil given the SELIC levels that motivate the study, and it models no transaction costs. Analysis will therefore use within-market Sharpe deltas, not cross-market levels.

## 9. Conclusion

This work contributes a Macro Economist Agent that gives a multi-agent LLM trading framework structured access to macroeconomic data through the brazilfi SDK; a validated no-lookahead backtest harness that connects that framework to classical baselines; and a 2×2 factorial design, with a built-in control ticker, for testing whether macro context matters asymmetrically across market types. H1 asks a question the multi-agent trading literature has not yet posed: whether architectures validated in macro-stable developed markets transfer to environments where macroeconomic volatility dominates price formation. The answer matters because most of the world's markets look more like Brazil than like the United States.

## References

Dong, H., Zhang, P., Gao, Y., et al. (2025). Finch: Benchmarking Finance & Accounting across Spreadsheet-Centric Enterprise Workflows. arXiv:2512.13168.

Shi, Y., Fu, Z., Chen, S., Zhao, B., Xu, W., Zhang, C., & Li, J. (2025). Kronos: A Foundation Model for the Language of Financial Markets. arXiv:2508.02739. AAAI 2026.

Xiao, Y., Sun, E., Luo, D., & Wang, W. (2024). TradingAgents: Multi-Agents LLM Financial Trading Framework. arXiv:2412.20138.

Yang, H., Liu, X.-Y., & Wang, C. D. (2023). FinGPT: Open-Source Financial Large Language Models. arXiv:2306.06031.
