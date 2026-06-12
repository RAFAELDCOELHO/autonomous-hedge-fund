# Architecture

This document describes how the system is put together at the code level: which modules exist, how data moves through them, and why the backtest harness is designed the way it is. For the research motivation, hypothesis, and experimental design, see [README.md](../README.md) and [PAPER.md](../PAPER.md); for the research narrative behind the design, see [CONTRIBUTION.md](../CONTRIBUTION.md).

## 1. Component Map

One line per component, grouped by layer. Paths are relative to the repository root.

```
Entry points
  main.py                                      Single trading-day pipeline run (one ticker, one date)
  run_backtest.py                              Backtest CLI: baselines and/or agent runs over a date range
  cli/                                         Interactive terminal UI
  scripts/smoke_macro_agent.py                 Manual smoke test for the Macro Economist Agent (real API calls)

Agents (tradingagents/agents/)
  analysts/market_analyst.py                   Market Analyst (technical, Kronos-backed)
  analysts/social_media_analyst.py             Social Analyst (sentiment, FinGPT methodology)
  analysts/news_analyst.py                     News Analyst (event analysis)
  analysts/fundamentals_analyst.py             Fundamentals Analyst (financial statements)
  analysts/macro_economist.py                  Macro Economist Agent (original contribution; opt-in)
  researchers/bull_researcher.py               Bull side of the research debate
  researchers/bear_researcher.py               Bear side of the research debate
  managers/research_manager.py                 Adjudicates the debate, writes the investment plan
  managers/portfolio_manager.py                Final trading decision after risk review
  risk_mgmt/{aggressive,conservative,neutral}_debator.py   Three-agent risk committee
  trader/trader.py                             Converts the investment plan into a proposed trade
  utils/agent_states.py                        AgentState TypedDict — the shared state all nodes read/write
  utils/macro_tools.py                         get_selic / get_inflation / get_gdp / get_exchange_rate (brazilfi-backed)
  utils/{kronos_tool,fingpt_tool,...}.py       Remaining LangChain tools bound to analysts

Orchestration (tradingagents/graph/)
  trading_graph.py                             TradingAgentsGraph facade; propagate(ticker, date) entry point
  setup.py                                     Builds the LangGraph StateGraph from selected_analysts
  conditional_logic.py                         Routing: ReAct tool loops per analyst, debate round limits
  propagation.py                               Initial-state construction for a trading day
  signal_processing.py                         SignalProcessor — extracts a 5-class rating from the final decision
  reflection.py                                Post-trade reflection / memory updates

Data adapters (tradingagents/dataflows/)
  y_finance.py, yfinance_news.py               Price and news data (yfinance)
  alpha_vantage_*.py                           Alpha Vantage stock/news/fundamentals/indicators
  indicator_fallback.py                        Fallback chain when an indicator source fails
  kronos_analyst.py                            Kronos time-series forecasting adapter
  fingpt_analyst.py                            FinGPT-methodology sentiment adapter
  stockstats_utils.py                          load_ohlcv — cached OHLCV loader used by the backtest runner

Backtest (tradingagents/backtest/)
  baselines.py                                 BuyAndHold, MACD(12,26,9), SMA(50/200) — shared decide() interface
  metrics.py                                   CR, Annualized Return, Sharpe, MDD from an equity curve
  runner.py                                    Day-by-day execution engine; the no-lookahead loop
  agent_integration.py                         map_signal, make_decide_fn, run_tradingagents_backtest
  report.py                                    Comparison tables and equity curves

LLM clients (tradingagents/llm_clients/)
  factory.py + {anthropic,openai,google,azure}_client.py   Provider-agnostic client construction
  model_catalog.py, validators.py              Model registry and config validation

Tests (tests/)                                 56 offline unit tests (metrics, adapters, fallbacks, graph construction)
```

## 2. Data Flow

A full trading-day decision costs roughly 17–25 LLM calls depending on selected analysts and debate rounds.

**What enters.** A ticker symbol and a trade date — from `main.py` (single day), the CLI, or the backtest runner (one call per trading day in the range). `propagation.py` turns these into the initial `AgentState`.

**Step 1 — Analyst reports.** The compiled LangGraph runs the selected analysts *in sequence* (graph order: Market → Social → News → Fundamentals → Macro). They are conceptually independent — none reads another's report — but execute one after another, each in its own ReAct loop: the analyst node calls its bound tools (routed by `conditional_logic.py` through a per-analyst `ToolNode`), the tools fetch external data, and the loop repeats until the analyst writes its report. Data sources per analyst: yfinance/Alpha Vantage for Market (plus Kronos forecasts), FinGPT-methodology sentiment for Social, news adapters for News, statement data for Fundamentals, and brazilfi (Bacen, IBGE) for Macro. Each analyst writes one field into `AgentState`: `market_report`, `sentiment_report`, `news_report`, `fundamentals_report`, `macro_report`. A message-clear node after each analyst prunes its tool-call chatter from the state.

**Step 2 — Research debate.** Bull and Bear researchers argue over the five reports for a bounded number of rounds (`conditional_logic.py` enforces the limit). The Research Manager adjudicates and writes an investment plan.

**Step 3 — Trade proposal and risk review.** The Trader converts the plan into a proposed trade. The three risk debaters (aggressive, conservative, neutral) stress-test it in rounds; the Portfolio Manager reads the risk debate and writes the final trade decision as free text.

**Step 4 — Signal extraction.** `SignalProcessor.process_signal` (a quick-thinking LLM call) reduces the final decision text to exactly one of five classes: `BUY`, `OVERWEIGHT`, `HOLD`, `UNDERWEIGHT`, `SELL`.

**What exits.** In single-day mode, the decision is printed/returned. In backtest mode, `agent_integration.map_signal` collapses the 5 classes to 3 actions, `runner.run_agent_strategy` executes the action at that day's close (full-position, long-only), and the loop emits a daily equity `pd.Series`. `metrics.py` computes CR/AR/Sharpe/MDD from that curve and `report.py` renders comparison tables — the same path classical baselines exit through.

## 3. Agent Responsibilities

**Market Analyst** (`analysts/market_analyst.py`). Produces the technical view: price action, momentum, and indicator state for the ticker. Unlike the upstream framework, its forecasting is backed by Kronos, a foundation model trained on financial time series, rather than generic LLM reasoning over price tables — the premise being that numerical forecasting belongs to a model trained on quantitative market structure. Output: `market_report`.

**Social Analyst** (`analysts/social_media_analyst.py`). Produces the sentiment view following the FinGPT methodology, using domain-specific sentiment modeling instead of a generic backbone. It summarizes the prevailing retail/social mood around the ticker and flags sentiment shifts. Output: `sentiment_report`.

**News Analyst** (`analysts/news_analyst.py`). Reads recent news for the ticker and the broader market through the news adapters and reports event-driven risks and catalysts — earnings, litigation, product, and general macro *narrative* (as text, not structured indicators; structured macro belongs to the Macro Economist). Output: `news_report`.

**Fundamentals Analyst** (`analysts/fundamentals_analyst.py`). Assesses the company's financial statements — earnings, margins, balance-sheet quality, valuation — through the fundamentals data tools, producing the bottom-up view that complements the technical and sentiment perspectives. Output: `fundamentals_report`.

**Macro Economist Agent** (`analysts/macro_economist.py`) — the original contribution of this work. Consumes four brazilfi-backed tools (`get_selic`, `get_inflation`, `get_gdp`, `get_exchange_rate`) and follows a procedural system prompt: establish the monetary policy environment, assess currency pressure, classify the macro regime (tightening / easing / stagflation risk / stable), and derive sector implications (banks vs. exporters vs. defensive names). It is opt-in via `selected_analysts`, which is what gives the 2×2 factorial design its "absent" condition without code changes. Output: `macro_report`. See PAPER.md §3.1 for the full design rationale.

**Bull and Bear Researchers** (`researchers/`). Adversarial pair that debates the analyst reports over multiple rounds — the Bull argues the strongest case for the position, the Bear the strongest case against, each citing specific reports (including `macro_report` when present). The debate forces analyst claims to survive opposition before they reach a decision, rather than being averaged together. Both maintain memories of past debates.

**Research Manager** (`managers/research_manager.py`). Adjudicates the Bull/Bear debate on a deep-thinking model: weighs the arguments, takes a side (or neither), and writes the investment plan that downstream agents act on. It is the first point where the five analyst perspectives are fused into a single directional view.

**Trader** (`trader/trader.py`). Translates the Research Manager's plan into a concrete proposed trade for the ticker — direction and conviction — bridging the research layer and the risk layer. It maintains memory of prior decisions so repeated mistakes can be reflected on.

**Risk Committee** (`risk_mgmt/`). Three debaters with fixed dispositions — aggressive, conservative, and neutral — stress-test the Trader's proposal in rounds. The role split is deliberate: each disposition is guaranteed a voice, so the proposal is always attacked from the risk-seeking and risk-averse sides regardless of what the rest of the pipeline concluded.

**Portfolio Manager** (`managers/portfolio_manager.py`). Reads the risk debate and makes the final call on a deep-thinking model. Its free-text decision is the pipeline's last reasoning step; everything after it is mechanical extraction and execution.

**Signal Processor** (`graph/signal_processing.py`). A single quick-LLM call that reduces the Portfolio Manager's free text to exactly one of five rating classes. Isolating extraction in one small component means the rest of the pipeline can reason in natural language while the backtest harness receives a clean categorical signal.

## 4. Backtest Infrastructure Design Decisions

The harness lives in `tradingagents/backtest/` (see PAPER.md §4 for the academic treatment; this section explains the *why* behind three load-bearing choices).

### Why `prices.iloc[:i+1]` — look-ahead prevention by construction

`runner.run_agent_strategy` walks the price series day by day and, at step `i`, hands the decision function only `prices.iloc[:i+1]` — bars up to and including the current day, never beyond (`runner.py:69`). The guarantee is structural: a strategy *cannot* peek at future bars because the data is not in scope, rather than being trusted not to. This matters because look-ahead bugs are the classic silent killer of backtest credibility — they inflate results without crashing anything. Two supporting details:

- The agent path's `decide_fn` ignores the price window argument (the graph fetches its own data), but the contract still holds because `propagate` is called with `curr_date` and all agent tools are date-keyed — they can only query data up to that date (`agent_integration.py:68-72`).
- A `decide_fn` that raises degrades to `HOLD` instead of aborting the run (`runner.py:72-74`), so a transient failure produces a conservative no-op rather than a corrupted equity curve.

### Why the 5→3 signal mapping

The SignalProcessor emits five classes (`BUY`, `OVERWEIGHT`, `HOLD`, `UNDERWEIGHT`, `SELL`) because graded conviction is natural for LLM reasoning and useful in reports. The runner, however, models binary positioning only — 100% equity or 100% cash, no shorting, no sizing — so `OVERWEIGHT` and `BUY` have no *executable* difference. `map_signal` collapses `OVERWEIGHT → BUY` and `UNDERWEIGHT → SELL` (`agent_integration.py:27-33`) rather than inventing position-sizing semantics, for two reasons: position sizing would add a free parameter that confounds the factorial comparison (any Sharpe delta could be a sizing artifact instead of an information effect), and it would break comparability with the classical baselines, which are also binary. Malformed or missing output maps to `HOLD` — the defensive fallback that makes parsing failure a no-op rather than a trade.

### Why mock-BUY must equal BuyAndHold

Before any hypothesis test, the integration layer itself had to be shown not to bias outcomes. The validation: drive the runner with a mock `propagate_fn` that returns `BUY` on every call, and require the result to be *numerically identical* to the `BuyAndHold` baseline — same CR, same Sharpe, same MDD (PAPER.md Table 1). This is a differential test of the plumbing: if the agent path added hidden transaction costs, executed a day late (off-by-one), or leaked state between steps, the two equity curves would diverge even with identical decision streams. Identity across all three metrics closes both loops at once — `map_signal` routes the BUY class correctly, and the runner treats an agent's decisions exactly like a baseline's. This is the precondition that lets any ΔSharpe in the factorial experiment be attributed to the agents rather than the harness. The `propagate_fn` injection point in `make_decide_fn` exists precisely to make this test possible without API calls. Known scope limit: the check exercises only the BUY path over one month; SELL/HOLD-path and B3-ticker validations are planned before the experiment (see [ROADMAP.md](../ROADMAP.md)).

## 5. Future Architecture

Planned extensions, in dependency order (research motivation in PAPER.md §8):

**ANBIMA yield curves.** The Macro Economist currently sees the *spot* SELIC series but not rate *expectations*. Adding ANBIMA term-structure data (DI futures curve) as a fifth brazilfi-backed tool would let the agent reason about where the market thinks rates are going — likely the dominant channel for repricing the banking names (ITUB4, BPAC11) in the ticker set. Architecturally this is one new provider in brazilfi plus one new `@tool` in `macro_tools.py`; the agent's prompt procedure gains a "rate expectations" step.

**CVM filings.** Regulatory filings from CVM (the Brazilian securities regulator) would give the Fundamentals Analyst a native Brazilian data path, mirroring what Alpha Vantage provides for US tickers. This is a `dataflows/` adapter plus a tool registration — no graph changes — but requires parsing Portuguese-language structured filings, which connects to the fine-tuning track in the roadmap.

**Multi-country expansion.** The Macro Economist's only Brazil-specific surface is its four tools; the agent factory, prompt structure (regime classification → sector implications), and graph integration are country-agnostic. Generalizing means defining a country data-layer interface — the four-function contract `get_policy_rate / get_inflation / get_gdp / get_exchange_rate` that `macro_tools.py` already implements for Brazil — and supplying per-country implementations (Mexico: Banxico/INEGI; India: RBI/MoSPI). The `selected_analysts` toggle and the backtest harness need no changes, so each new country costs one data adapter and one ticker-set definition, which is what makes the cross-country generalization of H1 tractable.
