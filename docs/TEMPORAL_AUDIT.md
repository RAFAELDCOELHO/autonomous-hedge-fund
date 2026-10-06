# Temporal look-ahead audit (P4.1)

Audited SHA: `80d2f8049d090f4efadd58ee591d82bc2663e7f7` (main tip after PR #59)  
Audit date: 2026-10-06 (BRT)  
Scope: agent-tool temporal leakage assessment for H1 execution paths

## 1) Method and scope constraints

- Read-only audit of code paths used by the current TradingAgents H1 workflow.
- Verified every citation in this document against the live tree at the audited SHA.
- Off-limits in this audit PR by policy: `benchmark/results/`, `macro_agent_draft/`, `data/rf/`, and signed preregistration body edits.
- This PR is documentation only (no runtime behavior changes).

## 2) Summary of confirmed temporal leaks

| Leak ID | Severity | Arms affected | Verified file:line | Mechanism |
|---|---|---|---|---|
| L1 | High | both | `tradingagents/dataflows/y_finance.py:245`; `tradingagents/agents/utils/fundamental_data_tools.py:20` | Fundamentals come from `ticker_obj.info` (current snapshot), routed directly as fundamentals tool output. |
| L2 | High | both | `tradingagents/dataflows/stockstats_utils.py:90`; `tradingagents/dataflows/stockstats_utils.py:100` | Financial statements are filtered by fiscal period end timestamp (`data.columns <= cutoff`), not publication/filing availability date. |
| L3 | High | both | `tradingagents/agents/utils/news_data_tools.py:53`; `tradingagents/dataflows/y_finance.py:397` | Insider transactions are exposed with no `curr_date` filter in tool wrapper or yfinance fetch path. |
| L4 | High | both | `tradingagents/dataflows/yfinance_news.py:69`; `tradingagents/dataflows/yfinance_news.py:85`; `tradingagents/dataflows/yfinance_news.py:87`; `tradingagents/dataflows/yfinance_news.py:182` | Stock news fetches fixed `count=20`; date filtering is weak (`end_dt + 1 day`) and articles with missing `pub_date` pass through. Global news filter only applies to nested-content branch, not flat branch. |
| L5 | Medium | both | `tradingagents/dataflows/y_finance.py:47`; `tradingagents/dataflows/y_finance.py:287`; `tradingagents/dataflows/y_finance.py:319`; `tradingagents/dataflows/y_finance.py:351`; `tradingagents/dataflows/y_finance.py:383`; `tradingagents/dataflows/y_finance.py:407` | Tool outputs stamp retrieval wall-clock time (`datetime.now()`), exposing post-cut temporal metadata to the model. |
| L6 | Medium | both | `tradingagents/agents/utils/core_stock_tools.py:22`; `tradingagents/dataflows/y_finance.py:23` | `get_stock_data` accepts caller `end_date` and forwards it to history fetch with no cap against simulation date. |
| L7 | Medium | both | `tradingagents/dataflows/stockstats_utils.py:58`; `tradingagents/dataflows/stockstats_utils.py:77`; `tradingagents/dataflows/stockstats_utils.py:41` | OHLCV cache pulls a 5-year window ending at today with `auto_adjust=True`; gap handling includes backfill (`bfill`) before date truncation. |
| L8 | Medium | macro-only | `tradingagents/agents/utils/macro_tools.py:118`; `tradingagents/agents/utils/macro_tools.py:119`; `tradingagents/agents/utils/macro_tools.py:124` | GDP uses `IBGE().pib(last=...)` from present-day-relative windows, then applies a fixed lag gate, not true point-in-time vintages. |
| L9 | High | both | `tradingagents/backtest/runner.py:79`; `tradingagents/backtest/runner.py:102`; `tradingagents/dataflows/yfinance_news.py:87` | Strategy executes signal at same-day close while news filter can admit items through end-of-next-day bound (`+1 day`), enabling t+1 contamination. |
| L10 | Medium | both | `tradingagents/agents/utils/fingpt_tool.py:57`; `tradingagents/agents/utils/news_data_tools.py:6` | FinGPT wrapper calls `route_to_vendor("get_news", symbol, curr_date, look_back_days)` but `get_news` expects `(ticker, start_date, end_date)`, creating temporal argument mismatch. |

## 3) Per-leak details (why this breaks as-of semantics)

- **L1 (fundamentals snapshot)**: `ticker_obj.info` is a mutable present snapshot, not a historical-as-of record, so historical simulation dates can observe future-updated fields.
- **L2 (financials by fiscal period end)**: filtering statement columns by period end date omits publication lag and amendment timing, allowing information not yet public at decision time.
- **L3 (insiders no cutoff)**: insider transaction fetch path does not receive or apply decision-date filtering.
- **L4 (news weak cutoff / missing pub_date pass-through)**: query shape plus permissive filter admits potentially late articles; unparseable publication timestamps are not excluded.
- **L5 (retrieval timestamp leakage)**: explicit current-time stamps can reveal run-time date beyond simulated date context.
- **L6 (uncapped end_date)**: tool caller can request windows extending beyond simulation date unless external guard is added.
- **L7 (adjusted prices + backfill over long cache window)**: `auto_adjust=True` and backfilling create dependence on revised/corporate-action-aware series built with future knowledge relative to earlier timestamps.
- **L8 (GDP vintage)**: quarter availability is approximated by lag; true vintages/revisions are not tracked.
- **L9 (same-day execution + permissive news window)**: trade is committed at day `t` close, but news guard can include material up to `t+1` boundary.
- **L10 (FinGPT args)**: wrapper argument mismatch undermines consistent temporal news retrieval contract.

## 4) Checked-clean paths

| Check | Verified file:line | Result |
|---|---|---|
| Indicator truncation to simulation date | `tradingagents/dataflows/stockstats_utils.py:85` | `load_ohlcv` truncates OHLCV rows to `Date <= curr_date`. |
| SELIC/FX as-of handling | `tradingagents/agents/utils/macro_tools.py:40`; `tradingagents/agents/utils/macro_tools.py:42`; `tradingagents/agents/utils/macro_tools.py:64`; `tradingagents/agents/utils/macro_tools.py:149` | Daily macro tools pull strictly before trade date. |
| Memory reflection not active in run path | `main.py:31`; `tradingagents/graph/trading_graph.py:289` | `reflect_and_remember` exists but is only commented in demo entrypoint, not invoked in normal backtest loop. |
| In-process BM25 memory (no external retrieval cache) | `tradingagents/agents/utils/memory.py:25`; `tradingagents/agents/utils/memory.py:40`; `tradingagents/agents/utils/memory.py:67` | Memory is local in-process BM25 index initialized empty. |
| Risk-free dated t-1 logic | `tradingagents/backtest/risk_free.py:76`; `tradingagents/backtest/risk_free.py:79` | BR compounds in `[t-1, t)` and US uses `asof(dates[:-1])`. |
| Prompts use injected simulation date token | `tradingagents/agents/analysts/market_analyst.py:69`; `tradingagents/agents/analysts/social_media_analyst.py:40`; `tradingagents/agents/analysts/news_analyst.py:38`; `tradingagents/agents/analysts/fundamentals_analyst.py:44`; `tradingagents/agents/analysts/macro_economist.py:63` | Analyst prompts frame context with `{current_date}` rather than wall-clock text. |

## 5) Explicitly open / not fully verified

- **H1 ticker survivorship/selection provenance remains open**: registry is fixed in code (`tradingagents/backtest/cells.py:35`; `scripts/h1_stats.py:63`), including control ticker (`scripts/h1_stats.py:62`), but selection criterion provenance is not encoded in implementation.

## 6) Vendor routing note (context for impact)

- All major agent-tool calls route via `route_to_vendor` (`tradingagents/dataflows/interface.py:134`).
- Default vendor configuration uses yfinance in all relevant categories (`tradingagents/default_config.py:27`; `tradingagents/default_config.py:28`; `tradingagents/default_config.py:29`; `tradingagents/default_config.py:30`; `tradingagents/default_config.py:31`) with no tool-level overrides by default (`tradingagents/default_config.py:34`; `tradingagents/dataflows/interface.py:127`).
- H1 arms differ by macro analyst inclusion only (`scripts/headline_arena_arms.py:15`; `scripts/headline_arena_arms.py:20`; `scripts/headline_arena_arms.py:24`), so non-macro leaks affect both arms.

## 7) Historical paper-cited script context (non-agent path)

These are standalone script paths, not the current tool-routed agent loop:

- `scripts/qwen_coldstart_n10.py` uses committed fixture `benchmark/prices/PETR4.csv` (`scripts/qwen_coldstart_n10.py:10`; `scripts/qwen_coldstart_n10.py:42`; `scripts/qwen_coldstart_n10.py:75`), with no agent-tool calls.
- `brazilbench_mistral_test.py` uses `yf.download(..., auto_adjust=True)` over 2019-06..2020-06 (`brazilbench_mistral_test.py:36`; `brazilbench_mistral_test.py:37`) and includes same-day volume z-score in prompt (`brazilbench_mistral_test.py:66`; `brazilbench_mistral_test.py:82`).
- `scripts/reliability_diagram.py` performs offline joins from existing logs plus fixture (`scripts/reliability_diagram.py:3`; `scripts/reliability_diagram.py:35`; `scripts/reliability_diagram.py:90`).
- `scripts/survivorship_distress.py` runs classical baselines from committed fixtures (`scripts/survivorship_distress.py:1`; `scripts/survivorship_distress.py:53`; `scripts/survivorship_distress.py:73`).

Minor caveats on those standalone scripts (not the core four agent-tool leaks): adjusted-price usage in yfinance pulls, same-day volume z-score usage in the mistral script, and hard-coded macro constants in prompt assembly (`scripts/qwen_coldstart_n10.py:49`; `brazilbench_mistral_test.py:108`).

## 8) Planning map to follow-up fixes (no implementation in this PR)

| Planned item | Intended remediation scope |
|---|---|
| P4.2 | Point-in-time fundamentals snapshots |
| P4.3 | Financials filtered by publication date (not fiscal end only) |
| P4.4 | Insider transactions filtered by decision-date availability |
| P4.5 | News temporal filtering hardening |
| P4.6 | Cross-tool temporal guardrail injection |
| P4.7 | Amendment / prereg alignment updates |
| P4.8 | Remove retrieval wall-clock stamp from tool outputs |
| P4.9 | Cap `get_stock_data` at simulation date |
| P4.10 | As-of pricing policy clarification and enforcement |
| P4.11 | Execution-timing update (t+1 policy discussion) |
| P4.12 | GDP vintage-safe macro data path |
| P4.13 | FinGPT argument contract fix |

Note: P4.5, P4.11, and any design-level workflow changes require explicit approval before implementation.
