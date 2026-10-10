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

## 9) P4.3/P4.4 availability rule (approximation)

These fixes use a deterministic proxy for "publicly available" timestamps because neither yfinance nor Alpha Vantage exposes filing/publication dates for statement rows/reports or insider rows.

- **Statements (P4.3, strict boundary):** a statement period is visible only when `available_date < curr_date`.
  - Quarterly periods: `available_date = period_end + 45 calendar days`.
  - Annual periods: `available_date = period_end + 3 calendar months` (`relativedelta(months=3)`).
  - **Q4 in quarterly series:** if a quarterly period end matches fiscal year-end month/day, it is treated as annual timing (`+3 calendar months`) rather than 45 days.
  - Fiscal year-end month/day is inferred from annual statement period ends for that ticker/report path when available; fallback is **Dec 31** when annual inference is unavailable.
- **Insiders (P4.4, strict boundary):** an insider transaction is visible only when `available_date < curr_date`, where `available_date = transaction_date + 2 US business days` (Form 4 proxy; `CustomBusinessDay` with `USFederalHolidayCalendar`).
- **B3 insider vendor check:** spot checks for `PETR4.SA`, `VALE3.SA`, and `ITUB4.SA` returned zero `insider_transactions` rows via yfinance at implementation time, so the CVM month-end + 10-day lag rule was not enabled in runtime filtering for this PR.
- On main, no analyst currently binds `get_insider_transactions`; this P4.4 filter is therefore defensive for any future binding, and exposing that tool to an analyst remains a separate design decision.

Residual risk remains by design:
- Late filers (or issuers with filing extensions) can still publish after the proxy lag and therefore leak if interpreted as public immediately at `available_date`.
- Early filers can be hidden longer than necessary (false delay) because the lag is conservative and not issuer/event-specific.
- FYE is inferred from the most recent annual period returned by yfinance, which may be after `curr_date`; no ticker in the universe changed FYE, and restricting inference to past periods would be worse because yfinance returns only ~4 annual periods.

## 10) Decision-timing rule: decide before the open of D

Decided by Rafa on 2026-10-09 19:48 BRT. Code at `e5ea062` is not changed by this section.

- **Information set for the decision of day D:** everything published up to the close of D-1 (B3 close for prices; for other sources, anything whose publication timestamp is at or before that close).
- **Execution:** at the open of D.
- Consequence: any value first known at or after the open of D (the close of D, returns or statistics that use it, data published during D) is look-ahead for the decision of D.

Known places that violate the rule at `e5ea062` (references only; nothing fixed here):

- `scripts/qwen_coldstart_n10.py:84,94` - the prompt's `close` is `closes[idx]` (close of D), and the 20d/60d returns are computed to that close.
- `brazilbench_mistral_test.py:52,55-56` - same in `compute_stats`: `close = prices[idx]` and `ret_20d`/`ret_60d` to the close of D.
- `scripts/reliability_diagram.py:102-103` - scoring uses the close-D to close-D+1 return, which does not match execution at the open of D.
- `tradingagents/backtest/runner.py:79,100,102` - the agent loop passes prices up to and including D (`prices.iloc[: i + 1]`) and executes at the close of D (see L9 and P4.11 above).

Any code fix is a separate PR and needs Rafa's explicit approval before implementation.

### 10.1) Deliberate conservative cuts when tools receive D-1

With the runner fix (P4.11) the date the agent and its tools receive for the decision of D is D-1, the previous session of the ticker's exchange. Every existing ceiling then applies unchanged at D-1. Two of them are strict (`<`), so they cut one day earlier than the rule requires. This is a deliberate choice: it is conservative and does not leak.

- **Statements (P4.3):** visible only when `available_date < D-1` (`tradingagents/dataflows/stockstats_utils.py:165`). A statement whose proxy availability date is D-1 is excluded, even if it was filed after the D-1 close and was public before the D open.
- **Insider transactions (P4.4):** visible only when `available_date < D-1` (`tradingagents/dataflows/stockstats_utils.py:211`). Same consequence.
- **Daily SELIC and USD/BRL** (`tradingagents/agents/utils/macro_tools.py:44`, `df.index < cutoff`) also read strictly before the date they receive, so the D-1 value is excluded and the latest value shown is D-2 (same direction: conservative; PTAX for D-1 is in fact published around 13:00 BRT on D-1).
- **IPCA and GDP** (`macro_tools.py:93,126`) use `published <= cutoff` evaluated at D-1, with the existing approximate publication dates (IPCA month M on the 15th of M+1; GDP quarter end + 90 days).
- **News: NOT conservative; post-close news from D-1 is admitted (decision 4b, needs Rafa's yes).** Both yfinance paths share one guard, `_published_in_window` (`tradingagents/dataflows/yfinance_news.py:43-56`). An item is kept when `start_date 00:00 UTC <= pub < (end_date + 1 day) 00:00 UTC` (strict). With `end_date` = D-1, the cutoff is **`pub < D 00:00:00 UTC`**: midnight UTC at the start of D, not a local midnight and not the close. During the 2024 H1 window it falls at 19:00 ET (EST, until 2024-03-09) or 20:00 ET (EDT, from 2024-03-10), and at 21:00 BRT (UTC-3, no DST). Both are after the D-1 regular close of NYSE and of B3.
  - Consequence: news published between the D-1 close and 00:00 UTC is visible to the decision of D. That is about 3-4 h of after-hours news for NYSE and a few hours for B3. It is **not** look-ahead relative to execution at the open of D, because it was public before the open. It **does** differ from the stated rule ("published up to the D-1 close"), and it goes the **opposite way** from the conservative statement, insider and macro cuts above. Tightening it to the exchange close would be a code change. News from 00:00 UTC on D to the open of D (pre-market) is excluded.
  - **Residuals fixed in P4.11** (all were open at `e5ea062`):
    - The bound is strict: an item stamped exactly D 00:00:00 UTC is excluded, and D-1 23:59:59 UTC is included.
    - `pubDate` is converted to UTC. A non-`Z` offset is no longer dropped, and a naive stamp is read as UTC, never as the process's local zone.
    - Flat-format items are filtered on `providerPublishTime` (epoch seconds, UTC). At `e5ea062` they skipped the filter: `get_global_news_yfinance('2024-01-15', 7, 5)` returned 2026 headlines under a "2024-01-08 to 2024-01-15" label (Lingxi's live repro).
    - Items with a missing or unparseable date, in either format, are dropped, as insider rows with no date are.
    - Global news filters before dedup and before `limit`, counts only the items it keeps, and over-fetches (`news_count=max(50, 10*limit)`). At `e5ea062` it took the first `limit` results and filtered afterwards.
    - `look_back_days` is now a real lower bound (`start_date` = `curr_date` - `look_back_days`, 00:00 UTC). At `e5ea062` it was only a label.
    - An empty result says "No ... news found" in both tools. At `e5ea062`, global news could return only the header.
  - **Tool caps:** the ticker `get_news` (`end_date`) and the `get_global_news` (`curr_date`) tools take `min(model date, trade_date)`, with `trade_date` = D-1 from graph state (section 10.2).
  - **Practical effect:** Yahoo serves only recent news. With the live vendor and 2024 decision dates, news essentially disappears after this fix. That is correct: before it, the agent was shown current (2026) headlines.
  - The non-default vendor, Alpha Vantage (`alpha_vantage_news.py`), sends `time_to = format_datetime_for_api(end_date)` = `YYYYMMDDT0000` of D-1, i.e. 00:00 at the start of D-1 in Alpha Vantage's own time convention. That excludes the whole of D-1 and is conservative. It is not used by the default config.
- **Fundamentals snapshot (known, not fixed here):** `y_finance.get_fundamentals` ignores the date and returns Yahoo's current company snapshot. Its date is capped like every other tool's, but that cannot help. This is P4.2 (PR #63), which is not on `e5ea062`.

The statement, insider and macro cuts can withhold information that was public before the open of D. The news cut (decision 4b) is the exception: it admits news from after the D-1 close up to D 00:00 UTC, which is still before the open of D. Nothing published at or after D 00:00 UTC enters. Changing any of these cuts would be a separate change and needs Rafa's approval.

### 10.2) How P4.11 (PR7) implements the rule in the backtest runner

- **Dates.** The decision date D iterates the sessions of the ticker's exchange in `[start, end]`, from a fixed trading calendar (`tradingagents/backtest/calendar.py`, `exchange_calendars` XNYS / BVMF). D-1 is the previous session of that calendar, never "the previous day with a vendor bar". Vendor bars on non-session days are ignored with a warning.
- **What the agent receives.** `decide_fn(D-1, window)`, with `window` = bars dated up to and including D-1. The graph's `trade_date` and the prompt's current date are D-1. **Every agent tool** takes its date as `min(model-supplied date, trade_date)` through one helper, `cap_date` (`tradingagents/agents/utils/temporal.py`), with `trade_date` injected from graph state (`InjectedState`). The model therefore cannot ask for data past D-1, whatever date it passes. Capped tools:
  - `get_stock_data`
  - `get_indicators`
  - `get_fundamentals`
  - `get_balance_sheet`
  - `get_cashflow`
  - `get_income_statement`
  - `get_news`
  - `get_global_news`
  - `get_insider_transactions`
  - `get_fingpt_sentiment_tool`
  - `get_kronos_forecast`

  The macro tools already read `InjectedState("trade_date")`. The ceilings of 10.1 apply at the capped date. The LLM-visible schemas and descriptions are unchanged from `e5ea062`; a test compares them with a snapshot.
- **Execution and return.** The decision of D fills at Open(D). Equity is indexed by D and marked at Open(D+1), so the decision of D earns Open(D+1)/Open(D) - 1. The last decision is marked at the open of the session after `end` (the defined exit price). Opens and closes come from one `auto_adjust=True` download, loaded without forward-fill of prices; the price cache file name records the adjustment (`-adj`). The classical baselines (Buy & Hold, MACD, SMA) run through the same simulator and follow the same convention. The paper scripts that simulate close-to-close on their own (`scripts/regime_lib.py`, `tradingagents/backtest/brazilbench.py`) are not changed and keep their convention.
- **Known open item (B2, pending a decision).** The curve has no initial-capital point. `equity[D_0]` is already the value at Open(D_1), so the metrics compute 60 returns from 61 points and the first decision's return, Open(D_0) to Open(D_1), is not counted. Adding the point changes the curve convention that the independent test package locks (index = decision dates), so it is left for an explicit decision.
- **Logs.** `equity.attrs["decision_log"]` and the graph state log record `decision_date` = D and `data_cutoff` = D-1; `cells.csv` keeps `start`/`end` as decision dates and adds `data_cutoff` (the D-1 of the first session).
- **Fail closed.** These vendor-data problems raise `DataDefectError` (a subclass of both `ValueError` and `LookupError`): a missing bar on the first D-1, on any D or on the exit session; a missing, NaN or non-positive Open on any D or on the exit session; or a duplicate vendor date. No day is skipped, and no close is used in place of an open. `validate_snapshot` raises a subclass of it.
  - In `run_backtest.py --cells-out`, only `DataDefectError` is caught in the baselines, so code bugs still propagate. A data defect writes one `status=failed` row per planned arm, with `sharpe` empty and the reason in `failure_reason`, and the run exits non-zero. The cell is then a counted exclusion for `scripts/h1_stats.py` and never goes missing.
  - A failed agent arm writes `failure_reason` (exception type and message) and `data_cutoff`.
  - A data defect is deterministic: a re-run under a new seed (PREREGISTRATION §5 "Re-runs") raises it again.
- **Snapshot check.** `tradingagents.backtest.snapshot.validate_snapshot` checks the whole pre-registered grid (9 tickers x 61 sessions per exchange for 2024-01-02..2024-03-28, plus D-1 of the first session, 2023-12-29 NYSE / 2023-12-28 B3, and the exit session 2024-04-01) and lists every missing bar or open. PR7 only ships and tests it; running it before the pilot is a separate step.
