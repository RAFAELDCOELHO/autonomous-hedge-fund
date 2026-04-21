# Research Contribution

## Origin: brazilfi

Brazilfi started as a practical exercise in building a Python SDK, motivated 
by my interest in both computer science and financial markets. Upon exploring 
the existing landscape of Brazilian financial APIs — Bacen SGS, IBGE SIDRA, 
Tesouro Direto — I found the state of the art genuinely poor: fragmented 
documentation, heterogeneous data formats, no modern typing, no stable 
unified interface. This matched a common pain point for anyone building 
fintech products, macroeconomic research tools, or investment analysis 
software in Brazil.

The decision to invest seriously in the library emerged from this observation: 
the cost of reinventing data-access infrastructure was being paid silently 
by every Brazilian developer in this space. Brazilfi aimed to eliminate 
that cost.

Current state (v0.3.0):
- 4 providers: Bacen, IBGE, Tesouro Direto, B3
- Pydantic v2 typed models
- 76% test coverage, 36 tests
- CI with ruff + mypy strict + pytest on Python 3.11/3.12
- Trusted Publishing to PyPI via GitHub Actions OIDC## 
Independent Research: Multi-Agent LLM Trading

Separately from brazilfi, I developed an implementation inspired by 
TradingAgents (Xiao et al., 2024, arXiv:2412.20138), a multi-agent LLM 
framework for financial trading from UCLA/MIT's Tauric Research. My 
implementation replaced the original generic LLM backbones with specialized 
models:

- **Claude (Anthropic)** for agent reasoning and debate
- **Kronos** (AAAI 2026) for time series forecasting
- **FinGPT methodology** for sentiment analysis

The initial research question was empirical: do specialized backbones 
(each optimized for its specific task) outperform generic LLMs across 
the full multi-agent pipeline, when measured against the same benchmarks 
used in the original paper (CR, Sharpe, MDD on AAPL, GOOGL, AMZN during 
Jan-Mar 2024)?

This work was developed independently and targets US equities, matching 
the original paper's evaluation setup.
## Emerging Research Question: Macro Context in Multi-Agent Trading

During the planning of evaluation phases beyond the paper's original 
US-stocks benchmark, a research question emerged naturally from the 
architecture analysis:

The TradingAgents framework includes four analyst agents — Market 
(technical), Social (sentiment), News (general macro), and Fundamentals 
(financials). None of these agents consumes explicit macroeconomic 
indicators like interest rates, inflation, or currency as structured 
inputs. This omission is reasonable in markets with low macroeconomic 
volatility, but becomes a potential blind spot in emerging markets.

**Motivating observation:** In the past decade, US Federal Funds Rate 
varied between 0% and 5.5%. Brazilian SELIC varied between 2% and 14.25% 
in the same period, with multiple 300-500 basis point swings occurring 
within 12-month windows. A rate move of this magnitude materially 
affects equity pricing through multiple channels (cost of credit, 
relative attractiveness vs fixed income, currency-driven effects on 
exporters). A multi-agent system that does not explicitly model these 
channels may miss information that dominates price formation in 
emerging markets.

**Hypothesis (H1):** Adding a Macro Economist Agent that consumes 
structured macroeconomic data (interest rates, inflation, GDP growth, 
exchange rates) to the TradingAgents pipeline improves trading 
performance (measured by Sharpe ratio and cumulative return) 
significantly more in emerging markets than in developed markets, 
due to differences in macroeconomic volatility.
## Convergence: brazilfi as Substrate

The Macro Economist Agent requires structured, reliable access to 
Brazilian macroeconomic data — specifically SELIC from Bacen, IPCA 
inflation from IBGE and Bacen, GDP growth from IBGE SIDRA, and 
BRL/USD exchange rate from Bacen. These are exactly the data sources 
that brazilfi was built to unify under a single typed interface.

This convergence was not planned from the outset. The two projects 
were developed independently: brazilfi as a general-purpose SDK 
addressing the fragmentation of Brazilian financial data sources, 
and the hedge-fund research as an empirical question about 
specialized backbones in multi-agent trading systems. The convergence 
became apparent when extending the research into emerging markets 
required macroeconomic data access as infrastructure.

The practical consequence is that implementing the Macro Economist 
Agent becomes trivial infrastructure-wise:

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

Without brazilfi, each of these would require writing custom parsers, 
managing heterogeneous response formats, and handling the quirks of 
each agency's API (Bacen's SGS XML endpoints, IBGE's SIDRA nested JSON, 
Tesouro Direto's deprecated endpoints behind Cloudflare). With brazilfi, 
the Macro Economist Agent's tool layer is a handful of wrapper 
functions — typed, tested, and CI-validated.
## Experimental Plan

### H1 Test: Macro Context Asymmetry Across Markets

The hypothesis will be tested through a 2x2 factorial design:

**Factor A — Market:**
- Developed (US): AAPL, GOOGL, AMZN (matching original paper)
- Emerging (BR): ITUB4, BPAC11, PETR4, VALE3, WEGE3, RADL3

**Factor B — Macro Agent:**
- Absent: baseline TradingAgents pipeline with 4 analysts
- Present: baseline + Macro Economist Agent consuming brazilfi tools

**Ticker selection rationale (BR):**
- ITUB4, BPAC11: banking sector, high SELIC sensitivity
- PETR4: energy, dual sensitivity to FX and commodity prices
- VALE3: mining, FX sensitivity and China demand exposure
- WEGE3: industrial exporter, FX sensitivity (different profile from commodity exporters)
- RADL3: defensive consumer, low macro sensitivity — acts as control

All tickers are native B3 listings (no BDRs). Sector coverage spans 
banking, energy, mining, industrial exports, and defensive consumer, 
ensuring the result is not driven by a single sector's behavior.

### Metrics

Following the original paper:
- Cumulative Return (CR)
- Annualized Return (ARR)
- Sharpe Ratio (SR)
- Maximum Drawdown (MDD)

### Evaluation Period

- US tickers: Jan-Mar 2024 (matching original paper for direct comparability)
- BR tickers: same period (Jan-Mar 2024) for within-study consistency

### Expected Outcomes

Under H1, the Macro Agent's contribution (measured as delta in Sharpe 
ratio between "present" and "absent" conditions) should be:
- Small or neutral on US tickers
- Substantially positive on emerging market tickers, excluding the 
  defensive control (RADL3)

A result pattern consistent with this prediction supports H1. Other 
patterns either refute H1 or require re-examination of mechanism 
(e.g., if the agent helps uniformly across markets, macro context 
matters everywhere, not just emerging markets).

### Honest Limitations

- Single-country emerging market test (Brazil only). Generalization 
  to other emerging markets (Mexico, India, South Africa) requires 
  future work.
- Time period is short (3 months). Results may not generalize to 
  different macro regimes (e.g., disinflationary cycles vs tightening cycles).
- Infrastructure constraint: brazilfi currently covers four data sources; 
  additional macro series (CVM filings, ANBIMA yield curves) could 
  enrich the Macro Agent in future iterations.