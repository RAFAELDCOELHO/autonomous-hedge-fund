"""Brazilian macroeconomic data tools for the Macro Economist Agent.

These tools wrap brazilfi (https://pypi.org/project/brazilfi/) and expose
Brazilian Central Bank (Bacen) and IBGE data as LangChain-compatible tools.

Point-in-time: every tool receives ``trade_date`` from the graph state
(injected by the ToolNode, hidden from the LLM) and returns only observations
that were public before the market opened on that date:

- SELIC (SGS 11) and BRL/USD PTAX (SGS 1), daily: dates strictly before
  ``trade_date``. Day D's values are published after the open of day D.
- IPCA (SGS 433), monthly: reference month M counts as known from day
  ``IPCA_RELEASE_DAY`` (15) of M+1. IBGE usually releases it around the 10th.
- GDP (IBGE SIDRA 1620), quarterly: a quarter counts as known
  ``GDP_LAG_DAYS`` (90) days after it ends. IBGE releases it after about 60.
  IBGE.pib only supports last=N counted back from today, so the tool
  over-fetches; brazilfi 0.2.1 dates quarter Q of year Y as Y-Q-01.
"""

from typing import Annotated

import pandas as pd
from brazilfi import Bacen, IBGE
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

TradeDate = Annotated[str, InjectedState("trade_date")]

IPCA_RELEASE_DAY = 15
GDP_LAG_DAYS = 90


def _daily_before(fetch, trade_date: str, last_days: int) -> pd.DataFrame:
    cutoff = pd.Timestamp(trade_date)
    start = cutoff - pd.Timedelta(days=2 * last_days + 10)
    end = cutoff - pd.Timedelta(days=1)
    df = fetch(start=start.date(), end=end.date()).to_dataframe()
    return df[df.index < cutoff].tail(last_days)


@tool
def get_selic(trade_date: TradeDate, last_days: int = 90) -> str:
    """Get the Brazilian SELIC rate (benchmark interest rate set by the Central Bank).

    SELIC is the primary monetary policy tool in Brazil. High SELIC typically
    signals tight monetary policy; low SELIC signals accommodative policy.
    Historical range (2020-2024): 2.0% to 14.25% annualized.

    Only dates strictly before the current trading date are returned.

    Args:
        last_days: Number of most recent business days to retrieve. Default 90.

    Returns:
        Markdown-formatted table with date and daily SELIC rate (in percent).
    """
    return _daily_before(Bacen().selic, trade_date, last_days).to_markdown()


@tool
def get_inflation(trade_date: TradeDate, last_months: int = 12) -> str:
    """Get the Brazilian IPCA (official consumer price index / inflation rate).

    IPCA is the official inflation measure in Brazil, calculated monthly by IBGE
    and used as the target for the Central Bank's monetary policy. The inflation
    target is 3.0% per year (with a tolerance band of +/- 1.5%).

    Values returned are monthly inflation rates (not annualized). To understand
    the inflation environment, sum the last 12 months or compare recent trends.

    Only months already published by the current trading date are returned
    (month M is treated as published on the 15th of month M+1).

    Args:
        last_months: Number of most recent months to retrieve. Default 12.

    Returns:
        Markdown-formatted table with date (first day of each month) and
        monthly inflation rate (in percent).
    """
    cutoff = pd.Timestamp(trade_date)
    start = cutoff - pd.DateOffset(months=last_months + 3)
    df = Bacen().ipca(start=start.date(), end=cutoff.date()).to_dataframe()
    published = df.index + pd.DateOffset(months=1, days=IPCA_RELEASE_DAY - 1)
    return df[published <= cutoff].tail(last_months).to_markdown()


@tool
def get_gdp(trade_date: TradeDate, last_quarters: int = 8) -> str:
    """Get Brazilian GDP (Gross Domestic Product) quarterly data from IBGE.

    GDP growth is the primary measure of economic activity. Positive growth
    signals economic expansion; negative growth signals recession. Brazilian
    GDP data from IBGE SIDRA is published quarterly with approximately 2-month lag.

    Use this tool to assess whether the Brazilian economy is expanding or
    contracting, which affects corporate earnings expectations and equity pricing.

    Only quarters already published by the current trading date are returned
    (a quarter is treated as published 90 days after it ends).

    Args:
        last_quarters: Number of most recent quarters to retrieve. Default 8 (two years).

    Returns:
        Markdown-formatted table with quarterly GDP data, indexed by quarter.
    """
    cutoff = pd.Timestamp(trade_date)
    n =last_quarters + (pd.Timestamp.today() - cutoff).days // 90 + 2
    df = IBGE().pib(last=max(n, last_quarters)).to_dataframe()
    quarters = pd.PeriodIndex(
        [pd.Period(year=d.year, quarter=d.month, freq="Q") for d in df.index],
        name="quarter",
    )
    published = quarters.end_time.normalize() + pd.Timedelta(days=GDP_LAG_DAYS)
    df = df.set_axis(quarters.astype(str))
    return df[published <= cutoff].tail(last_quarters).to_markdown()


@tool
def get_exchange_rate(trade_date: TradeDate, last_days: int = 90) -> str:
    """Get BRL/USD exchange rate (Brazilian Real to US Dollar) from Bacen.

    The exchange rate is critical for Brazilian equities: exporters (VALE3, PETR4,
    SUZB3, WEGE3) benefit from a weaker real (higher USD/BRL), while importers
    and companies with dollar-denominated debt suffer. A strengthening real
    (lower USD/BRL) has the opposite effects.

    Historical context (2020-2024): BRL/USD ranged from ~4.00 to ~5.80.

    Only dates strictly before the current trading date are returned.

    Args:
        last_days: Number of most recent business days to retrieve. Default 90.

    Returns:
        Markdown-formatted table with date and BRL per USD.
    """
    return _daily_before(Bacen().dolar, trade_date, last_days).to_markdown()
