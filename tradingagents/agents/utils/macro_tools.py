"""Brazilian macroeconomic data tools for the Macro Economist Agent.

These tools wrap brazilfi (https://pypi.org/project/brazilfi/) and expose
Brazilian Central Bank (Bacen) and IBGE data as LangChain-compatible tools.
"""

from langchain_core.tools import tool
from brazilfi import Bacen, IBGE


@tool
def get_selic(last_days: int = 90) -> str:
    """Get the Brazilian SELIC rate (benchmark interest rate set by the Central Bank).
    
    SELIC is the primary monetary policy tool in Brazil. High SELIC typically
    signals tight monetary policy; low SELIC signals accommodative policy.
    Historical range (2020-2024): 2.0% to 14.25% annualized.
    
    Args:
        last_days: Number of most recent business days to retrieve. Default 90.
    
    Returns:
        Markdown-formatted table with date and daily SELIC rate (in percent).
    """
    df = Bacen().selic(last=last_days).to_dataframe()
    return df.to_markdown()
@tool
def get_inflation(last_months: int = 12) -> str:
    """Get the Brazilian IPCA (official consumer price index / inflation rate).
    
    IPCA is the official inflation measure in Brazil, calculated monthly by IBGE
    and used as the target for the Central Bank's monetary policy. The inflation
    target is 3.0% per year (with a tolerance band of +/- 1.5%).
    
    Values returned are monthly inflation rates (not annualized). To understand
    the inflation environment, sum the last 12 months or compare recent trends.
    
    Args:
        last_months: Number of most recent months to retrieve. Default 12.
    
    Returns:
        Markdown-formatted table with date (first day of each month) and
        monthly inflation rate (in percent).
    """
    df = Bacen().ipca(last=last_months).to_dataframe()
    return df.to_markdown()


@tool
def get_gdp(last_quarters: int = 8) -> str:
    """Get Brazilian GDP (Gross Domestic Product) quarterly data from IBGE.
    
    GDP growth is the primary measure of economic activity. Positive growth
    signals economic expansion; negative growth signals recession. Brazilian
    GDP data from IBGE SIDRA is published quarterly with approximately 2-month lag.
    
    Use this tool to assess whether the Brazilian economy is expanding or
    contracting, which affects corporate earnings expectations and equity pricing.
    
    Args:
        last_quarters: Number of most recent quarters to retrieve. Default 8 (two years).
    
    Returns:
        Markdown-formatted table with quarterly GDP data.
    """
    df = IBGE().pib(last=last_quarters).to_dataframe()
    return df.to_markdown()


@tool
def get_exchange_rate(last_days: int = 90) -> str:
    """Get BRL/USD exchange rate (Brazilian Real to US Dollar) from Bacen.
    
    The exchange rate is critical for Brazilian equities: exporters (VALE3, PETR4,
    SUZB3, WEGE3) benefit from a weaker real (higher USD/BRL), while importers
    and companies with dollar-denominated debt suffer. A strengthening real
    (lower USD/BRL) has the opposite effects.
    
    Historical context (2020-2024): BRL/USD ranged from ~4.00 to ~5.80.
    
    Args:
        last_days: Number of most recent business days to retrieve. Default 90.
    
    Returns:
        Markdown-formatted table with date and BRL per USD.
    """
    df = Bacen().dolar(last=last_days).to_dataframe()
    return df.to_markdown()