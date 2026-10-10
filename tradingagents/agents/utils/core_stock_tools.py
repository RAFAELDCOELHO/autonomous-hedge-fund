from langchain_core.tools import tool
from typing import Annotated, Optional
from langgraph.prebuilt import InjectedState
from tradingagents.agents.utils.temporal import cap_date
from tradingagents.dataflows.interface import route_to_vendor


@tool
def get_stock_data(
    symbol: Annotated[str, "ticker symbol of the company"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
    curr_date: Annotated[str, "Current trading date in yyyy-mm-dd format"],
    trade_date: Annotated[Optional[str], InjectedState("trade_date")] = None,
) -> str:
    """
    Retrieve stock price data (OHLCV) for a given ticker symbol.
    Uses the configured core_stock_apis vendor.
    Args:
        symbol (str): Ticker symbol of the company, e.g. AAPL, TSM
        start_date (str): Start date in yyyy-mm-dd format
        end_date (str): End date in yyyy-mm-dd format
    Returns:
        str: A formatted dataframe containing the stock price data for the specified ticker symbol in the specified date range.
    """
    # P4.9 end_date <= curr_date, with curr_date capped at the graph's trade_date (P4.11).
    effective_end_date = cap_date(end_date, cap_date(curr_date, trade_date))
    return route_to_vendor("get_stock_data", symbol, start_date, effective_end_date)
