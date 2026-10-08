from langchain_core.tools import tool
from typing import Annotated
from datetime import datetime
from tradingagents.dataflows.interface import route_to_vendor


@tool
def get_stock_data(
    symbol: Annotated[str, "ticker symbol of the company"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
    curr_date: Annotated[str, "Current trading date in yyyy-mm-dd format"],
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
    effective_end_date = min(
        datetime.strptime(end_date, "%Y-%m-%d"),
        datetime.strptime(curr_date, "%Y-%m-%d"),
    ).strftime("%Y-%m-%d")
    return route_to_vendor("get_stock_data", symbol, start_date, effective_end_date)
