from langchain_core.tools import tool
from typing import Annotated, Optional
from langgraph.prebuilt import InjectedState
from tradingagents.agents.utils.temporal import cap_date
from tradingagents.dataflows.interface import route_to_vendor

@tool
def get_news(
    ticker: Annotated[str, "Ticker symbol"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
    trade_date: Annotated[Optional[str], InjectedState("trade_date")] = None,
) -> str:
    """
    Retrieve news data for a given ticker symbol.
    Uses the configured news_data vendor.
    Args:
        ticker (str): Ticker symbol
        start_date (str): Start date in yyyy-mm-dd format
        end_date (str): End date in yyyy-mm-dd format
    Returns:
        str: A formatted string containing news data
    """
    # P4.9-style ceiling: end_date is capped at the graph's trade_date (the data
    # cutoff), injected from state and hidden from the LLM. None outside a graph.
    end_date = cap_date(end_date, trade_date)
    return route_to_vendor("get_news", ticker, start_date, end_date)

@tool
def get_global_news(
    curr_date: Annotated[str, "Current date in yyyy-mm-dd format"],
    look_back_days: Annotated[int, "Number of days to look back"] = 7,
    limit: Annotated[int, "Maximum number of articles to return"] = 5,
    trade_date: Annotated[Optional[str], InjectedState("trade_date")] = None,
) -> str:
    """
    Retrieve global news data.
    Uses the configured news_data vendor.
    Args:
        curr_date (str): Current date in yyyy-mm-dd format
        look_back_days (int): Number of days to look back (default 7)
        limit (int): Maximum number of articles to return (default 5)
    Returns:
        str: A formatted string containing global news data
    """
    # Same ceiling as get_news (P4.11): the model-supplied curr_date is capped at
    # the graph's trade_date (the data cutoff D-1), injected from state and hidden
    # from the LLM. The look-back window is then counted back from the capped date.
    # None outside a graph.
    curr_date = cap_date(curr_date, trade_date)
    return route_to_vendor("get_global_news", curr_date, look_back_days, limit)

@tool
def get_insider_transactions(
    ticker: Annotated[str, "ticker symbol"],
    curr_date: Annotated[str, "current date in yyyy-mm-dd format"],
    trade_date: Annotated[Optional[str], InjectedState("trade_date")] = None,
) -> str:
    """
    Retrieve insider transaction information about a company.
    Uses the configured news_data vendor.
    Args:
        ticker (str): Ticker symbol of the company
    Returns:
        str: A report of insider transaction data
    """
    # P4.11: curr_date capped at the graph's trade_date (cap_date).
    return route_to_vendor("get_insider_transactions", ticker, cap_date(curr_date, trade_date))
