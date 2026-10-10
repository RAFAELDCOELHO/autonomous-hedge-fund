from langchain_core.tools import tool
from typing import Annotated, Optional
from langgraph.prebuilt import InjectedState
from tradingagents.agents.utils.temporal import cap_date
from tradingagents.dataflows.interface import route_to_vendor

# P4.11: every tool below caps the model-supplied curr_date at the graph's
# trade_date (InjectedState, hidden from the LLM) before the vendor's own
# availability rules apply at that date.
TradeDate = Annotated[Optional[str], InjectedState("trade_date")]


@tool
def get_fundamentals(
    ticker: Annotated[str, "ticker symbol"],
    curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"],
    trade_date: TradeDate = None,
) -> str:
    """
    Retrieve comprehensive fundamental data for a given ticker symbol.
    Uses the configured fundamental_data vendor.
    Args:
        ticker (str): Ticker symbol of the company
        curr_date (str): Current date you are trading at, yyyy-mm-dd
    Returns:
        str: A formatted report containing comprehensive fundamental data
    """
    return route_to_vendor("get_fundamentals", ticker, cap_date(curr_date, trade_date))


@tool
def get_balance_sheet(
    ticker: Annotated[str, "ticker symbol"],
    curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"],
    freq: Annotated[str, "reporting frequency: annual/quarterly"] = "quarterly",
    trade_date: TradeDate = None,
) -> str:
    """
    Retrieve balance sheet data for a given ticker symbol.
    Uses the configured fundamental_data vendor.
    Args:
        ticker (str): Ticker symbol of the company
        freq (str): Reporting frequency: annual/quarterly (default quarterly)
        curr_date (str): Current date you are trading at, yyyy-mm-dd
    Returns:
        str: A formatted report containing balance sheet data
    """
    return route_to_vendor("get_balance_sheet", ticker, freq, cap_date(curr_date, trade_date))


@tool
def get_cashflow(
    ticker: Annotated[str, "ticker symbol"],
    curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"],
    freq: Annotated[str, "reporting frequency: annual/quarterly"] = "quarterly",
    trade_date: TradeDate = None,
) -> str:
    """
    Retrieve cash flow statement data for a given ticker symbol.
    Uses the configured fundamental_data vendor.
    Args:
        ticker (str): Ticker symbol of the company
        freq (str): Reporting frequency: annual/quarterly (default quarterly)
        curr_date (str): Current date you are trading at, yyyy-mm-dd
    Returns:
        str: A formatted report containing cash flow statement data
    """
    return route_to_vendor("get_cashflow", ticker, freq, cap_date(curr_date, trade_date))


@tool
def get_income_statement(
    ticker: Annotated[str, "ticker symbol"],
    curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"],
    freq: Annotated[str, "reporting frequency: annual/quarterly"] = "quarterly",
    trade_date: TradeDate = None,
) -> str:
    """
    Retrieve income statement data for a given ticker symbol.
    Uses the configured fundamental_data vendor.
    Args:
        ticker (str): Ticker symbol of the company
        freq (str): Reporting frequency: annual/quarterly (default quarterly)
        curr_date (str): Current date you are trading at, yyyy-mm-dd
    Returns:
        str: A formatted report containing income statement data
    """
    return route_to_vendor("get_income_statement", ticker, freq, cap_date(curr_date, trade_date))
