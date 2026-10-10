"""Point-in-time date ceiling shared by every agent tool (P4.11)."""

from __future__ import annotations

from datetime import datetime
from typing import Optional


def cap_date(llm_date: Optional[str], trade_date: Optional[str]) -> str:
    """The date a tool may use: min(llm_date, trade_date), as YYYY-MM-DD.

    trade_date is the graph's data cutoff (the runner's D-1), injected from
    state and hidden from the LLM, so a model-supplied date after it never
    reaches a vendor. trade_date None (a tool called outside a graph) keeps
    llm_date; llm_date None falls back to trade_date.
    """
    if trade_date is None:
        return llm_date
    if llm_date is None:
        return trade_date
    return min(
        datetime.strptime(llm_date, "%Y-%m-%d"),
        datetime.strptime(trade_date, "%Y-%m-%d"),
    ).strftime("%Y-%m-%d")
