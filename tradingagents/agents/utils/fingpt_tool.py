"""LangChain tool wrapping FinGPT sentiment for the social_media_analyst."""

import re
from typing import Annotated, Optional
from datetime import datetime
from dateutil.relativedelta import relativedelta

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

from tradingagents.agents.utils.temporal import cap_date
from tradingagents.dataflows.fingpt_analyst import get_fingpt_sentiment
from tradingagents.dataflows.interface import route_to_vendor


_MIN_HEADLINE_CHARS = 20
_MAX_HEADLINES = 30


def _extract_headlines(news_text: str) -> list[str]:
    """Heuristic extraction of headline-like lines from a vendor news blob.

    Vendor news responses are human-readable strings (see yfinance_news,
    alpha_vantage_news). We split on blank-line or newline boundaries and
    keep lines with at least _MIN_HEADLINE_CHARS that are not markdown
    headers or metadata. Sprint 6 will replace this with a structured
    fetcher once the OpenBB data layer lands.
    """
    if not news_text:
        return []
    chunks = re.split(r"\n{2,}", news_text)
    headlines: list[str] = []
    for chunk in chunks:
        for line in chunk.splitlines():
            line = line.strip()
            if len(line) < _MIN_HEADLINE_CHARS:
                continue
            if line.startswith("#"):
                continue
            if line.lower().startswith(("date:", "url:", "source:")):
                continue
            headlines.append(line)
            if len(headlines) >= _MAX_HEADLINES:
                return headlines
    return headlines


@tool
def get_fingpt_sentiment_tool(
    symbol: Annotated[str, "ticker symbol of the company"],
    curr_date: Annotated[str, "current trading date, YYYY-mm-dd"],
    look_back_days: Annotated[int, "how many days of news to analyze"] = 7,
    trade_date: Annotated[Optional[str], InjectedState("trade_date")] = None,
) -> str:
    """Obtain an aggregate FinGPT-style sentiment score over recent headlines.

    Call once per analysis as a complementary signal to your qualitative
    social/news review. If the tool reports unavailable, proceed with
    manual analysis.
    """
    # P4.11: this calls the news vendor directly (not through get_news), so it
    # applies the same trade_date cap itself; the window ends at the capped date.
    curr_date = cap_date(curr_date, trade_date)
    start_date = (
        datetime.strptime(curr_date, "%Y-%m-%d") - relativedelta(days=look_back_days)
    ).strftime("%Y-%m-%d")

    try:
        news_text = route_to_vendor("get_news", symbol, start_date, curr_date)
    except Exception as e:
        return f"FinGPT sentiment unavailable (news fetch failed: {e})."

    headlines = _extract_headlines(news_text)
    if not headlines:
        return "FinGPT sentiment unavailable (no headlines extracted)."

    result = get_fingpt_sentiment(headlines, symbol, curr_date)
    if result is None:
        return "FinGPT sentiment unavailable for this run."

    pos = sum(1 for s in result["scores"] if s == 1)
    neu = sum(1 for s in result["scores"] if s == 0)
    neg = sum(1 for s in result["scores"] if s == -1)
    return (
        f"FinGPT sentiment ({result['n']} headlines analyzed, {result['n_failed']} failed): "
        f"{result['label']} (score {result['score']}, "
        f"{pos}+ / {neu}o / {neg}-)."
    )
