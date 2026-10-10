"""yfinance-based news data fetching functions."""

import yfinance as yf
from datetime import UTC, datetime, timedelta
from dateutil.relativedelta import relativedelta

from .stockstats_utils import yf_retry


def _parse_pub_date(article: dict) -> datetime | None:
    """Publication instant as an aware UTC datetime, or None if missing/unparseable.

    ``content`` items carry an ISO ``pubDate`` (any UTC offset, converted to UTC, never
    dropped); flat items carry ``providerPublishTime`` in epoch seconds. A naive ISO
    stamp is read as UTC. Anything missing or unparseable gives None.
    """
    if "content" in article:
        raw = (article.get("content") or {}).get("pubDate")
        if not isinstance(raw, str) or not raw:
            return None
        try:
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            return None
        if parsed.tzinfo is None:
            # Yahoo stamps are UTC; a naive stamp is read as UTC (never as the process's
            # local zone), as the legacy filter effectively did.
            parsed = parsed.replace(tzinfo=UTC)
        return parsed.astimezone(UTC)
    raw = article.get("providerPublishTime")
    if isinstance(raw, bool) or not isinstance(raw, (int, float, str)):
        return None
    try:
        return datetime.fromtimestamp(float(raw), tz=UTC)
    except (ValueError, OverflowError, OSError):
        return None


def _utc_midnight(date_str: str) -> datetime:
    return datetime.strptime(date_str + "+0000", "%Y-%m-%d%z")


def _published_in_window(article: dict, start_date: str | None, end_date: str) -> bool:
    """Look-ahead guard shared by the ticker and global paths (P4.11).

    Kept: start_date 00:00 UTC <= pub < (end_date + 1 day) 00:00 UTC, i.e. strictly
    before midnight UTC at the start of the day after end_date. With end_date = D-1
    that is ``pub < D 00:00:00 UTC``. Items with no parseable publication instant
    are dropped (same as insider rows with no date).
    """
    pub = _parse_pub_date(article)
    if pub is None:
        return False
    if pub >= _utc_midnight(end_date) + timedelta(days=1):
        return False
    return start_date is None or pub >= _utc_midnight(start_date)


def _extract_article_data(article: dict) -> dict:
    """Extract article data from yfinance news format (handles nested 'content' structure)."""
    pub_date = _parse_pub_date(article)
    if "content" in article:
        content = article["content"]
        provider = content.get("provider", {})
        # Get URL from canonicalUrl or clickThroughUrl
        url_obj = content.get("canonicalUrl") or content.get("clickThroughUrl") or {}
        return {
            "title": content.get("title", "No title"),
            "summary": content.get("summary", ""),
            "publisher": provider.get("displayName", "Unknown"),
            "link": url_obj.get("url", ""),
            "pub_date": pub_date,
        }
    # Fallback for flat structure
    return {
        "title": article.get("title", "No title"),
        "summary": article.get("summary", ""),
        "publisher": article.get("publisher", "Unknown"),
        "link": article.get("link", ""),
        "pub_date": pub_date,
    }


def get_news_yfinance(
    ticker: str,
    start_date: str,
    end_date: str,
) -> str:
    """
    Retrieve news for a specific stock ticker using yfinance.

    Args:
        ticker: Stock ticker symbol (e.g., "AAPL")
        start_date: Start date in yyyy-mm-dd format
        end_date: End date in yyyy-mm-dd format

    Returns:
        Formatted string containing news articles
    """
    try:
        stock = yf.Ticker(ticker)
        news = yf_retry(lambda: stock.get_news(count=20))

        if not news:
            return f"No news found for {ticker}"

        news_str = ""
        filtered_count = 0

        for article in news:
            # start_date 00:00 UTC <= pub < (end_date + 1 day) 00:00 UTC; undated dropped.
            if not _published_in_window(article, start_date, end_date):
                continue
            data = _extract_article_data(article)

            news_str += f"### {data['title']} (source: {data['publisher']})\n"
            if data["summary"]:
                news_str += f"{data['summary']}\n"
            if data["link"]:
                news_str += f"Link: {data['link']}\n"
            news_str += "\n"
            filtered_count += 1

        if filtered_count == 0:
            return f"No news found for {ticker} between {start_date} and {end_date}"

        return f"## {ticker} News, from {start_date} to {end_date}:\n\n{news_str}"

    except Exception as e:
        return f"Error fetching news for {ticker}: {str(e)}"


def get_global_news_yfinance(
    curr_date: str,
    look_back_days: int = 7,
    limit: int = 10,
) -> str:
    """
    Retrieve global/macro economic news using yfinance Search.

    Args:
        curr_date: Current date in yyyy-mm-dd format
        look_back_days: Number of days to look back
        limit: Maximum number of articles to return

    Returns:
        Formatted string containing global news articles
    """
    # Search queries for macro/global news
    search_queries = [
        "stock market economy",
        "Federal Reserve interest rates",
        "inflation economic outlook",
        "global markets trading",
    ]

    # Date window: curr_date - look_back_days .. curr_date (see _published_in_window).
    curr_dt = datetime.strptime(curr_date, "%Y-%m-%d")
    start_date = (curr_dt - relativedelta(days=look_back_days)).strftime("%Y-%m-%d")

    all_news = []
    seen_titles = set()

    try:
        for query in search_queries:
            search = yf_retry(lambda q=query: yf.Search(
                query=q,
                # Over-fetch: the date filter drops items, and only kept items count
                # toward `limit` (live 2024 queries return mostly recent news).
                news_count=max(50, 10 * limit),
                enable_fuzzy_query=True,
            ))

            for article in search.news or []:
                # The date filter runs before dedup and before the limit, so items
                # dropped by it never use up a slot.
                if not _published_in_window(article, start_date, curr_date):
                    continue
                title = _extract_article_data(article)["title"]
                if title and title not in seen_titles:
                    seen_titles.add(title)
                    all_news.append(article)

            if len(all_news) >= limit:
                break

        kept = all_news[:max(limit, 0)]
        if not kept:  # limit <= 0 too: never a bare header
            return f"No global news found for {curr_date}"

        news_str = ""
        for article in kept:
            data = _extract_article_data(article)
            title, publisher, link = data["title"], data["publisher"], data["link"]
            summary = data["summary"] if "content" in article else ""

            news_str += f"### {title} (source: {publisher})\n"
            if summary:
                news_str += f"{summary}\n"
            if link:
                news_str += f"Link: {link}\n"
            news_str += "\n"

        return f"## Global Market News, from {start_date} to {curr_date}:\n\n{news_str}"

    except Exception as e:
        return f"Error fetching global news: {str(e)}"
