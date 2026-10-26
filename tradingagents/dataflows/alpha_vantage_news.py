import json

from .alpha_vantage_common import _make_api_request, format_datetime_for_api
from .stockstats_utils import US_BUSINESS_DAY
import pandas as pd

def get_news(ticker, start_date, end_date) -> dict[str, str] | str:
    """Returns live and historical market news & sentiment data from premier news outlets worldwide.

    Covers stocks, cryptocurrencies, forex, and topics like fiscal policy, mergers & acquisitions, IPOs.

    Args:
        ticker: Stock symbol for news articles.
        start_date: Start date for news search.
        end_date: End date for news search.

    Returns:
        Dictionary containing news sentiment data or JSON string.
    """

    params = {
        "tickers": ticker,
        "time_from": format_datetime_for_api(start_date),
        "time_to": format_datetime_for_api(end_date),
    }

    return _filter_news_response(_make_api_request("NEWS_SENTIMENT", params), params["time_to"])

def get_global_news(curr_date, look_back_days: int = 7, limit: int = 50) -> dict[str, str] | str:
    """Returns global market news & sentiment data without ticker-specific filtering.

    Covers broad market topics like financial markets, economy, and more.

    Args:
        curr_date: Current date in yyyy-mm-dd format.
        look_back_days: Number of days to look back (default 7).
        limit: Maximum number of articles (default 50).

    Returns:
        Dictionary containing global news sentiment data or JSON string.
    """
    from datetime import datetime, timedelta

    # Calculate start date
    curr_dt = datetime.strptime(curr_date, "%Y-%m-%d")
    start_dt = curr_dt - timedelta(days=look_back_days)
    start_date = start_dt.strftime("%Y-%m-%d")

    params = {
        "topics": "financial_markets,economy_macro,economy_monetary",
        "time_from": format_datetime_for_api(start_date),
        "time_to": format_datetime_for_api(curr_date),
        "limit": str(limit),
    }

    return _filter_news_response(_make_api_request("NEWS_SENTIMENT", params), params["time_to"])


def _published_before(item, cutoff) -> bool:
    """True only when item's time_published (YYYYMMDDTHHMM[SS]) parses and is < cutoff.

    Alpha Vantage does not document the timezone of time_published (its docs
    only give time_from/time_to as YYYYMMDDTHHMM). This compares naive
    timestamps: time_published as written against the time_to sent in the
    request (00:00 of the end date), with no timezone conversion.
    """
    from datetime import datetime

    raw = item.get("time_published") if isinstance(item, dict) else None
    if not isinstance(raw, str):
        return False
    for fmt in ("%Y%m%dT%H%M%S", "%Y%m%dT%H%M"):
        try:
            return datetime.strptime(raw, fmt) < cutoff
        except ValueError:
            continue
    return False


def _filter_news_payload(payload, cutoff):
    if not isinstance(payload, dict) or (_is_stamped(payload) and not _published_before(payload, cutoff)):
        return {"feed": []}  # unknown shape or a single late item: never echo it
    if "feed" not in payload:
        # AV error/info messages carry no items and pass unchanged; any
        # time-stamped item found elsewhere is still held to the cutoff.
        return _scrub_stamped(payload, cutoff)
    feed = payload["feed"]
    kept = [item for item in feed if _published_before(item, cutoff)] if isinstance(feed, list) else []
    payload["feed"] = kept
    if "items" in payload:
        payload["items"] = str(len(kept))
    return payload


def _is_stamped(value) -> bool:
    return isinstance(value, dict) and "time_published" in value


def _scrub_stamped(node, cutoff):
    """Recursively apply the cutoff to items carrying time_published.

    A list holding any stamped dict keeps only stamped dicts published before
    the cutoff (fail closed: unstamped/non-dict siblings go too); a stamped dict
    under a key is removed unless published before the cutoff. Anything with no
    stamped item is returned unchanged.
    """
    if isinstance(node, list):
        if any(_is_stamped(v) for v in node):
            return [_scrub_stamped(v, cutoff) for v in node if _is_stamped(v) and _published_before(v, cutoff)]
        return [_scrub_stamped(v, cutoff) for v in node]
    if isinstance(node, dict):
        return {
            k: _scrub_stamped(v, cutoff)
            for k, v in node.items()
            if not _is_stamped(v) or _published_before(v, cutoff)
        }
    return node


def _filter_news_response(result, time_to: str):
    """Drop news items published at/after time_to, client-side (AV may ignore time_to).

    Fail closed: items with a missing or unparseable time_published are dropped.
    Non-JSON text (no items) is returned as-is; JSON text comes back as JSON text.
    """
    from datetime import datetime

    cutoff = datetime.strptime(time_to, "%Y%m%dT%H%M")
    if isinstance(result, str):
        try:
            parsed = json.loads(result)
        except json.JSONDecodeError:
            return result
        filtered = _filter_news_payload(parsed, cutoff)
        if isinstance(parsed, dict) and "feed" not in parsed and filtered == parsed:
            return result  # nothing stamped: keep the vendor text byte-for-byte
        return json.dumps(filtered, indent=2)
    return _filter_news_payload(result, cutoff)


def get_insider_transactions(symbol: str, curr_date: str = None) -> dict[str, str] | str:
    """Returns latest and historical insider transactions by key stakeholders.

    Covers transactions by founders, executives, board members, etc.

    Args:
        symbol: Ticker symbol. Example: "IBM".

    Returns:
        Dictionary containing insider transaction data or JSON string.
    """

    params = {
        "symbol": symbol,
    }

    result = _make_api_request("INSIDER_TRANSACTIONS", params)
    if not curr_date:
        return result
    # _make_api_request returns the raw response text, not a dict.
    if isinstance(result, str):
        try:
            parsed = json.loads(result)
        except json.JSONDecodeError:
            return result
        if not isinstance(parsed, dict) or "data" not in parsed:
            return result
        return json.dumps(_filter_insider_payload(parsed, curr_date), indent=2)
    if not isinstance(result, dict) or "data" not in result:
        return result
    return _filter_insider_payload(result, curr_date)


def _filter_insider_payload(result: dict, curr_date: str) -> dict:
    cutoff = pd.Timestamp(curr_date)
    filtered_rows = []
    for row in result.get("data", []):
        tx_date = pd.to_datetime(row.get("transaction_date"), errors="coerce")
        if pd.isna(tx_date):
            continue
        tx_date = tx_date.tz_localize(None) if getattr(tx_date, "tzinfo", None) else tx_date
        available_date = tx_date + (2 * US_BUSINESS_DAY)
        if available_date < cutoff:
            filtered_rows.append(row)

    result["data"] = filtered_rows
    return result