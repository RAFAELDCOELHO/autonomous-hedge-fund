import time
import logging

import pandas as pd
import yfinance as yf
from dateutil.relativedelta import relativedelta
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
from yfinance.exceptions import YFRateLimitError
from typing import Annotated
import os
from .config import get_config

logger = logging.getLogger(__name__)
US_BUSINESS_DAY = CustomBusinessDay(calendar=USFederalHolidayCalendar())


def yf_retry(func, max_retries=3, base_delay=2.0):
    """Execute a yfinance call with exponential backoff on rate limits.

    yfinance raises YFRateLimitError on HTTP 429 responses but does not
    retry them internally. This wrapper adds retry logic specifically
    for rate limits. Other exceptions propagate immediately.
    """
    for attempt in range(max_retries + 1):
        try:
            return func()
        except YFRateLimitError:
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt)
                logger.warning(f"Yahoo Finance rate limited, retrying in {delay:.0f}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
            else:
                raise


def _clean_dataframe(data: pd.DataFrame, fill_prices: bool = True) -> pd.DataFrame:
    """Normalize a stock DataFrame for stockstats: parse dates, drop invalid rows, fill price gaps.

    fill_prices=False leaves NaN prices (other than Close) in place, so a caller
    can fail closed on a missing Open instead of trading on a filled one.
    """
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data = data.dropna(subset=["Date"])

    price_cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in data.columns]
    data[price_cols] = data[price_cols].apply(pd.to_numeric, errors="coerce")
    data = data.dropna(subset=["Close"])
    if fill_prices:
        data[price_cols] = data[price_cols].ffill().bfill()

    return data


# A calendar day ends in "-DD". That is not an offset. These are: "Z", "+",
# "±HH:MM", "±HHMM", and a short negative offset after a time ("T00:30:00-05").
_OFFSET_MARKER_RE = r"(?:Z|\+|[+-]\d{2}:\d{2}|[+-]\d{4}|T\d{2}:\d{2}:\d{2}-\d{2}$)"

# A civil DST transition shifts the clock by at most one hour (US/Eastern
# -05:00 vs -04:00; Lord Howe by 30 minutes). A wider mix is not that case.
_MAX_DST_OFFSET_SPAN_MINUTES = 60


def _missing_date_cell(value) -> bool:
    if isinstance(value, str):
        return not value.strip()
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return bool(missing)


def _reject_mixed_date_zones(dates: pd.Series, symbol: str) -> pd.Series | None:
    """Fail closed on mixed date zones. Rewrite a DST-sized offset mix.

    Column-wise ``pd.to_datetime`` turns the other kind into NaT, and the
    following drop deletes those rows, or it raises ``Mixed timezones detected``.
    A datetime64 column already has one timezone, including a named zone whose
    UTC offset changes across the year, and is left for the written-day
    conversion. Naive stamps are not re-parsed here. ``None`` means unchanged.

    Text offsets spanning more than 60 minutes raise. A span of at most 60
    minutes is one civil zone across DST, so each cell becomes the calendar
    day written in it and the column parser does not see mixed offsets. Two
    cells that name one calendar day raise.
    """
    if pd.api.types.is_datetime64_any_dtype(dates):
        return None
    if not dates.astype("string").str.contains(_OFFSET_MARKER_RE, regex=True, na=False).any():
        return None

    saw_naive = False
    offsets: set[int] = set()
    parsed_at: list[tuple[object, pd.Timestamp]] = []
    for idx, value in dates.items():
        if _missing_date_cell(value):
            continue
        parsed = pd.to_datetime(value, errors="coerce")
        if pd.isna(parsed):
            continue
        if getattr(parsed, "tzinfo", None) is None:
            saw_naive = True
        else:
            offsets.add(int(parsed.utcoffset().total_seconds() // 60))
            parsed_at.append((idx, parsed))
        if saw_naive and offsets:
            raise ValueError(f"{symbol}: Date mixes timezone-aware and naive timestamps")
    if len(offsets) <= 1:
        return None
    if max(offsets) - min(offsets) > _MAX_DST_OFFSET_SPAN_MINUTES:
        raise ValueError(f"{symbol}: Date mixes UTC offsets")

    out = dates.copy()
    seen: set[pd.Timestamp] = set()
    for idx, parsed in parsed_at:
        day = parsed.tz_localize(None).normalize()
        if day in seen:
            raise ValueError(f"{symbol}: Date normalizes to duplicate calendar days")
        seen.add(day)
        out.at[idx] = f"{day:%Y-%m-%d}"
    return out


def _calendar_day_as_written(dates: pd.Series) -> pd.Series:
    """Calendar day printed on the stamp. Drop a UTC offset; do not shift the clock.

    ``2024-01-05T00:30+09:00`` is 5 Jan as written (UTC would make it 4 Jan).
    Naive values are returned unchanged, including a time that is not midnight.
    """
    if getattr(dates.dt, "tz", None) is None:
        return dates
    return dates.dt.tz_localize(None).dt.normalize()


def load_ohlcv(symbol: str, curr_date: str, fill_prices: bool = True) -> pd.DataFrame:
    """Fetch OHLCV data with caching, filtered to prevent look-ahead bias.

    Downloads 5 years of data up to today and caches per symbol. On
    subsequent calls the cache is reused. Rows after curr_date are
    filtered out so backtests never see future prices. A timestamp with a
    UTC offset is the calendar day written in the cell: the offset is
    removed without shifting the clock to UTC, then the time is floored to
    midnight. Naive timestamps are left as parsed. Text offsets that differ by
    at most 60 minutes are that same reading per cell, so a US/Eastern file
    with ``-05:00`` and ``-04:00`` across DST keeps each written day. A column
    that mixes naive timestamps with timezone-aware ones raises ValueError.
    So does a column whose offsets span more than 60 minutes: the rows are
    not dropped and the offsets are not forced into one zone. Written days
    that collide after that reading also raise ValueError. The 60-minute
    bound is the span of the offsets, not a zone name, so two zones an hour
    apart are read the same way when each shows only one offset. All of OHLC comes
    from one auto_adjust=True download; the cache file name records that
    setting. With fill_prices=True (the default), gaps in Open, High, Low,
    Close, and Volume are forward-filled and then backfilled only on the
    rows still dated on or before curr_date. The in-window bfill does use
    later bars up to curr_date. A bar after curr_date is not a source.
    fill_prices=False leaves those gaps as NaN.
    """
    config = get_config()
    curr_date_dt = pd.to_datetime(curr_date)

    # Cache uses a fixed window (5y to today) so one file per symbol
    today_date = pd.Timestamp.today()
    start_date = today_date - pd.DateOffset(years=5)
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = today_date.strftime("%Y-%m-%d")

    os.makedirs(config["data_cache_dir"], exist_ok=True)
    data_file = os.path.join(
        config["data_cache_dir"],
        f"{symbol}-YFin-data-{start_str}-{end_str}-adj.csv",  # auto_adjust=True
    )

    if os.path.exists(data_file):
        data = pd.read_csv(data_file, on_bad_lines="skip")
    else:
        data = yf_retry(lambda: yf.download(
            symbol,
            start=start_str,
            end=end_str,
            multi_level_index=False,
            progress=False,
            auto_adjust=True,
        ))
        data = data.reset_index()
        data.to_csv(data_file, index=False)

    # Parse and drop bad rows first, then cut the window, then fill. Backfill
    # before the cut takes a leading NaN from a bar after curr_date.
    # Mixed naive/aware values become NaT and would be deleted; mixed offsets
    # raise pandas' "Mixed timezones detected". A span of at most an hour is
    # rewritten to each written day first. A wider span is named and rejected.
    rewritten = _reject_mixed_date_zones(data["Date"], symbol)
    if rewritten is not None:
        data["Date"] = rewritten
    data = _clean_dataframe(data, fill_prices=False)
    # Wall clock, not UTC. A tz-aware Date cannot be compared with naive curr_date.
    # Naive stamps are left untouched so an existing cache still filters as before.
    dates = data["Date"]
    if getattr(dates.dt, "tz", None) is not None:
        data["Date"] = _calendar_day_as_written(dates)
    data = data.loc[data["Date"] <= curr_date_dt].copy()
    if fill_prices:
        price_cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in data.columns]
        if price_cols:
            data[price_cols] = data[price_cols].ffill().bfill()

    return data


def _coerce_period_end(value) -> pd.Timestamp | None:
    period_end = pd.to_datetime(value, errors="coerce")
    if pd.isna(period_end):
        return None
    return period_end.tz_localize(None) if getattr(period_end, "tzinfo", None) else period_end


def get_fiscal_year_end_month_day(annual_period_ends) -> tuple[int, int]:
    """Infer fiscal-year-end month/day from annual period ends; fallback to Dec 31."""
    if annual_period_ends is None:
        return 12, 31

    valid_periods = []
    for value in annual_period_ends:
        period_end = _coerce_period_end(value)
        if period_end is not None:
            valid_periods.append((period_end.year, period_end.month, period_end.day))

    if not valid_periods:
        return 12, 31

    valid_periods.sort()
    _, month, day = valid_periods[-1]
    return month, day


def _statement_available_date(
    period_end: pd.Timestamp, freq: str, fiscal_year_end_month_day: tuple[int, int]
) -> pd.Timestamp:
    # Mirror y_finance's branching: anything other than "quarterly" fetched annual data.
    normalized_freq = (freq or "quarterly").strip().lower()
    if normalized_freq != "quarterly":
        return period_end + relativedelta(months=3)

    if _is_fiscal_year_end_quarter(period_end, fiscal_year_end_month_day):
        return period_end + relativedelta(months=3)
    return period_end + pd.Timedelta(days=45)


def _is_fiscal_year_end_quarter(
    period_end: pd.Timestamp, fiscal_year_end_month_day: tuple[int, int], tolerance_days: int = 7
) -> bool:
    """True if period_end is within a few days of the fiscal year end.

    Tolerates Feb-28/29 leap-year drift and 52/53-week fiscal years, where an
    exact month/day match would misclassify Q4 as a regular 45-day quarter.
    """
    month, day = fiscal_year_end_month_day
    for year in (period_end.year - 1, period_end.year, period_end.year + 1):
        last_day = pd.Timestamp(year=year, month=month, day=1).days_in_month
        anchor = pd.Timestamp(year=year, month=month, day=min(day, last_day))
        if abs((period_end.normalize() - anchor).days) <= tolerance_days:
            return True
    return False


def statement_period_is_visible(
    period_end_value, curr_date: str, freq: str, fiscal_year_end_month_day: tuple[int, int]
) -> bool:
    period_end = _coerce_period_end(period_end_value)
    if period_end is None:
        return False
    cutoff = pd.Timestamp(curr_date)
    available_date = _statement_available_date(period_end, freq, fiscal_year_end_month_day)
    return available_date < cutoff


def filter_financials_by_date(
    data: pd.DataFrame,
    curr_date: str,
    freq: str = "quarterly",
    annual_period_ends=None,
) -> pd.DataFrame:
    """Drop columns not yet approximately public by curr_date (strict boundary)."""
    if not curr_date or data.empty:
        return data

    fiscal_year_end_month_day = get_fiscal_year_end_month_day(annual_period_ends)
    mask = [
        statement_period_is_visible(col, curr_date, freq, fiscal_year_end_month_day)
        for col in data.columns
    ]
    return data.loc[:, mask]


def _insider_available_date(transaction_date: pd.Timestamp) -> pd.Timestamp:
    return transaction_date + (2 * US_BUSINESS_DAY)


def filter_insider_transactions_by_date(
    data: pd.DataFrame,
    curr_date: str,
    date_columns: tuple[str, ...] = ("Start Date", "Transaction Date", "Date", "transaction_date"),
) -> pd.DataFrame:
    """Filter insider rows to those approximately public before curr_date."""
    if not curr_date or data is None or data.empty:
        return data

    date_column = next((col for col in date_columns if col in data.columns), None)
    if not date_column:
        # Fail closed: without a transaction date we cannot prove availability.
        return data.iloc[0:0]

    cutoff = pd.Timestamp(curr_date)
    transaction_dates = pd.to_datetime(data[date_column], errors="coerce")
    available_dates = transaction_dates.apply(
        lambda d: _insider_available_date(d.tz_localize(None) if getattr(d, "tzinfo", None) else d)
        if not pd.isna(d)
        else pd.NaT
    )
    return data.loc[available_dates < cutoff]


class StockstatsUtils:
    @staticmethod
    def get_stock_stats(
        symbol: Annotated[str, "ticker symbol for the company"],
        indicator: Annotated[
            str, "quantitative indicators based off of the stock data for the company"
        ],
        curr_date: Annotated[
            str, "curr date for retrieving stock price data, YYYY-mm-dd"
        ],
    ):
        from .indicator_fallback import compute_indicator_with_fallback

        data = load_ohlcv(symbol, curr_date)
        curr_date_str = pd.to_datetime(curr_date).strftime("%Y-%m-%d")

        series = compute_indicator_with_fallback(data, indicator, symbol=symbol, curr_date=curr_date)

        if curr_date_str in series.index:
            return series.loc[curr_date_str]
        return "N/A: Not a trading day (weekend or holiday)"
