import time
import logging

import pandas as pd
import yfinance as yf
from dateutil.relativedelta import relativedelta
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
from yfinance.exceptions import YFRateLimitError
from typing import Annotated, NoReturn
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


def _empty_date_mask(values: pd.Series) -> pd.Series:
    """True for a missing Date or a blank string. Those stay droppable NaT.

    A whitespace-only cell is blank. Any other token is non-empty, so a later
    NaT is a defect rather than a silent drop.
    """
    missing = values.isna()
    if pd.api.types.is_datetime64_any_dtype(values):
        return missing
    text = values.astype("string").str.strip()
    blank = text.isna() | text.eq("")
    return missing | blank.fillna(False)


def _raise_date_defect(source: str | None, reason: str, offenders: pd.Series) -> NoReturn:
    """Fail closed, naming the file or ticker, each row, and the raw value.

    DataDefectError lives in runner, which imports load_ohlcv before that
    class exists, so this import cannot sit at module level.
    """
    from tradingagents.backtest.runner import DataDefectError

    where = source or "OHLCV Date column"
    details = "; ".join(f"row {idx} raw={raw!r}" for idx, raw in offenders.items())
    raise DataDefectError(f"{where}: {reason}: {details}")


def _parse_ohlcv_dates(values: pd.Series, source: str | None) -> pd.Series:
    """Parse Date as ISO8601. Non-empty failures raise; blanks stay NaT.

    ``format="mixed"`` is not used: it chooses month and day per cell, so a
    column of ``01/02/2024`` and ``13/02/2024`` is read as Jan 2 and Feb 13.
    Naive timestamps mixed with offsets raise even under ``errors="coerce"``
    (pandas: ``Mixed timezones``); that becomes the same fail-closed error,
    naming each non-empty row.
    """
    try:
        # ISO8601, not mixed: mixed decides month/day independently per cell.
        parsed = pd.to_datetime(values, format="ISO8601", errors="coerce")
    except ValueError as exc:
        if "mixed timezones" in str(exc).lower():
            _raise_date_defect(
                source,
                "mixed timezones in Date",
                values.loc[~_empty_date_mask(values)],
            )
        raise
    empty = _empty_date_mask(values)
    bad = parsed.isna() & ~empty
    if bad.any():
        _raise_date_defect(source, "Date is not ISO8601", values.loc[bad])
    return parsed


def _clean_dataframe(
    data: pd.DataFrame,
    fill_prices: bool = True,
    *,
    source: str | None = None,
) -> pd.DataFrame:
    """Normalize a stock DataFrame for stockstats: parse dates, drop blank rows, fill price gaps.

    fill_prices=False leaves NaN prices (other than Close) in place, so a caller
    can fail closed on a missing Open instead of trading on a filled one.
    A date defect raises DataDefectError naming ``source`` (ticker and file,
    when the caller has them).
    """
    data = data.copy()
    data["Date"] = _parse_ohlcv_dates(data["Date"], source)
    data = data.dropna(subset=["Date"])

    price_cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in data.columns]
    data[price_cols] = data[price_cols].apply(pd.to_numeric, errors="coerce")
    data = data.dropna(subset=["Close"])
    if fill_prices:
        data[price_cols] = data[price_cols].ffill().bfill()

    return data


def load_ohlcv(symbol: str, curr_date: str, fill_prices: bool = True) -> pd.DataFrame:
    """Fetch OHLCV data with caching, filtered to prevent look-ahead bias.

    Downloads 5 years of data up to today and caches per symbol. On
    subsequent calls the cache is reused. Dates are parsed as ISO8601
    (``format="ISO8601"``, ``errors="coerce"``). A column that mixes
    ``2024-01-02T00:00:00`` and ``2024-01-03 00:00:00`` — naive, or every
    cell carrying one offset — keeps each written day.
    ``2024-01-04T22:00:00-05:00`` stays 4 Jan. A non-empty cell that is not
    ISO8601 (a US date such as ``12/29/2023``, or garbage) raises
    DataDefectError naming the ticker, the cache file, the row, and the raw
    value. The same error is raised when naive timestamps are mixed with
    offset timestamps. An empty or missing Date is still coerced to NaT and
    dropped. Rows after curr_date are filtered out so backtests never see
    future prices. All of OHLC comes from one auto_adjust=True download; the
    cache file name records that setting. With fill_prices=True (the
    default), gaps in Open, High, Low, Close, and Volume are forward-filled
    and then backfilled only on the rows still dated on or before
    curr_date. The in-window bfill does use later bars up to curr_date. A
    bar after curr_date is not a source, so a missing price is never taken
    from a later bar after curr_date. fill_prices=False leaves those gaps
    as NaN.
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

    # Cut the window, then fill. Backfill before the cut takes a leading NaN
    # from a bar after curr_date.
    data = _clean_dataframe(
        data,
        fill_prices=False,
        source=f"{symbol} ({data_file})",
    )
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
