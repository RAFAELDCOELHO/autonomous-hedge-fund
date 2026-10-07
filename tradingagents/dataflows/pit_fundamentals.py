import pandas as pd

from .stockstats_utils import _statement_available_date


def _safe_value(frame: pd.DataFrame | None, row: str, column) -> float | None:
    if frame is None or frame.empty or column not in frame.columns or row not in frame.index:
        return None
    value = frame.at[row, column]
    if pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _append_line(lines: list[str], label: str, value: float | None, precision: str) -> None:
    if value is None or pd.isna(value):
        return
    lines.append(f"{label}: {format(value, precision)}")


def build_point_in_time_fundamentals(
    ticker,
    curr_date,
    history: pd.DataFrame,
    income_stmt: pd.DataFrame,
    balance_sheet: pd.DataFrame,
    cashflow: pd.DataFrame,
) -> str:
    ticker_upper = str(ticker).upper()
    cutoff = pd.Timestamp(curr_date)

    basis_column = None
    if income_stmt is not None and not income_stmt.empty:
        valid_columns = [pd.to_datetime(col, errors="coerce") for col in income_stmt.columns]
        valid_columns = [col.tz_localize(None) if getattr(col, "tzinfo", None) else col for col in valid_columns if not pd.isna(col)]
        if valid_columns:
            basis_column = max(valid_columns)

    header = [f"# Company Fundamentals for {ticker_upper} (point-in-time as of {curr_date})"]

    if basis_column is not None:
        visible_date = _statement_available_date(basis_column, "annual", (basis_column.month, basis_column.day))
        header.append(
            "# Statement basis: fiscal year ending "
            f"{basis_column.strftime('%Y-%m-%d')} (treated as public from {visible_date.strftime('%Y-%m-%d')}; "
            "P4.3 availability rule)"
        )
    else:
        header.append(
            f"# No fiscal-year statements publicly available as of {curr_date} under the P4.3 availability rule."
        )

    history_df = history.copy() if history is not None else pd.DataFrame()
    if not history_df.empty:
        if history_df.index.tz is not None:
            history_df.index = history_df.index.tz_localize(None)
        history_df = history_df.loc[history_df.index.normalize() <= cutoff]
        history_df = history_df.sort_index()

    close = None
    close_date = None
    if not history_df.empty and "Close" in history_df.columns:
        close = pd.to_numeric(history_df["Close"], errors="coerce").dropna()
        if not close.empty:
            close_date = close.index[-1]
            close = float(close.iloc[-1])
        else:
            close = None

    if close is not None and close_date is not None:
        header.append(f"# Price basis: unadjusted close on {close_date.strftime('%Y-%m-%d')}")
    else:
        header.append("# Price basis: unavailable")

    header.append("")

    if basis_column is None:
        return "\n".join(header)

    shares = _safe_value(balance_sheet, "Ordinary Shares Number", basis_column)
    if shares is None or shares <= 0:
        shares = _safe_value(balance_sheet, "Share Issued", basis_column)
        if shares is not None and shares <= 0:
            shares = None
    equity = _safe_value(balance_sheet, "Stockholders Equity", basis_column)
    total_assets = _safe_value(balance_sheet, "Total Assets", basis_column)
    total_debt = _safe_value(balance_sheet, "Total Debt", basis_column)
    current_assets = _safe_value(balance_sheet, "Current Assets", basis_column)
    current_liabilities = _safe_value(balance_sheet, "Current Liabilities", basis_column)

    eps = _safe_value(income_stmt, "Diluted EPS", basis_column)
    if eps is None:
        eps = _safe_value(income_stmt, "Basic EPS", basis_column)
    revenue = _safe_value(income_stmt, "Total Revenue", basis_column)
    gross_profit = _safe_value(income_stmt, "Gross Profit", basis_column)
    ebitda = _safe_value(income_stmt, "EBITDA", basis_column)
    net_income = _safe_value(income_stmt, "Net Income Common Stockholders", basis_column)
    if net_income is None:
        net_income = _safe_value(income_stmt, "Net Income", basis_column)
    operating_income = _safe_value(income_stmt, "Operating Income", basis_column)
    free_cash_flow = _safe_value(cashflow, "Free Cash Flow", basis_column)

    market_cap = None
    if close is not None and shares is not None:
        market_cap = close * shares

    lines: list[str] = []
    _append_line(lines, "Market Cap", market_cap, ".0f")

    pe_ratio = None
    if close is not None and eps is not None and eps > 0:
        pe_ratio = close / eps
    _append_line(lines, "PE Ratio (latest FY)", pe_ratio, ".4f")

    price_to_book = None
    if market_cap is not None and equity is not None and equity > 0:
        price_to_book = market_cap / equity
    _append_line(lines, "Price to Book", price_to_book, ".4f")

    _append_line(lines, "EPS (latest FY, diluted)", eps, ".4f")

    if not history_df.empty:
        window = history_df.loc[
            (history_df.index > cutoff - pd.Timedelta(days=365)) & (history_df.index <= cutoff)
        ]
        if not window.empty:
            if "High" in window.columns:
                high = pd.to_numeric(window["High"], errors="coerce").dropna()
                if not high.empty:
                    _append_line(lines, "52 Week High", float(high.max()), ".4f")
            if "Low" in window.columns:
                low = pd.to_numeric(window["Low"], errors="coerce").dropna()
                if not low.empty:
                    _append_line(lines, "52 Week Low", float(low.min()), ".4f")

        if "Close" in history_df.columns:
            closes = pd.to_numeric(history_df["Close"], errors="coerce").dropna()
            if len(closes) >= 50:
                _append_line(lines, "50 Day Average", float(closes.tail(50).mean()), ".4f")
            if len(closes) >= 200:
                _append_line(lines, "200 Day Average", float(closes.tail(200).mean()), ".4f")

    _append_line(lines, "Revenue (latest FY)", revenue, ".0f")
    _append_line(lines, "Gross Profit (latest FY)", gross_profit, ".0f")
    _append_line(lines, "EBITDA (latest FY)", ebitda, ".0f")
    _append_line(lines, "Net Income (latest FY)", net_income, ".0f")

    profit_margin = None
    if net_income is not None and revenue is not None and revenue > 0:
        profit_margin = net_income / revenue
    _append_line(lines, "Profit Margin", profit_margin, ".4f")

    operating_margin = None
    if operating_income is not None and revenue is not None and revenue > 0:
        operating_margin = operating_income / revenue
    _append_line(lines, "Operating Margin", operating_margin, ".4f")

    roe = None
    if net_income is not None and equity is not None and equity > 0:
        roe = net_income / equity
    _append_line(lines, "Return on Equity", roe, ".4f")

    roa = None
    if net_income is not None and total_assets is not None and total_assets > 0:
        roa = net_income / total_assets
    _append_line(lines, "Return on Assets", roa, ".4f")

    debt_to_equity = None
    if total_debt is not None and equity is not None and equity > 0:
        debt_to_equity = total_debt / equity
    _append_line(lines, "Debt to Equity", debt_to_equity, ".4f")

    current_ratio = None
    if current_assets is not None and current_liabilities is not None and current_liabilities > 0:
        current_ratio = current_assets / current_liabilities
    _append_line(lines, "Current Ratio", current_ratio, ".4f")

    book_value_per_share = None
    if equity is not None and shares is not None and shares > 0:
        book_value_per_share = equity / shares
    _append_line(lines, "Book Value per Share", book_value_per_share, ".4f")

    _append_line(lines, "Free Cash Flow (latest FY)", free_cash_flow, ".0f")

    return "\n".join(header + lines)
