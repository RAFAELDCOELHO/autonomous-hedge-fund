import os
import requests
import pandas as pd
import json
from datetime import datetime
from io import StringIO

API_BASE_URL = "https://www.alphavantage.co/query"

def get_api_key() -> str:
    """Retrieve the API key for Alpha Vantage from environment variables."""
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if not api_key:
        raise ValueError("ALPHA_VANTAGE_API_KEY environment variable is not set.")
    return api_key

def format_datetime_for_api(date_input) -> str:
    """Convert various date formats to YYYYMMDDTHHMM format required by Alpha Vantage API."""
    if isinstance(date_input, str):
        # If already in correct format, return as-is
        if len(date_input) == 13 and 'T' in date_input:
            return date_input
        # Try to parse common date formats
        try:
            dt = datetime.strptime(date_input, "%Y-%m-%d")
            return dt.strftime("%Y%m%dT0000")
        except ValueError:
            try:
                dt = datetime.strptime(date_input, "%Y-%m-%d %H:%M")
                return dt.strftime("%Y%m%dT%H%M")
            except ValueError:
                raise ValueError(f"Unsupported date format: {date_input}")
    elif isinstance(date_input, datetime):
        return date_input.strftime("%Y%m%dT%H%M")
    else:
        raise ValueError(f"Date must be string or datetime object, got {type(date_input)}")

class AlphaVantageRateLimitError(Exception):
    """Exception raised when Alpha Vantage API rate limit is exceeded."""
    pass

def _make_api_request(function_name: str, params: dict) -> dict | str:
    """Helper function to make API requests and handle responses.
    
    Raises:
        AlphaVantageRateLimitError: When API rate limit is exceeded
    """
    # Create a copy of params to avoid modifying the original
    api_params = params.copy()
    api_params.update({
        "function": function_name,
        "apikey": get_api_key(),
        "source": "trading_agents",
    })
    
    # Handle entitlement parameter if present in params or global variable
    current_entitlement = globals().get('_current_entitlement')
    entitlement = api_params.get("entitlement") or current_entitlement
    
    if entitlement:
        api_params["entitlement"] = entitlement
    elif "entitlement" in api_params:
        # Remove entitlement if it's None or empty
        api_params.pop("entitlement", None)
    
    response = requests.get(API_BASE_URL, params=api_params)
    response.raise_for_status()

    response_text = response.text
    
    # Check if response is JSON (error responses are typically JSON)
    try:
        response_json = json.loads(response_text)
        # Check for rate limit error
        if "Information" in response_json:
            info_message = response_json["Information"]
            if "rate limit" in info_message.lower() or "api key" in info_message.lower():
                raise AlphaVantageRateLimitError(f"Alpha Vantage rate limit exceeded: {info_message}")
    except json.JSONDecodeError:
        # Response is not JSON (likely CSV data), which is normal
        pass

    return response_text



def _header_only_csv(csv_data: str) -> str:
    """CSV header and no body. A failed filter must not echo price rows."""
    lines = csv_data.splitlines()
    if not lines or not lines[0].strip():
        return ""
    return f"{lines[0]}\n"


def _local_calendar_day(values: pd.Series) -> pd.Series:
    """Calendar day written in the cell. A trailing ``Z`` or ``±hh:mm`` is removed, not converted.

    The clock stays put, so the day can differ from the UTC date:
    ``2024-01-12T20:00-05:00`` is 2024-01-12, not 2024-01-13.
    ``2024-01-09T22:00-05:00`` is 2024-01-09, not 2024-01-10.
    ``2024-01-13T00:30+09:00`` and ``2024-01-13T03:00+05:00`` are 2024-01-13,
    not 2024-01-12.
    """
    wall = (
        values.astype("string")
        .str.strip()
        .str.replace(r"(?i)(?:Z|[+-]\d{2}:?\d{2})$", "", regex=True)
    )
    # format="mixed": pandas 3 otherwise infers one format and nulls the rest.
    return pd.to_datetime(wall, errors="coerce", format="mixed").dt.normalize()


def _filter_csv_by_date_range(csv_data: str, start_date: str, end_date: str) -> str:
    """Keep rows whose first-column local calendar day is inside the window.

    The day is the date written in the cell. ``_local_calendar_day`` strips a
    trailing offset without shifting the clock, then floors to that calendar
    day, so ``2024-01-12 15:00`` is still 2024-01-12. An unreadable value is
    missing. A row stays when ``start_date <= that day <= end_date``. Cells are
    read as text (``dtype=str``, ``keep_default_na=False``) and the mask is
    applied to that frame, so the returned text is the original:
    ``2024-01-12T23:00-05:00,1`` is not rewritten, and a token such as ``NA``
    is not blanked.

    A parsed body is returned with LF record separators and always ends with
    a trailing newline, whether or not the input ended with one. That
    conversion applies only to the separators: a CRLF or LF inside a quoted
    cell is preserved.

    On any other failure the result is a header-only CSV. ``get_stock`` returns
    this string unchanged, and returning the original body would publish rows
    after ``end_date``.
    """
    if not csv_data or csv_data.strip() == "":
        return csv_data

    try:
        df = pd.read_csv(StringIO(csv_data), dtype=str, keep_default_na=False)
        date_col = df.columns[0]
        day = _local_calendar_day(df[date_col])
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        keep = day.notna() & (day >= start_dt) & (day <= end_dt)
        return df.loc[keep].to_csv(index=False)
    except Exception as exc:
        print(f"Warning: Failed to filter CSV data by date range: {exc}")
        return _header_only_csv(csv_data)
