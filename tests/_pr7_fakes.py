"""Offline fake vendors for the PR7 tests (no network, no new dependency).

The fake vendor bars are built from the *test's own* small holiday tables so
the data looks like real NYSE / B3 output. The expected D-1 dates in the tests
are hard-coded literals, never derived from these tables.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import pandas as pd

SENTINEL = 987654.32          # value that only exists on decision day D
SENTINEL_TEXT = "987654.32"
CUTOFF_SENTINEL = 876543.21   # value that only exists on the cutoff day D-1 (macro daily)
CUTOFF_SENTINEL_TEXT = "876543.21"

# Exchange closures used only to shape fake vendor data (2023-09 .. 2025-06).
NYSE_CLOSED = {
    "2023-09-04", "2023-11-23", "2023-12-25", "2024-01-01", "2024-01-15", "2024-02-19",
    "2024-03-29", "2024-05-27", "2024-06-19", "2024-07-04", "2024-09-02", "2024-11-28",
    "2024-12-25", "2025-01-01", "2025-01-09", "2025-01-20", "2025-02-17", "2025-04-18",
    "2025-05-26",
}
B3_CLOSED = {
    "2023-09-07", "2023-10-12", "2023-11-02", "2023-11-15", "2023-12-25", "2023-12-29",
    "2024-01-01", "2024-02-12", "2024-02-13", "2024-03-29", "2024-05-01", "2024-05-30",
    "2024-11-15", "2024-11-20", "2024-12-24", "2024-12-25", "2024-12-31", "2025-01-01",
    "2025-03-03", "2025-03-04", "2025-04-18", "2025-04-21", "2025-05-01",
}


def vendor_sessions(ticker: str, first: str = "2023-09-01", last: str = "2025-06-30") -> pd.DatetimeIndex:
    closed = B3_CLOSED if ticker.upper().endswith(".SA") else NYSE_CLOSED
    days = pd.bdate_range(first, last)
    return days[~days.strftime("%Y-%m-%d").isin(closed)]


def make_bars(ticker: str) -> pd.DataFrame:
    """Adjusted OHLCV bars. Open and Close follow different paths on purpose,
    so open-to-open and close-to-close returns never coincide."""
    idx = vendor_sessions(ticker)
    i = pd.Series(range(len(idx)), index=idx, dtype=float)
    df = pd.DataFrame(
        {
            "Open": 50.0 + i,
            "High": 400.0 + 3.0 * i,
            "Low": 10.0 + 0.1 * i,
            "Close": 80.0 + 2.0 * i,
            "Volume": 1_000_000.0 + i,
        },
        index=idx,
    )
    df.index.name = "Date"
    return df


@dataclass
class FakeYahoo:
    """Stands in for yfinance.download / Ticker / Search for one test."""

    bars: dict = field(default_factory=dict)            # ticker -> adjusted bars
    raw_factor: Optional[Callable[[pd.Timestamp], float]] = None
    download_calls: list = field(default_factory=list)
    history_calls: list = field(default_factory=list)
    statements: dict = field(default_factory=dict)      # attr -> DataFrame
    insider: Optional[pd.DataFrame] = None
    news: list = field(default_factory=list)

    def frame(self, ticker: str, auto_adjust: bool = True) -> pd.DataFrame:
        if ticker not in self.bars:
            self.bars[ticker] = make_bars(ticker)
        df = self.bars[ticker].copy()
        if not auto_adjust:
            factor = self.raw_factor or (lambda _d: 1.0)
            f = pd.Series([factor(d) for d in df.index], index=df.index)
            for col in ("Open", "High", "Low", "Close"):
                df[col] = df[col] * f
            df["Adj Close"] = self.bars[ticker]["Close"]
        return df

    # yfinance.download(symbol, start=, end=, auto_adjust=, ...)
    def download(self, symbol, start=None, end=None, auto_adjust=True, **kwargs):
        self.download_calls.append({"symbol": symbol, "auto_adjust": auto_adjust, **kwargs})
        df = self.frame(symbol, auto_adjust)
        if start is not None:
            df = df.loc[pd.Timestamp(start):]
        if end is not None:
            df = df.loc[: pd.Timestamp(end) - pd.Timedelta(days=1)]  # yfinance end is exclusive
        return df

    def ticker(self, symbol):
        return _FakeTicker(self, symbol)

    def search(self, query=None, news_count=10, **kwargs):
        return _FakeSearch(self.news)


class _FakeSearch:
    def __init__(self, news):
        self.news = list(news)


class _FakeTicker:
    def __init__(self, yahoo: FakeYahoo, symbol: str):
        self._y = yahoo
        self._symbol = symbol

    def history(self, start=None, end=None, auto_adjust=True, **kwargs):
        self._y.history_calls.append({"symbol": self._symbol, "auto_adjust": auto_adjust})
        df = self._y.frame(self._symbol, auto_adjust)
        if start is not None:
            df = df.loc[pd.Timestamp(start):]
        if end is not None:
            df = df.loc[: pd.Timestamp(end) - pd.Timedelta(days=1)]
        return df

    def get_news(self, count=20, **kwargs):
        return list(self._y.news)

    @property
    def insider_transactions(self):
        return self._y.insider

    def __getattr__(self, name):
        statements = self.__dict__["_y"].statements
        if name in statements:
            return statements[name]
        raise AttributeError(name)


def install_fake_yahoo(monkeypatch, tmp_path, yahoo: Optional[FakeYahoo] = None) -> FakeYahoo:
    """Route every yfinance entry point to `yahoo` and isolate the OHLCV cache."""
    import yfinance

    from tradingagents.dataflows import config as df_config

    yahoo = yahoo or FakeYahoo()
    monkeypatch.setattr(yfinance, "download", yahoo.download)
    monkeypatch.setattr(yfinance, "Ticker", yahoo.ticker)
    monkeypatch.setattr(yfinance, "Search", yahoo.search)
    df_config.initialize_config()
    monkeypatch.setitem(df_config._config, "data_cache_dir", str(tmp_path / "ohlcv-cache"))
    monkeypatch.setitem(
        df_config._config,
        "data_vendors",
        {
            "core_stock_apis": "yfinance",
            "technical_indicators": "yfinance",
            "fundamental_data": "yfinance",
            "news_data": "yfinance",
        },
    )
    monkeypatch.setitem(df_config._config, "tool_vendors", {})
    return yahoo


class RecordingAgent:
    """decide_fn stand-in. Records the date/window the runner hands the agent
    and optionally plays a tool-calling agent keyed to that date."""

    def __init__(self, actions=None, on_call: Optional[Callable[[str], None]] = None):
        self._actions = actions
        self._on_call = on_call
        self.dates: list[str] = []
        self.windows: list = []
        self.errors: list = []

    def __call__(self, *args, **kwargs):
        import _pr7_api as api

        date = api.agent_date(args, kwargs)
        self.dates.append(date)
        self.windows.append(api.agent_window(args, kwargs))
        if self._on_call is not None:
            try:
                self._on_call(date)
            except Exception as exc:  # the runner would swallow it as a decision error
                self.errors.append(exc)
        n = len(self.dates) - 1
        if self._actions is None:
            return "HOLD"
        if callable(self._actions):
            return self._actions(n, date)
        return self._actions[n] if n < len(self._actions) else "HOLD"
