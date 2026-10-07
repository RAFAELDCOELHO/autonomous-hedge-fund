"""P4.2 regression tests: point-in-time fundamentals (F4 PR4).

All tests are offline: yfinance is replaced by an in-memory fake ticker.
They target the public vendor function ``y_finance.get_fundamentals`` and the
agent tool wrapper, so they are agnostic to internal helper names.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from tradingagents.dataflows import y_finance

FY = [pd.Timestamp("2021-12-31"), pd.Timestamp("2022-12-31"),
      pd.Timestamp("2023-12-31"), pd.Timestamp("2024-12-31")]

# Per-fiscal-year values. FY2023/FY2024 carry sentinel magnitudes so any
# leak into a 2024-Q1 decision is unambiguous in the output.
INCOME = pd.DataFrame(
    {
        FY[0]: [80e9, 30e9, 20e9, 25e9, 10e9, 1.00],
        FY[1]: [100e9, 40e9, 30e9, 35e9, 20e9, 2.00],
        FY[2]: [777e9, 777e9, 777e9, 777e9, 777e9, 77.70],
        FY[3]: [888e9, 888e9, 888e9, 888e9, 888e9, 88.80],
    },
    index=["Total Revenue", "Gross Profit", "Operating Income", "EBITDA",
           "Net Income Common Stockholders", "Diluted EPS"],
)
BALANCE = pd.DataFrame(
    {
        FY[0]: [9e9, 150e9, 300e9, 50e9, 60e9, 40e9],
        FY[1]: [10e9, 200e9, 400e9, 100e9, 80e9, 40e9],
        FY[2]: [7.77e9, 777e9, 777e9, 777e9, 777e9, 777e9],
        FY[3]: [8.88e9, 888e9, 888e9, 888e9, 888e9, 888e9],
    },
    index=["Ordinary Shares Number", "Stockholders Equity", "Total Assets",
           "Total Debt", "Current Assets", "Current Liabilities"],
)
CASHFLOW = pd.DataFrame(
    {FY[0]: [5e9], FY[1]: [15e9], FY[2]: [777e9], FY[3]: [888e9]},
    index=["Free Cash Flow"],
)

SENTINEL_INFO = {
    "longName": "Sentinel Corp", "sector": "Energy", "industry": "Oil",
    "marketCap": 123456789, "trailingPE": 66.6, "forwardPE": 55.5,
    "pegRatio": 4.4, "forwardEps": 3.3, "beta": 2.2, "dividendYield": 0.11,
}


def _price_frame() -> pd.DataFrame:
    # Business days 2022-06-01..2024-12-31. Raw close = 20 + day index * 0.01
    # up to 2024-03-28; every row after 2024-03-28 is a 9999 sentinel.
    idx = pd.bdate_range("2022-06-01", "2024-12-31")
    close = 20.0 + np.arange(len(idx)) * 0.01
    future = idx > pd.Timestamp("2024-03-28")
    close[future] = 9999.0
    return pd.DataFrame(
        {"Open": close, "High": close + 0.5, "Low": close - 0.5,
         "Close": close, "Adj Close": close * 0.8, "Volume": 1000},
        index=idx.tz_localize("America/Sao_Paulo"),
    )


class FakeTicker:
    def __init__(self):
        self.info_reads = 0
        self.history_calls = []

    @property
    def info(self):
        self.info_reads += 1
        return dict(SENTINEL_INFO)

    balance_sheet = BALANCE
    income_stmt = INCOME
    cashflow = CASHFLOW
    quarterly_balance_sheet = pd.DataFrame()
    quarterly_income_stmt = pd.DataFrame()
    quarterly_cashflow = pd.DataFrame()

    def history(self, *args, **kwargs):
        # yfinance default is auto_adjust=True (Close becomes dividend-adjusted).
        # The fake deliberately ignores start/end so the implementation must
        # cap rows at curr_date itself.
        self.history_calls.append(kwargs)
        df = _price_frame()
        if kwargs.get("auto_adjust", True):
            df = df.assign(Close=df["Adj Close"]).drop(columns=["Adj Close"])
        return df


@pytest.fixture
def fake(monkeypatch):
    ticker = FakeTicker()

    class _SentinelDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2099, 1, 1, 12, 34, 56)

    monkeypatch.setattr(y_finance, "datetime", _SentinelDatetime)
    monkeypatch.setattr(y_finance, "yf_retry", lambda fn, *a, **k: fn())
    monkeypatch.setattr(y_finance.yf, "Ticker", lambda _sym: ticker)
    return ticker


def _fields(out: str) -> dict[str, float]:
    fields = {}
    for line in out.splitlines():
        if line.startswith("#") or ": " not in line:
            continue
        label, value = line.split(": ", 1)
        try:
            fields[label.strip()] = float(value.replace(",", ""))
        except ValueError:
            pass
    return fields


def _close_on(day: str) -> float:
    df = _price_frame()
    df.index = df.index.tz_localize(None)
    return float(df.loc[:day, "Close"].iloc[-1])


def test_p42_never_reads_current_info_snapshot(fake):
    out = y_finance.get_fundamentals("PETR4.SA", "2024-02-15")
    assert fake.info_reads == 0
    assert not out.startswith("Error")
    assert "Sentinel Corp" not in out
    assert not {123456789.0, 66.6} & set(_fields(out).values())


def test_p42_drops_fields_not_reconstructable_point_in_time(fake):
    out = y_finance.get_fundamentals("PETR4.SA", "2024-02-15")
    for label in ("Forward PE", "Forward EPS", "PEG Ratio", "Beta", "Dividend Yield"):
        assert label not in out
    assert not {55.5, 4.4, 3.3, 2.2, 0.11} & set(_fields(out).values())


def test_p42_statement_basis_is_latest_publicly_available_fiscal_year(fake):
    # FY2023 (period end 2023-12-31) is public only from 2024-04-01 under the
    # P4.3 annual rule (period_end + 3 months, strict <). On 2024-03-28 the
    # latest visible fiscal year is FY2022.
    out = y_finance.get_fundamentals("PETR4.SA", "2024-03-28")
    f = _fields(out)
    assert "fiscal year ending 2022-12-31" in out
    assert f["Revenue (latest FY)"] == pytest.approx(100e9)
    assert f["Net Income (latest FY)"] == pytest.approx(20e9)
    assert f["Free Cash Flow (latest FY)"] == pytest.approx(15e9)
    assert f["EPS (latest FY, diluted)"] == pytest.approx(2.0)
    assert f["Profit Margin"] == pytest.approx(0.20)
    assert f["Operating Margin"] == pytest.approx(0.30)
    assert f["Return on Equity"] == pytest.approx(0.10)
    assert f["Return on Assets"] == pytest.approx(0.05)
    assert f["Debt to Equity"] == pytest.approx(0.50)
    assert f["Current Ratio"] == pytest.approx(2.0)
    assert f["Book Value per Share"] == pytest.approx(20.0)
    leaked = {777e9, 888e9, 77.7, 88.8, 7.77e9, 8.88e9}
    assert not leaked & set(f.values())
    assert "2023-12-31" not in out and "2024-12-31" not in out


def test_p42_annual_boundary_matches_p43_rule(fake):
    on_deadline = y_finance.get_fundamentals("PETR4.SA", "2024-03-31")
    after = y_finance.get_fundamentals("PETR4.SA", "2024-04-01")
    assert "fiscal year ending 2022-12-31" in on_deadline
    assert "fiscal year ending 2023-12-31" in after


def test_p42_price_fields_use_unadjusted_close_capped_at_curr_date(fake):
    curr = "2024-02-15"
    out = y_finance.get_fundamentals("PETR4.SA", curr)
    f = _fields(out)
    close = _close_on(curr)
    assert "unadjusted close on 2024-02-15" in out
    assert f["Market Cap"] == pytest.approx(close * 10e9, rel=1e-3)
    assert f["PE Ratio (latest FY)"] == pytest.approx(close / 2.0, rel=1e-3)
    assert f["Price to Book"] == pytest.approx(close * 10e9 / 200e9, rel=1e-3)
    df = _price_frame()
    df.index = df.index.tz_localize(None)
    window = df.loc[(df.index > pd.Timestamp(curr) - pd.Timedelta(days=365))
                    & (df.index <= pd.Timestamp(curr))]
    assert f["52 Week High"] == pytest.approx(window["High"].max(), rel=1e-4)
    assert f["52 Week Low"] == pytest.approx(window["Low"].min(), rel=1e-4)
    upto = df.loc[:curr, "Close"]
    assert f["50 Day Average"] == pytest.approx(upto.tail(50).mean(), rel=1e-4)
    assert f["200 Day Average"] == pytest.approx(upto.tail(200).mean(), rel=1e-4)
    assert max(f.values()) < 9999 * 10e9 * 0.5  # no 9999 sentinel close leaked into Market Cap
    assert f["52 Week High"] < 9999
    assert any(c.get("auto_adjust") is False for c in fake.history_calls)


def test_p42_weekend_curr_date_uses_last_close_on_or_before(fake):
    out = y_finance.get_fundamentals("PETR4.SA", "2024-02-17")  # Saturday
    assert "unadjusted close on 2024-02-16" in out
    assert _fields(out)["Market Cap"] == pytest.approx(_close_on("2024-02-16") * 10e9, rel=1e-3)


def test_p42_no_visible_fiscal_year_omits_statement_fields(fake):
    # On 2022-03-15 FY2021 (avail 2022-03-31) is not yet public: no statements.
    out = y_finance.get_fundamentals("PETR4.SA", "2022-03-15")
    assert fake.info_reads == 0
    assert "No fiscal-year statements publicly available as of 2022-03-15" in out
    f = _fields(out)
    for label in ("Revenue (latest FY)", "Market Cap", "PE Ratio (latest FY)", "Book Value per Share"):
        assert label not in f


def test_p42_missing_curr_date_fails_closed(fake):
    out = y_finance.get_fundamentals("PETR4.SA", None)
    assert fake.info_reads == 0
    assert "curr_date" in out
    assert _fields(out) == {}


def test_p42_output_has_no_wall_clock_stamp(fake):
    out = y_finance.get_fundamentals("PETR4.SA", "2024-02-15")
    assert "2099" not in out
    assert "Data retrieved on" not in out
