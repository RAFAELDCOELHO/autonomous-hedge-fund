"""Regression tests for temporal-integrity fixes (P4.8, P4.9, P4.13)."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from tradingagents.agents.utils import core_stock_tools, fingpt_tool
from tradingagents.dataflows import y_finance


class _SentinelDatetime(datetime):
    @classmethod
    def now(cls, tz=None):  # pragma: no cover - regression guard only
        return cls(2099, 1, 1, 12, 34, 56)


class _FakeTicker:
    def history(self, start=None, end=None):
        return pd.DataFrame(
            {
                "Open": [10.1234],
                "High": [11.1234],
                "Low": [9.1234],
                "Close": [10.5678],
                "Adj Close": [10.5678],
            },
            index=pd.DatetimeIndex(["2024-01-03"]),
        )

    @property
    def info(self):
        return {"longName": "Acme Corp", "marketCap": 123456789}

    @property
    def quarterly_balance_sheet(self):
        return pd.DataFrame(
            {"2024-03-31": [1000], "2023-12-31": [900]},
            index=["Total Assets"],
        )

    @property
    def balance_sheet(self):
        return self.quarterly_balance_sheet

    @property
    def quarterly_cashflow(self):
        return pd.DataFrame(
            {"2024-03-31": [250], "2023-12-31": [200]},
            index=["Operating Cash Flow"],
        )

    @property
    def cashflow(self):
        return self.quarterly_cashflow

    @property
    def quarterly_income_stmt(self):
        return pd.DataFrame(
            {"2024-03-31": [400], "2023-12-31": [350]},
            index=["Net Income"],
        )

    @property
    def income_stmt(self):
        return self.quarterly_income_stmt

    @property
    def insider_transactions(self):
        return pd.DataFrame(
            {"Shares": [100], "Price": [10.5]},
            index=[0],
        )


def _setup_fake_yf(monkeypatch):
    monkeypatch.setattr(y_finance, "datetime", _SentinelDatetime)
    monkeypatch.setattr(y_finance, "yf_retry", lambda fn: fn())
    monkeypatch.setattr(y_finance, "filter_financials_by_date", lambda df, _: df)
    monkeypatch.setattr(y_finance.yf, "Ticker", lambda _: _FakeTicker())


@pytest.mark.parametrize(
    ("formatter_name", "call_args", "call_kwargs"),
    [
        (
            "get_YFin_data_online",
            ("AAPL", "2024-01-01", "2024-01-03"),
            {},
        ),
        ("get_fundamentals", ("AAPL",), {}),
        ("get_balance_sheet", ("AAPL",), {"freq": "quarterly", "curr_date": "2024-04-01"}),
        ("get_cashflow", ("AAPL",), {"freq": "quarterly", "curr_date": "2024-04-01"}),
        (
            "get_income_statement",
            ("AAPL",),
            {"freq": "quarterly", "curr_date": "2024-04-01"},
        ),
        ("get_insider_transactions", ("AAPL",), {}),
    ],
)
def test_p48_all_yfinance_formatters_exclude_wall_clock_stamp(
    monkeypatch,
    formatter_name,
    call_args,
    call_kwargs,
):
    _setup_fake_yf(monkeypatch)
    formatter = getattr(y_finance, formatter_name)
    out = formatter(*call_args, **call_kwargs)

    assert "Data retrieved on" not in out
    assert "2099-01-01 12:34:56" not in out


def test_p48_yfinance_module_source_has_no_stamp_literal():
    source = Path(y_finance.__file__).read_text(encoding="utf-8")
    assert "Data retrieved on" not in source


def test_p48_remove_only_wall_clock_stamp_from_stock_tool_output(monkeypatch):
    class _SentinelDatetime(datetime):
        @classmethod
        def now(cls, tz=None):  # pragma: no cover - regression guard only
            return cls(2099, 1, 1, 12, 34, 56)

    class _FakeTicker:
        def history(self, start=None, end=None):
            return pd.DataFrame(
                {
                    "Open": [10.1234],
                    "High": [11.1234],
                    "Low": [9.1234],
                    "Close": [10.5678],
                    "Adj Close": [10.5678],
                },
                index=pd.DatetimeIndex(["2024-01-03"]),
            )

    monkeypatch.setattr(y_finance, "datetime", _SentinelDatetime)
    monkeypatch.setattr(y_finance, "yf_retry", lambda fn: fn())
    monkeypatch.setattr(y_finance.yf, "Ticker", lambda _: _FakeTicker())

    out = y_finance.get_YFin_data_online("AAPL", "2024-01-01", "2024-01-03")

    expected_header = "# Stock data for AAPL from 2024-01-01 to 2024-01-03\n"
    expected_header += "# Total records: 1\n"
    csv_body = _FakeTicker().history().round(2).to_csv()
    expected_with_stamp = (
        expected_header
        + "# Data retrieved on: 2099-01-01 12:34:56\n\n"
        + csv_body
    )

    assert out == expected_with_stamp.replace(
        "# Data retrieved on: 2099-01-01 12:34:56\n", ""
    )
    assert "2099-01-01 12:34:56" not in out


def test_p49_cap_is_inclusive_of_curr_date_and_excludes_next_day(monkeypatch):
    class _FakeTicker:
        def history(self, start=None, end=None):
            all_rows = pd.DataFrame(
                {
                    "Open": [100, 101, 102],
                    "High": [110, 111, 112],
                    "Low": [90, 91, 92],
                    "Close": [105, 106, 107],
                    "Adj Close": [105, 106, 107],
                },
                index=pd.DatetimeIndex(["2024-01-02", "2024-01-03", "2024-01-04"]),
            )
            start_dt = pd.Timestamp(start)
            end_dt = pd.Timestamp(end)
            return all_rows[(all_rows.index >= start_dt) & (all_rows.index < end_dt)]

    monkeypatch.setattr(y_finance, "yf_retry", lambda fn: fn())
    monkeypatch.setattr(y_finance.yf, "Ticker", lambda _: _FakeTicker())

    seen = {}

    def _route(method, symbol, start_date, end_date):
        seen["args"] = (method, symbol, start_date, end_date)
        return y_finance.get_YFin_data_online(symbol, start_date, end_date)

    monkeypatch.setattr(core_stock_tools, "route_to_vendor", _route)

    out = core_stock_tools.get_stock_data.invoke(
        {
            "symbol": "AAPL",
            "start_date": "2024-01-01",
            "end_date": "2024-01-10",
            "curr_date": "2024-01-03",
        }
    )

    assert seen["args"] == ("get_stock_data", "AAPL", "2024-01-01", "2024-01-03")
    rows = [line.split(",")[0] for line in out.splitlines() if line.startswith("2024-")]
    assert "2024-01-03" in rows
    assert "2024-01-04" not in rows


def test_p413_fingpt_news_call_uses_start_and_end_dates(monkeypatch):
    seen = {}

    def _route(method, symbol, start_date, end_date):
        seen["args"] = (method, symbol, start_date, end_date)
        return "Company announces strong quarterly growth outlook"

    monkeypatch.setattr(fingpt_tool, "route_to_vendor", _route)
    monkeypatch.setattr(
        fingpt_tool,
        "get_fingpt_sentiment",
        lambda headlines, symbol, curr_date: {
            "scores": [1],
            "n": 1,
            "n_failed": 0,
            "label": "positive",
            "score": 1.0,
        },
    )

    out = fingpt_tool.get_fingpt_sentiment_tool.invoke(
        {"symbol": "PETR4", "curr_date": "2024-01-10", "look_back_days": 2}
    )

    assert seen["args"] == ("get_news", "PETR4", "2024-01-08", "2024-01-10")
    assert "positive" in out
