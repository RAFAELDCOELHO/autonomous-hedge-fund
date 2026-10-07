"""B3 insider transactions fail closed until the CVM availability rule exists (F4 PR4, commit 2).

#62 (P4.4) applies a US Form 4 proxy (transaction_date + 2 US business days).
That proxy is wrong for B3 issuers (CVM: month-end + 10 days), so for
``*.SA`` symbols with a curr_date the insider path must return no rows and a
clear note, without fetching vendor data. Offline: vendors are faked.
"""

from __future__ import annotations

import pandas as pd
import pytest

from tradingagents.agents.utils import news_data_tools
from tradingagents.dataflows import alpha_vantage_news, y_finance

# Old enough to pass the US 2-business-day proxy for curr_date 2024-02-15.
_ROWS = pd.DataFrame(
    {
        "Insider": ["SENTINEL INSIDER", "OTHER INSIDER"],
        "Shares": [111111, 222222],
        "Start Date": pd.to_datetime(["2024-01-05", "2024-01-10"]),
    }
)


class _FakeTicker:
    def __init__(self):
        self.insider_reads = 0

    @property
    def insider_transactions(self):
        self.insider_reads += 1
        return _ROWS.copy()


@pytest.fixture
def fake_yf(monkeypatch):
    ticker = _FakeTicker()
    monkeypatch.setattr(y_finance, "yf_retry", lambda fn, *a, **k: fn())
    monkeypatch.setattr(y_finance.yf, "Ticker", lambda _sym: ticker)
    return ticker


@pytest.mark.parametrize("symbol", ["PETR4.SA", "itub4.sa"])
def test_b3_yfinance_insider_rows_are_refused(fake_yf, symbol):
    out = y_finance.get_insider_transactions(symbol, "2024-02-15")
    assert "SENTINEL INSIDER" not in out and "111111" not in out
    assert "withheld" in out and "CVM" in out
    assert fake_yf.insider_reads == 0  # fails closed before any vendor fetch


def test_us_yfinance_insider_rows_still_use_form4_proxy(fake_yf):
    out = y_finance.get_insider_transactions("AAPL", "2024-02-15")
    assert "SENTINEL INSIDER" in out and "OTHER INSIDER" in out
    assert "withheld" not in out


def test_b3_alpha_vantage_insider_rows_are_refused(monkeypatch):
    calls = []

    def _fake_request(function, params):
        calls.append((function, params))
        return {"data": [{"transaction_date": "2024-01-05", "executive": "SENTINEL INSIDER"}]}

    monkeypatch.setattr(alpha_vantage_news, "_make_api_request", _fake_request)
    out = alpha_vantage_news.get_insider_transactions("PETR4.SA", "2024-02-15")
    assert calls == []
    assert "SENTINEL INSIDER" not in str(out)
    assert "withheld" in str(out) and "CVM" in str(out)


def test_b3_insider_tool_end_to_end_is_refused(fake_yf):
    out = news_data_tools.get_insider_transactions.invoke(
        {"ticker": "VALE3.SA", "curr_date": "2024-02-15"}
    )
    assert "SENTINEL INSIDER" not in out
    assert "withheld" in out and "CVM" in out
    assert fake_yf.insider_reads == 0
