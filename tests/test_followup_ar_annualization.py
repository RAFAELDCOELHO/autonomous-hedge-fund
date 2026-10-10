"""Documents (does not endorse) how metrics.py annualizes a B2 curve.

metrics.py is frozen. compute() uses n_days = len(equity) (curve points), so
the PREREG curve (62 points: D-1 anchor + 61 decisions, 61 returns) is
annualized with 252/62, not 252/61. See TEMPORAL_AUDIT 10.2 "AR annualization
note". If this test fails, the convention changed: update the docs on purpose.
"""

from __future__ import annotations

import pytest

import _pr7_api as api
from _pr7_fakes import RecordingAgent, install_fake_yahoo
from tradingagents.backtest.metrics import FLAT_RF_SENSITIVITY, ExtendedMetricsCalculator

CAPITAL = 100_000.0


@pytest.mark.parametrize("ticker", ["AAPL", "PETR4.SA"])
def test_ar_exponent_counts_curve_points_not_returns(monkeypatch, tmp_path, ticker):
    install_fake_yahoo(monkeypatch, tmp_path)
    start, end = api.prereg_window()
    eq = api.run_agent(RecordingAgent(actions=lambda n, d: "BUY"), ticker, start, end, CAPITAL)
    assert len(eq) == 62 and eq.attrs["n_days"] == 61

    m = ExtendedMetricsCalculator(annual_rf_rate=FLAT_RF_SENSITIVITY).compute(eq)
    cr = eq.iloc[-1] / CAPITAL - 1
    assert m["n_days"] == 62
    assert m["ar"] == pytest.approx((1 + cr) ** (252 / 62) - 1, rel=1e-12)
    assert m["ar"] != pytest.approx((1 + cr) ** (252 / 61) - 1, rel=1e-6)
