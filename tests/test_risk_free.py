"""P3.7: market rf from committed snapshots, cash accrual, excess Sharpe, decision errors. Offline."""

from __future__ import annotations

import hashlib
import importlib.util
import shutil
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from tradingagents.backtest.metrics import (
    FLAT_RF_SENSITIVITY,
    ExtendedMetricsCalculator,
    h1_cell_metrics,
)
from tradingagents.backtest.agent_integration import (
    UnparseableSignal,
    make_decide_fn,
    run_tradingagents_backtest,
)
from tradingagents.backtest.risk_free import RF_DIR, RF_SOURCE, daily_rf, verify_snapshots
from tradingagents.backtest.runner import MAX_DECISION_ERROR_RATE, run_agent_strategy

REPO = Path(__file__).resolve().parents[1]
US_DATES = pd.DatetimeIndex(["2024-01-12", "2024-01-16", "2024-01-17", "2024-01-18"])


def _dtb3_daily(pct: float) -> float:
    return (1 + pct / 100) ** (1 / 252) - 1


def _run(decider, dates, market=None, capital=1_000.0):
    df = pd.DataFrame({"Date": dates, "Close": np.linspace(100.0, 110.0, len(dates))})
    with patch("tradingagents.backtest.runner.load_ohlcv", return_value=df):
        return run_agent_strategy(decider, "X", str(dates[0].date()), str(dates[-1].date()),
                                  capital, market=market)


def test_committed_snapshots_pass_sha_check():
    verify_snapshots()


def test_tampered_snapshot_fails_loudly(tmp_path):
    rf_dir = tmp_path / "rf"
    shutil.copytree(RF_DIR, rf_dir)
    cdi = next(rf_dir.glob("bcb_*.csv"))
    cdi.write_text(cdi.read_text().replace("0,043739", "0,000001", 1))
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_snapshots(rf_dir)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        daily_rf("BR", pd.DatetimeIndex(["2024-01-02", "2024-01-03"]), rf_dir=rf_dir)


def test_cdi_is_daily_percent_dated_t_minus_1_and_compounds_skipped_days():
    rf = daily_rf("BR", pd.DatetimeIndex(["2023-12-29", "2024-01-02", "2024-01-04"]))
    assert rf.index.tolist() == [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-04")]
    assert rf.iloc[0] == pytest.approx(0.043739 / 100, rel=1e-12)
    assert rf.iloc[1] == pytest.approx((1 + 0.043739 / 100) ** 2 - 1, rel=1e-12)


def test_dtb3_uses_previous_trading_day_with_forward_fill():
    rf = daily_rf("US", US_DATES)
    # 01-16: t-1 = 01-12 (01-15 MLK) -> 5.22; 01-18 uses 01-17's 5.24, not 01-18's.
    expected = [_dtb3_daily(5.22), _dtb3_daily(5.22), _dtb3_daily(5.24)]
    assert rf.to_numpy() == pytest.approx(expected, rel=1e-12)
    # t-1 is the run calendar's previous date (01-12: 5.22), not calendar day 01-17 (5.24).
    rf = daily_rf("US", pd.DatetimeIndex(["2024-01-12", "2024-01-18"]))
    assert rf.to_numpy() == pytest.approx([_dtb3_daily(5.22)], rel=1e-12)


def test_rf_outside_snapshot_window_fails():
    with pytest.raises(ValueError, match="snapshot covers"):
        daily_rf("US", pd.DatetimeIndex(["2024-06-03", "2024-06-04"]))


def test_cash_earns_rf_and_has_zero_excess_sharpe():
    eq = _run(lambda d, w: "HOLD", US_DATES, market="US")
    rf = daily_rf("US", US_DATES)
    assert eq.to_numpy() == pytest.approx(1_000.0 * np.cumprod([1.0, *(1 + rf)]), rel=1e-12)
    assert h1_cell_metrics(eq, "US")["sharpe"] == 0.0
    assert np.allclose(_run(lambda d, w: "HOLD", US_DATES).to_numpy(), 1_000.0)


def test_sharpe_is_on_excess_return_over_daily_rf():
    eq = pd.Series([100.0, 101.0, 100.5, 102.0], index=US_DATES)
    excess = eq.pct_change().dropna() - daily_rf("US", US_DATES)
    expected = np.sqrt(252) * excess.mean() / excess.std(ddof=1)
    assert h1_cell_metrics(eq, "US")["sharpe"] == pytest.approx(expected, rel=1e-12)
    flat = ExtendedMetricsCalculator(annual_rf_rate=FLAT_RF_SENSITIVITY).compute(eq)["sharpe"]
    assert flat != pytest.approx(expected, rel=1e-9)


def test_decision_errors_are_counted_and_exported_for_cells_csv():
    bad = {"2024-01-12": "raise", "2024-01-16": "MAYBE", "2024-01-17": None}

    def decider(date, _window):
        action = bad.get(date, "BUY")
        if action == "raise":
            raise RuntimeError("api down")
        return action

    eq = _run(decider, US_DATES, market="US")
    assert eq.attrs == {"n_days": 4, "n_decision_errors": 3, "decision_errors_exceed_limit": True}
    assert h1_cell_metrics(eq, "US") | {"sharpe": None} == {
        "n_days": 4, "n_decision_errors": 3, "sharpe": None, "rf_source": "FRED-DTB3",
    }


@pytest.mark.parametrize("n_errors, exceeded", [(1, False), (2, True)])
def test_error_limit_is_exclusive_at_5_percent(n_errors, exceeded, caplog):
    dates = pd.bdate_range("2024-01-02", periods=20)
    bad = {d.strftime("%Y-%m-%d") for d in dates[:n_errors]}
    with caplog.at_level("WARNING", logger="tradingagents.backtest.runner"):
        eq = _run(lambda d, w: "MAYBE" if d in bad else "HOLD", dates)
    assert eq.attrs["n_days"] == 20 and eq.attrs["n_decision_errors"] == n_errors
    assert eq.attrs["decision_errors_exceed_limit"] is exceeded
    assert ("exclusion rule E4" in caplog.text) is exceeded


def _h1_stats():
    spec = importlib.util.spec_from_file_location("h1_stats", REPO / "scripts" / "h1_stats.py")
    h1 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(h1)
    return h1


def test_rf_sources_match_h1_stats():
    assert RF_SOURCE == _h1_stats().RF_SOURCE


def test_error_limit_matches_h1_stats():
    assert MAX_DECISION_ERROR_RATE == _h1_stats().MAX_ERROR_RATE


@pytest.mark.parametrize("raw", ["the outlook is unclear", None, "", "   ", "WAIT", 42])
def test_decide_fn_raises_on_unparseable_signal(raw):
    decide = make_decide_fn("X", {}, propagate_fn=lambda t, d: ({}, raw))
    with pytest.raises(UnparseableSignal):
        decide("2024-01-12", None)


@pytest.mark.parametrize("raw, action", [("**OVERWEIGHT**", "BUY"), ("Rating: HOLD", "HOLD"), ("sell", "SELL"), ("do not BUY", "HOLD")])
def test_decide_fn_returns_mapped_label(raw, action):
    assert make_decide_fn("X", {}, propagate_fn=lambda t, d: ({}, raw))("2024-01-12", None) == action


def test_tradingagents_backtest_counts_propagate_and_parse_errors():
    def propagate(_ticker, date):
        if date == "2024-01-12":
            raise RuntimeError("api down")
        return {}, "no idea" if date == "2024-01-16" else "BUY"

    df = pd.DataFrame({"Date": US_DATES, "Close": [100.0, 101.0, 102.0, 103.0]})
    with patch("tradingagents.backtest.runner.load_ohlcv", return_value=df):
        eq = run_tradingagents_backtest("X", "2024-01-12", "2024-01-18", {}, 1_000.0,
                                        propagate_fn=propagate, market="US")
    assert eq.attrs["n_decision_errors"] == 2
    assert eq.attrs["n_days"] == 4
    assert eq.iloc[1] > 1_000.0


def test_flat_4_34_only_in_explicit_sensitivity_path():
    with pytest.raises(ValueError, match="rf"):
        ExtendedMetricsCalculator().compute(pd.Series([1.0, 1.1, 1.2], index=US_DATES[:3]))
    hits = [
        p.relative_to(REPO).as_posix()
        for p in (REPO / "tradingagents").rglob("*.py")
        if "0.0434" in p.read_text() or "4.34" in p.read_text()
    ]
    assert hits == ["tradingagents/backtest/metrics.py"]


def test_cdi_uses_prior_trading_day_not_date_t():
    rf = daily_rf("BR", pd.DatetimeIndex(["2024-03-20", "2024-03-21", "2024-03-22"]))
    # rf_t = CDI of t-1: 03-21 <- CDI(03-20) 0,041957; 03-22 <- CDI(03-21) 0,040168.
    assert rf[pd.Timestamp("2024-03-21")] == pytest.approx(0.041957 / 100, rel=1e-12)
    assert rf[pd.Timestamp("2024-03-22")] == pytest.approx(0.040168 / 100, rel=1e-12)


def test_only_cash_earns_rf_stock_position_does_not():
    df = pd.DataFrame({"Date": US_DATES, "Close": 100.0})
    with patch("tradingagents.backtest.runner.load_ohlcv", return_value=df):
        eq = run_agent_strategy(lambda d, w: "BUY", "X", "2024-01-12", "2024-01-18", 1_000.0, market="US")
    assert eq.tolist() == [1_000.0] * len(US_DATES)


def test_partial_rf_coverage_raises_no_zero_fill():
    eq = pd.Series([100.0, 101.0, 100.5, 102.0], index=US_DATES)
    rf = daily_rf("US", US_DATES).iloc[:-1]
    with pytest.raises(ValueError, match="does not cover"):
        ExtendedMetricsCalculator().compute(eq, rf=rf)


def test_dtb3_forward_fills_missing_t_minus_1_never_backward(tmp_path):
    rf_dir = tmp_path / "rf"
    shutil.copytree(RF_DIR, rf_dir)
    dtb3 = next(rf_dir.glob("fred_*.csv"))
    dtb3.write_text(dtb3.read_text().replace("2024-01-16,5.22", "2024-01-16,9.99", 1))
    (rf_dir / "SHA256SUMS").write_text("".join(
        f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}\n" for p in sorted(rf_dir.glob("*.csv"))
    ))
    dates = pd.DatetimeIndex(["2024-01-12", "2024-01-15", "2024-01-16"])
    # rf at 01-16: t-1 = 01-15 (MLK, no print) -> prior print 01-12 (5.22), not next print 01-16.
    rf = daily_rf("US", dates, rf_dir=rf_dir)[pd.Timestamp("2024-01-16")]
    assert rf == pytest.approx(_dtb3_daily(5.22), rel=1e-12)
    assert rf != pytest.approx(_dtb3_daily(9.99), rel=1e-6)
    assert daily_rf("US", dates)[pd.Timestamp("2024-01-16")] == pytest.approx(_dtb3_daily(5.22), rel=1e-12)
