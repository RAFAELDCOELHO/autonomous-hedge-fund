"""PR7 (P4.11) decision-before-the-open: runner, execution, calendar, logs.

Rule: for decision day D the agent sees data up to the D-1 close only (D-1 =
previous session of the ticker's exchange, fixed B3/NYSE calendar) and the
decision executes at the D open; decision D earns open(D) -> open(D+1).

Every test here is expected to FAIL on main e5ea062 (close-of-D execution, agent
keyed to D) and PASS once PR7 lands. Assumed PR7 names: tests/_pr7_api.py.
Fake vendors: tests/_pr7_fakes.py (offline; yfinance entry points patched).
"""

from __future__ import annotations

import csv

import numpy as np
import pandas as pd
import pytest

import _pr7_api as api
from _pr7_fakes import SENTINEL, FakeYahoo, RecordingAgent, install_fake_yahoo, make_bars

CAPITAL = 100_000.0


@pytest.fixture
def yahoo(monkeypatch, tmp_path):
    return install_fake_yahoo(monkeypatch, tmp_path)


def _opens(yahoo: FakeYahoo, ticker: str, auto_adjust: bool = True) -> pd.Series:
    return yahoo.frame(ticker, auto_adjust)["Open"].astype(float)


def _ts(d: str) -> pd.Timestamp:
    return pd.Timestamp(d)


# --------------------------------------------------------------------------
# Information set handed to the agent
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("ticker", "day", "cutoff"),
    [("AAPL", "2024-05-15", "2024-05-14"), ("PETR4.SA", "2024-05-15", "2024-05-14")],
)
def test_agent_window_ends_at_previous_session_and_never_contains_day_d(yahoo, ticker, day, cutoff):
    bars = make_bars(ticker)
    bars.loc[_ts(day), ["Open", "High", "Low", "Close"]] = SENTINEL
    yahoo.bars[ticker] = bars
    agent = RecordingAgent()

    api.run_agent(agent, ticker, day, day, CAPITAL)

    assert agent.dates == [cutoff]
    window = agent.windows[0]
    assert window is not None
    assert window.index.max() == _ts(cutoff)
    assert _ts(day) not in window.index
    assert not (window.select_dtypes("number") == SENTINEL).any().any()


def test_prompt_date_given_to_propagate_is_data_cutoff_not_decision_date(yahoo):
    seen = []

    def propagate(ticker, date_str):
        seen.append(date_str)
        return {}, "HOLD"

    eq = api.run_tradingagents("AAPL", "2024-07-01", "2024-07-08", propagate, CAPITAL)

    decision_dates = ["2024-07-01", "2024-07-02", "2024-07-03", "2024-07-05", "2024-07-08"]
    assert [d.strftime("%Y-%m-%d") for d in eq.index] == decision_dates
    assert seen == ["2024-06-28", "2024-07-01", "2024-07-02", "2024-07-03", "2024-07-05"]
    assert not set(seen) & {"2024-07-08"}


# --------------------------------------------------------------------------
# Execution at the open
# --------------------------------------------------------------------------


def test_fill_at_open_of_d_and_pnl_is_open_d_to_open_next(yahoo):
    ticker = "AAPL"
    days = ["2024-05-13", "2024-05-14", "2024-05-15", "2024-05-16"]
    exit_day = "2024-05-17"
    o = _opens(yahoo, ticker)
    agent = RecordingAgent(actions=["BUY", "HOLD", "SELL", "HOLD"])

    eq = api.run_agent(agent, ticker, days[0], days[-1], CAPITAL)

    assert [d.strftime("%Y-%m-%d") for d in eq.index] == days
    shares = CAPITAL / o[_ts(days[0])]  # filled at open(D0)
    cash_after_sell = shares * o[_ts(days[2])]  # sold at open(D2)
    expected = [shares * o[_ts(days[1])], shares * o[_ts(days[2])], cash_after_sell, cash_after_sell]
    np.testing.assert_allclose(eq.values, expected, rtol=1e-12)

    pnl = api.pnl_by_decision_date(eq, CAPITAL)
    np.testing.assert_allclose(
        pnl.values,
        [
            o[_ts(days[1])] / o[_ts(days[0])] - 1,
            o[_ts(days[2])] / o[_ts(days[1])] - 1,
            0.0,
            0.0,
        ],
        atol=1e-12,
    )
    assert _ts(exit_day) not in eq.index


def test_open_prices_and_marks_come_from_one_auto_adjust_setting(monkeypatch, tmp_path):
    # Raw (unadjusted) prices differ from adjusted ones before 2024-03-01.
    yahoo = FakeYahoo(raw_factor=lambda d: 1.25 if d < pd.Timestamp("2024-03-01") else 1.0)
    install_fake_yahoo(monkeypatch, tmp_path, yahoo)
    ticker = "AAPL"
    days = ["2024-02-27", "2024-02-28", "2024-02-29", "2024-03-01", "2024-03-04"]
    agent_eq = api.run_agent(RecordingAgent(actions=lambda n, d: "BUY"), ticker, days[0], days[-1], CAPITAL)
    bh_eq = api.run_buy_and_hold(ticker, days[0], days[-1], CAPITAL)

    used = {call["auto_adjust"] for call in yahoo.download_calls}
    assert len(used) == 1, f"open/close fetched with mixed auto_adjust settings: {used}"
    o = _opens(yahoo, ticker, auto_adjust=used.pop())
    sessions = [_ts(d) for d in days] + [_ts("2024-03-05")]
    expected = [o[sessions[i + 1]] / o[sessions[i]] - 1 for i in range(len(days))]
    np.testing.assert_allclose(api.pnl_by_decision_date(agent_eq, CAPITAL).values, expected, atol=1e-12)
    np.testing.assert_allclose(api.pnl_by_decision_date(bh_eq, CAPITAL).values, expected, atol=1e-12)


@pytest.mark.parametrize("missing_day", ["2024-05-13", "2024-05-15"])
def test_missing_open_on_decision_day_fails_closed(yahoo, missing_day):
    ticker = "AAPL"
    bars = make_bars(ticker)
    bars.loc[_ts(missing_day), "Open"] = np.nan  # Close, High, Low still present
    yahoo.bars[ticker] = bars

    with pytest.raises(api.FAIL_CLOSED_ERRORS):
        api.run_agent(RecordingAgent(actions=lambda n, d: "BUY"), ticker, "2024-05-13", "2024-05-16", CAPITAL)


# --------------------------------------------------------------------------
# Fixed trading calendar for D-1
# --------------------------------------------------------------------------

CALENDAR_CASES = [
    # (ticker, D, expected D-1, why)
    ("AAPL", "2024-01-08", "2024-01-05", "Monday -> Friday (NYSE)"),
    ("PETR4.SA", "2024-01-08", "2024-01-05", "Monday -> Friday (B3)"),
    ("AAPL", "2024-07-05", "2024-07-03", "NYSE-only holiday: Independence Day Thu 2024-07-04"),
    ("PETR4.SA", "2024-07-05", "2024-07-04", "B3 open on US Independence Day"),
    ("AAPL", "2024-02-20", "2024-02-16", "NYSE-only holiday: Presidents' Day Mon 2024-02-19"),
    ("PETR4.SA", "2024-02-20", "2024-02-19", "B3 open on Presidents' Day"),
    ("PETR4.SA", "2024-02-14", "2024-02-09", "B3-only holiday: Carnival Mon/Tue 2024-02-12/13"),
    ("AAPL", "2024-02-14", "2024-02-13", "NYSE open during Carnival"),
    ("PETR4.SA", "2025-04-22", "2025-04-17", "B3-only holiday: Tiradentes Mon 2025-04-21 (+ Good Friday)"),
    ("AAPL", "2025-04-22", "2025-04-21", "NYSE open on Tiradentes"),
    ("PETR4.SA", "2024-01-02", "2023-12-28", "B3-only closure: last business day Fri 2023-12-29"),
    ("AAPL", "2024-01-02", "2023-12-29", "NYSE open on 2023-12-29"),
]


@pytest.mark.parametrize(("ticker", "day", "cutoff", "why"), CALENDAR_CASES, ids=[c[3] for c in CALENDAR_CASES])
def test_data_cutoff_is_previous_session_of_ticker_exchange(yahoo, ticker, day, cutoff, why):
    agent = RecordingAgent()
    api.run_agent(agent, ticker, day, day, CAPITAL)
    assert agent.dates == [cutoff], why
    assert agent.windows[0].index.max() == _ts(cutoff), why


@pytest.mark.parametrize(
    ("ticker", "day", "bogus_bars", "cutoff"),
    [
        ("PETR4.SA", "2025-04-22", ["2025-04-18", "2025-04-21"], "2025-04-17"),
        ("AAPL", "2024-07-05", ["2024-07-04"], "2024-07-03"),
    ],
)
def test_calendar_not_vendor_bars_defines_previous_session(yahoo, ticker, day, bogus_bars, cutoff):
    """Vendors sometimes emit rows on exchange holidays; they must not become D-1."""
    bars = make_bars(ticker)
    for b in bogus_bars:
        bars.loc[_ts(b)] = bars.iloc[0]
    yahoo.bars[ticker] = bars.sort_index()
    agent = RecordingAgent()

    api.run_agent(agent, ticker, day, day, CAPITAL)

    assert agent.dates == [cutoff]
    assert agent.windows[0].index.max() == _ts(cutoff)


@pytest.mark.parametrize(
    ("ticker", "day", "drop"),
    [("AAPL", "2024-07-05", "2024-07-03"), ("PETR4.SA", "2024-02-14", "2024-02-09")],
)
def test_missing_vendor_bar_at_previous_session_fails_closed(yahoo, ticker, day, drop):
    """No silent slide back to an earlier bar when the vendor lacks D-1."""
    yahoo.bars[ticker] = make_bars(ticker).drop(index=_ts(drop))
    agent = RecordingAgent()

    with pytest.raises(api.FAIL_CLOSED_ERRORS):
        api.run_agent(agent, ticker, day, day, CAPITAL)
    assert all(d != drop for d in agent.dates)


# --------------------------------------------------------------------------
# Window boundaries
# --------------------------------------------------------------------------


@pytest.mark.parametrize(("ticker", "cutoff"), [("AAPL", "2023-12-29"), ("PETR4.SA", "2023-12-28")])
def test_first_day_cutoff_lies_before_window_and_inside_lookback(yahoo, ticker, cutoff):
    start, end = "2024-01-02", "2024-01-04"  # PREREG window start
    agent = RecordingAgent()

    eq = api.run_agent(agent, ticker, start, end, CAPITAL)

    assert eq.index[0] == _ts(start)
    assert agent.dates[0] == cutoff
    first_window = agent.windows[0]
    assert first_window.index.max() == _ts(cutoff)
    assert (first_window.index < _ts(start)).all()
    assert len(first_window) >= 60  # pre-window history available for look-back


@pytest.mark.parametrize("ticker", ["AAPL", "PETR4.SA"])
def test_last_day_exit_price_is_open_of_next_session(yahoo, ticker):
    # 2024-03-29 (Good Friday) is closed on both exchanges -> exit at open(2024-04-01).
    start, end = "2024-03-25", "2024-03-28"  # PREREG window end
    o = _opens(yahoo, ticker)

    eq = api.run_agent(RecordingAgent(actions=["BUY"]), ticker, start, end, CAPITAL)

    assert eq.index[-1] == _ts(end)
    assert np.isfinite(eq.iloc[-1])
    assert eq.iloc[-1] == pytest.approx(CAPITAL * o[_ts("2024-04-01")] / o[_ts(start)], rel=1e-12)


def test_last_day_without_next_session_bar_fails_closed(yahoo):
    yahoo.bars["AAPL"] = make_bars("AAPL").drop(index=_ts("2024-04-01"))
    with pytest.raises(api.FAIL_CLOSED_ERRORS):
        api.run_agent(RecordingAgent(actions=["BUY"]), "AAPL", "2024-03-25", "2024-03-28", CAPITAL)


@pytest.mark.parametrize("ticker", ["AAPL", "PETR4.SA"])
def test_buy_and_hold_uses_same_open_to_open_convention(yahoo, ticker):
    start, end = "2024-05-13", "2024-05-17"
    o = _opens(yahoo, ticker)

    bh = api.run_buy_and_hold(ticker, start, end, CAPITAL)
    agent = api.run_agent(RecordingAgent(actions=lambda n, d: "BUY"), ticker, start, end, CAPITAL)

    assert list(bh.index) == list(agent.index)
    np.testing.assert_allclose(bh.values, agent.values, rtol=1e-12)
    assert bh.iloc[0] == pytest.approx(CAPITAL * o[_ts("2024-05-14")] / o[_ts(start)], rel=1e-12)
    assert bh.iloc[-1] == pytest.approx(CAPITAL * o[_ts("2024-05-20")] / o[_ts(start)], rel=1e-12)


def test_baseline_signal_for_d_uses_only_data_up_to_d_minus_1(yahoo):
    """Same convention for every run_strategy baseline: the signal observed at
    the D-1 close decides the position taken at open(D)."""
    from tradingagents.backtest.baselines import _Strategy

    class OneDaySignal(_Strategy):
        name = "one-day"

        def signals(self, prices):
            return pd.Series(prices.index == _ts("2024-05-14"), index=prices.index)

    from tradingagents.backtest.runner import run_strategy

    o = _opens(yahoo, "AAPL")
    eq = run_strategy(OneDaySignal(), "AAPL", "2024-05-13", "2024-05-17", CAPITAL)

    # Signal on 05-14 -> long from open(05-15) to open(05-16), flat otherwise.
    gain = o[_ts("2024-05-16")] / o[_ts("2024-05-15")]
    np.testing.assert_allclose(
        eq.values, [CAPITAL, CAPITAL, CAPITAL * gain, CAPITAL * gain, CAPITAL * gain], rtol=1e-12
    )


# --------------------------------------------------------------------------
# Logs and cells.csv keep D as the decision date, plus data_cutoff = D-1
# --------------------------------------------------------------------------


def test_decision_log_records_decision_date_d_and_data_cutoff_d_minus_1(yahoo):
    eq = api.run_agent(RecordingAgent(), "AAPL", "2024-07-01", "2024-07-08", CAPITAL)

    log = api.decision_log(eq)
    assert list(log[api.DECISION_DATE_FIELD]) == [
        "2024-07-01", "2024-07-02", "2024-07-03", "2024-07-05", "2024-07-08",
    ]
    assert list(log[api.DATA_CUTOFF_FIELD]) == [
        "2024-06-28", "2024-07-01", "2024-07-02", "2024-07-03", "2024-07-05",
    ]
    assert list(log[api.DECISION_DATE_FIELD]) == [d.strftime("%Y-%m-%d") for d in eq.index]


@pytest.mark.parametrize(("ticker", "cutoff"), [("AAPL", "2023-12-29"), ("PETR4.SA", "2023-12-28")])
def test_cells_csv_keeps_start_as_decision_date_and_adds_data_cutoff(yahoo, monkeypatch, tmp_path, ticker, cutoff):
    run_backtest = api.run_backtest_module()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-dummy")
    monkeypatch.setattr(run_backtest, "load_dotenv", lambda *a, **k: None)
    monkeypatch.setattr(run_backtest, "make_decide_fn", lambda **_kw: RecordingAgent())
    cells = tmp_path / "cells.csv"

    rc = run_backtest.main(
        ["--ticker", ticker, "--start", "2024-01-02", "--end", "2024-03-28",  # PREREG window
         "--arms", "baseline", "--cells-out", str(cells), "--seed", "0"]
    )

    assert rc == 0
    with cells.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 1
    assert rows[0]["start"] == "2024-01-02"
    assert rows[0]["end"] == "2024-03-28"
    assert rows[0].get(api.CELLS_DATA_CUTOFF_COLUMN) == cutoff


# --------------------------------------------------------------------------
# Fail closed means RAISE, never a silently skipped day or cell (Lingxi)
# --------------------------------------------------------------------------

BAD_DAY = "2024-02-07"  # Wed inside the PREREG window; also D-1 of Thu 2024-02-08


def _damage(bars: pd.DataFrame, kind: str) -> pd.DataFrame:
    if kind == "missing_open":
        bars.loc[_ts(BAD_DAY), "Open"] = np.nan
        return bars
    if kind == "missing_bar":  # missing bar for D and for the next day's D-1
        return bars.drop(index=_ts(BAD_DAY))
    raise AssertionError(kind)


@pytest.mark.parametrize("kind", ["missing_open", "missing_bar"])
@pytest.mark.parametrize("ticker", ["AAPL", "PETR4.SA"])
def test_multi_day_run_raises_on_one_bad_day_instead_of_skipping_it(yahoo, ticker, kind):
    yahoo.bars[ticker] = _damage(make_bars(ticker), kind)
    agent = RecordingAgent(actions=lambda n, d: "HOLD")

    with pytest.raises(api.FAIL_CLOSED_ERRORS):
        api.run_agent(agent, ticker, "2024-02-01", "2024-02-15", CAPITAL)
    with pytest.raises(api.FAIL_CLOSED_ERRORS):
        api.run_buy_and_hold(ticker, "2024-02-01", "2024-02-15", CAPITAL)


@pytest.mark.parametrize("kind", ["missing_open", "missing_bar"])
def test_cells_grid_never_silently_drops_the_bad_cell(yahoo, monkeypatch, tmp_path, kind):
    """A bad (ticker, day) must surface: either the invocation raises a fail-closed
    error (and writes nothing for that ticker), or every grid row exists and the
    bad ticker's rows are status=failed with a non-zero exit. A cells.csv that is
    simply missing the bad cell after a clean exit, or an `ok` row built from a
    skipped day, is the bug."""
    tickers, arms, seeds = ["AAPL", "PETR4.SA", "VALE3.SA"], ["baseline", "macro"], [0]
    bad = "PETR4.SA"
    yahoo.bars[bad] = _damage(make_bars(bad), kind)
    cells = tmp_path / "cells.csv"

    outcomes = api.write_cells_grid(cells, tickers, arms, seeds, RecordingAgent, monkeypatch)

    rows = []
    if cells.exists():
        with cells.open(newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
    by_key = {(r["ticker"], r["arm"], r["seed"]): r for r in rows}
    assert len(by_key) == len(rows), "duplicate keys in cells.csv"
    grid = {(t.replace(".SA", ""), a, str(s)) for t in tickers for a in ("absent", "present") for s in seeds}

    for out in outcomes:
        keys = {k for k in grid if k[0] == out["ticker"].replace(".SA", "") and k[2] == str(out["seed"])}
        if out["ticker"] != bad:
            assert out["exc"] is None and out["rc"] == 0, out
            assert keys <= set(by_key) and all(by_key[k]["status"] == "ok" for k in keys)
            continue
        if out["exc"] is not None:
            assert isinstance(out["exc"], api.FAIL_CLOSED_ERRORS), repr(out["exc"])
            assert not keys & set(by_key), "raised but left partial rows for the bad cell"
        else:
            assert out["rc"] not in (0, None), "bad cell finished with exit 0"
            assert keys <= set(by_key), "bad cell silently dropped from cells.csv"
            assert all(by_key[k]["status"] == "failed" for k in keys), [by_key[k] for k in keys]
    # Nothing outside the grid, and no silent shortfall: every missing key belongs
    # to an invocation that raised.
    assert set(by_key) <= grid
    raised = {(o["ticker"].replace(".SA", ""), str(o["seed"])) for o in outcomes if o["exc"] is not None}
    assert all((k[0], k[2]) in raised for k in grid - set(by_key)), sorted(grid - set(by_key))
    assert any(o["ticker"] == bad and (o["exc"] is not None or o["rc"]) for o in outcomes)
