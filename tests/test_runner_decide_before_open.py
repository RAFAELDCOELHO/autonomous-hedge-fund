"""P4.11: decide with information up to the close of D-1, execute at the open of D. Offline.

Sessions and D-1 come from the fixed exchange calendar (B3 for .SA, NYSE otherwise).
"""

from __future__ import annotations

import json
from itertools import pairwise
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from tradingagents.backtest.agent_integration import make_decide_fn
from tradingagents.backtest.calendar import exchange_for, previous_session, sessions
from tradingagents.backtest.runner import run_agent_strategy

LOAD = "tradingagents.backtest.runner.load_ohlcv"
OPEN_SENTINEL, CLOSE_SENTINEL = 777777.0, 999999.0


def _ohlcv(dates, opens, closes=None) -> pd.DataFrame:
    opens = np.asarray(opens, dtype=float)
    closes = opens if closes is None else np.asarray(closes, dtype=float)
    return pd.DataFrame({
        "Date": pd.to_datetime(dates),
        "Open": opens,
        "High": np.fmax(opens, closes),
        "Low": np.fmin(opens, closes),
        "Close": closes,
        "Volume": 1_000.0,
    })


def _recording(action="HOLD"):
    calls = []

    def decide(date, window):
        calls.append((date, window.copy()))
        return action(date) if callable(action) else action

    return decide, calls


def _ds(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


def _nyse(start, end):
    return sessions("X", start, end)  # "X" has no .SA suffix -> NYSE


def test_exchange_mapping_follows_market_of():
    assert exchange_for("PETR4.SA") == exchange_for("vale3.sa") == "BVMF"
    assert exchange_for("AAPL") == "XNYS"


@pytest.mark.parametrize(
    "ticker, holidays",
    [
        ("PETR4.SA", ["2024-02-12", "2024-02-13", "2024-03-29", "2024-05-30"]),
        ("AAPL", ["2024-01-15", "2024-02-19", "2024-03-29", "2024-07-04"]),
    ],
)
def test_calendar_pins_known_holidays(ticker, holidays):
    open_days = sessions(ticker, "2024-01-02", "2024-07-31")
    for day in holidays:
        assert pd.Timestamp(day) not in open_days
        assert previous_session(ticker, pd.Timestamp(day) + pd.Timedelta(days=1)) < pd.Timestamp(day)
    # The other exchange trades on each one-sided holiday.
    assert pd.Timestamp("2024-01-15") in sessions("PETR4.SA", "2024-01-15", "2024-01-15")
    assert pd.Timestamp("2024-02-12") in sessions("AAPL", "2024-02-12", "2024-02-12")


def test_decision_sees_only_previous_session_never_d():
    dates = _nyse("2024-01-02", "2024-01-10")[:6]
    opens = [10.0, 11.0, 12.0, 13.0, 14.0, OPEN_SENTINEL]
    closes = [10.5, 11.5, 12.5, 13.5, 14.5, CLOSE_SENTINEL]
    decide, calls = _recording()
    with patch(LOAD, return_value=_ohlcv(dates, opens, closes)):  # dates[-1]: exit session
        run_agent_strategy(decide, "X", _ds(dates[1]), _ds(dates[-2]), 1_000.0)

    window_sessions = dates[1:-1]
    assert len(calls) == len(window_sessions)
    for d, prev, (date, window) in zip(window_sessions, dates[:-2], calls):
        assert date == _ds(prev)
        assert window.index.max() == prev < d
        assert d not in window.index
        assert window.attrs == {"decision_date": _ds(d), "data_cutoff": _ds(prev)}
        assert not window.isin([OPEN_SENTINEL, CLOSE_SENTINEL]).any().any()


def test_orders_fill_and_mark_at_opens_closes_ignored():
    dates = _nyse("2023-12-29", "2024-01-08")  # 12-29 pre-window bar, 01-08 exit session
    assert len(dates) == 6
    opens = [10.0, 20.0, 25.0, 40.0, 50.0, 60.0]

    def action(date):  # BUY for 01-02, SELL for 01-04
        return {"2023-12-29": "BUY", "2024-01-03": "SELL"}.get(date, "HOLD")

    curves = []
    for closes in ([CLOSE_SENTINEL] * 6, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]):
        decide, _ = _recording(action)
        with patch(LOAD, return_value=_ohlcv(dates, opens, closes)):
            curves.append(run_agent_strategy(decide, "X", "2024-01-02", "2024-01-05", 1_000.0))

    eq = curves[0]
    shares = 1_000.0 / 20.0  # capital / Open[first session]
    assert eq.iloc[0] == pytest.approx(shares * 25.0)  # marked at the next session's open
    assert eq.iloc[1] == pytest.approx(shares * 40.0)
    assert eq.tolist() == pytest.approx([1_250.0, 2_000.0, 2_000.0, 2_000.0])  # SELL at Open 40
    pd.testing.assert_series_equal(curves[0], curves[1])


def test_first_session_d_minus_1_is_the_calendar_previous_session():
    dates = _nyse("2023-12-27", "2024-01-04")  # 12-27, 12-28, 12-29, 01-02, 01-03, exit 01-04
    decide, calls = _recording()
    with patch(LOAD, return_value=_ohlcv(dates, [10.0] * 6)):
        run_agent_strategy(decide, "X", "2024-01-02", "2024-01-03", 1_000.0)
    date, window = calls[0]
    assert date == "2023-12-29"  # 01-01 is an NYSE holiday
    assert window.index.tolist() == list(dates[:3])


def test_no_pre_window_bar_fails_closed():
    dates = _nyse("2024-01-02", "2024-01-04")
    decide, calls = _recording()
    with patch(LOAD, return_value=_ohlcv(dates, [10.0] * 3)), pytest.raises(
        ValueError, match=r"X: no vendor bar on 2023-12-29.*look-back must cover"
    ):
        run_agent_strategy(decide, "X", "2024-01-02", "2024-01-04", 1_000.0)
    assert calls == []


def test_missing_d_minus_1_bar_never_slides_back_to_an_older_bar():
    # 12-29 (D-1 of 01-02) is missing; 12-28 must not stand in for it.
    df = _ohlcv(["2023-12-28", "2024-01-02", "2024-01-03"], [10.0] * 3)
    decide, calls = _recording()
    with patch(LOAD, return_value=df), pytest.raises(ValueError, match=r"X: no vendor bar on 2023-12-29"):
        run_agent_strategy(decide, "X", "2024-01-02", "2024-01-03", 1_000.0)
    assert calls == []


def test_missing_window_session_bar_fails_closed():
    df = _ohlcv(["2023-12-29", "2024-01-02", "2024-01-04", "2024-01-05"], [10.0] * 4)
    decide, calls = _recording()
    with patch(LOAD, return_value=df), pytest.raises(
        ValueError, match=r"X: no vendor bar on session 2024-01-03"
    ):
        run_agent_strategy(decide, "X", "2024-01-02", "2024-01-05", 1_000.0)
    assert calls == []


@pytest.mark.parametrize(
    "ticker, dates, start, end, holiday",
    [
        ("X", ["2024-01-11", "2024-01-12", "2024-01-15", "2024-01-16", "2024-01-17"], "2024-01-12",
         "2024-01-16", "2024-01-15"),
        ("PETR4.SA", ["2024-02-08", "2024-02-09", "2024-02-12", "2024-02-14", "2024-02-15"], "2024-02-09",
         "2024-02-14", "2024-02-12"),
        # A holiday bar between D-1 of start and start is ignored too.
        ("X", ["2023-12-29", "2024-01-01", "2024-01-02", "2024-01-03"], "2024-01-02", "2024-01-02",
         "2024-01-01"),
    ],
)
def test_vendor_bar_on_a_holiday_is_ignored_with_a_warning(ticker, dates, start, end, holiday, caplog):
    decide, calls = _recording()
    with patch(LOAD, return_value=_ohlcv(dates, [10.0] * len(dates))), caplog.at_level(
        "WARNING", logger="tradingagents.backtest.runner"
    ):
        eq = run_agent_strategy(decide, ticker, start, end, 1_000.0)
    assert f"ignoring vendor bars on non-{exchange_for(ticker)} sessions: {holiday}" in caplog.text
    assert pd.Timestamp(holiday) not in eq.index
    assert [c[0] for c in calls] == [_ds(previous_session(ticker, d)) for d in eq.index]
    assert all(pd.Timestamp(holiday) not in window.index for _, window in calls)


def test_last_session_is_decided_and_marked_at_the_exit_open():
    dates = _nyse("2023-12-29", "2024-01-05")  # 01-05: exit session after end
    decide, calls = _recording("BUY")
    with patch(LOAD, return_value=_ohlcv(dates, [10.0, 20.0, 30.0, 40.0, 50.0], [5.0] * 5)):
        eq = run_agent_strategy(decide, "X", "2024-01-02", "2024-01-04", 1_000.0)
    assert eq.index[-1] == dates[-2]
    assert eq.iloc[-1] == pytest.approx(1_000.0 / 20.0 * 50.0)
    assert len(calls) == 3 and calls[-1][0] == _ds(dates[-3])
    assert all(window.index.max() < dates[-2] for _, window in calls)
    assert eq.attrs["execution"] == "open"
    assert eq.attrs["information_cutoff"] == "previous session close"
    assert eq.attrs["n_days"] == 3
    assert eq.attrs["data_cutoff"] == "2023-12-29"  # D-1 of the first session


@pytest.mark.parametrize(
    "dates, dup",
    [
        (["2023-12-29", "2024-01-02", "2024-01-02", "2024-01-03"], "2024-01-02"),  # a decision session
        (["2023-12-28", "2023-12-28", "2023-12-29", "2024-01-02", "2024-01-03"], "2023-12-28"),  # look-back
        (["2023-12-29", "2024-01-02", "2024-01-03", "2024-01-03"], "2024-01-03"),  # exit session
    ],
)
def test_duplicate_vendor_dates_fail_closed(dates, dup):
    from tradingagents.backtest.runner import DataDefectError, run_buy_and_hold

    decide, calls = _recording("BUY")
    with patch(LOAD, return_value=_ohlcv(dates, [10.0 + i for i in range(len(dates))])):
        for run in (lambda: run_agent_strategy(decide, "X", "2024-01-02", "2024-01-02", 1_000.0),
                    lambda: run_buy_and_hold("X", "2024-01-02", "2024-01-02", 1_000.0)):
            with pytest.raises(DataDefectError, match=rf"X: duplicate vendor bars on {dup}"):
                run()
    assert calls == []


@pytest.mark.parametrize(
    "frame",
    [
        _ohlcv(["2024-01-02", "2024-01-03"], [10.0] * 2),  # no D-1 bar
        _ohlcv(["2023-12-29", "2024-01-03"], [10.0] * 2),  # no bar on the session
        _ohlcv(["2023-12-29", "2024-01-02"], [10.0] * 2),  # no exit bar
        _ohlcv(["2023-12-29", "2024-01-02", "2024-01-03"], [10.0] * 3).drop(columns="Open"),
        _ohlcv(["2023-12-29", "2024-01-02", "2024-01-03"], [10.0, np.nan, 10.0]),
    ],
    ids=["d_minus_1", "session", "exit", "open_column", "bad_open"],
)
def test_every_fail_closed_check_is_a_data_defect_error(frame):
    from tradingagents.backtest.runner import DataDefectError
    from tradingagents.backtest.snapshot import SnapshotIncomplete

    with patch(LOAD, return_value=frame), pytest.raises(DataDefectError):
        run_agent_strategy(lambda d, w: "BUY", "X", "2024-01-02", "2024-01-02", 1_000.0)
    assert issubclass(DataDefectError, ValueError) and issubclass(DataDefectError, LookupError)
    assert issubclass(SnapshotIncomplete, DataDefectError)


def test_missing_exit_session_bar_fails_closed():
    dates = _nyse("2023-12-29", "2024-01-04")  # no bar for the exit session 01-05
    with patch(LOAD, return_value=_ohlcv(dates, [10.0] * 4)), pytest.raises(
        ValueError, match=r"X: no vendor bar on 2024-01-05, the exit session"
    ):
        run_agent_strategy(lambda d, w: "BUY", "X", "2024-01-02", "2024-01-04", 1_000.0)


def test_missing_open_column_fails_closed():
    df = _ohlcv(_nyse("2023-12-29", "2024-01-03"), [10.0] * 3).drop(columns="Open")
    with patch(LOAD, return_value=df), pytest.raises(ValueError, match="Open"):
        run_agent_strategy(lambda d, w: "BUY", "X", "2024-01-02", "2024-01-03", 1_000.0)


@pytest.mark.parametrize("pos, day", [(2, "2024-01-04"), (4, "2024-01-08")])  # decision / exit session
@pytest.mark.parametrize("bad", [np.nan, 0.0])
def test_bad_open_names_ticker_and_date_never_uses_close(bad, pos, day):
    dates = [previous_session("PETR4.SA", "2024-01-03"), *sessions("PETR4.SA", "2024-01-03", "2024-01-08")]
    assert _ds(dates[pos]) == day
    opens = [10.0, 11.0, 12.0, 13.0, 14.0]
    opens[pos] = bad
    df = _ohlcv(dates, opens, [10.0] * 5)
    with patch(LOAD, return_value=df), pytest.raises(ValueError, match=rf"PETR4\.SA.*{day}"):
        run_agent_strategy(lambda d, w: "BUY", "PETR4.SA", "2024-01-03", "2024-01-05", 1_000.0)


def test_runner_loads_prices_unfilled_so_a_missing_open_reaches_the_check():
    dates = _nyse("2023-12-29", "2024-01-03")
    with patch(LOAD, return_value=_ohlcv(dates, [10.0] * 3)) as load:
        run_agent_strategy(lambda d, w: "HOLD", "X", "2024-01-02", "2024-01-02", 1_000.0)
    load.assert_called_once_with("X", "2024-01-03", fill_prices=False)  # through the exit session


def test_monday_decision_uses_previous_friday():
    dates = ["2024-01-04", "2024-01-05", "2024-01-08", "2024-01-09", "2024-01-10"]  # Thu..Wed (exit)
    decide, calls = _recording()
    with patch(LOAD, return_value=_ohlcv(dates, [10.0] * 5)):
        run_agent_strategy(decide, "X", "2024-01-05", "2024-01-09", 1_000.0)
    assert [c[0] for c in calls] == ["2024-01-04", "2024-01-05", "2024-01-08"]
    assert calls[1][1].attrs == {"decision_date": "2024-01-08", "data_cutoff": "2024-01-05"}


@pytest.mark.parametrize(
    "start, end, frames, expected",
    [
        # B3 Carnival (02-12/13): PETR4.SA skips it, AAPL trades through it.
        (
            "2024-02-09", "2024-02-15",
            {
                "PETR4.SA": ["2024-02-08", "2024-02-09", "2024-02-14", "2024-02-15", "2024-02-16"],
                "AAPL": ["2024-02-08", "2024-02-09", "2024-02-12", "2024-02-13", "2024-02-14", "2024-02-15",
                         "2024-02-16"],
            },
            {
                "PETR4.SA": ["2024-02-08", "2024-02-09", "2024-02-14"],
                "AAPL": ["2024-02-08", "2024-02-09", "2024-02-12", "2024-02-13", "2024-02-14"],
            },
        ),
        # US MLK (01-15): AAPL skips it, PETR4.SA trades.
        (
            "2024-01-12", "2024-01-17",
            {
                "AAPL": ["2024-01-11", "2024-01-12", "2024-01-16", "2024-01-17", "2024-01-18"],
                "PETR4.SA": ["2024-01-11", "2024-01-12", "2024-01-15", "2024-01-16", "2024-01-17", "2024-01-18"],
            },
            {
                "AAPL": ["2024-01-11", "2024-01-12", "2024-01-16"],
                "PETR4.SA": ["2024-01-11", "2024-01-12", "2024-01-15", "2024-01-16"],
            },
        ),
    ],
)
def test_previous_session_follows_each_tickers_own_calendar(start, end, frames, expected):
    data = {t: _ohlcv(d, [10.0] * len(d)) for t, d in frames.items()}
    for ticker in frames:
        decide, calls = _recording()
        with patch(LOAD, side_effect=lambda t, *_a, **_k: data[t]):
            eq = run_agent_strategy(decide, ticker, start, end, 1_000.0)
        assert [c[0] for c in calls] == expected[ticker]
        # Logs record the decision as D and its cutoff as the calendar D-1.
        log = eq.attrs["decision_log"]
        assert [e["decision_date"] for e in log] == [_ds(d) for d in eq.index]
        assert [e["data_cutoff"] for e in log] == expected[ticker]
        assert [e["data_cutoff"] for e in log] == [_ds(previous_session(ticker, d)) for d in eq.index]
        assert eq.attrs["data_cutoff"] == expected[ticker][0]


def test_decision_log_records_action_and_error_per_session():
    dates = _nyse("2023-12-29", "2024-01-08")  # 01-08: exit session

    def action(date):
        if date == "2024-01-02":
            raise RuntimeError("api down")
        return "BUY"

    decide, _ = _recording(action)
    with patch(LOAD, return_value=_ohlcv(dates, [10.0] * 6)):
        eq = run_agent_strategy(decide, "X", "2024-01-02", "2024-01-05", 1_000.0)
    assert eq.attrs["decision_log"] == [
        {"decision_date": "2024-01-02", "data_cutoff": "2023-12-29", "action": "BUY", "error": False},
        {"decision_date": "2024-01-03", "data_cutoff": "2024-01-02", "action": "HOLD", "error": True},
        {"decision_date": "2024-01-04", "data_cutoff": "2024-01-03", "action": "BUY", "error": False},
        {"decision_date": "2024-01-05", "data_cutoff": "2024-01-04", "action": "BUY", "error": False},
    ]
    assert eq.attrs["data_cutoff"] == "2023-12-29"  # D-1 of the first session


def test_buy_and_hold_is_the_always_buy_agent():
    from tradingagents.backtest.runner import run_buy_and_hold

    dates = _nyse("2023-12-29", "2024-01-08")  # 01-08: exit session
    opens = [9.0, 10.0, 12.0, 11.0, 15.0, 16.0]
    df = _ohlcv(dates, opens, [CLOSE_SENTINEL] * 6)
    with patch(LOAD, return_value=df):
        bh = run_buy_and_hold("X", "2024-01-02", "2024-01-05", 1_000.0)
        agent = run_agent_strategy(lambda d, w: "BUY", "X", "2024-01-02", "2024-01-05", 1_000.0)
    assert bh.index.equals(agent.index)
    assert bh.tolist() == pytest.approx(agent.tolist(), rel=1e-12)
    assert bh.iloc[0] == pytest.approx(1_000.0 * 12.0 / 10.0)
    assert bh.iloc[-1] / 1_000.0 - 1 == pytest.approx(16.0 / 10.0 - 1)


def test_make_decide_fn_propagates_previous_session_dates_only():
    dates = _nyse("2023-12-29", "2024-01-08")  # 01-08: exit session
    seen = []

    def propagate(_ticker, date):
        seen.append(date)
        return {}, "HOLD"

    with patch(LOAD, return_value=_ohlcv(dates, [10.0] * 6)):
        run_agent_strategy(make_decide_fn("AAPL", {}, propagate_fn=propagate), "AAPL",
                           "2024-01-02", "2024-01-05", 1_000.0)
    assert seen == [_ds(d) for d in dates[:-2]]


def test_make_decide_fn_real_graph_gets_cutoff_as_trade_date_and_d_as_decision_date():
    dates = _nyse("2023-12-29", "2024-01-05")  # decisions 01-02..01-04; 01-05: exit session
    seen = []

    class FakeGraph:
        def __init__(self, **_kwargs):
            pass

        def propagate(self, ticker, trade_date, decision_date=None):
            seen.append((trade_date, decision_date))
            return {}, "HOLD"

    with patch("tradingagents.graph.trading_graph.TradingAgentsGraph", FakeGraph), patch(
        LOAD, return_value=_ohlcv(dates, [10.0] * 5)
    ):
        run_agent_strategy(make_decide_fn("AAPL", {}), "AAPL", "2024-01-02", "2024-01-04", 1_000.0)
    assert seen == [(_ds(p), _ds(d)) for p, d in pairwise(dates[:-1])]


def _final_state(trade_date):
    debate = {k: "" for k in ("bull_history", "bear_history", "history", "current_response", "judge_decision")}
    risk = {k: "" for k in ("aggressive_history", "conservative_history", "neutral_history", "history",
                            "judge_decision")}
    return {
        "company_of_interest": "AAPL", "trade_date": trade_date, "market_report": "m",
        "sentiment_report": "s", "news_report": "n", "fundamentals_report": "f",
        "investment_debate_state": debate, "trader_investment_plan": "t", "risk_debate_state": risk,
        "investment_plan": "p", "final_trade_decision": "HOLD",
    }


def _bare_graph(tmp_path, final_state):
    from tradingagents.graph.trading_graph import TradingAgentsGraph

    g = TradingAgentsGraph.__new__(TradingAgentsGraph)
    g.config = {"results_dir": str(tmp_path)}
    g.log_states_dict = {}
    g.debug = False
    g.propagator = SimpleNamespace(
        create_initial_state=lambda name, date: {"company_of_interest": name, "trade_date": date},
        get_graph_args=dict,
    )
    g.graph = SimpleNamespace(invoke=lambda state, **_a: final_state)
    g.process_signal = lambda decision: decision
    return g


def test_graph_state_log_is_keyed_by_decision_date_with_cutoff(tmp_path):
    g = _bare_graph(tmp_path, _final_state("2024-01-11"))
    g.propagate("AAPL", "2024-01-11", decision_date="2024-01-12")
    log_dir = tmp_path / "AAPL" / "TradingAgentsStrategy_logs"
    assert [p.name for p in log_dir.iterdir()] == ["full_states_log_2024-01-12.json"]
    logged = json.loads((log_dir / "full_states_log_2024-01-12.json").read_text())
    assert logged["decision_date"] == "2024-01-12"
    assert logged["data_cutoff"] == "2024-01-11"
    assert logged["trade_date"] == "2024-01-11"  # the graph and tools ran at the cutoff
    assert list(g.log_states_dict) == ["2024-01-12"]


def test_graph_state_log_unchanged_without_decision_date(tmp_path):
    state = _final_state("2024-01-11")
    g = _bare_graph(tmp_path, state)
    g.propagate("AAPL", "2024-01-11")
    path = tmp_path / "AAPL" / "TradingAgentsStrategy_logs" / "full_states_log_2024-01-11.json"
    expected = {
        "company_of_interest": "AAPL", "trade_date": "2024-01-11", "market_report": "m",
        "sentiment_report": "s", "news_report": "n", "fundamentals_report": "f",
        "investment_debate_state": state["investment_debate_state"],
        "trader_investment_decision": "t", "risk_debate_state": state["risk_debate_state"],
        "investment_plan": "p", "final_trade_decision": "HOLD",
    }
    assert path.read_text(encoding="utf-8") == json.dumps(expected, indent=4)
    assert list(g.log_states_dict) == ["2024-01-11"]


def test_get_stock_data_at_runner_date_hides_session_d():
    from tradingagents.agents.utils.core_stock_tools import get_stock_data

    dates = _nyse("2023-12-29", "2024-01-08")  # 01-08: exit session, sentinel too
    df = _ohlcv(dates, [10.0, 11.0, 12.0, 13.0, OPEN_SENTINEL, OPEN_SENTINEL],
                [10.5, 11.5, 12.5, 13.5, CLOSE_SENTINEL, CLOSE_SENTINEL])

    def vendor(_method, _symbol, start_date, end_date):
        rows = df[(df["Date"] >= start_date) & (df["Date"] <= end_date)]
        return rows.to_string()

    texts = []

    def decide(date, _window):
        texts.append(get_stock_data.invoke({
            "symbol": "X", "start_date": "2023-12-01", "end_date": "2024-12-31", "curr_date": date,
        }))
        return "HOLD"

    with patch(LOAD, return_value=df), patch(
        "tradingagents.agents.utils.core_stock_tools.route_to_vendor", side_effect=vendor
    ):
        run_agent_strategy(decide, "X", "2024-01-02", "2024-01-05", 1_000.0)

    assert len(texts) == 4
    assert "13.5" in texts[-1]  # D-1's close is visible to the decision for D
    for text in texts:
        assert "777777" not in text and "999999" not in text
