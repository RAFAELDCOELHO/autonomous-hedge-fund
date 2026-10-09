"""P3.9: cells.csv writer and the agent-arm H1 Sharpe column. Offline, no LLM."""

from __future__ import annotations

import csv
import importlib.util
import math
import os
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from tradingagents.backtest.calendar import previous_session, sessions
from tradingagents.backtest.cells import (
    COLUMNS,
    HARNESS_TO_ARM,
    PREREG_TICKERS,
    SHARPE_FLAT_FIELD,
    append_cells,
    make_cell_row,
)
from tradingagents.backtest.metrics import (
    FLAT_RF_SENSITIVITY,
    ExtendedMetricsCalculator,
    flat_rf_metrics,
    h1_cell_metrics,
)
from tradingagents.backtest.report import (
    H1_SHARPE_COL,
    SHARPE_FLAT_COL,
    build_comparison_table,
    format_table_markdown,
    print_comparison,
)
from tradingagents.backtest.runner import run_agent_strategy

REPO = Path(__file__).resolve().parents[1]
DATES = pd.DatetimeIndex(["2024-01-12", "2024-01-16", "2024-01-17", "2024-01-18"])
OPENS = [100.0, 101.0, 100.5, 102.0]
PRE_WINDOW = pd.Timestamp("2024-01-11")  # D-1 of the first window session


@pytest.fixture(autouse=True)
def _offline_buy_and_hold_at_open(monkeypatch):
    # main() adds a "Buy & Hold (open)" curve when agents run; keep it offline.
    monkeypatch.setattr(
        "tradingagents.backtest.runner.run_buy_and_hold_at_open",
        lambda *_a, **_k: pd.Series([100.0, 101.0, 102.0], index=pd.bdate_range("2024-02-01", periods=3)),
    )


def _h1_stats():
    spec = importlib.util.spec_from_file_location("h1_stats", REPO / "scripts" / "h1_stats.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


h1 = _h1_stats()


def _load_run_backtest():
    spec = importlib.util.spec_from_file_location("run_backtest", REPO / "run_backtest.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _equity(market: str, bad_dates: set[str] | None = None) -> pd.Series:
    bad_dates = bad_dates or set()
    opens = [OPENS[0], *OPENS]
    df = pd.DataFrame({"Date": [PRE_WINDOW, *DATES], "Open": opens, "Close": opens})

    # decider gets D-1: "2024-01-11" is the decision for the first session (BUY at its open).
    def decider(date, _window):
        if date in bad_dates:
            raise RuntimeError("api down")
        return "BUY" if date == "2024-01-11" else "HOLD"

    with patch("tradingagents.backtest.runner.load_ohlcv", return_value=df):
        return run_agent_strategy(
            decider, "X", "2024-01-12", "2024-01-18", 1_000.0, market=market
        )


def _raw_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_writer_constants_match_h1_stats():
    assert COLUMNS == h1.COLUMNS
    assert PREREG_TICKERS == h1.TICKERS
    assert HARNESS_TO_ARM == {"baseline": "absent", "macro": "present"}
    assert set(HARNESS_TO_ARM.values()) == set(h1.ARMS)


def test_rows_round_trip_through_h1_stats_validation(tmp_path):
    us = _equity("US")
    br = _equity("BR", bad_dates={"2024-01-16"})
    path = tmp_path / "cells.csv"
    specs = [
        ("AAPL", "baseline", 0, us),
        ("AAPL", "macro", 0, us),
        ("PETR4.SA", "baseline", 1, br),
        ("vale3.sa", "present", 2, br),
    ]
    for ticker, arm, seed, equity in specs:
        append_cells(path, [make_cell_row(ticker, arm, seed, equity)])

    text = path.read_text(encoding="utf-8")
    assert text.splitlines()[0] == ",".join((*COLUMNS, SHARPE_FLAT_FIELD))
    assert text.count(text.splitlines()[0]) == 1

    raw = _raw_rows(path)
    assert [(r["ticker"], r["market"], r["arm"], r["seed"], r["rf_source"]) for r in raw] == [
        ("AAPL", "US", "absent", "0", "FRED-DTB3"),
        ("AAPL", "US", "present", "0", "FRED-DTB3"),
        ("PETR4", "BR", "absent", "1", "BCB-SGS-12"),
        ("VALE3", "BR", "present", "2", "BCB-SGS-12"),
    ]
    assert raw[2]["n_decision_errors"] == "1"
    assert raw[2]["n_days"] == str(len(br))
    assert float(raw[0]["sharpe"]) == h1_cell_metrics(us, "US")["sharpe"]
    assert float(raw[2]["sharpe"]) == h1_cell_metrics(br, "BR")["sharpe"]

    loaded = h1.load_cells(path)
    assert [(r["ticker"], r["arm"], r["seed"], r["market"]) for r in loaded] == [
        ("AAPL", "absent", 0, "US"),
        ("AAPL", "present", 0, "US"),
        ("PETR4", "absent", 1, "BR"),
        ("VALE3", "present", 2, "BR"),
    ]
    assert loaded[0]["sharpe"] == pytest.approx(h1_cell_metrics(us, "US")["sharpe"])
    assert loaded[2]["n_decision_errors"] == 1
    assert h1.main([str(path)]) == 0

    # A later replicate appends; the same (ticker, arm, seed) does not.
    append_cells(path, [make_cell_row("AAPL", "baseline", 3, us)])
    before = path.read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        append_cells(path, [make_cell_row("AAPL", "baseline", 0, us)])
    assert path.read_text(encoding="utf-8") == before
    h1.load_cells(path)


def test_append_preserves_extra_columns(tmp_path):
    path = tmp_path / "cells.csv"
    path.write_text(
        "model," + ",".join(COLUMNS) + f",{SHARPE_FLAT_FIELD}\n"
        "claude,AAPL,US,absent,0,ok,4,0,1.0,FRED-DTB3,\n",
        encoding="utf-8",
    )
    append_cells(path, [make_cell_row("AAPL", "macro", 0, _equity("US"))])
    raw = _raw_rows(path)
    assert raw[0]["model"] == "claude"
    assert raw[1]["model"] == ""
    assert raw[1]["arm"] == "present"
    h1.load_cells(path)


def test_append_refuses_sharpe_flat_on_legacy_header_but_accepts_failed_rows(tmp_path):
    path = tmp_path / "cells.csv"
    path.write_text(",".join(COLUMNS) + "\nAAPL,US,absent,0,ok,4,0,1.0,FRED-DTB3\n", encoding="utf-8")
    before = path.read_bytes()
    with pytest.raises(ValueError, match="has no sharpe_flat column; migrate it first"):
        append_cells(path, [make_cell_row("AAPL", "macro", 0, _equity("US"))])
    assert path.read_bytes() == before
    append_cells(path, [make_cell_row("AAPL", "macro", 0, status="failed")])
    raw = _raw_rows(path)
    assert list(raw[1]) == list(COLUMNS)
    assert raw[1]["status"] == "failed"
    h1.load_cells(path)


def test_append_new_file_header_includes_sharpe_flat(tmp_path):
    path = tmp_path / "cells.csv"
    equity = _equity("US")
    append_cells(path, [make_cell_row("AAPL", "baseline", 0, equity)])
    assert path.read_text(encoding="utf-8").splitlines()[0] == ",".join((*COLUMNS, SHARPE_FLAT_FIELD))
    assert float(_raw_rows(path)[0][SHARPE_FLAT_FIELD]) == flat_rf_metrics(equity)["sharpe"]


def test_failed_row_leaves_sharpe_empty_and_still_validates(tmp_path):
    path = tmp_path / "cells.csv"
    row = make_cell_row("GOOGL", "macro", 4, status="failed")
    assert row["sharpe"] == ""
    assert row["arm"] == "present"
    assert row["n_days"] == "0"
    assert row["rf_source"] == "FRED-DTB3"
    append_cells(path, [row])
    loaded = h1.load_cells(path)
    assert loaded[0]["status"] == "failed"
    assert math.isnan(loaded[0]["sharpe"])
    assert h1.main([str(path)]) == 0


def test_sharpe_flat_is_the_flat_rf_table_sharpe_and_empty_when_failed():
    equity = _equity("US")
    flat = ExtendedMetricsCalculator(annual_rf_rate=FLAT_RF_SENSITIVITY).compute(equity)["sharpe"]
    r = equity.pct_change().dropna() - FLAT_RF_SENSITIVITY / 252
    assert flat == pytest.approx(math.sqrt(252) * r.mean() / r.std(ddof=1))

    row = make_cell_row("AAPL", "baseline", 0, equity)
    assert float(row[SHARPE_FLAT_FIELD]) == flat
    assert float(row[SHARPE_FLAT_FIELD]) != float(row["sharpe"])
    table = build_comparison_table({"TradingAgents (baseline)": equity}, market="US")
    assert table[SHARPE_FLAT_COL].iloc[0] == f"{flat:.3f}"
    assert make_cell_row("AAPL", "macro", 0, status="failed")[SHARPE_FLAT_FIELD] == ""


@pytest.mark.parametrize(
    "ticker, message",
    [
        ("MSFT", "not pre-registered"),
        ("PETR4", r"\.SA"),
        ("petr4", r"\.SA"),
    ],
)
def test_writer_rejects_tickers_that_cannot_validate(ticker, message):
    with pytest.raises(ValueError, match=message):
        make_cell_row(ticker, "baseline", 0, _equity("US"))


def test_h1_column_sits_beside_flat_sharpe_for_agent_arms_only(capsys):
    outside = pd.bdate_range("2020-01-01", periods=8)
    agent = _equity("US")
    curves = {
        "Buy & Hold": pd.Series(range(100, 108), index=outside, dtype=float),
        "MACD(12,26,9)": pd.Series(range(100, 108), index=outside, dtype=float),
        "TradingAgents (baseline)": agent,
    }
    df = print_comparison(curves, market="US")
    cols = list(df.columns)
    assert cols[cols.index(SHARPE_FLAT_COL) + 1] == H1_SHARPE_COL
    assert df.loc[df["Strategy"] == "Buy & Hold", H1_SHARPE_COL].iloc[0] == "—"
    assert df.loc[df["Strategy"] == "MACD(12,26,9)", H1_SHARPE_COL].iloc[0] == "—"

    h1_sharpe = h1_cell_metrics(agent, "US")["sharpe"]
    flat = ExtendedMetricsCalculator(annual_rf_rate=FLAT_RF_SENSITIVITY).compute(agent)["sharpe"]
    shown = df.loc[df["Strategy"] == "TradingAgents (baseline)", H1_SHARPE_COL].iloc[0]
    assert shown == f"{h1_sharpe:.3f}"
    assert shown != f"{flat:.3f}"
    header = format_table_markdown(df).splitlines()[0]
    assert header.index(SHARPE_FLAT_COL) < header.index(H1_SHARPE_COL)

    # Rich wraps the long header inside a narrow console; the fragments stay visible.
    printed = capsys.readouterr().out
    assert "H1 Sharpe" in printed
    assert "excess over" in printed
    assert "daily rf" in printed
    assert shown in printed
    assert "exploratory" in printed


def test_cli_appends_mapped_rows_and_prints_h1_sharpe(tmp_path, capsys):
    run_backtest = _load_run_backtest()
    equity = _equity("BR")
    path = tmp_path / "nested" / "cells.csv"

    class FakeGraph:
        def __init__(self, *args, **kwargs):
            pass

        def propagate(self, ticker: str, curr_date: str):
            return {}, "HOLD"

    def fake_run_strategy(*_args, **_kwargs):
        idx = pd.bdate_range("2024-02-01", periods=5)
        return pd.Series([100.0, 101.0, 102.0, 101.5, 103.0], index=idx)

    def fake_run_agent(*_args, **_kwargs):
        out = equity.copy()
        out.attrs = dict(equity.attrs)
        return out

    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "ci-dummy-key"}), patch.object(
        run_backtest, "load_dotenv"
    ), patch("tradingagents.graph.trading_graph.TradingAgentsGraph", FakeGraph), patch.object(
        run_backtest, "run_strategy", side_effect=fake_run_strategy
    ), patch.object(run_backtest, "run_agent_strategy", side_effect=fake_run_agent):
        rc = run_backtest.main(
            [
                "--ticker", "PETR4.SA",
                "--start", "2024-01-02",
                "--end", "2024-03-28",
                "--arms", "baseline,macro",
                "--cells-out", str(path),
                "--seed", "2",
            ]
        )

    assert rc == 0
    raw = _raw_rows(path)
    assert [(r["ticker"], r["market"], r["arm"], r["seed"], r["status"]) for r in raw] == [
        ("PETR4", "BR", "absent", "2", "ok"),
        ("PETR4", "BR", "present", "2", "ok"),
    ]
    assert raw[0]["rf_source"] == "BCB-SGS-12"
    assert float(raw[0]["sharpe"]) == h1_cell_metrics(equity, "BR")["sharpe"]
    assert float(raw[0][SHARPE_FLAT_FIELD]) == flat_rf_metrics(equity)["sharpe"]
    loaded = h1.load_cells(path)
    assert h1.main([str(path)]) == 0
    assert loaded[0]["n_days"] == len(equity)

    printed = capsys.readouterr().out
    shown = f"{h1_cell_metrics(equity, 'BR')['sharpe']:.3f}"
    assert "H1 Sharpe" in printed
    assert shown in printed

    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "ci-dummy-key"}), patch.object(
        run_backtest, "load_dotenv"
    ), patch("tradingagents.graph.trading_graph.TradingAgentsGraph", FakeGraph), patch.object(
        run_backtest, "run_strategy", side_effect=fake_run_strategy
    ), patch.object(run_backtest, "run_agent_strategy", side_effect=fake_run_agent), patch.object(
        run_backtest, "print_comparison"
    ):
        with pytest.raises(SystemExit) as exc:
            run_backtest.main(
                [
                    "--ticker", "PETR4.SA",
                    "--start", "2024-01-02",
                    "--end", "2024-03-28",
                    "--arms", "baseline",
                    "--cells-out", str(path),
                    "--seed", "2",
                ]
            )
    assert exc.value.code == 2
    assert len(_raw_rows(path)) == 2
    h1.load_cells(path)


def test_cli_failed_arm_is_a_failed_row(tmp_path):
    run_backtest = _load_run_backtest()
    path = tmp_path / "cells.csv"

    class FakeGraph:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("pipeline down")

    def fake_run_strategy(*_args, **_kwargs):
        idx = pd.bdate_range("2024-02-01", periods=4)
        return pd.Series([100.0, 101.0, 102.0, 103.0], index=idx)

    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "ci-dummy-key"}), patch.object(
        run_backtest, "load_dotenv"
    ), patch("tradingagents.graph.trading_graph.TradingAgentsGraph", FakeGraph), patch.object(
        run_backtest, "run_strategy", side_effect=fake_run_strategy
    ), patch.object(run_backtest, "print_comparison"):
        rc = run_backtest.main(
            [
                "--ticker", "AMZN",
                "--start", "2024-01-02",
                "--end", "2024-03-28",
                "--arms", "macro",
                "--cells-out", str(path),
                "--seed", "0",
            ]
        )

    assert rc == 1
    raw = _raw_rows(path)
    assert raw == [
        {
            "ticker": "AMZN",
            "market": "US",
            "arm": "present",
            "seed": "0",
            "status": "failed",
            "n_days": "0",
            "n_decision_errors": "0",
            "sharpe": "",
            "rf_source": "FRED-DTB3",
            "start": "2024-01-02",
            "end": "2024-03-28",
            "sharpe_flat": "",
            "deep_think_llm": "claude-sonnet-4-6",
            "quick_think_llm": "claude-haiku-4-5-20251001",
            "temperature": "provider-default",
            "max_debate_rounds": "1",
            "max_risk_discuss_rounds": "1",
            "data_cutoff": "",
        }
    ]
    assert h1.load_cells(path)[0]["status"] == "failed"


def test_cli_row_records_data_cutoff_and_keeps_decision_dates(tmp_path):
    """start/end stay the decision dates; data_cutoff is the calendar D-1 of `end`."""
    run_backtest = _load_run_backtest()
    path = tmp_path / "cells.csv"
    start, end = "2024-01-02", "2024-03-28"
    dates = [previous_session("PETR4.SA", start), *sessions("PETR4.SA", start, end)]
    prices = pd.DataFrame({"Date": dates, "Open": 100.0, "Close": 100.0})
    decision_dates = []

    class FakeGraph:
        def __init__(self, *args, **kwargs):
            pass

        def propagate(self, ticker, trade_date, decision_date=None):
            decision_dates.append(decision_date)
            return {}, "HOLD"

    def fake_run_strategy(*_args, **_kwargs):
        return pd.Series([100.0, 101.0], index=pd.DatetimeIndex(dates[1:3]))

    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "ci-dummy-key"}), patch.object(
        run_backtest, "load_dotenv"
    ), patch("tradingagents.graph.trading_graph.TradingAgentsGraph", FakeGraph), patch(
        "tradingagents.backtest.runner.load_ohlcv", return_value=prices
    ), patch.object(run_backtest, "run_strategy", side_effect=fake_run_strategy), patch.object(
        run_backtest, "print_comparison"
    ):
        rc = run_backtest.main(
            ["--ticker", "PETR4.SA", "--start", start, "--end", end,
             "--arms", "baseline", "--cells-out", str(path)]
        )

    assert rc == 0
    [row] = _raw_rows(path)
    assert (row["start"], row["end"]) == (start, end)
    assert row["data_cutoff"] == str(previous_session("PETR4.SA", end).date()) == "2024-03-27"
    assert row["n_days"] == str(len(dates) - 1)
    assert decision_dates[0] == start and decision_dates[-1] == end
    h1.load_cells(path)


def test_cli_skip_agents_and_negative_seed(tmp_path):
    run_backtest = _load_run_backtest()
    path = tmp_path / "cells.csv"

    def fake_run_strategy(*_args, **_kwargs):
        idx = pd.bdate_range("2024-02-01", periods=3)
        return pd.Series([100.0, 101.0, 102.0], index=idx)

    with patch.object(run_backtest, "run_strategy", side_effect=fake_run_strategy), patch.object(
        run_backtest, "print_comparison"
    ):
        rc = run_backtest.main(
            [
                "--ticker", "AAPL",
                "--start", "2024-01-02",
                "--end", "2024-03-28",
                "--skip-agents",
                "--cells-out", str(path),
            ]
        )
    assert rc == 0
    assert not path.exists()

    with pytest.raises(SystemExit) as exc:
        run_backtest.main(
            ["--ticker", "AAPL", "--start", "2024-01-02", "--end", "2024-01-10", "--seed", "-1"]
        )
    assert exc.value.code == 2
