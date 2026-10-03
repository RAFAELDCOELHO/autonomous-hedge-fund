"""Offline regression test for selected_analysts wiring in the harness."""

from __future__ import annotations

import importlib.util
import os
import csv
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest


REPO = Path(__file__).resolve().parents[1]
RUN_BACKTEST = REPO / "run_backtest.py"


def _load_run_backtest():
    spec = importlib.util.spec_from_file_location("run_backtest", RUN_BACKTEST)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_macro_arm_instantiates_macro_analyst_and_baseline_does_not():
    run_backtest = _load_run_backtest()
    selected_calls: list[list[str] | None] = []

    class FakeGraph:
        def __init__(self, *args, **kwargs):
            selected_calls.append(kwargs.get("selected_analysts"))

        def propagate(self, ticker: str, curr_date: str):
            return {}, "HOLD"

    def fake_run_strategy(*_args, **_kwargs):
        return [100_000.0]

    def fake_run_agent_strategy(*_args, **_kwargs):
        return [100_000.0]

    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "ci-dummy-key"}), patch.object(
        run_backtest, "load_dotenv"
    ), patch("tradingagents.graph.trading_graph.TradingAgentsGraph", FakeGraph), patch.object(
        run_backtest, "run_strategy", side_effect=fake_run_strategy
    ), patch.object(run_backtest, "run_agent_strategy", side_effect=fake_run_agent_strategy), patch.object(
        run_backtest, "print_comparison"
    ):
        rc = run_backtest.main(
            ["--ticker", "AAPL", "--start", "2024-01-01", "--end", "2024-01-10"]
        )

    assert rc == 0
    assert len(selected_calls) == 2
    baseline_analysts, macro_analysts = selected_calls
    assert baseline_analysts is not None and "macro" not in baseline_analysts
    assert macro_analysts is not None and "macro" in macro_analysts


def test_arms_macro_runs_only_macro_arm():
    run_backtest = _load_run_backtest()
    selected_calls: list[list[str] | None] = []

    class FakeGraph:
        def __init__(self, *args, **kwargs):
            selected_calls.append(kwargs.get("selected_analysts"))

        def propagate(self, ticker: str, curr_date: str):
            return {}, "HOLD"

    def fake_run_strategy(*_args, **_kwargs):
        return [100_000.0]

    def fake_run_agent_strategy(*_args, **_kwargs):
        return [100_000.0]

    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "ci-dummy-key"}), patch.object(
        run_backtest, "load_dotenv"
    ), patch("tradingagents.graph.trading_graph.TradingAgentsGraph", FakeGraph), patch.object(
        run_backtest, "run_strategy", side_effect=fake_run_strategy
    ), patch.object(run_backtest, "run_agent_strategy", side_effect=fake_run_agent_strategy), patch.object(
        run_backtest, "print_comparison"
    ):
        rc = run_backtest.main(
            ["--ticker", "AAPL", "--start", "2024-01-01", "--end", "2024-01-10", "--arms", "macro"]
        )

    assert rc == 0
    assert len(selected_calls) == 1
    assert selected_calls[0] is not None and "macro" in selected_calls[0]


def test_agents_fail_fast_without_anthropic_key():
    run_backtest = _load_run_backtest()

    with patch.dict(os.environ, {}, clear=True), patch.object(run_backtest, "load_dotenv"), patch.object(
        run_backtest, "run_strategy"
    ) as run_strategy_mock, patch.object(run_backtest, "_run_agent_decider") as run_agent_decider_mock, patch.object(
        run_backtest, "print_comparison"
    ) as print_comparison_mock:
        rc = run_backtest.main(
            ["--ticker", "AAPL", "--start", "2024-01-01", "--end", "2024-01-10"]
        )

    assert rc == 2
    run_agent_decider_mock.assert_not_called()
    run_strategy_mock.assert_not_called()
    print_comparison_mock.assert_not_called()


def test_skip_agents_runs_classic_without_anthropic_key():
    run_backtest = _load_run_backtest()

    def fake_run_strategy(*_args, **_kwargs):
        return [100_000.0]

    with patch.dict(os.environ, {}, clear=True), patch.object(
        run_backtest, "load_dotenv"
    ) as load_dotenv_mock, patch.object(
        run_backtest, "run_strategy", side_effect=fake_run_strategy
    ) as run_strategy_mock, patch.object(run_backtest, "print_comparison"):
        rc = run_backtest.main(
            [
                "--ticker",
                "AAPL",
                "--start",
                "2024-01-01",
                "--end",
                "2024-01-10",
                "--skip-agents",
            ]
        )

    assert rc == 0
    assert run_strategy_mock.call_count == 3
    load_dotenv_mock.assert_not_called()


def test_invalid_arms_exits_with_code_2_and_clear_message(capsys):
    run_backtest = _load_run_backtest()

    with pytest.raises(SystemExit) as exc:
        run_backtest.main(
            [
                "--ticker",
                "AAPL",
                "--start",
                "2024-01-01",
                "--end",
                "2024-01-10",
                "--arms",
                "baseline,invalid",
            ]
        )

    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert "invalid arm(s): invalid" in captured.err
    assert "Allowed: baseline,macro" in captured.err


def test_cells_out_rejects_non_preregistered_ticker_before_any_strategy_or_llm(capsys, tmp_path):
    run_backtest = _load_run_backtest()
    cells_path = tmp_path / "cells.csv"

    with patch.object(run_backtest, "run_strategy") as run_strategy_mock, patch.object(
        run_backtest, "_run_agent_decider"
    ) as run_agent_decider_mock:
        with pytest.raises(SystemExit) as exc:
            run_backtest.main(
                [
                    "--ticker",
                    "MSFT",
                    "--start",
                    "2024-01-01",
                    "--end",
                    "2024-01-10",
                    "--cells-out",
                    str(cells_path),
                ]
            )

    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert "--cells-out requires a pre-registered ticker" in captured.err
    run_strategy_mock.assert_not_called()
    run_agent_decider_mock.assert_not_called()


def test_cells_out_rejects_duplicate_key_before_any_strategy_or_llm(capsys, tmp_path):
    run_backtest = _load_run_backtest()
    cells_path = tmp_path / "cells.csv"
    cells_path.write_text(
        "ticker,market,arm,seed,status,n_days,n_decision_errors,sharpe,rf_source\n"
        "AAPL,US,absent,0,ok,2,0,1.0,FRED-DTB3\n",
        encoding="utf-8",
    )

    with patch.object(run_backtest, "run_strategy") as run_strategy_mock, patch.object(
        run_backtest, "_run_agent_decider"
    ) as run_agent_decider_mock:
        with pytest.raises(SystemExit) as exc:
            run_backtest.main(
                [
                    "--ticker",
                    "AAPL",
                    "--start",
                    "2024-01-01",
                    "--end",
                    "2024-01-10",
                    "--cells-out",
                    str(cells_path),
                    "--arms",
                    "baseline",
                    "--seed",
                    "0",
                ]
            )

    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert "--cells-out already contains (ticker, arm, seed) key(s) for this run" in captured.err
    run_strategy_mock.assert_not_called()
    run_agent_decider_mock.assert_not_called()


def test_cells_out_writes_failed_row_and_continues_next_arm_on_exception(tmp_path):
    run_backtest = _load_run_backtest()
    cells_path = tmp_path / "cells.csv"
    equity = pd.Series(
        [100_000.0, 101_000.0, 100_500.0],
        index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
    )

    def fake_run_strategy(*_args, **_kwargs):
        return [100_000.0]

    arm_calls = {"count": 0}

    def fake_run_agent_decider(*_args, **_kwargs):
        arm_calls["count"] += 1
        if arm_calls["count"] == 1:
            raise RuntimeError("boom")
        return equity

    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "ci-dummy-key"}), patch.object(
        run_backtest, "load_dotenv"
    ), patch.object(run_backtest, "run_strategy", side_effect=fake_run_strategy), patch.object(
        run_backtest, "_selected_analysts_by_arm", return_value={"baseline": ["market"], "macro": ["macro"]}
    ), patch.object(run_backtest, "_run_agent_decider", side_effect=fake_run_agent_decider), patch.object(
        run_backtest, "print_comparison"
    ):
        rc = run_backtest.main(
            [
                "--ticker",
                "AAPL",
                "--start",
                "2024-01-02",
                "--end",
                "2024-01-04",
                "--cells-out",
                str(cells_path),
            ]
        )

    assert rc == 0
    with cells_path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 2
    assert {row["arm"] for row in rows} == {"absent", "present"}
    failed = next(row for row in rows if row["arm"] == "absent")
    ok = next(row for row in rows if row["arm"] == "present")
    assert failed["status"] == "failed"
    assert failed["sharpe"] == ""
    assert failed["start"] == "2024-01-02"
    assert failed["end"] == "2024-01-04"
    assert ok["status"] == "ok"
    assert ok["sharpe"] != ""
    assert ok["start"] == "2024-01-02"
    assert ok["end"] == "2024-01-04"
