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
                    "2024-01-02",
                    "--end",
                    "2024-03-28",
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


def test_cells_out_rejects_bare_b3_ticker_before_any_strategy_or_llm(capsys, tmp_path):
    run_backtest = _load_run_backtest()
    cells_path = tmp_path / "cells.csv"

    with patch.object(run_backtest, "run_strategy") as run_strategy_mock, patch.object(
        run_backtest, "_run_agent_decider"
    ) as run_agent_decider_mock:
        with pytest.raises(SystemExit) as exc:
            run_backtest.main(
                [
                    "--ticker",
                    "PETR4",
                    "--start",
                    "2024-01-02",
                    "--end",
                    "2024-03-28",
                    "--cells-out",
                    str(cells_path),
                ]
            )

    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert "B3 tickers require .SA" in captured.err
    run_strategy_mock.assert_not_called()
    run_agent_decider_mock.assert_not_called()


def test_cells_out_rejects_bad_existing_header_before_any_strategy_or_llm(capsys, tmp_path):
    run_backtest = _load_run_backtest()
    cells_path = tmp_path / "cells.csv"
    cells_path.write_text(
        "ticker,market,arm,seed,status,n_days,n_decision_errors,rf_source\n",
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
                    "2024-01-02",
                    "--end",
                    "2024-03-28",
                    "--cells-out",
                    str(cells_path),
                    "--arms",
                    "baseline",
                ]
            )

    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert "missing required columns" in captured.err
    run_strategy_mock.assert_not_called()
    run_agent_decider_mock.assert_not_called()


def test_cells_out_rejects_header_with_only_start_column(capsys, tmp_path):
    run_backtest = _load_run_backtest()
    cells_path = tmp_path / "cells.csv"
    cells_path.write_text(
        "ticker,market,arm,seed,status,n_days,n_decision_errors,sharpe,rf_source,start\n",
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
                    "2024-01-02",
                    "--end",
                    "2024-03-28",
                    "--cells-out",
                    str(cells_path),
                ]
            )

    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert "must include both start and end columns together" in captured.err
    run_strategy_mock.assert_not_called()
    run_agent_decider_mock.assert_not_called()


def test_cells_out_rejects_wrong_window_before_any_strategy_or_llm(capsys, tmp_path):
    run_backtest = _load_run_backtest()
    cells_path = tmp_path / "cells.csv"

    with patch.object(run_backtest, "run_strategy") as run_strategy_mock, patch.object(
        run_backtest, "_run_agent_decider"
    ) as run_agent_decider_mock:
        with pytest.raises(SystemExit) as exc:
            run_backtest.main(
                [
                    "--ticker",
                    "AAPL",
                    "--start",
                    "2023-01-02",
                    "--end",
                    "2023-03-28",
                    "--cells-out",
                    str(cells_path),
                ]
            )

    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert "--cells-out requires the preregistered H1 window" in captured.err
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
        run_backtest.logging, "exception"
    ) as log_exception_mock, patch.object(
        run_backtest, "print_comparison"
    ):
        rc = run_backtest.main(
            [
                "--ticker",
                "AAPL",
                "--start",
                "2024-01-02",
                "--end",
                "2024-03-28",
                "--cells-out",
                str(cells_path),
            ]
        )

    assert rc == 1
    log_exception_mock.assert_called_once()
    assert "failed; recording status=failed" in str(log_exception_mock.call_args.args[0])
    assert log_exception_mock.call_args.args[1] == "baseline"
    with cells_path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 2
    assert {row["arm"] for row in rows} == {"absent", "present"}
    failed = next(row for row in rows if row["arm"] == "absent")
    ok = next(row for row in rows if row["arm"] == "present")
    assert failed["status"] == "failed"
    assert failed["sharpe"] == ""
    assert failed["start"] == "2024-01-02"
    assert failed["end"] == "2024-03-28"
    assert ok["status"] == "ok"
    assert ok["sharpe"] != ""
    assert ok["start"] == "2024-01-02"
    assert ok["end"] == "2024-03-28"


def test_ensure_cells_extra_columns_uses_atomic_replace_and_preserves_original_on_failure(tmp_path):
    run_backtest = _load_run_backtest()
    path = tmp_path / "cells.csv"
    path.write_text(
        "ticker,market,arm,seed,status,n_days,n_decision_errors,sharpe,rf_source\n"
        "AAPL,US,absent,0,ok,2,0,1.0,FRED-DTB3\n",
        encoding="utf-8",
    )
    before = path.read_text(encoding="utf-8")

    with patch.object(run_backtest.os, "replace", side_effect=OSError("replace failed")) as replace_mock:
        with pytest.raises(OSError, match="replace failed"):
            run_backtest._ensure_cells_extra_columns(path, ("start", "end"))

    replace_mock.assert_called_once()
    assert path.read_text(encoding="utf-8") == before
    assert list(tmp_path.glob("*.tmp")) == []


@pytest.mark.parametrize(
    ("start", "end"),
    [
        ("2024-01-03", "2024-03-28"),  # only start differs
        ("2024-01-02", "2024-03-27"),  # only end differs
    ],
    ids=["start-only", "end-only"],
)
def test_cells_out_rejects_window_when_only_one_endpoint_differs(capsys, tmp_path, start, end):
    run_backtest = _load_run_backtest()
    cells_path = tmp_path / "cells.csv"

    with patch.object(run_backtest, "run_strategy") as run_strategy_mock, patch.object(
        run_backtest, "_run_agent_decider"
    ) as run_agent_decider_mock:
        with pytest.raises(SystemExit) as exc:
            run_backtest.main(
                ["--ticker", "AAPL", "--start", start, "--end", end, "--cells-out", str(cells_path)]
            )

    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert "--cells-out requires the preregistered H1 window" in captured.err
    run_strategy_mock.assert_not_called()
    run_agent_decider_mock.assert_not_called()


_VALID_HEADER = "ticker,market,arm,seed,status,n_days,n_decision_errors,sharpe,rf_source\n"


@pytest.mark.parametrize(
    ("existing", "ticker", "start", "end"),
    [
        (None, "MSFT", "2024-01-02", "2024-03-28"),  # non-preregistered ticker
        (None, "PETR4", "2024-01-02", "2024-03-28"),  # bare B3 ticker
        (None, "AAPL", "2024-01-03", "2024-03-28"),  # wrong window
        (_VALID_HEADER + "AAPL,US,absent,0,ok,2,0,1.0,FRED-DTB3\n", "AAPL", "2024-01-02", "2024-03-28"),  # duplicate
        ("ticker,market,arm,seed,status,n_days,n_decision_errors,rf_source\n", "AAPL", "2024-01-02", "2024-03-28"),  # bad header
        (_VALID_HEADER.rstrip("\n") + ",start\n", "AAPL", "2024-01-02", "2024-03-28"),  # start without end
    ],
    ids=["unregistered-ticker", "bare-b3", "wrong-window", "duplicate-key", "bad-header", "start-without-end"],
)
def test_cells_out_preflight_rejection_never_touches_cells_file(tmp_path, existing, ticker, start, end):
    run_backtest = _load_run_backtest()
    cells_path = tmp_path / "cells.csv"
    if existing is not None:
        cells_path.write_text(existing, encoding="utf-8")
        os.utime(cells_path, ns=(1_000_000_000, 1_000_000_000))
        before_bytes = cells_path.read_bytes()
        before_stat = cells_path.stat()
        before_ino = before_stat.st_ino
        before_mtime = before_stat.st_mtime_ns

    with patch.object(run_backtest, "run_strategy") as run_strategy_mock, patch.object(
        run_backtest, "_run_agent_decider"
    ) as run_agent_decider_mock:
        with pytest.raises(SystemExit) as exc:
            run_backtest.main(
                [
                    "--ticker",
                    ticker,
                    "--start",
                    start,
                    "--end",
                    end,
                    "--cells-out",
                    str(cells_path),
                    "--arms",
                    "baseline",
                    "--seed",
                    "0",
                ]
            )

    assert exc.value.code == 2
    run_strategy_mock.assert_not_called()
    run_agent_decider_mock.assert_not_called()
    if existing is None:
        assert not cells_path.exists()
        assert list(tmp_path.iterdir()) == []
    else:
        assert cells_path.read_bytes() == before_bytes
        after_stat = cells_path.stat()
        assert after_stat.st_ino == before_ino
        assert after_stat.st_mtime_ns == before_mtime
        assert list(tmp_path.iterdir()) == [cells_path]
