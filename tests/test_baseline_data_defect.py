"""A fail-closed price-data defect in the baselines is a counted exclusion, not a missing cell.

run_backtest.main over the fake vendor (tests/_pr7_fakes.py): the bad (ticker, seed)
gets one status=failed row per planned arm with empty Sharpe columns and a
failure_reason, exit 1; re-running under a new seed hits the defect again; the
real scripts/h1_stats.py counts those rows as ``failed`` exclusions. Offline.
"""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

import _pr7_api as api
import numpy as np
import pandas as pd
import pytest
from _pr7_fakes import RecordingAgent, install_fake_yahoo, make_bars

REPO = Path(__file__).resolve().parents[1]
START, END = "2024-01-02", "2024-03-28"
BAD_DAY = pd.Timestamp("2024-02-07")


@pytest.fixture
def env(monkeypatch, tmp_path):
    yahoo = install_fake_yahoo(monkeypatch, tmp_path)
    module = api.run_backtest_module()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-dummy")
    monkeypatch.setattr(module, "load_dotenv", lambda *a, **k: None)
    # BUY on the first session then HOLD: finite, non-zero Sharpe on ok rows.
    monkeypatch.setattr(
        module, "make_decide_fn",
        lambda **_kw: RecordingAgent(actions=lambda n, _d: "BUY" if n == 0 else "HOLD"),
    )
    return yahoo, module, tmp_path


def _run(module, ticker, seed, cells):
    return module.main(["--ticker", ticker, "--start", START, "--end", END,
                        "--cells-out", str(cells), "--seed", str(seed)])


def _rows(cells):
    with cells.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _assert_failed_cell(rows, bare, seed):
    cell = [r for r in rows if r["ticker"] == bare and r["seed"] == str(seed)]
    assert sorted(r["arm"] for r in cell) == ["absent", "present"]  # every planned arm
    for r in cell:
        assert r["status"] == "failed"
        assert r["sharpe"] == r["sharpe_flat"] == r["data_cutoff"] == ""  # empty, never 0
        assert r["failure_reason"].startswith("data defect:")
        assert "2024-02-07" in r["failure_reason"]


@pytest.mark.parametrize("kind", ["missing_open", "missing_bar"])
def test_baseline_data_defect_writes_failed_rows_and_exits_1(env, kind):
    yahoo, module, tmp_path = env
    bars = make_bars("PETR4.SA")
    yahoo.bars["PETR4.SA"] = (
        bars.drop(index=BAD_DAY) if kind == "missing_bar"
        else bars.assign(Open=bars["Open"].where(bars.index != BAD_DAY, np.nan))
    )
    cells = tmp_path / "cells.csv"

    assert _run(module, "AAPL", 0, cells) == 0
    assert _run(module, "PETR4.SA", 0, cells) == 1
    assert _run(module, "VALE3.SA", 0, cells) == 0

    rows = _rows(cells)
    assert len(rows) == 6  # never fewer than planned for the bad cell
    _assert_failed_cell(rows, "PETR4", 0)
    good = [r for r in rows if r["ticker"] != "PETR4"]
    assert all(r["status"] == "ok" and r["sharpe"] and r["failure_reason"] == "" for r in good)


def test_data_defect_reraises_on_new_seed_and_resume_is_refused(env):
    yahoo, module, tmp_path = env
    bars = make_bars("AAPL")
    bars.loc[BAD_DAY, "Open"] = np.nan
    yahoo.bars["AAPL"] = bars
    cells = tmp_path / "cells.csv"

    assert _run(module, "AAPL", 0, cells) == 1
    with pytest.raises(SystemExit) as exc:  # same key: duplicate, refused before any download
        _run(module, "AAPL", 0, cells)
    assert exc.value.code == 2
    # PREREG §5 re-run under a new seed: the cached vendor frame still carries the defect.
    assert _run(module, "AAPL", 1, cells) == 1

    rows = _rows(cells)
    assert len(rows) == 4
    _assert_failed_cell(rows, "AAPL", 0)
    _assert_failed_cell(rows, "AAPL", 1)


def test_non_data_baseline_error_still_propagates(env, monkeypatch):
    _yahoo, module, tmp_path = env
    cells = tmp_path / "cells.csv"

    def boom(*_a, **_k):
        raise ValueError("code bug, not a data defect")  # plain ValueError: no longer caught

    monkeypatch.setattr(module, "run_strategy", boom)
    with pytest.raises(ValueError) as exc:
        _run(module, "AAPL", 0, cells)
    assert type(exc.value) is ValueError
    assert not cells.exists()


def test_data_defect_error_from_a_baseline_gives_failed_rows(env, monkeypatch):
    from tradingagents.backtest.runner import DataDefectError

    _yahoo, module, tmp_path = env
    cells = tmp_path / "cells.csv"

    def defect(*_a, **_k):
        raise DataDefectError("AAPL: duplicate vendor bars on 2024-02-07")

    monkeypatch.setattr(module, "run_strategy", defect)
    assert _run(module, "AAPL", 0, cells) == 1
    _assert_failed_cell(_rows(cells), "AAPL", 0)


def test_h1_stats_counts_baseline_failure_as_failed_exclusion(env):
    yahoo, module, tmp_path = env
    cells = tmp_path / "cells.csv"
    for seed in (0, 1, 2):
        assert _run(module, "AAPL", seed, cells) == 0
    ok_sharpe = {r["arm"]: float(r["sharpe"]) for r in _rows(cells)}

    bars = make_bars("AAPL")
    bars.loc[BAD_DAY, "Open"] = np.nan
    yahoo.bars["AAPL"] = bars
    shutil.rmtree(tmp_path / "ohlcv-cache")  # serve the damaged frame from now on
    assert _run(module, "AAPL", 3, cells) == 1
    _assert_failed_cell(_rows(cells), "AAPL", 3)

    out = tmp_path / "h1.json"
    proc = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "h1_stats.py"), str(cells), "--out", str(out)],
        capture_output=True, text=True, cwd=REPO, check=False,
    )
    assert proc.returncode == 0, proc.stderr
    result = json.loads(out.read_text(encoding="utf-8"))
    assert result["n_rows"] == 8
    assert result["n_valid_runs"] == 6
    assert result["exclusion_counts"]["failed"] == 2
    assert sorted((e["arm"], e["seed"]) for e in result["excluded"] if e["reason"] == "failed") == [
        ("absent", 3), ("present", 3)
    ]
    [aapl] = [t for t in result["per_ticker"] if t["ticker"] == "AAPL"]
    # Empty sharpe is not a 0 entering the mean: only the three ok seeds count.
    assert (aapl["n_absent"], aapl["n_present"]) == (3, 3)
    assert aapl["mean_sharpe_absent"] == pytest.approx(ok_sharpe["absent"], rel=1e-12)
    assert aapl["mean_sharpe_present"] == pytest.approx(ok_sharpe["present"], rel=1e-12)
    assert ok_sharpe["absent"] != 0.0
