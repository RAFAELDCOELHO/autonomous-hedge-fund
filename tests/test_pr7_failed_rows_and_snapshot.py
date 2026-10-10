"""PR7 (P4.11): baseline errors become `failed` cells that the REAL
scripts/h1_stats.py excludes, and the H1 price snapshot is validated up front.

Expected to FAIL on e5ea062 (baseline errors are not caught there and the
runner does not fail closed; there is no snapshot validator or calendar) and to
PASS after PR7. Assumed names: tests/_pr7_api.py (A6c, A6d).
"""

from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import _pr7_api as api
from _pr7_fakes import RecordingAgent, install_fake_yahoo, make_bars

REPO = Path(__file__).resolve().parents[1]
H1_STATS = REPO / "scripts" / "h1_stats.py"
ARMS = ["baseline", "macro"]
SEEDS = [0, 1, 2, 3, 4]  # PREREGISTRATION §2: 5 seeds per ticker x arm
BAD_TICKER, BAD_SEED, BAD_DAY = "PETR4.SA", 4, "2024-02-07"


def _h1_stats_module():
    spec = importlib.util.spec_from_file_location("h1_stats_pr7", H1_STATS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read(path: Path):
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        return list(reader.fieldnames or []), list(reader)


def _produce_baseline_failed_rows(monkeypatch, tmp_path, kind):
    """Run the real cells.csv producer for the bad (ticker, seed) with a price
    defect that makes the classical baselines fail."""
    yahoo = install_fake_yahoo(monkeypatch, tmp_path)
    bars = make_bars(BAD_TICKER)
    if kind == "missing_open":
        bars.loc[pd.Timestamp(BAD_DAY), "Open"] = np.nan
    else:
        bars = bars.drop(index=pd.Timestamp(BAD_DAY))
    yahoo.bars[BAD_TICKER] = bars
    cells = tmp_path / "produced.csv"
    outcomes = api.write_cells_grid(cells, [BAD_TICKER], ARMS, [BAD_SEED], RecordingAgent, monkeypatch)
    assert len(outcomes) == 1
    return outcomes[0], cells


@pytest.mark.parametrize("kind", ["missing_open", "missing_bar"])
def test_baseline_error_becomes_failed_row_in_existing_h1_format(monkeypatch, tmp_path, kind):
    outcome, cells = _produce_baseline_failed_rows(monkeypatch, tmp_path, kind)

    assert outcome["exc"] is None, f"baseline error must be caught, got {outcome['exc']!r}"
    assert outcome["rc"] not in (0, None)
    assert cells.exists()
    fieldnames, rows = _read(cells)
    assert len(rows) == len(ARMS), "one failed row per arm; none may be dropped"
    assert api.FAILURE_REASON_COLUMN in fieldnames

    from tradingagents.backtest.cells import make_cell_row

    for arm, row in zip(ARMS, rows):
        expected = api.baseline_failed_row(BAD_TICKER, arm, BAD_SEED)
        # The adapter's literal format is the one the existing writer produces.
        existing = make_cell_row(BAD_TICKER, arm, BAD_SEED, status="failed")
        assert {k: existing[k] for k in expected} == expected
        assert {k: row.get(k) for k in expected} == expected
        assert (row.get(api.FAILURE_REASON_COLUMN) or "").strip(), "failed row needs a reason"
        # Same as the agent failed row: data_cutoff = calendar D-1 of the first session.
        assert row.get(api.CELLS_DATA_CUTOFF_COLUMN) == api.first_data_cutoff(BAD_TICKER, api.prereg_window()[0])


def _ok_sharpe(ticker: str, arm: str, seed: int) -> float:
    names = sorted(api.baseline_failed_row(t, "baseline", 0)["ticker"] for t in api.prereg_tickers_yahoo())
    return round(0.4 + 0.07 * names.index(ticker) + 0.11 * seed + (0.25 if arm == "present" else 0.0), 6)


@pytest.mark.parametrize("kind", ["missing_open", "missing_bar"])
def test_real_h1_stats_excludes_baseline_failed_cell_without_zero_filling(monkeypatch, tmp_path, kind):
    outcome, produced = _produce_baseline_failed_rows(monkeypatch, tmp_path, kind)
    fieldnames, failed_rows = _read(produced)
    assert [r["status"] for r in failed_rows] == ["failed"] * len(ARMS)

    ok_rows = []
    for ticker in api.prereg_tickers_yahoo():
        for arm in ARMS:
            for seed in SEEDS:
                if ticker == BAD_TICKER and seed == BAD_SEED:
                    continue
                row = {k: "" for k in fieldnames}
                row.update(api.baseline_failed_row(ticker, arm, seed))
                row.update(status="ok", n_days="61", n_decision_errors="0",
                           sharpe=repr(_ok_sharpe(row["ticker"], row["arm"], seed)), sharpe_flat="0.1")
                ok_rows.append(row)

    def write(path, rows):
        with path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
            w.writeheader()
            w.writerows(rows)

    with_failed, without = tmp_path / "with_failed.csv", tmp_path / "without.csv"
    write(with_failed, ok_rows[:5] + failed_rows + ok_rows[5:])
    write(without, ok_rows)

    results = {}
    for name, path in (("with", with_failed), ("without", without)):
        out_json = tmp_path / f"{name}.json"
        proc = subprocess.run(
            [sys.executable, str(H1_STATS), str(path), "--out", str(out_json)],
            cwd=REPO, capture_output=True, text=True, timeout=300,
        )
        assert proc.returncode == 0, proc.stderr  # does not crash
        results[name] = json.loads(out_json.read_text())

    a, b = results["with"], results["without"]
    bare = api.baseline_failed_row(BAD_TICKER, "baseline", 0)["ticker"]
    # Counted as an exclusion with h1_stats' own reason code.
    assert a["exclusion_counts"].get("failed", 0) == b["exclusion_counts"].get("failed", 0) + len(ARMS)
    for arm in ("absent", "present"):
        assert {"ticker": bare, "arm": arm, "seed": BAD_SEED, "reason": "failed"} in a["excluded"]
    # Empty metrics are not zeros: every aggregate equals the run without the cell.
    assert a["n_rows"] == b["n_rows"] + len(ARMS)
    assert a["n_valid_runs"] == b["n_valid_runs"]
    for key in ("per_ticker", "primary", "secondary"):
        assert a[key] == b[key], key
    petr = next(r for r in a["per_ticker"] if r["ticker"] == bare)
    assert petr["n_absent"] == petr["n_present"] == len(SEEDS) - 1
    assert petr["mean_sharpe_absent"] == pytest.approx(
        np.mean([_ok_sharpe(bare, "absent", s) for s in SEEDS if s != BAD_SEED])
    )


# --------------------------------------------------------------------------
# H1 snapshot validation (Lingxi)
# --------------------------------------------------------------------------


def test_prereg_grid_is_9_tickers_and_61_sessions_on_each_exchange_calendar():
    tickers = api.prereg_tickers_yahoo()
    start, end = api.prereg_window()
    assert (start, end) == ("2024-01-02", "2024-03-28")
    assert len(tickers) == 9
    assert {api.baseline_failed_row(t, "baseline", 0)["ticker"] for t in tickers} == set(_h1_stats_module().TICKERS)
    for t in tickers:
        sessions = api.exchange_sessions(t, start, end)
        assert len(sessions) == 61, (t, len(sessions))
        assert sessions[0] == start and sessions[-1] == end
        # Per-month split locks the exchange: NYSE 21/20/20, B3 22/19/20 (Jan/Feb/Mar 2024).
        per_month = [sum(d.startswith(f"2024-{m:02d}") for d in sessions) for m in (1, 2, 3)]
        assert per_month == ([22, 19, 20] if t.endswith(".SA") else [21, 20, 20]), (t, per_month)
        if t.endswith(".SA"):
            assert {"2024-02-12", "2024-02-13"}.isdisjoint(sessions), t  # Carnaval: B3 closed
            assert "2024-01-15" in sessions, t
        else:
            assert "2024-01-15" not in sessions, t  # MLK Day: NYSE closed
            assert {"2024-02-12", "2024-02-13"} <= set(sessions), t


@pytest.fixture
def snapshot(monkeypatch, tmp_path):
    yahoo = install_fake_yahoo(monkeypatch, tmp_path)
    for t in api.prereg_tickers_yahoo():
        yahoo.bars[t] = make_bars(t)
    return yahoo


def _nan_open(yahoo, ticker, day):
    yahoo.bars[ticker].loc[pd.Timestamp(day), "Open"] = np.nan


def _drop(yahoo, ticker, day):
    yahoo.bars[ticker] = yahoo.bars[ticker].drop(index=pd.Timestamp(day))


def test_clean_snapshot_passes(snapshot):
    api.validate_snapshot()  # must not raise


BAR, OPEN = api.SNAPSHOT_FIELD_BAR, api.SNAPSHOT_FIELD_OPEN
SINGLE_DEFECTS = [
    ("missing_open_mid", _nan_open, "AAPL", "2024-02-07", OPEN),
    ("missing_bar_mid", _drop, "PETR4.SA", "2024-03-01", BAR),
    ("missing_first_day_cutoff_us", _drop, "AAPL", "2023-12-29", BAR),
    ("missing_first_day_cutoff_b3", _drop, "VALE3.SA", "2023-12-28", BAR),
    ("missing_post_last_open", _nan_open, "GOOGL", "2024-04-01", OPEN),
    ("missing_post_last_bar", _drop, "ITUB4.SA", "2024-04-01", BAR),
    ("missing_open_first_session", _nan_open, "WEGE3.SA", "2024-01-02", OPEN),
    ("missing_open_last_session", _nan_open, "AMZN", "2024-03-28", OPEN),
]


@pytest.mark.parametrize(("name", "damage", "ticker", "day", "field"), SINGLE_DEFECTS, ids=[d[0] for d in SINGLE_DEFECTS])
def test_single_snapshot_defect_raises_and_is_listed(snapshot, name, damage, ticker, day, field):
    damage(snapshot, ticker, day)
    with pytest.raises(api.FAIL_CLOSED_ERRORS) as info:
        api.validate_snapshot()
    assert api.snapshot_missing(info.value) == {(ticker, day, field)}


def test_every_snapshot_defect_is_listed_not_just_the_first(snapshot):
    for _, damage, ticker, day, _ in SINGLE_DEFECTS:
        damage(snapshot, ticker, day)
    with pytest.raises(api.FAIL_CLOSED_ERRORS) as info:
        api.validate_snapshot()
    assert api.snapshot_missing(info.value) == {(t, d, f) for _, _, t, d, f in SINGLE_DEFECTS}


def test_single_exchange_holidays_are_not_flagged_as_missing(snapshot):
    """B3 has no bars on Carnival / 2023-12-29; NYSE none on MLK, Presidents'
    Day; neither is a defect for the other exchange's tickers."""
    _drop(snapshot, "BPAC11.SA", "2024-03-01")  # one real defect so the full list is visible
    with pytest.raises(api.FAIL_CLOSED_ERRORS) as info:
        api.validate_snapshot()
    missing = api.snapshot_missing(info.value)
    assert missing == {("BPAC11.SA", "2024-03-01", BAR)}
    for t in api.prereg_tickers_yahoo():
        bars = snapshot.bars[t].index.strftime("%Y-%m-%d")
        if t.endswith(".SA"):
            assert {"2024-02-12", "2024-02-13", "2023-12-29"}.isdisjoint(bars)
            assert {"2024-01-15", "2024-02-19"} <= set(bars)
        else:
            assert {"2024-01-15", "2024-02-19"}.isdisjoint(bars)
            assert {"2024-02-12", "2024-02-13", "2023-12-29"} <= set(bars)
