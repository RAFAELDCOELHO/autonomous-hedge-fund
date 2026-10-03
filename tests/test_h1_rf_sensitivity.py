"""P3.15: exploratory flat-rf H1 sensitivity (scripts/h1_rf_sensitivity.py). Offline."""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pytest

from tradingagents.backtest.report import SHARPE_FLAT_COL

REPO = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


h1 = _load("h1_stats")
sens = _load("h1_rf_sensitivity")


def _rows() -> list[dict]:
    rows = []
    for i, (ticker, market) in enumerate(h1.TICKERS.items()):
        for arm in h1.ARMS:
            present = arm == "present"
            for s in range(h1.MIN_VALID_SEEDS):
                rows.append({
                    "ticker": ticker, "market": market, "arm": arm, "seed": s, "status": "ok",
                    "n_days": 61, "n_decision_errors": 0,
                    "sharpe": f"{0.1 * i + 0.01 * s + (0.2 if present else 0.0):.6f}",
                    "rf_source": h1.RF_SOURCE[market],
                    "sharpe_flat": f"{0.1 * i - 0.07 * s * (1 if present else -1) + (0.3 + 0.05 * i if present else 0.0):.6f}",
                })
    # Empty sharpe_flat -> NaN -> h1_stats' own missing_sharpe exclusion.
    rows.append({**rows[0], "arm": "present", "seed": 9, "sharpe_flat": ""})
    # failed / truncated / decision_errors exclusions; finite outliers so skipping them moves D.
    rows += [
        {**rows[0], "seed": 10, "status": "failed", "n_days": 0, "sharpe": "", "sharpe_flat": ""},
        {**rows[0], "seed": 11, "n_days": 60, "sharpe": "9.0", "sharpe_flat": "50.0"},
        {**rows[0], "seed": 12, "n_decision_errors": 4, "sharpe": "9.0", "sharpe_flat": "50.0"},
    ]
    return rows


def _write(path: Path, rows: list[dict], columns) -> Path:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(columns), extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return path


def test_script_reports_hand_computed_flat_rf_contrast(tmp_path, capsys):
    rows = _rows()
    cells = _write(tmp_path / "cells.csv", rows, (*h1.COLUMNS, "sharpe_flat"))
    out = tmp_path / "out.json"
    assert sens.main([str(cells), "--out", str(out)]) == 0

    def delta(ticker):
        means = []
        for arm in ("present", "absent"):
            vals = [float(r["sharpe_flat"]) for r in rows
                    if r["ticker"] == ticker and r["arm"] == arm and r["seed"] < h1.MIN_VALID_SEEDS]
            means.append(sum(vals) / len(vals))
        return means[0] - means[1]

    br = [delta(t) for t in ("ITUB4", "BPAC11", "PETR4", "VALE3", "WEGE3")]
    us = [delta(t) for t in ("AAPL", "GOOGL", "AMZN")]
    expected = sum(br) / len(br) - sum(us) / len(us)

    raw = out.read_text(encoding="utf-8")
    result = json.loads(raw)
    assert result["D"] == pytest.approx(expected)
    assert result["exclusion_counts"] == {"missing_sharpe": 1, "failed": 1, "truncated": 1, "decision_errors": 1}
    assert set(result) == {"exploratory", "n_rows", "n_valid_runs", "exclusion_counts", "excluded", "per_ticker", "D",
                           "exclusions_differ_from_primary"}
    printed = capsys.readouterr().out
    assert printed.startswith("EXPLORATORY (PREREGISTRATION §7")
    assert SHARPE_FLAT_COL in printed
    assert "RADL3" in printed and "(control)" in printed
    assert f"D (flat rf, exploratory) = mean dSharpe(BR sensitive) - mean dSharpe(US) = {expected:+.4f}" in printed
    for text in (printed, raw):
        for banned in ("p =", "p_value", "reject", "REJECT", "alpha", "Holm", "pre-registered analysis"):
            assert banned not in text


def test_script_matches_h1_stats_analyze(tmp_path, capsys):
    cells = _write(tmp_path / "cells.csv", _rows(), (*h1.COLUMNS, "sharpe_flat"))
    out = tmp_path / "out.json"
    assert sens.main([str(cells), "--out", str(out)]) == 0
    result = json.loads(out.read_text(encoding="utf-8"))
    ref = json.loads(json.dumps(h1.analyze(sens.load_rows(cells)[1])))
    for key in ("per_ticker", "n_rows", "n_valid_runs", "exclusion_counts", "excluded"):
        assert result[key] == ref[key]
    assert result["D"] == pytest.approx(ref["primary"]["statistic"], rel=0, abs=1e-12)


def test_script_skips_permutation_code(tmp_path, capsys, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("permutation path called")

    for name in ("analyze", "primary_test", "_within_ticker_perm"):
        monkeypatch.setattr(sens.h1_stats, name, boom)
    cells = _write(tmp_path / "cells.csv", _rows(), (*h1.COLUMNS, "sharpe_flat"))
    assert sens.main([str(cells)]) == 0
    assert "D (flat rf, exploratory) = " in capsys.readouterr().out


def test_script_warns_when_exclusions_diverge_from_primary(tmp_path, capsys):
    rows = _rows() + [{**_rows()[0], "arm": "present", "seed": 13, "sharpe": "", "sharpe_flat": "1.0"}]
    cells = _write(tmp_path / "cells.csv", rows, (*h1.COLUMNS, "sharpe_flat"))
    out = tmp_path / "out.json"
    assert sens.main([str(cells), "--out", str(out)]) == 0
    err = capsys.readouterr().err
    assert err.startswith("WARNING: 2 excluded row(s) differ")
    assert "excluded only in sensitivity: AAPL/present/seed 9 (missing_sharpe)" in err
    assert "excluded only in primary: AAPL/present/seed 13 (missing_sharpe)" in err
    assert json.loads(out.read_text(encoding="utf-8"))["exclusions_differ_from_primary"] == [
        {"ticker": "AAPL", "arm": "present", "seed": 9, "reason": "missing_sharpe", "only_in": "sensitivity"},
        {"ticker": "AAPL", "arm": "present", "seed": 13, "reason": "missing_sharpe", "only_in": "primary"},
    ]


def test_script_no_warning_when_exclusions_match_primary(tmp_path, capsys):
    rows = [r for r in _rows() if r["seed"] != 9]
    cells = _write(tmp_path / "cells.csv", rows, (*h1.COLUMNS, "sharpe_flat"))
    out = tmp_path / "out.json"
    assert sens.main([str(cells), "--out", str(out)]) == 0
    assert "WARNING" not in capsys.readouterr().err
    assert json.loads(out.read_text(encoding="utf-8"))["exclusions_differ_from_primary"] == []


def test_script_not_evaluable_without_us_tickers(tmp_path, capsys):
    rows = [r for r in _rows() if r["market"] != "US"]
    cells = _write(tmp_path / "cells.csv", rows, (*h1.COLUMNS, "sharpe_flat"))
    assert sens.main([str(cells)]) == 0
    assert "D (flat rf, exploratory) not evaluable: no BR-sensitive or no US ticker left" in capsys.readouterr().out


@pytest.mark.parametrize("columns, mutate", [
    (h1.COLUMNS, None),
    ((*h1.COLUMNS, "sharpe_flat"), "high"),
])
def test_script_rejects_missing_or_non_numeric_sharpe_flat(tmp_path, capsys, columns, mutate):
    rows = _rows()
    if mutate:
        rows[1]["sharpe_flat"] = mutate
    cells = _write(tmp_path / "cells.csv", rows, columns)
    assert sens.main([str(cells)]) == 2
    assert "sharpe_flat" in capsys.readouterr().err


def test_h1_stats_output_identical_with_or_without_sharpe_flat(tmp_path, capsys):
    rows = _rows()
    outputs = []
    for name, columns in (("plain", h1.COLUMNS), ("flat", (*h1.COLUMNS, "sharpe_flat"))):
        cells = _write(tmp_path / f"{name}.csv", rows, columns)
        out = tmp_path / f"{name}.json"
        assert h1.main([str(cells), "--out", str(out)]) == 0
        outputs.append((capsys.readouterr().out, out.read_bytes()))
    assert outputs[0] == outputs[1]
