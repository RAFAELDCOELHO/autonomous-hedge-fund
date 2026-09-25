"""Offline tests for the pre-registered H1 test (scripts/h1_stats.py).

All inputs are synthetic cells.csv files written to tmp_path; no network, no LLM.
"""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "h1_stats.py"


def _load():
    spec = importlib.util.spec_from_file_location("h1_stats", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


h1 = _load()


def _rows(effect_br: float, effect_us: float, seed: int = 0, n_seeds: int = 5) -> list[dict]:
    """Synthetic runs: present-arm Sharpe shifted by the market's effect."""
    rng = np.random.default_rng(seed)
    rows = []
    for ticker, market in h1.TICKERS.items():
        effect = effect_us if market == "US" else effect_br
        base = rng.normal(0.5, 0.5)
        for arm in h1.ARMS:
            for s in range(n_seeds):
                sharpe = base + rng.normal(0, 0.1) + (effect if arm == "present" else 0.0)
                rows.append({
                    "ticker": ticker, "market": market, "arm": arm, "seed": s, "status": "ok",
                    "n_days": 61, "n_decision_errors": 0, "sharpe": f"{sharpe:.6f}",
                    "rf_source": h1.RF_SOURCE[market],
                })
    return rows


def _write(path: Path, rows: list[dict], columns=h1.COLUMNS) -> Path:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(columns), extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return path


def _run(tmp_path: Path, rows: list[dict]) -> dict:
    return h1.analyze(h1.load_cells(_write(tmp_path / "cells.csv", rows)))


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda r: r[0].update(ticker="MSFT"), "not pre-registered"),
        (lambda r: r[0].update(market="BR"), "must have market=US"),
        (lambda r: r[0].update(arm="both"), "arm must be one of"),
        (lambda r: r[0].update(status="error"), "status must be one of"),
        (lambda r: r[0].update(rf_source="flat-4.34%"), "rf_source=FRED-DTB3"),
        (lambda r: r[0].update(seed="-1"), "must be >= 0"),
        (lambda r: r[0].update(n_days="sixty"), "not an integer"),
        (lambda r: r[0].update(sharpe="high"), "not a number"),
        (lambda r: r.append(dict(r[0])), "duplicate"),
    ],
)
def test_schema_violations_raise(tmp_path, mutate, message):
    rows = _rows(0.0, 0.0)
    mutate(rows)
    with pytest.raises(ValueError, match=message):
        h1.load_cells(_write(tmp_path / "cells.csv", rows))


def test_missing_column_and_empty_file_raise(tmp_path):
    cols = [c for c in h1.COLUMNS if c != "rf_source"]
    with pytest.raises(ValueError, match="missing required columns"):
        h1.load_cells(_write(tmp_path / "a.csv", _rows(0, 0), cols))
    with pytest.raises(ValueError, match="no data rows"):
        h1.load_cells(_write(tmp_path / "b.csv", []))


def test_cli_returns_2_on_schema_error(tmp_path, capsys):
    rows = _rows(0, 0)
    rows[0]["ticker"] = "MSFT"
    assert h1.main([str(_write(tmp_path / "cells.csv", rows))]) == 2
    assert "not pre-registered" in capsys.readouterr().err


def test_exclusion_rules(tmp_path):
    rows = _rows(1.0, 0.0)
    by_key = {(r["ticker"], r["arm"], r["seed"]): r for r in rows}
    by_key[("AAPL", "present", 0)].update(status="failed", sharpe="")
    by_key[("AAPL", "present", 1)].update(sharpe="")
    by_key[("AAPL", "absent", 0)].update(n_days=58)
    by_key[("AAPL", "absent", 1)].update(n_decision_errors=4)  # 4/61 > 5%
    by_key[("GOOGL", "absent", 2)].update(n_decision_errors=3)  # 3/61 <= 5%: kept
    for s in range(3):  # PETR4 present: only 2 valid runs left -> ticker dropped
        by_key[("PETR4", "present", s)].update(status="failed")

    result = _run(tmp_path, rows)
    assert result["exclusion_counts"] == {
        "failed": 4, "missing_sharpe": 1, "truncated": 1, "decision_errors": 1, "ticker_dropped": 1,
    }
    per = {r["ticker"]: r for r in result["per_ticker"]}
    assert "PETR4" not in per
    assert (per["AAPL"]["n_absent"], per["AAPL"]["n_present"]) == (3, 3)
    assert per["GOOGL"]["n_absent"] == 5
    assert result["primary"]["br_tickers"] == ["ITUB4", "BPAC11", "VALE3", "WEGE3"]
    assert result["primary"]["n_permutations"] == 35  # C(7, 3)
    assert result["n_valid_runs"] == 76  # 90 rows - 7 row exclusions - PETR4's 7 remaining runs


def test_clear_positive_effect_rejects_h0(tmp_path):
    result = _run(tmp_path, _rows(effect_br=1.0, effect_us=0.0))
    primary = result["primary"]
    assert primary["n_permutations"] == 56  # C(8, 3)
    assert primary["p_value"] == pytest.approx(1 / 56)
    assert primary["reject_h0"] is True
    s1, s2 = result["secondary"]
    assert s1["reject_h0"] is True and s1["p_holm"] < 0.001
    assert s2["reject_h0"] is False
    control = [r for r in result["per_ticker"] if r["role"] == "control"]
    assert [r["ticker"] for r in control] == ["RADL3"]


def test_null_and_uniform_effect_do_not_reject(tmp_path):
    for effect_br, effect_us in ((0.0, 0.0), (1.0, 1.0)):
        result = _run(tmp_path, _rows(effect_br, effect_us, seed=1))
        assert result["primary"]["reject_h0"] is False
        assert result["primary"]["p_value"] > h1.ALPHA


def test_primary_not_evaluable_without_us_tickers(tmp_path):
    rows = [r for r in _rows(1.0, 0.0) if r["market"] == "BR"]
    result = _run(tmp_path, rows)
    assert result["primary"] == {
        "evaluable": False, "reason": "no BR-sensitive or no US ticker left", "reject_h0": False,
    }
    assert result["secondary"][1]["p_value"] is None


def test_holm_adjustment():
    assert h1.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_deterministic_json_output(tmp_path, capsys):
    cells = _write(tmp_path / "cells.csv", _rows(0.3, 0.1, seed=7))
    outs = []
    for name in ("a.json", "b.json"):
        assert h1.main([str(cells), "--out", str(tmp_path / name)]) == 0
        outs.append((tmp_path / name).read_text(encoding="utf-8"))
    assert outs[0] == outs[1]
    assert json.loads(outs[0])["rng_seed"] == h1.RNG_SEED
    printed = capsys.readouterr().out
    assert "PRIMARY D =" in printed and "S1_br_sensitive_delta_gt_0" in printed
