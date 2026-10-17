"""Extra edge coverage for cells.csv row helpers."""

from __future__ import annotations

import csv

import pandas as pd

from tradingagents.backtest.cells import COLUMNS, SHARPE_FLAT_FIELD, append_cells, make_cell_row


def _equity_without_attrs() -> pd.Series:
    # Keep attrs empty to cover the fallback path in make_cell_row.
    return pd.Series(
        [100.0, 101.0, 100.5, 102.0],
        index=pd.DatetimeIndex(["2024-01-12", "2024-01-16", "2024-01-17", "2024-01-18"]),
    )


def _raw_rows(path):
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_make_cell_row_ok_works_without_optional_equity_attrs():
    row = make_cell_row("AAPL", "baseline", 0, _equity_without_attrs())
    assert row["status"] == "ok"
    # With no attrs["n_days"], writer falls back to the metrics count.
    assert row["n_days"] == "4"
    assert row["n_decision_errors"] == "0"
    assert row["rf_source"] == "FRED-DTB3"
    assert row["sharpe"] != ""
    assert row[SHARPE_FLAT_FIELD] != ""


def test_failed_row_ignores_incompatible_curve_attrs():
    curve = _equity_without_attrs()
    curve.attrs["rf_source"] = "none"
    curve.attrs["n_days"] = 999
    row = make_cell_row("AAPL", "macro", 3, equity=curve, status="failed")
    assert row["arm"] == "present"
    assert row["status"] == "failed"
    assert row["n_days"] == "0"
    assert row["n_decision_errors"] == "0"
    assert row["sharpe"] == ""
    assert row[SHARPE_FLAT_FIELD] == ""
    assert row["rf_source"] == "FRED-DTB3"


def test_append_cells_preserves_existing_column_order(tmp_path):
    path = tmp_path / "cells.csv"
    custom_order = [
        "status",
        "seed",
        "ticker",
        "arm",
        "market",
        "n_days",
        "n_decision_errors",
        "rf_source",
        "sharpe",
        SHARPE_FLAT_FIELD,
    ]
    assert set(custom_order) == set(COLUMNS) | {SHARPE_FLAT_FIELD}
    path.write_text(",".join(custom_order) + "\n", encoding="utf-8")

    append_cells(path, [make_cell_row("AAPL", "macro", 7, status="failed")])

    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == ",".join(custom_order)
    rows = _raw_rows(path)
    assert len(rows) == 1
    assert list(rows[0].keys()) == custom_order
    assert rows[0]["ticker"] == "AAPL"
    assert rows[0]["arm"] == "present"
    assert rows[0]["seed"] == "7"
    assert rows[0]["status"] == "failed"
