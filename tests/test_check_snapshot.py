"""CLI wrapper around validate_snapshot (scripts/check_snapshot.py). Offline."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from tradingagents.backtest.snapshot import FIELD_BAR, FIELD_CLOSE, prereg_tickers

REPO = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "check_snapshot", REPO / "scripts" / "check_snapshot.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


check_snapshot = _load()


def _frame() -> pd.DataFrame:
    idx = pd.bdate_range("2023-12-01", "2024-04-30", name="Date")
    return pd.DataFrame({"Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 101.0, "Volume": 1.0}, index=idx)


def _snapshot() -> dict[str, pd.DataFrame]:
    return {ticker: _frame() for ticker in prereg_tickers()}


def _write(directory: Path, prices: dict[str, pd.DataFrame]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    for ticker, frame in prices.items():
        out = frame.reset_index() if "Date" not in frame.columns else frame
        out.to_csv(directory / f"{ticker}.csv", index=False)
    return directory


def test_clean_snapshot_exits_0(tmp_path, capsys):
    path = _write(tmp_path / "snap", _snapshot())
    assert check_snapshot.main([str(path)]) == 0
    assert capsys.readouterr().out == ""


def test_field_bar_defect_exits_1_and_prints_code(tmp_path, capsys):
    prices = _snapshot()
    prices["AAPL"] = prices["AAPL"].drop(pd.Timestamp("2024-01-10"))
    path = _write(tmp_path / "snap", prices)
    assert check_snapshot.main([str(path)]) == 1
    assert capsys.readouterr().out == f"{FIELD_BAR} AAPL 2024-01-10\n"


def test_field_close_defect_exits_1_and_prints_code(tmp_path, capsys):
    prices = _snapshot()
    prices["AAPL"].loc[pd.Timestamp("2024-01-10"), "Close"] = np.nan
    path = _write(tmp_path / "snap", prices)
    assert check_snapshot.main([str(path)]) == 1
    assert capsys.readouterr().out == f"{FIELD_CLOSE} AAPL 2024-01-10\n"


def test_missing_path_exits_2(tmp_path, capsys):
    missing = tmp_path / "missing-snapshot"
    assert check_snapshot.main([str(missing)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "missing-snapshot" in captured.err
