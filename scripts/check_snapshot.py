#!/usr/bin/env python3
"""Check an offline price snapshot with ``validate_snapshot``.

The snapshot is a directory of per-ticker CSVs, one file ``{ticker}.csv``
for each pre-registered Yahoo symbol (``AAPL.csv``, ``PETR4.SA.csv``, ...).
Each CSV has the columns the vendor loader leaves for ``validate_snapshot``:
a ``Date`` column (or a ``Date`` index) and ``Open`` / ``Close``. Rows are
passed through unchanged; this script does not fill prices, drop bad closes,
or download anything.

``validate_snapshot`` applies the runner's completeness rules, including the
D-1 Close rule. One stdout line per defect::

    <field code> <ticker> <YYYY-MM-DD>

Field codes are the ones ``validate_snapshot`` reports (``bar``, ``Open``,
``Close``, ``duplicate``). Exit 0 when the snapshot is complete, 1 when any
defect is listed, 2 on usage or I/O error.

Usage::

    python scripts/check_snapshot.py SNAPSHOT_DIR
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from tradingagents.backtest.snapshot import (
    SnapshotIncomplete,
    prereg_tickers,
    validate_snapshot,
)


def load_snapshot(path: Path) -> dict[str, pd.DataFrame]:
    """Load ``{ticker}.csv`` frames for the pre-registered grid.

    A ticker with no file is omitted. ``validate_snapshot`` then lists every
    required session of that ticker as a missing bar.
    """
    prices: dict[str, pd.DataFrame] = {}
    for ticker in prereg_tickers():
        csv_path = path / f"{ticker}.csv"
        if not csv_path.is_file():
            continue
        frame = pd.read_csv(csv_path, encoding="utf-8")
        if "Date" not in frame.columns:
            raise ValueError(f"missing Date column in {csv_path}")
        prices[ticker] = frame
    return prices


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run validate_snapshot on a directory of per-ticker OHLCV CSVs."
    )
    parser.add_argument(
        "snapshot",
        type=Path,
        help="directory of {ticker}.csv files (Date, Open, Close; Yahoo names, B3 with .SA)",
    )
    args = parser.parse_args(argv)
    path: Path = args.snapshot
    if not path.exists():
        print(f"error: {path}: no such file or directory", file=sys.stderr)
        return 2
    if not path.is_dir():
        print(f"error: {path}: not a snapshot directory", file=sys.stderr)
        return 2
    try:
        prices = load_snapshot(path)
        validate_snapshot(prices_by_ticker=prices)
    except SnapshotIncomplete as exc:
        for ticker, day, field in exc.missing:
            print(f"{field} {ticker} {day}")
        return 1
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
