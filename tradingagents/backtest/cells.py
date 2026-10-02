"""cells.csv writer for one H1 run (docs/PREREGISTRATION.md §4).

Column order and allowed values match ``scripts/h1_stats.py``. Harness labels
from ``run_backtest.py --arms`` map ``baseline`` → ``absent`` and ``macro`` →
``present``. B3 Yahoo symbols (``PETR4.SA``) are stored without the ``.SA``
suffix. Metric fields come from ``h1_cell_metrics``; market from ``market_of``.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

from .metrics import h1_cell_metrics
from .risk_free import RF_SOURCE, market_of

# Locked to scripts/h1_stats.py COLUMNS / TICKERS by tests/test_cells_csv.py.
COLUMNS = (
    "ticker",
    "market",
    "arm",
    "seed",
    "status",
    "n_days",
    "n_decision_errors",
    "sharpe",
    "rf_source",
)
HARNESS_TO_ARM = {"baseline": "absent", "macro": "present"}
PREREG_TICKERS = {
    "AAPL": "US",
    "GOOGL": "US",
    "AMZN": "US",
    "ITUB4": "BR",
    "BPAC11": "BR",
    "PETR4": "BR",
    "VALE3": "BR",
    "WEGE3": "BR",
    "RADL3": "BR",
}


def bare_ticker(ticker: str) -> str:
    """Pre-registered symbol: uppercase, B3 names without the Yahoo ``.SA`` suffix."""
    name = ticker.strip().upper()
    if name.endswith(".SA"):
        name = name[:-3]
    return name


def prereg_arm(arm: str) -> str:
    """Map a harness label or an already-canonical arm onto absent/present."""
    key = arm.strip().lower()
    if key in HARNESS_TO_ARM:
        return HARNESS_TO_ARM[key]
    if key in ("absent", "present"):
        return key
    raise ValueError(f"arm must be baseline or macro, got {arm!r}")


def _format_sharpe(value: float | None) -> str:
    if value is None:
        return ""
    # repr() round-trips through float() on the Python we run.
    return repr(float(value))


def make_cell_row(
    ticker: str,
    arm: str,
    seed: int,
    equity: pd.Series | None = None,
    status: str = "ok",
) -> dict[str, str]:
    """One cells.csv row. ``status='failed'`` leaves sharpe empty (schema §4)."""
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError(f"seed must be an integer >= 0, got {seed!r}")
    if status not in ("ok", "failed"):
        raise ValueError(f"status must be ok or failed, got {status!r}")

    symbol = ticker.strip()
    bare = bare_ticker(symbol)
    expected = PREREG_TICKERS.get(bare)
    if expected is None:
        raise ValueError(f"ticker {bare!r} is not pre-registered")
    market = market_of(symbol)
    if market != expected:
        raise ValueError(
            f"{symbol} resolves to market {market} via market_of, but {bare} is {expected}. "
            "Pass the Yahoo symbol (B3 tickers need the .SA suffix)."
        )

    if status == "ok":
        if equity is None:
            raise ValueError("ok rows need an equity curve")
        metrics = h1_cell_metrics(equity, market)
        n_days = metrics["n_days"]
        n_errors = metrics["n_decision_errors"]
        sharpe = _format_sharpe(metrics["sharpe"])
        rf_source = metrics["rf_source"]
    else:
        n_days = 0
        n_errors = 0
        sharpe = ""
        rf_source = RF_SOURCE[market]

    return {
        "ticker": bare,
        "market": market,
        "arm": prereg_arm(arm),
        "seed": str(seed),
        "status": status,
        "n_days": str(n_days),
        "n_decision_errors": str(n_errors),
        "sharpe": sharpe,
        "rf_source": rf_source,
    }


def append_cells(path: Path, rows: list[dict[str, str]]) -> None:
    """Append rows, writing the header only when the file is new or empty.

    The header must include COLUMNS. Extra columns already in the file are
    kept (empty on new rows); h1_stats ignores them. A repeated
    (ticker, arm, seed) is refused so the file stays valid.
    """
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: set[tuple[str, str, str]] = set()
    fieldnames = list(COLUMNS)
    write_header = True
    if path.exists() and path.stat().st_size > 0:
        with path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            fieldnames = list(reader.fieldnames or [])
            missing = [column for column in COLUMNS if column not in fieldnames]
            if missing:
                raise ValueError(f"{path} is missing required columns: {missing}")
            for raw in reader:
                existing.add((raw.get("ticker", ""), raw.get("arm", ""), raw.get("seed", "")))
        write_header = False

    encoded: list[dict[str, str]] = []
    for row in rows:
        key = (row["ticker"], row["arm"], row["seed"])
        if key in existing:
            raise ValueError(f"duplicate (ticker, arm, seed) {key}")
        existing.add(key)
        encoded.append({name: row.get(name, "") for name in fieldnames})

    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        if write_header:
            writer.writeheader()
        writer.writerows(encoded)
