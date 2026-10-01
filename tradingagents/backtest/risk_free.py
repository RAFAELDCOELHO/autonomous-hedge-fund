"""Daily risk-free series for the H1 runs (docs/PREREGISTRATION.md §3).

Reads only the committed snapshots in data/rf/, verified against
data/rf/SHA256SUMS before every use. No network access.

- BR: BCB SGS 12 CDI (% per business day). rf_t compounds the CDI of every
  CDI business day in [t-1, t), t-1 being the previous trading day.
- US: FRED DTB3 (% p.a., discount basis). rf_t = (1 + DTB3/100)^(1/252) - 1
  using the last DTB3 published on or before t-1, the previous trading day
  of the run's calendar (same t-1 as BR).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

RF_DIR = Path(__file__).resolve().parents[2] / "data" / "rf"
RF_FILES = {
    "BR": "bcb_sgs_12_cdi_daily_2023-12-01_2024-04-30.csv",
    "US": "fred_dtb3_2023-12-01_2024-04-30.csv",
}
RF_SOURCE = {"BR": "BCB-SGS-12", "US": "FRED-DTB3"}
ONE_DAY = pd.Timedelta(days=1)


def market_of(ticker: str) -> str:
    return "BR" if ticker.upper().endswith(".SA") else "US"


def verify_snapshots(rf_dir: Path = RF_DIR) -> None:
    """Raise ValueError unless every file listed in SHA256SUMS matches its checksum."""
    rf_dir = Path(rf_dir)
    expected = {}
    for line in (rf_dir / "SHA256SUMS").read_text().splitlines():
        if line.strip():
            digest, name = line.split()
            expected[name.lstrip("*")] = digest
    missing = set(RF_FILES.values()) - set(expected)
    if missing:
        raise ValueError(f"rf snapshots not listed in SHA256SUMS: {sorted(missing)}")
    for name, digest in expected.items():
        actual = hashlib.sha256((rf_dir / name).read_bytes()).hexdigest()
        if actual != digest:
            raise ValueError(f"SHA-256 mismatch for {rf_dir / name}: {actual} != {digest}")


def load_series(market: str, rf_dir: Path = RF_DIR) -> pd.Series:
    """Raw snapshot series in its source units (CDI % a.d. / DTB3 % p.a.), NaNs dropped."""
    verify_snapshots(rf_dir)
    path = Path(rf_dir) / RF_FILES[market]
    if market == "BR":
        df = pd.read_csv(path, sep=";", decimal=",")
        index = pd.to_datetime(df["data"], format="%d/%m/%Y")
        values = df["valor"]
    else:
        df = pd.read_csv(path)
        index = pd.to_datetime(df["observation_date"], format="%Y-%m-%d")
        values = pd.to_numeric(df["DTB3"], errors="coerce")
    return pd.Series(values.to_numpy(float), index=index).dropna().sort_index()


def daily_rf(market: str, dates: pd.DatetimeIndex, rf_dir: Path = RF_DIR) -> pd.Series:
    """rf for each return in `dates` (trading days), indexed by dates[1:]."""
    series = load_series(market, rf_dir)
    dates = pd.DatetimeIndex(dates)
    if len(dates) and (dates[0] < series.index[0] or dates[-1] - ONE_DAY > series.index[-1]):
        raise ValueError(
            f"{RF_SOURCE[market]} snapshot covers {series.index[0].date()}..{series.index[-1].date()}, "
            f"not {dates[0].date()}..{dates[-1].date()}"
        )
    if market == "BR":
        growth = 1.0 + series / 100.0
        rates = [growth[(growth.index >= prev) & (growth.index < t)].prod() - 1.0
                 for prev, t in zip(dates[:-1], dates[1:])]
    else:
        rates = (1.0 + series.asof(dates[:-1]).to_numpy() / 100.0) ** (1 / 252) - 1.0
    return pd.Series(rates, index=dates[1:], dtype=float, name="rf")
