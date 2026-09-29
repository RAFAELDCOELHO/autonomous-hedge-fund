# Risk-free series snapshots for H1 preregistration

Download date: 2026-09-28

These files are immutable snapshots committed for reproducibility of the preregistered H1 setup in `docs/PREREGISTRATION.md`.

## Files

- `bcb_sgs_12_cdi_daily_2023-12-01_2024-04-30.csv`
  - Source: Banco Central do Brasil (BCB), SGS series 12 (CDI daily)
  - URL: `https://api.bcb.gov.br/dados/serie/bcdata.sgs.12/dados?formato=csv&dataInicial=01/12/2023&dataFinal=30/04/2024`
  - Units: percent per business day (% a.d.; taxa diária)
  - Snapshot sanity check: first values are `0,045513`, `0,045513`, ... (daily percent points), consistent with BCB SGS 12 daily CDI quoting
  - Interval in file: 2023-12-01 to 2024-04-30
- `fred_dtb3_2023-12-01_2024-04-30.csv`
  - Source: Federal Reserve Economic Data (FRED), series DTB3
  - URL: `https://fred.stlouisfed.org/graph/fredgraph.csv?id=DTB3&cosd=2023-12-01&coed=2024-04-30`
  - Units: percent per annum (% p.a.), discount basis (3-month T-bill secondary market rate)
  - Snapshot sanity check: values are around `5.2x` (e.g., `5.23`, `5.26`), consistent with annualized DTB3 levels
  - Interval in file: 2023-12-01 to 2024-04-30
- `SHA256SUMS`
  - SHA-256 checksums for the two CSV snapshots above.

## Conversion used in preregistration

For US rows in the preregistration, DTB3 is converted to a daily rate as:

`rf_t = (1 + DTB3_{t-1}/100)^(1/252) - 1`

using the last DTB3 value published on or before `t-1` (forward-filled across non-publication days). For BR rows, SGS 12 CDI is already daily (% a.d.) and is used as the daily cash benchmark.

## Coverage rationale

The experiment window in the preregistration is Jan–Mar 2024 (decided endpoints 2024-01-02 to 2024-03-28). The snapshot interval includes extra padding before and after this window to cover t-1 lookback alignment and calendar mismatches.
