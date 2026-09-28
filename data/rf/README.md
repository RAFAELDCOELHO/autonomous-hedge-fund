# Risk-free series snapshots for H1 preregistration

Download date: 2026-09-28

These files are immutable snapshots committed for reproducibility of the preregistered H1 setup in `docs/PREREGISTRATION.md`.

## Files

- `bcb_sgs_12_cdi_daily_2023-12-01_2024-04-30.csv`
  - Source: Banco Central do Brasil (BCB), SGS series 12 (CDI daily)
  - URL: `https://api.bcb.gov.br/dados/serie/bcdata.sgs.12/dados?formato=csv&dataInicial=01/12/2023&dataFinal=30/04/2024`
  - Interval in file: 2023-12-01 to 2024-04-30
- `fred_dtb3_2023-12-01_2024-04-30.csv`
  - Source: Federal Reserve Economic Data (FRED), series DTB3
  - URL: `https://fred.stlouisfed.org/graph/fredgraph.csv?id=DTB3&cosd=2023-12-01&coed=2024-04-30`
  - Interval in file: 2023-12-01 to 2024-04-30
- `SHA256SUMS`
  - SHA-256 checksums for the two CSV snapshots above.

## Coverage rationale

The experiment window in the preregistration is Jan–Mar 2024 (proposed endpoints 2024-01-02 to 2024-03-28). The snapshot interval includes extra padding before and after this window to cover t-1 lookback alignment and calendar mismatches.
