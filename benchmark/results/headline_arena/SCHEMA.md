# Headline Arena dry-run fixture schema

Produced by `make arena-dry-run` / `scripts/headline_arena_dry_run.py`.

- `mode`: `"dry_run"`
- `network`: always `false` on this path
- `issue`: link to GitHub issue #3
- `arms.macro` / `arms.no_macro`: each has `arena_agent`, `analysts`,
  `credential_slots` (env **names** + credentials_file paths only),
  `forecast` (synthetic, `status: dry_run_not_submitted`), and `scorecard` stub

Not a live Headline Arena submission. See `docs/HEADLINE_ARENA.md`.
