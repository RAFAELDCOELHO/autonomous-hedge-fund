# Headline Arena — dual-agent runbook (P2.1)

Forward-only live leg of the BrazilBench macro-agent hypothesis
([issue #3](https://github.com/RAFAELDCOELHO/autonomous-hedge-fund/issues/3),
Kopei / Headline Arena). Two TradingAgents graphs differ **only** in whether the
Macro Economist analyst is on; they register as **two separate Arena agents**
with **separate credentials** so public scorecards stay separable.

| Arm | `arena_agent` | Analysts |
|---|---|---|
| `macro` | `ahf-tradingagents-macro` | market, social, news, fundamentals, **macro** |
| `no_macro` | `ahf-tradingagents-no-macro` | market, social, news, fundamentals |

Defined in [`scripts/headline_arena_arms.py`](../scripts/headline_arena_arms.py).
Config template: [`config/headline_arena.example.yaml`](../config/headline_arena.example.yaml).

## Cost

| Path | Cost | Needs |
|------|------|--------|
| `make arena-help` / `make arena-dry-run` / list arms | **$0** | No Anthropic key, no Arena secrets, no network for dry-run |
| Live registration + daily forecasts | Arena API is free; **LLM inference** for producing forecasts needs your own model budget | Separate `client_secret` per arm |

CI keeps the `$0` path green: dry-run never calls the network or Anthropic.

## Install the plugin

1. Skill / onboarding guide: https://headlinearena.com/api/v1/agent/onboarding/guide.txt
2. Plugin repo: [headlinearena-agent-plugin](https://github.com/headlinearena/headlinearena-agent-plugin) (ships `scripts/ha.py`)
3. Raw API: https://headlinearena.com/api/docs
4. Operator UI: https://headlinearena.com/agent-onboarding

## Register each arm with SEPARATE credentials

**Do not share one `client_secret` (or one credentials file) across arms.**
Sharing collapses the factorial into a single Arena identity.

For **each** arm (`macro`, then `no_macro`):

1. Register via the plugin or `POST /api/v1/agent/registry/register` using the
   `arena_agent` name from the table (distinct bio/provider strings help).
2. Save `agent_id` and `client_secret` immediately (`client_secret` is shown once).
   Map them to the env names in the example config, e.g.
   `HEADLINE_ARENA_MACRO_AGENT_ID` / `HEADLINE_ARENA_MACRO_CLIENT_SECRET` and the
   `no_macro` pair. Optional second credential files:
   `~/.headlinearena/credentials-macro.json` vs
   `~/.headlinearena/credentials-no-macro.json`.
3. Copy `config/headline_arena.example.yaml` → `config/headline_arena.local.yaml`
   (gitignored). Never commit `*.secrets.yaml` or real secrets.

## Claim OAuth / activate

1. Registration returns a `claim_url`.
2. Open it in a browser and complete claim / OAuth
   (`GET`/`POST /api/v1/agent/claim/{claim_token}`), or poll with
   `ha.py status --wait` / plugin `ha_status`.
3. Repeat for the **second** arm with its own claim URL.
4. Exchange `agent_id` + `client_secret` for a short-lived JWT via
   `POST /api/v1/agent/auth/token` when submitting (plugin handles this).

Sandbox registrations may auto-activate; production requires the claim step.

## Submit forecasts

1. List open challenges: `ha.py challenges` or
   `GET /api/v1/eval/challenges?status=open`.
2. Submit directional forecast + confidence + rationale before the deadline
   (plugin predict path / API docs).
3. Drive both arms through the same code path
   (`scripts/headline_arena_arms.py`) so only the macro analyst differs.
4. Offline check: `make arena-dry-run` →
   `benchmark/results/headline_arena/dry_run.json`.

`scripts/headline_arena_dry_run.py --live` only checks that both credential env
slots are set (exit 2 if missing); it does not call the network. Use the plugin
for live submit.

## Where public scorecards / calibration appear

After forecasts resolve (no secrets required to read):

- Per-agent scorecard: `GET /api/v1/eval/agents/{agent_id}/scorecard`
- Predictions: `GET /api/v1/eval/agents/{agent_id}/predictions`
- Rankings / leaderboard: `GET /api/v1/eval/rankings`, `GET /api/v1/eval/leaderboard`
- Site rankings / calibration: https://headlinearena.com/
- Methodology: https://headlinearena.com/methodology

Compare `ahf-tradingagents-macro` vs `ahf-tradingagents-no-macro` on the same
challenge set — that public delta is the live factorial contrast for issue #3.

## Quick commands

```bash
make arena-help          # setup reminder ($0)
make arena-dry-run       # validate config + write fixture JSON ($0)
uv run python scripts/headline_arena_arms.py   # list arms ($0)
```

## Related

- Scaffold merged in PR #5 (`scripts/headline_arena_arms.py`)
- Do **not** merge this P2.1 PR until Rafa reviews; wiring only
