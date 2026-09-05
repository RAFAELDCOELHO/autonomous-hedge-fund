# Headline Arena — dual-agent runbook (P2.1)

Forward-only live leg of the BrazilBench macro hypothesis ([issue #3](https://github.com/RAFAELDCOELHO/autonomous-hedge-fund/issues/3)).
Two TradingAgents graphs differ **only** in whether the Macro Economist analyst is on; they register as **two separate Arena agents** so public scorecards stay separable ([Kopei](https://github.com/RAFAELDCOELHO/autonomous-hedge-fund/issues/3#issuecomment-5552449154)).

| Arm | `arena_agent` | Analysts |
|---|---|---|
| `macro` | `ahf-tradingagents-macro` | market, social, news, fundamentals, **macro** |
| `no_macro` | `ahf-tradingagents-no-macro` | market, social, news, fundamentals |

Defined in `scripts/headline_arena_arms.py`. Config template: `config/headline_arena.example.yaml`.

## $0 dry-run (CI-safe)

```bash
make arena-dry-run
# → benchmark/results/headline_arena/dry_run.json
```

No network, no secrets, no Anthropic API. Validates distinct agent names and credential *slots*.

## Live path (needs credentials + plugin)

1. Install the [headlinearena agent plugin](https://github.com/headlinearena/headlinearena-agent-plugin) (ships `scripts/ha.py`).
2. **Register each arm separately** (two `agent_id` / `client_secret` pairs). Do not reuse one credential file for both arms.
3. Copy `config/headline_arena.example.yaml` → `config/headline_arena.local.yaml` (gitignored) and point env vars / credential files at each pair.
4. Complete OAuth claim for each agent (`ha.py status --wait` or plugin `ha_status` poll).
5. Discover challenges (`ha.py challenges`) and submit forecasts per arm before lock.
6. Public scorecards and calibration curves appear on [headlinearena.com](https://headlinearena.com) after settlement — no login.

Live runs may require an LLM API for the TradingAgents graph (`--build`); that is **out of the $0 path**. Anthropic API remains forbidden in gabinete Max policy unless Rafa says otherwise — prefer local/Codex for graph builds when testing.

## Secrets

Never commit `client_secret`, access tokens, or `config/headline_arena.local.yaml`.
Placeholders and env **names** only in the example file.

## Related

- Plugin API docs: https://headlinearena.com/api/docs
- `make arena-help` — short checklist
- Scaffold merged in PR #5
