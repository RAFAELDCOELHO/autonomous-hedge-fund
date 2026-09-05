# BrazilBench: frozen B3 Close fixtures, Buy & Hold / MACD / SMA only.
# No API key. No LLM. No TradingAgents graph. Env lock: uv.lock + .python-version.
.PHONY: bench reproduce reliability docker-bench arena-help arena-dry-run

PY = uv run python

# Fast printout: 3 strategies x 6 tickers x 4 regimes.
bench: | .venv
	$(PY) scripts/run_brazilbench.py

# Regenerate the paper artifacts from committed fixtures (offline). A clean
# `git status` afterwards means byte-identical; tests/test_reproduce.py pins it.
reproduce: | .venv
	$(PY) scripts/run_random_n100.py
	$(PY) scripts/run_ew_portfolio.py
	$(PY) scripts/run_brazilbench.py --write

# P1.6: reliability diagram (stated confidence vs next-day win rate) from the
# committed mistral:7b logs + PETR4 fixture. Offline, stdlib only, no LLM call.
reliability: | .venv
	$(PY) scripts/reliability_diagram.py

# `make reproduce` inside a container built from uv.lock. Outputs land in
# ./benchmark/results and ./docs. No .env, no keys, no GPU.
docker-bench:
	docker build -f Dockerfile.bench -t brazilbench .
	docker run --rm -v "$$PWD/benchmark/results:/app/benchmark/results" -v "$$PWD/docs:/app/docs" brazilbench

# Headline Arena: forward-only live arm (issue #3). Printing / dry-run costs $0.
arena-help:
	@echo "Headline Arena setup (two arms: macro vs no_macro; SEPARATE credentials)"
	@echo "  Runbook: docs/HEADLINE_ARENA.md"
	@echo "  1. Install the plugin: https://github.com/headlinearena/headlinearena-agent-plugin"
	@echo "  2. Copy config/headline_arena.example.yaml → config/headline_arena.local.yaml (gitignored)"
	@echo "  3. Register one arena agent per arm with SEPARATE secrets (names in scripts/headline_arena_arms.py):"
	@echo "       $(PY) scripts/headline_arena_arms.py"
	@echo "  4. Claim OAuth via each claim_url; submit forecasts via the plugin"
	@echo "  5. Public scorecards: GET /api/v1/eval/agents/{agent_id}/scorecard — https://headlinearena.com/"
	@echo '  $$0: make arena-dry-run (no network). Live LLM forecasts need ANTHROPIC_API_KEY (.env).'

# Offline fixture: validate dual-arm example config + write dry_run.json ($0, no network).
arena-dry-run: | .venv
	$(PY) scripts/headline_arena_dry_run.py

.venv:
	uv sync
