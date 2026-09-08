# BrazilBench: frozen B3 Close fixtures, Buy & Hold / MACD / SMA only.
# No API key. No paid LLM (qwen-n10 is local Ollama). No TradingAgents graph. Env lock: uv.lock + .python-version.
# P0.2 audit map (seeds, JSONL schemas, model versions, open-PR gaps):
#   docs/REPRODUCIBILITY.md  —  pinned by tests/test_repro_manifest.py
.PHONY: bench reproduce reliability qwen-n10 hmm-regimes multi-asset survivorship chronos docker-bench arena-help arena-dry-run
.PHONY: docs-smoke

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

# Minimal offline docs smoke: required files exist and README links key
# artifact entry points. No network, no API keys, no third-party deps.
docs-smoke:
	python3 -m unittest -q tests/test_docs_smoke.py

# P1.5: Qwen 2.5-7B via local Ollama, N=10 independent cold-start sessions per
# critical PETR4/crisis_2020 date (server killed between runs), mean +/- std into
# benchmark/results/qwen_n10/. Needs `ollama` + qwen2.5:7b pulled; no key, $0.
qwen-n10: | .venv
	$(PY) scripts/qwen_coldstart_n10.py

# P1.7: Hamilton HMM regimes vs the hand-defined regimes on the committed
# ^BVSP fixture. NumPy-only Baum-Welch, offline, no LLM call.
hmm-regimes: | .venv
	$(PY) scripts/hmm_regimes.py

# P1.8: multi-asset portfolios (EW / inverse-vol / long-only min-variance) that
# use the daily correlation structure of the paper-five fixtures. Offline.
multi-asset: | .venv
	$(PY) scripts/multi_asset_corr.py

# P1.9: survivorship bracket. Classical baselines on distressed OIBR3/MGLU3/
# AMER3 (GOLL4 unavailable on Yahoo) vs the liquid paper-five. Offline fixtures.
survivorship: | .venv
	$(PY) scripts/survivorship_distress.py

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

# P1.10: Chronos-t5-tiny comparator on paper fixtures (optional extra).
# Downloads HF weights once on first run; writes benchmark/results/chronos/.
chronos: | .venv
	uv run --extra chronos python scripts/chronos_comparator.py

.venv:
	uv sync
