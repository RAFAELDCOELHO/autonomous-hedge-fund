# BrazilBench: frozen B3 Close fixtures, Buy & Hold / MACD / SMA only.
# No API key. No paid LLM (qwen-n10 is local Ollama). No TradingAgents graph. Env lock: uv.lock + .python-version.
.PHONY: bench reproduce reliability qwen-n10 hmm-regimes survivorship docker-bench arena-help

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

# P1.5: Qwen 2.5-7B via local Ollama, N=10 independent cold-start sessions per
# critical PETR4/crisis_2020 date (server killed between runs), mean +/- std into
# benchmark/results/qwen_n10/. Needs `ollama` + qwen2.5:7b pulled; no key, $0.
qwen-n10: | .venv
	$(PY) scripts/qwen_coldstart_n10.py

# P1.7: Hamilton HMM regimes vs the hand-defined regimes on the committed
# ^BVSP fixture. NumPy-only Baum-Welch, offline, no LLM call.
hmm-regimes: | .venv
	$(PY) scripts/hmm_regimes.py

# P1.9: survivorship bracket. Classical baselines on distressed OIBR3/MGLU3/
# AMER3 (GOLL4 unavailable on Yahoo) vs the liquid paper-five. Offline fixtures.
survivorship: | .venv
	$(PY) scripts/survivorship_distress.py

# `make reproduce` inside a container built from uv.lock. Outputs land in
# ./benchmark/results and ./docs. No .env, no keys, no GPU.
docker-bench:
	docker build -f Dockerfile.bench -t brazilbench .
	docker run --rm -v "$$PWD/benchmark/results:/app/benchmark/results" -v "$$PWD/docs:/app/docs" brazilbench

# Headline Arena: forward-only live arm (issue #3). Printing this costs $0.
arena-help:
	@echo "Headline Arena setup (two arms: macro vs no_macro)"
	@echo "  1. Install the plugin: https://github.com/headlinearena/headlinearena-agent-plugin"
	@echo "  2. Register one arena agent per arm (names in scripts/headline_arena_arms.py):"
	@echo "       python scripts/headline_arena_arms.py"
	@echo "  3. Submit daily forecasts via the plugin; API docs: https://headlinearena.com/api/docs"
	@echo "  Running an arm live needs ANTHROPIC_API_KEY (.env). Listing arms does not."

.venv:
	uv sync
