# BrazilBench: frozen B3 Close fixtures, Buy & Hold / MACD / SMA only.
# No API key. No LLM. No TradingAgents graph. Env lock: uv.lock + .python-version.
.PHONY: bench reproduce docker-bench

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

# `make reproduce` inside a container built from uv.lock. Outputs land in
# ./benchmark/results and ./docs. No .env, no keys, no GPU.
docker-bench:
	docker build -f Dockerfile.bench -t brazilbench .
	docker run --rm -v "$$PWD/benchmark/results:/app/benchmark/results" -v "$$PWD/docs:/app/docs" brazilbench

.venv:
	uv sync
