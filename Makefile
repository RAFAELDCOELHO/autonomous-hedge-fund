# BrazilBench: frozen B3 Close fixtures, Buy & Hold / MACD / SMA only.
# No API key. No LLM. No TradingAgents graph.
.PHONY: bench

bench:
	@if [ ! -d .venv ]; then uv sync; fi
	uv run python scripts/run_brazilbench.py
