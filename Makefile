# BrazilBench one-click path. Offline Close fixtures, frozen B3 splits.
# Strategies: Buy & Hold, MACD(12,26,9), SMA(50/200). No API key. No LLM.
.PHONY: bench

bench:
	@if [ ! -d .venv ]; then uv sync; fi
	uv run python scripts/run_brazilbench.py
