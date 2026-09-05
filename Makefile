# BrazilBench: frozen B3 Close fixtures, Buy & Hold / MACD / SMA only.
# No API key. No LLM. No TradingAgents graph.
.PHONY: bench arena-help

bench:
	@if [ ! -d .venv ]; then uv sync; fi
	uv run python scripts/run_brazilbench.py

# Headline Arena: forward-only live arm (issue #3). Printing this costs $0.
arena-help:
	@echo "Headline Arena setup (two arms: macro vs no_macro)"
	@echo "  1. Install the plugin: https://github.com/headlinearena/headlinearena-agent-plugin"
	@echo "  2. Register one arena agent per arm (names in scripts/headline_arena_arms.py):"
	@echo "       python scripts/headline_arena_arms.py"
	@echo "  3. Submit daily forecasts via the plugin; API docs: https://headlinearena.com/api/docs"
	@echo "  Running an arm live needs ANTHROPIC_API_KEY (.env). Listing arms does not."
