#!/usr/bin/env python3
"""CLI for `make bench`: offline BrazilBench baselines, no LLM."""

from tradingagents.backtest.brazilbench import main

if __name__ == "__main__":
    raise SystemExit(main())
