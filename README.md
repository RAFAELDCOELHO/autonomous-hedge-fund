# Autonomous Hedge Fund — Multi-Agent LLM Trading System

> Multi-agent trading system combining Kronos (AAAI 2026), FinGPT and TradingAgents, powered by Anthropic Claude.

## Academic References

| Component | Paper |
|-----------|-------|
| TradingAgents | arXiv:2412.20138 |
| Kronos | arXiv:2508.02739 (AAAI 2026) |
| FinGPT | AI4Finance Foundation |
| FINCH | arXiv:2512.13168 |

## Stack

| Layer | Technology |
|-------|-----------|
| LLM | Claude Sonnet 4.6 + Claude Haiku 4.5 |
| Time Series | Kronos-small (45 global exchanges, MPS) |
| Sentiment | FinGPT v3 methodology |
| Data | Yahoo Finance + OpenBB |
| Language | Python 3.12 |

## Quickstart

```bash
git clone https://github.com/RAFAELDCOELHO/autonomous-hedge-fund.git
cd autonomous-hedge-fund
uv sync --python 3.12
cp .env.example .env
uv run python main.py
```

## Baseline Results (AAPL 2023)

| Strategy | CR (%) | Sharpe | MDD (%) |
|----------|--------|--------|---------|
| Buy & Hold | 54.80 | 2.100 | -14.93 |
| MACD(12,26,9) | 36.04 | 2.047 | -6.40 |
| SMA(50/200) | 9.64 | 0.722 | -5.09 |

## Tests

40/40 passing — model validation, indicator fallback, Kronos, FinGPT.

## License

MIT
