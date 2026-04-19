"""CLI for running the academic backtest comparison.

Compares Buy & Hold, MACD(12/26/9), SMA(50/200), and (optionally) the
TradingAgents pipeline over a single-ticker window. Prints a rich table
of CR / AR / Sharpe / MDD.

Usage:
    uv run python run_backtest.py --ticker AAPL --start 2023-01-01 --end 2024-01-01
    uv run python run_backtest.py --ticker AAPL --start 2023-01-01 --end 2024-01-01 --skip-agents
"""

from __future__ import annotations

import argparse
import logging
import sys

from tradingagents.backtest import (
    BuyAndHold,
    MACDStrategy,
    SMACrossStrategy,
    print_comparison,
    run_strategy,
    run_agent_strategy,
)


def _run_agent_decider(ticker: str, start: str, end: str, capital: float):
    """Run TradingAgents once per trading day and return an equity curve.

    Falls back to None if the pipeline cannot be constructed.
    """
    try:
        from tradingagents.graph.trading_graph import TradingAgentsGraph
    except Exception as e:
        logging.warning("TradingAgents pipeline unavailable (%s)", e)
        return None

    graph = TradingAgentsGraph()

    def decide(curr_date: str, _prices):
        try:
            _, signal = graph.propagate(ticker, curr_date)
            action = (signal or "HOLD").upper()
            if action not in {"BUY", "SELL", "HOLD"}:
                action = "HOLD"
            return action
        except Exception as e:
            logging.warning("Agent decision failed on %s: %s", curr_date, e)
            return "HOLD"

    return run_agent_strategy(decide, ticker, start, end, capital)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run academic backtest comparison.")
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    parser.add_argument("--capital", type=float, default=100_000.0)
    parser.add_argument("--skip-agents", action="store_true",
                        help="Do not run the TradingAgents pipeline (baselines only)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    curves = {}
    for strat in (BuyAndHold(), MACDStrategy(), SMACrossStrategy()):
        curves[strat.name] = run_strategy(strat, args.ticker, args.start, args.end, args.capital)

    if not args.skip_agents:
        agent_curve = _run_agent_decider(args.ticker, args.start, args.end, args.capital)
        if agent_curve is not None:
            curves["TradingAgents"] = agent_curve

    print_comparison(curves)
    return 0


if __name__ == "__main__":
    sys.exit(main())
