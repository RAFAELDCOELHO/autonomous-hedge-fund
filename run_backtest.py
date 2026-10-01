"""CLI for running the academic backtest comparison.

Compares Buy & Hold, MACD(12/26/9), SMA(50/200), and TradingAgents arms
over a single-ticker window. TradingAgents runs two harness labels:
`baseline` (mapped from `no_macro` in scripts/headline_arena_arms.py) and
`macro`. Use `--arms` to choose which arms to run (default: baseline,macro)
or `--skip-agents` to run only classical baselines. Prints a rich table
of CR / AR / Sharpe / MDD. The TradingAgents arms' cash earns the market's
daily rf (CDI for .SA tickers, DTB3 otherwise) from data/rf/, so the agent
window must lie inside 2023-12-01..2024-04-30 (daily_rf raises otherwise).
print_comparison still shows Sharpe at the flat exploratory rf.

Usage:
    uv run python run_backtest.py --ticker AAPL --start 2024-01-02 --end 2024-03-28
    uv run python run_backtest.py --ticker AAPL --start 2023-01-01 --end 2024-01-01 --skip-agents
    uv run python run_backtest.py --ticker AAPL --start 2024-01-02 --end 2024-03-28 --arms macro
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from runpy import run_path
import sys

from dotenv import load_dotenv

from tradingagents.backtest import (
    BuyAndHold,
    MACDStrategy,
    SMACrossStrategy,
    print_comparison,
    run_strategy,
    run_agent_strategy,
)
from tradingagents.backtest.agent_integration import make_decide_fn
from tradingagents.backtest.risk_free import market_of
from tradingagents.default_config import DEFAULT_CONFIG


def _load_headline_arena_arms() -> dict[str, dict[str, object]]:
    script = Path(__file__).resolve().parent / "scripts" / "headline_arena_arms.py"
    return run_path(str(script))["ARMS"]


def _selected_analysts_by_arm() -> dict[str, list[str]]:
    arms_data = _load_headline_arena_arms()
    # no_macro (arms file) == baseline (harness label) == macro absent.
    baseline_arm = arms_data.get("no_macro")
    macro_arm = arms_data.get("macro")
    arms: dict[str, list[str]] = {}
    if baseline_arm and "selected_analysts" in baseline_arm:
        arms["baseline"] = list(baseline_arm["selected_analysts"])
    if macro_arm and "selected_analysts" in macro_arm:
        arms["macro"] = list(macro_arm["selected_analysts"])
    return arms


def _run_agent_decider(
    ticker: str,
    start: str,
    end: str,
    capital: float,
    selected_analysts: list[str] | None = None,
):
    """Run TradingAgents once per trading day and return an equity curve.

    Falls back to None if the pipeline cannot be constructed.
    """
    config = DEFAULT_CONFIG.copy()
    if selected_analysts is not None:
        config["selected_analysts"] = list(selected_analysts)

    try:
        decide_fn = make_decide_fn(ticker=ticker, config=config)
    except Exception as e:
        logging.warning("TradingAgents pipeline unavailable (%s)", e)
        return None

    return run_agent_strategy(decide_fn, ticker, start, end, capital, market=market_of(ticker))


def _parse_arms_csv(value: str) -> list[str]:
    allowed = {"baseline", "macro"}
    parsed = [token.strip().lower() for token in value.split(",") if token.strip()]
    if not parsed:
        raise ValueError("--arms must include at least one of: baseline,macro")
    invalid = [token for token in parsed if token not in allowed]
    if invalid:
        raise ValueError(
            f"--arms received invalid arm(s): {','.join(invalid)}. Allowed: baseline,macro"
        )
    deduped: list[str] = []
    for token in parsed:
        if token not in deduped:
            deduped.append(token)
    return deduped


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run academic backtest comparison.")
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    parser.add_argument("--capital", type=float, default=100_000.0)
    parser.add_argument("--skip-agents", action="store_true",
                        help="Do not run the TradingAgents pipeline (baselines only)")
    parser.add_argument(
        "--arms",
        default="baseline,macro",
        help="Comma-separated TradingAgents arms to run: baseline,macro (default: baseline,macro)",
    )
    args = parser.parse_args(argv)
    try:
        requested_arms = _parse_arms_csv(args.arms)
    except ValueError as e:
        parser.error(str(e))

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if not args.skip_agents and requested_arms:
        load_dotenv()
        if not os.getenv("ANTHROPIC_API_KEY"):
            logging.error(
                "ANTHROPIC_API_KEY is required when running agents; set it or use --skip-agents."
            )
            return 2

    curves = {}
    for strat in (BuyAndHold(), MACDStrategy(), SMACrossStrategy()):
        curves[strat.name] = run_strategy(strat, args.ticker, args.start, args.end, args.capital)

    if not args.skip_agents:
        arms = _selected_analysts_by_arm()
        if not arms:
            logging.warning("No selected_analysts arms configured; skipping TradingAgents run")
        for arm_name in requested_arms:
            selected_analysts = arms.get(arm_name)
            if selected_analysts is None:
                logging.warning(
                    "Requested arm '%s' is unavailable in harness config; skipping", arm_name
                )
                continue
            agent_curve = _run_agent_decider(
                args.ticker,
                args.start,
                args.end,
                args.capital,
                selected_analysts=selected_analysts,
            )
            if agent_curve is not None:
                curves[f"TradingAgents ({arm_name})"] = agent_curve

    print_comparison(curves)
    return 0


if __name__ == "__main__":
    sys.exit(main())
