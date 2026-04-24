"""Integration between TradingAgentsGraph and the backtest runner.

This module provides the bridge that lets TradingAgents (with Macro Economist
Agent) be evaluated through the same harness as classical baselines
(BuyAndHold, MACD, SMACrossover).

Key functions:
- map_signal(raw): normalize 5-class SignalProcessor output into the 3-class
  BUY/HOLD/SELL that run_agent_strategy expects.
- make_decide_fn(config, propagate_fn): factory that returns a decide_fn
  callable suitable for run_agent_strategy.
- run_tradingagents_backtest(...): high-level wrapper that stitches
  TradingAgentsGraph construction, decide_fn creation, and the runner call.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import pandas as pd

from tradingagents.graph.trading_graph import TradingAgentsGraph

from .runner import run_agent_strategy


_SIGNAL_MAP: Dict[str, str] = {
    "BUY": "BUY",
    "OVERWEIGHT": "BUY",
    "HOLD": "HOLD",
    "UNDERWEIGHT": "SELL",
    "SELL": "SELL",
}


def map_signal(raw: Optional[str]) -> str:
    """Normalize a SignalProcessor output to BUY/HOLD/SELL.

    SignalProcessor returns one of:
        BUY, OVERWEIGHT, HOLD, UNDERWEIGHT, SELL

    run_agent_strategy only accepts:
        BUY, HOLD, SELL

    Mapping:
        BUY, OVERWEIGHT       -> BUY
        HOLD                  -> HOLD
        SELL, UNDERWEIGHT     -> SELL
        (anything else / None -> HOLD, defensive fallback)
    """
    if raw is None:
        return "HOLD"
    cleaned = raw.strip().upper()
    return _SIGNAL_MAP.get(cleaned, "HOLD")


def make_decide_fn(
    ticker: str,
    config: Dict[str, Any],
    propagate_fn: Optional[Callable[[str, str], tuple]] = None,
    debug: bool = False,
) -> Callable[[str, pd.DataFrame], str]:
    """Build a decide_fn compatible with run_agent_strategy.

    The returned function has the signature:
        decide_fn(curr_date_str, prices_up_to_date) -> "BUY"|"HOLD"|"SELL"

    The prices_up_to_date argument is accepted (to honor run_agent_strategy's
    look-ahead prevention contract) but not used — TradingAgentsGraph fetches
    its own data internally, keyed by date. Look-ahead safety is preserved
    because propagate is called with curr_date, so agents only query data up
    to that date.

    Args:
        ticker: Symbol passed to propagate each day (e.g. "AAPL", "PETR4.SA").
        config: Config dict forwarded to TradingAgentsGraph.
        propagate_fn: Optional injection point for testing. If provided,
            this callable is used instead of constructing a real
            TradingAgentsGraph. Signature: (ticker, date_str) -> (state, signal).
        debug: Forwarded as the graph's debug flag when propagate_fn is None.

    Returns:
        A decide_fn closure suitable for run_agent_strategy.
    """
    if propagate_fn is None:
        ta = TradingAgentsGraph(debug=debug, config=config)
        _propagate = ta.propagate
    else:
        _propagate = propagate_fn

    def decide_fn(curr_date: str, prices_up_to_date: pd.DataFrame) -> str:
        _, raw_signal = _propagate(ticker, curr_date)
        return map_signal(raw_signal)

    return decide_fn


def run_tradingagents_backtest(
    ticker: str,
    start: str,
    end: str,
    config: Dict[str, Any],
    initial_capital: float = 100_000.0,
    propagate_fn: Optional[Callable[[str, str], tuple]] = None,
    debug: bool = False,
) -> pd.Series:
    """Run a day-by-day backtest of TradingAgents over [start, end].

    This is the main entry point for evaluating TradingAgents (with or
    without the Macro Economist agent, depending on config) through the
    same harness used for the baseline strategies.

    Args:
        ticker: Symbol to backtest (e.g. "AAPL" or "PETR4.SA").
        start, end: Date range, "YYYY-MM-DD" inclusive.
        config: Config dict for TradingAgentsGraph. Control whether the
            Macro Economist is active via config's selected_analysts
            (pass through whatever graph.setup expects).
        initial_capital: Starting portfolio value in USD (or BRL,
            depending on ticker).
        propagate_fn: Optional mock for testing without API calls.
        debug: Forwarded to TradingAgentsGraph when propagate_fn is None.

    Returns:
        pd.Series of daily equity values indexed by date. Feed this
        directly to ExtendedMetricsCalculator.compute().
    """
    decide_fn = make_decide_fn(
        ticker=ticker,
        config=config,
        propagate_fn=propagate_fn,
        debug=debug,
    )
    return run_agent_strategy(
        decide_fn=decide_fn,
        ticker=ticker,
        start=start,
        end=end,
        initial_capital=initial_capital,
    )
