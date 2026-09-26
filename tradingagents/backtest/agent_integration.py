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

import re
from typing import Any, Callable, Dict, Optional

import pandas as pd

from .runner import run_agent_strategy


_SIGNAL_MAP: Dict[str, str] = {
    "BUY": "BUY",
    "OVERWEIGHT": "BUY",
    "HOLD": "HOLD",
    "UNDERWEIGHT": "SELL",
    "SELL": "SELL",
}

_TEXT_SIGNAL_PATTERN = re.compile(
    r"\b(?P<buy>OVER\s+WEIGHT|OVERWEIGHT|BUY(?:ING)?)\b"
    r"|\b(?P<hold>HOLD(?:ING)?)\b"
    r"|\b(?P<sell>UNDER\s+WEIGHT|UNDERWEIGHT|SELL(?:ING)?)\b"
)
_NEGATION_PATTERN = re.compile(
    r"\b(?:DO\s+NOT|DON'T|DONT|NOT|NO|NEVER|AVOID|AVOIDING)\b"
    r"(?:\s+[A-Z']+){0,2}\s*$"
)


def _is_negated_label(text: str, label_start: int) -> bool:
    """Return True when a short negation appears right before a label."""
    # Keep the check local to the current clause, then inspect only a short
    # window of words before the label to stay lightweight and predictable.
    clause_start = max(text.rfind(ch, 0, label_start) for ch in ".!?;\n")
    local_prefix = text[clause_start + 1 : label_start]
    return _NEGATION_PATTERN.search(local_prefix) is not None


def map_signal(raw: Optional[str]) -> str:
    """Normalize a raw LLM/SignalProcessor output to BUY/HOLD/SELL.

    SignalProcessor should return one of:
        BUY, OVERWEIGHT, HOLD, UNDERWEIGHT, SELL.
    In practice, LLM output can be verbose markdown/text such as
    "**BUY**" or "Rating: OVERWEIGHT.". We defensively extract the first
    valid label and map it to the 3-class action space.

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
    if not cleaned:
        return "HOLD"

    mapped = _SIGNAL_MAP.get(cleaned)
    if mapped is not None:
        return mapped

    normalized_text = cleaned.replace("_", " ").replace("-", " ")
    for match in _TEXT_SIGNAL_PATTERN.finditer(normalized_text):
        if _is_negated_label(normalized_text, match.start()):
            continue
        if match.lastgroup == "buy":
            return "BUY"
        if match.lastgroup == "sell":
            return "SELL"
        return "HOLD"
    return "HOLD"


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
        from tradingagents.graph.trading_graph import TradingAgentsGraph

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
