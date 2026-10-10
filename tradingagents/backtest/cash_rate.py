"""The cash-rate rule shared by every backtest entry point (baselines and agent).

Kept dependency-free (no yfinance, no runner import) so agent_integration and
run_backtest can use it without importing the runner module.
"""

from __future__ import annotations

from typing import Optional

from .risk_free import RF_SOURCE

_REQUIRED = object()


def _cash_rate(market, no_rf: bool) -> Optional[str]:
    """The market whose daily rf the cash earns, or None under an explicit no_rf=True.

    Fail closed: every strategy (baselines and agent) must say which cash rate it
    uses. Neither market nor no_rf=True (market=None counts as not given) raises
    TypeError (no silent 0%); both raises ValueError (ambiguous).
    """
    if no_rf is not True and no_rf is not False:
        raise TypeError(f"no_rf must be True or False, got {no_rf!r}")
    given = market is not _REQUIRED and market is not None
    if given and no_rf:
        raise ValueError(f"ambiguous cash rate: market={market!r} and no_rf=True; pass exactly one")
    if no_rf:
        return None
    if not given:
        raise TypeError("cash rate required: market='BR'/'US' (daily rf) or no_rf=True (explicit 0% cash)")
    if market not in RF_SOURCE:
        raise ValueError(f"unknown market {market!r}; expected one of {sorted(RF_SOURCE)}")
    return market
