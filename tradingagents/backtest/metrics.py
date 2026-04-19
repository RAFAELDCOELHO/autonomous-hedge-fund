"""Academic performance metrics for backtests.

All values are returned as fractions (not percentages). Formatting to
percentage is the responsibility of the report layer.

Metrics:
- CR  : Cumulative Return = (V_end - V_start) / V_start
- AR  : Annualized Return = (1 + CR)^(252/n_days) - 1
- Sharpe : sqrt(252) * mean(excess) / std(excess), rf = annual_rf_rate/252
- MDD : min((V_t - cummax(V))/cummax(V))  (negative fraction)
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


class ExtendedMetricsCalculator:
    def __init__(
        self,
        annual_trading_days: int = 252,
        annual_rf_rate: float = 0.0434,
    ) -> None:
        self.annual_trading_days = annual_trading_days
        self.annual_rf_rate = annual_rf_rate

    def compute(self, equity: pd.Series) -> dict:
        """Compute {cr, ar, sharpe, mdd, mdd_date, n_days}.

        equity: pd.Series of portfolio values indexed by date (ascending).
        Returns None-valued metrics when the series is too short.
        """
        empty = {
            "cr": None,
            "ar": None,
            "sharpe": None,
            "mdd": None,
            "mdd_date": None,
            "n_days": 0,
        }
        if equity is None or len(equity) < 2:
            return empty

        equity = equity.dropna()
        if len(equity) < 2:
            return empty

        v_start = float(equity.iloc[0])
        v_end = float(equity.iloc[-1])
        if v_start <= 0:
            return empty

        cr = (v_end - v_start) / v_start
        n_days = len(equity)

        if n_days > 1:
            ar = (1.0 + cr) ** (self.annual_trading_days / n_days) - 1.0
        else:
            ar = None

        returns = equity.pct_change().dropna()
        if len(returns) < 2:
            sharpe = None
        else:
            daily_rf = self.annual_rf_rate / self.annual_trading_days
            excess = returns - daily_rf
            std = float(excess.std())
            if std > 1e-12:
                sharpe = float(np.sqrt(self.annual_trading_days) * excess.mean() / std)
            else:
                sharpe = 0.0

        rolling_max = equity.cummax()
        drawdown = (equity - rolling_max) / rolling_max
        if len(drawdown) > 0:
            mdd = float(drawdown.min())
            mdd_date = drawdown.idxmin() if mdd < 0 else None
            if mdd_date is not None and hasattr(mdd_date, "strftime"):
                mdd_date = mdd_date.strftime("%Y-%m-%d")
        else:
            mdd = 0.0
            mdd_date = None

        return {
            "cr": cr,
            "ar": ar,
            "sharpe": sharpe,
            "mdd": mdd,
            "mdd_date": mdd_date,
            "n_days": n_days,
        }
