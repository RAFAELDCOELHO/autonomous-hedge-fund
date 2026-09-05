#!/usr/bin/env python3
"""P1.8 - Multi-asset portfolios with correlation structure ($0, offline).

Replaces the "left for future work" gap: Appendix EW is the mean of per-ticker
terminal returns, which ignores how the five assets co-move day to day. This
script builds a daily portfolio from the aligned daily-return matrix and uses
the covariance / correlation structure explicitly.

Data: the committed paper-five Close fixtures (benchmark/prices/paper/) via
regime_lib.load_prices; dates inner-joined across tickers. No download, no key.

Rules (fixed, documented here so the numbers are interpretable)
    * Signals: the same long-only binary signals as regime_lib / the paper
      (BuyAndHold, MACD, SMA, Momentum, Random policy #seed). s_i(t) = 1 means
      "hold asset i into the close of day t", so it earns r_i(t+1).
    * Weights: a fixed target vector w over the five assets per (regime, method),
      rebalanced daily (constant-mix). Asset i contributes w_i * s_i(t-1) * r_i(t);
      when the agent is flat on an asset that slice sits in cash at 0%.
      So Buy & Hold is the pure weighting comparison and every other agent
      only changes *which* slices are in cash.
    * equal_weight: w = 1/5.
    * inv_vol:      w ∝ 1/σ_i, σ from the WARMUP_DAYS aligned daily returns
                    strictly before the regime start (no look-ahead).
    * min_var:      long-only minimum-variance weights from the warmup covariance
                    (sum w = 1, w >= 0), solved exactly by enumerating supports
                    (N=5 -> 31 subsets) with numpy only.
    * Metrics: total return %, annualised vol (sqrt(252) * daily std, ddof=1),
      Sharpe = mean/std * sqrt(252) with rf = 0 (SELIC is not subtracted; the
      Sharpe is therefore a raw return/vol ratio, comparable across methods only).
    * Random: mean over the N=100 policies (seed 0..99) of each metric.
    * ew_of_crs_pct: Appendix tab:ew-baselines number (mean of per-asset CRs),
      recomputed here as a cross-check; it is NOT a daily-rebalanced portfolio.

Outputs (benchmark/results/multi_asset/)
    corr_by_regime.csv       Pearson corr of realised in-regime daily returns
    weights.csv              target weights per (regime, method, ticker)
    portfolio_summary.csv    agent x regime x method metrics
    README.md                key deltas + limitations

Usage
    make multi-asset            (= uv run python scripts/multi_asset_corr.py)
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import regime_lib as rl  # noqa: E402
from tradingagents.backtest.baselines import (  # noqa: E402
    BuyAndHold, MACDStrategy, MomentumStrategy, SMACrossStrategy,
)

OUT_DIR = rl.REPO / "benchmark" / "results" / "multi_asset"
CORR_CSV = OUT_DIR / "corr_by_regime.csv"
WEIGHTS_CSV = OUT_DIR / "weights.csv"
SUMMARY_CSV = OUT_DIR / "portfolio_summary.csv"
README_MD = OUT_DIR / "README.md"

WARMUP_DAYS = 252
TRADING_DAYS = 252
METHODS = ["equal_weight", "inv_vol", "min_var"]
DETERMINISTIC = {
    "buy_and_hold": BuyAndHold(),
    "macd": MACDStrategy(),
    "sma_crossover": SMACrossStrategy(),
    "momentum": MomentumStrategy(),
}
AGENT_ORDER = [*DETERMINISTIC, "random"]


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_raw() -> dict[str, pd.DataFrame]:
    return {t: rl.load_prices(t) for t in rl.TICKERS}


def aligned_closes(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Date x ticker Close frame, inner-joined across the paper-five fixtures.
    (Only 2018-01-25 is dropped, inside the warmup; regime windows are identical.)"""
    return pd.concat({t: df["Close"] for t, df in raw.items()}, axis=1, join="inner")


def regime_returns(closes: pd.DataFrame, regime: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(warmup, window) simple daily returns. `window` starts at the regime's
    second bar (the first bar's return belongs to the day before the window,
    exactly as the single-asset simulation which starts at bar 0's close).
    `warmup` is the WARMUP_DAYS returns strictly before the regime start."""
    start, end = rl.REGIMES[regime]
    rets = closes.pct_change().dropna()
    warmup = rets[rets.index < pd.Timestamp(start)].tail(WARMUP_DAYS)
    return warmup, rets.loc[start:end].iloc[1:]


# --------------------------------------------------------------------------- #
# Signals: (window dates) x ticker boolean "hold into this close"
# --------------------------------------------------------------------------- #
def strategy_signals(raw: dict[str, pd.DataFrame], regime: str, strat) -> pd.DataFrame:
    """Same per-ticker warmed signals as regime_lib.run_strategy_cell (paper-exact)."""
    start, end = rl.REGIMES[regime]
    return pd.DataFrame({t: strat.signals(df.loc[:end]).loc[start:end] for t, df in raw.items()})


def random_signals(index: pd.DatetimeIndex, seed: int) -> pd.DataFrame:
    """Policy #seed (regime_lib.random_actions) as a hold-state signal; the same
    action stream drives all five tickers, as in Task 1 (one policy per seed)."""
    held, state = [], False
    for a in rl.random_actions(len(index), seed):
        state = True if a == "BUY" else False if a == "SELL" else state
        held.append(state)
    return pd.DataFrame({t: held for t in rl.TICKERS}, index=index)


# --------------------------------------------------------------------------- #
# Weights (warmup only)
# --------------------------------------------------------------------------- #
def min_var_long_only(cov: np.ndarray) -> np.ndarray:
    """Exact long-only minimum-variance weights: the optimum of a convex QP is
    the equality-constrained solution on its support, so enumerate supports."""
    n = len(cov)
    cov = cov + np.eye(n) * 1e-10 * np.trace(cov) / n  # ridge against singularity
    best, best_var = None, np.inf
    for k in range(1, n + 1):
        for sup in itertools.combinations(range(n), k):
            idx = list(sup)
            w = np.linalg.solve(cov[np.ix_(idx, idx)], np.ones(k))
            w /= w.sum()
            if (w < -1e-12).any():
                continue
            var = w @ cov[np.ix_(idx, idx)] @ w
            if var < best_var:
                best_var, best = var, np.zeros(n)
                best[idx] = np.clip(w, 0, None)
    return best / best.sum()


def weights_for(warmup: pd.DataFrame) -> dict[str, np.ndarray]:
    n = warmup.shape[1]
    sigma = warmup.std(ddof=1).to_numpy()
    inv = 1.0 / sigma
    return {
        "equal_weight": np.full(n, 1.0 / n),
        "inv_vol": inv / inv.sum(),
        "min_var": min_var_long_only(warmup.cov().to_numpy()),
    }


# --------------------------------------------------------------------------- #
# Portfolio metrics
# --------------------------------------------------------------------------- #
def exposure(window: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
    """0/1 frame: asset i earns r_i(t) iff it was held into the close of t-1."""
    return signals.astype(float).shift(1).reindex(window.index).fillna(0.0)


def portfolio_returns(window: pd.DataFrame, signals: pd.DataFrame, w: np.ndarray) -> pd.Series:
    """Daily constant-mix portfolio return: sum_i w_i * s_i(t-1) * r_i(t)."""
    return (exposure(window, signals) * window * w).sum(axis=1)


def metrics(rp: pd.Series) -> dict[str, float]:
    std = float(rp.std(ddof=1))
    return {
        "total_return_pct": float((1.0 + rp).prod() - 1.0) * 100.0,
        "ann_vol_pct": std * np.sqrt(TRADING_DAYS) * 100.0,
        "sharpe": float(rp.mean()) / std * np.sqrt(TRADING_DAYS) if std > 0 else float("nan"),
    }


def ew_of_crs(window: pd.DataFrame, signals: pd.DataFrame) -> float:
    """Appendix number: mean over assets of the gated compounded return (%)."""
    return float(((1.0 + exposure(window, signals) * window).prod() - 1.0).mean()) * 100.0


def compute() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = load_raw()
    closes = aligned_closes(raw)
    corr_rows, weight_rows, rows = [], [], []
    for regime in rl.REGIME_ORDER:
        warmup, window = regime_returns(closes, regime)
        corr = window.corr()
        for a, b in itertools.combinations_with_replacement(rl.TICKERS, 2):
            corr_rows.append({"regime": regime, "ticker_a": a, "ticker_b": b,
                              "corr": round(float(corr.loc[a, b]), 4)})
        weights = weights_for(warmup)
        for m, w in weights.items():
            for t, wi in zip(rl.TICKERS, w):
                weight_rows.append({"regime": regime, "method": m, "ticker": t,
                                    "weight": round(float(wi), 4)})

        sig_sets: dict[str, list[pd.DataFrame]] = {
            a: [strategy_signals(raw, regime, s)] for a, s in DETERMINISTIC.items()
        }
        rand_index = closes.loc[slice(*rl.REGIMES[regime])].index
        sig_sets["random"] = [random_signals(rand_index, s) for s in range(rl.N_SEEDS)]

        for agent in AGENT_ORDER:
            sigs = sig_sets[agent]
            naive = np.mean([ew_of_crs(window, s) for s in sigs])
            for m, w in weights.items():
                ms = pd.DataFrame([metrics(portfolio_returns(window, s, w)) for s in sigs]).mean()
                rows.append({
                    "agent": agent, "regime": regime, "method": m,
                    "n_days": len(window), "n_policies": len(sigs),
                    "total_return_pct": round(float(ms["total_return_pct"]), 4),
                    "ann_vol_pct": round(float(ms["ann_vol_pct"]), 4),
                    "sharpe": round(float(ms["sharpe"]), 4),
                    "ew_of_crs_pct": round(float(naive), 4),
                })
    return pd.DataFrame(corr_rows), pd.DataFrame(weight_rows), pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def _md_table(df: pd.DataFrame, fmt: str = "{:.2f}") -> str:
    idx_name = " / ".join(str(n) for n in (df.index.names if df.index.nlevels > 1 else [df.index.name or ""]))
    head = "| " + " | ".join([idx_name, *map(str, df.columns)]) + " |"
    sep = "|" + "---|" * (len(df.columns) + 1)
    body = []
    for i, r in zip(df.index, df.to_numpy()):
        label = " / ".join(map(str, i)) if isinstance(i, tuple) else str(i)
        cells = [fmt.format(v) if isinstance(v, float) else str(v) for v in r]
        body.append("| " + " | ".join([label, *cells]) + " |")
    return "\n".join([head, sep, *body])


def readme(corr: pd.DataFrame, weights: pd.DataFrame, summary: pd.DataFrame) -> str:
    off = corr[corr.ticker_a != corr.ticker_b].groupby("regime")["corr"].mean().reindex(rl.REGIME_ORDER)
    wpiv = weights.pivot_table(index=["regime", "method"], columns="ticker", values="weight")
    wpiv = wpiv.reindex(index=pd.MultiIndex.from_product([rl.REGIME_ORDER, METHODS]), columns=rl.TICKERS)
    pairs = pd.MultiIndex.from_product([AGENT_ORDER, rl.REGIME_ORDER])
    ret = summary.pivot_table(index=["agent", "regime"], columns="method", values="total_return_pct")
    ret = ret.reindex(index=pairs, columns=METHODS)
    ret["ew_of_crs(appendix)"] = summary.drop_duplicates(["agent", "regime"]).set_index(
        ["agent", "regime"])["ew_of_crs_pct"].reindex(ret.index)
    ret["min_var-ew"] = ret["min_var"] - ret["equal_weight"]
    ret["inv_vol-ew"] = ret["inv_vol"] - ret["equal_weight"]
    sh = summary.pivot_table(index=["agent", "regime"], columns="method", values="sharpe")
    sh = sh.reindex(index=pairs, columns=METHODS)
    gap = ret[["min_var-ew", "inv_vol-ew"]].abs().stack()
    ga, gr, gm = gap.idxmax()
    signed = ret.loc[(ga, gr), gm]

    parts = [
        "# P1.8 Multi-asset portfolios with correlation structure",
        "",
        "Generated by `scripts/multi_asset_corr.py` (`make multi-asset`) from the committed",
        "paper-five fixtures. Offline, numpy/pandas only, no LLM, no download.",
        "",
        "## What this adds over the Appendix EW table",
        "",
        "The Appendix `tab:ew-baselines` number is the *mean of five terminal returns*: it never",
        "sees how the assets co-move. Here the portfolio is built on the aligned daily-return",
        "matrix with fixed target weights rebalanced daily, and two of the three weightings are",
        "derived from the warmup covariance (252 aligned trading days strictly before each",
        "regime start, so no look-ahead). Column `ew_of_crs(appendix)` is that Appendix number",
        "recomputed here as a cross-check.",
        "",
        "## Mean pairwise correlation of in-regime daily returns",
        "",
        _md_table(off.to_frame("mean_offdiag_corr"), "{:.3f}"),
        "",
        "Full matrices: `corr_by_regime.csv`.",
        "",
        "## Target weights (from warmup only)",
        "",
        _md_table(wpiv, "{:.3f}"),
        "",
        "## Total return % by agent x regime x method",
        "",
        _md_table(ret),
        "",
        "## Sharpe (rf=0, annualised sqrt(252)) by method",
        "",
        _md_table(sh),
        "",
        "## Biggest EW vs correlation-aware gap",
        "",
        f"`{ga}` in `{gr}`: `{gm}` = {signed:+.2f} pp of total return.",
        "",
        "## Limitations",
        "",
        "- Rule-based agents only. No LLM agent is run here (that would cost money and is not",
        "  part of the $0 reproduce path); nothing here is a claim about the LLM pipeline.",
        "- Five assets, four of which are constituents of the fifth (^BVSP), so correlations are",
        "  structurally high and diversification benefit is small by construction.",
        "- Constant-mix with daily rebalancing, frictionless, no slippage, no borrowing; flat",
        "  slices earn 0% (no SELIC). Sharpe uses rf=0 and is only comparable across methods.",
        "- Weights are static per regime (estimated once on warmup). A rolling re-estimate is a",
        "  one-line change but adds a tuning knob the paper does not need.",
        "- Warmup covariance from (up to) 252 days on 5 assets is noisy (bull_2019 has only the",
        "  2018 history, ~247 returns); min_var concentrates weight in",
        "  the lowest-variance asset (typically ^BVSP), which is expected behaviour, not a finding.",
        "- Random rows average N=100 seeded policies; the same action stream drives all five",
        "  tickers (one policy per seed), matching `run_random_n100.py`.",
    ]
    return "\n".join(parts) + "\n"


def main() -> int:
    corr, weights, summary = compute()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    corr.to_csv(CORR_CSV, index=False)
    weights.to_csv(WEIGHTS_CSV, index=False)
    summary.to_csv(SUMMARY_CSV, index=False)
    text = readme(corr, weights, summary)
    README_MD.write_text(text)
    print(text)
    print(f"wrote {CORR_CSV}, {WEIGHTS_CSV}, {SUMMARY_CSV}, {README_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
