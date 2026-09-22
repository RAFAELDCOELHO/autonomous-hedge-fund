"""P1.7: Hamilton (1989) HMM regimes vs the hand-defined BrazilBench regimes.

Offline and $0 by construction: reads the committed ^BVSP fixture through
``regime_lib.load_prices`` (no download), fits a two-state Gaussian hidden
Markov model on daily log returns with Baum-Welch in plain NumPy (no third-party HMM/stats packages), and writes the alignment between the smoothed
HMM state path and the four hand-dated regime windows.

State labelling is canonical: state 0 = "calm" (lower return variance),
state 1 = "turbulent" (higher variance). Everything is deterministic:
quantile-based initialisation, no random numbers, fixed iteration budget.

Outputs (benchmark/results/hmm_regimes/):
    states.csv       one row per trading day: close, log return, P(turbulent),
                     smoothed state, hand regime label ("" between windows)
    alignment.csv    per hand regime: days, share turbulent, majority state,
                     purity; plus the unlabelled gap days and the pooled row
    breakpoints.csv  each hand boundary vs the nearest HMM switch date
    README.md        metrics + limitations (regenerated, byte-stable)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from regime_lib import REGIME_ORDER, REGIMES, load_prices  # noqa: E402

OUT_DIR = Path(__file__).resolve().parents[1] / "benchmark" / "results" / "hmm_regimes"
N_STATES = 2
MIN_RUN = 5  # trading days; shorter state runs are absorbed into the previous run
STATE_NAME = {0: "calm", 1: "turbulent"}


# --------------------------------------------------------------------------- #
# Gaussian HMM (Baum-Welch) in NumPy
# --------------------------------------------------------------------------- #
def _forward_backward(logb: np.ndarray, log_pi: np.ndarray, log_a: np.ndarray):
    """Forward-backward in log space. Returns (gamma, xi summed over t, loglik)."""
    n, k = logb.shape
    la = np.empty((n, k))
    la[0] = log_pi + logb[0]
    for t in range(1, n):
        la[t] = logb[t] + np.logaddexp.reduce(la[t - 1][:, None] + log_a, axis=0)
    loglik = float(np.logaddexp.reduce(la[-1]))
    lb = np.zeros((n, k))
    for t in range(n - 2, -1, -1):
        lb[t] = np.logaddexp.reduce(log_a + (logb[t + 1] + lb[t + 1])[None, :], axis=1)
    lg = la + lb
    gamma = np.exp(lg - np.logaddexp.reduce(lg, axis=1)[:, None])
    # xi_t(i,j) ∝ alpha_t(i) a_ij b_{t+1}(j) beta_{t+1}(j)
    lxi = la[:-1, :, None] + log_a[None] + (logb[1:] + lb[1:])[:, None, :]
    lxi -= np.logaddexp.reduce(lxi.reshape(n - 1, -1), axis=1)[:, None, None]
    return gamma, np.exp(lxi).sum(axis=0), loglik


def fit_hmm(x: np.ndarray, k: int = N_STATES, iters: int = 500, tol: float = 1e-8) -> dict:
    """Fit a k-state Gaussian HMM to a 1-D series. Deterministic init: split
    observations into k volatility buckets (|x - median| quantiles).
    Returns dict(mu, var, a, pi, gamma, loglik, iters); states sorted by var.
    """
    x = np.asarray(x, dtype=float)
    dev = np.abs(x - np.median(x))
    edges = np.quantile(dev, np.linspace(0, 1, k + 1))
    init = np.clip(np.searchsorted(edges[1:-1], dev, side="right"), 0, k - 1)
    mu = np.array([x[init == s].mean() for s in range(k)])
    var = np.array([x[init == s].var() for s in range(k)])
    a = np.full((k, k), 0.05 / (k - 1)) + np.eye(k) * (0.95 - 0.05 / (k - 1))
    pi = np.full(k, 1.0 / k)
    prev = -np.inf
    for it in range(1, iters + 1):
        logb = -0.5 * (np.log(2 * np.pi * var) + (x[:, None] - mu) ** 2 / var)
        with np.errstate(divide="ignore"):  # log(0) = -inf is fine in logaddexp
            gamma, xi, loglik = _forward_backward(logb, np.log(pi), np.log(a))
        pi = gamma[0]
        a = xi / xi.sum(axis=1, keepdims=True)
        w = gamma.sum(axis=0)
        mu = (gamma * x[:, None]).sum(axis=0) / w
        var = np.maximum((gamma * (x[:, None] - mu) ** 2).sum(axis=0) / w, 1e-12)
        if loglik - prev < tol:
            break
        prev = loglik
    order = np.argsort(var)  # canonical: 0 = calm, 1 = turbulent
    return dict(mu=mu[order], var=var[order], a=a[np.ix_(order, order)], pi=pi[order],
                gamma=gamma[:, order], loglik=loglik, iters=it)


def smooth_states(states: np.ndarray, min_run: int = MIN_RUN) -> np.ndarray:
    """Absorb runs shorter than ``min_run`` into the preceding run."""
    s = states.copy()
    i, n = 0, len(s)
    while i < n:
        j = i
        while j < n and s[j] == s[i]:
            j += 1
        if i > 0 and j - i < min_run:
            s[i:j] = s[i - 1]
        i = j
    return s


# --------------------------------------------------------------------------- #
# Alignment
# --------------------------------------------------------------------------- #
def hand_labels(index: pd.DatetimeIndex) -> pd.Series:
    lab = pd.Series("", index=index, dtype=object)
    for name, (start, end) in REGIMES.items():
        lab.loc[start:end] = name
    return lab


def adjusted_rand(a: np.ndarray, b: np.ndarray) -> float:
    """Adjusted Rand index between two labelings."""
    _, ai = np.unique(a, return_inverse=True)
    _, bi = np.unique(b, return_inverse=True)
    ct = np.zeros((ai.max() + 1, bi.max() + 1))
    np.add.at(ct, (ai, bi), 1)
    comb = lambda v: float((v * (v - 1) / 2).sum())  # noqa: E731
    sum_ij, sa, sb = comb(ct), comb(ct.sum(1)), comb(ct.sum(0))
    expected = sa * sb / comb(np.array([len(a)]))
    max_idx = 0.5 * (sa + sb)
    return (sum_ij - expected) / (max_idx - expected) if max_idx != expected else 1.0


def alignment_table(df: pd.DataFrame) -> pd.DataFrame:
    groups = [(r, df[df.hand_regime == r]) for r in REGIME_ORDER]
    groups += [("(unlabelled gaps)", df[df.hand_regime == ""]), ("(all days)", df)]
    rows = []
    for name, g in groups:
        share = float(g.state.mean())
        rows.append(dict(
            regime=name, days=len(g),
            share_turbulent=round(share, 4),
            majority_state=STATE_NAME[1 if share >= 0.5 else 0],
            purity=round(max(share, 1 - share), 4),
            mean_daily_logret=round(float(g.logret.mean()), 6),
            daily_vol=round(float(g.logret.std(ddof=0)), 6),
        ))
    return pd.DataFrame(rows)


def weighted_purity(align: pd.DataFrame) -> float:
    lab = align[align.regime.isin(REGIME_ORDER)]
    return float((lab.purity * lab.days).sum() / lab.days.sum())


def breakpoint_table(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.index
    switch_pos = np.flatnonzero(np.diff(df.state.to_numpy()) != 0) + 1
    rows = []
    for name, (start, end) in REGIMES.items():
        # Hand dates may fall on non-trading days (e.g. 2019-12-31): snap the
        # start forward and the end backward to the nearest trading day.
        for kind, pos in (("start", int(idx.searchsorted(pd.Timestamp(start)))),
                          ("end", int(idx.searchsorted(pd.Timestamp(end), side="right")) - 1)):
            day = start if kind == "start" else end
            near = switch_pos[np.argmin(np.abs(switch_pos - pos))]
            rows.append(dict(
                regime=name, boundary=kind, hand_date=day,
                nearest_hmm_switch=idx[near].strftime("%Y-%m-%d"),
                switch_to=STATE_NAME[int(df.state.iloc[near])],
                distance_days=int(near - pos),  # trading days; + = HMM switched later
            ))
    return pd.DataFrame(rows)


def write_note(path: Path, fit: dict, align: pd.DataFrame, bps: pd.DataFrame,
               ari: float, n_switches: int) -> None:
    a = fit["a"]
    dur = 1.0 / (1.0 - np.diag(a))
    n_days = int(align.loc[align.regime == "(all days)", "days"].iloc[0])
    wp = weighted_purity(align)
    lines = [
        "# P1.7 — Hamilton HMM regimes vs hand-defined BrazilBench regimes (^BVSP)",
        "",
        "Generated by `make hmm-regimes` (`scripts/hmm_regimes.py`). Offline: committed",
        "`benchmark/prices/paper/IDX_BVSP.csv` fixture, NumPy-only Baum-Welch, no LLM,",
        "no download, no random numbers. Do not edit by hand.",
        "",
        "## Model",
        "",
        f"Two-state Gaussian HMM on daily log returns, {n_days} trading days",
        f"({fit['first']} to {fit['last']}), EM stopped after {fit['iters']} iterations,",
        f"log-likelihood {fit['loglik']:.2f}. States are labelled by variance:",
        "",
        "| state | mean daily logret | daily vol | ann. vol | stay prob | expected duration (days) |",
        "|---|---|---|---|---|---|",
    ]
    for s in range(N_STATES):
        lines.append(f"| {STATE_NAME[s]} | {fit['mu'][s]:+.5f} | {np.sqrt(fit['var'][s]):.5f} "
                     f"| {np.sqrt(fit['var'][s] * 252):.3f} | {a[s, s]:.4f} | {dur[s]:.1f} |")
    lines += [
        "",
        f"Smoothed path (argmax posterior, runs < {MIN_RUN} days absorbed into the",
        f"previous run): {n_switches} state switches over the sample.",
        "",
        "## Alignment with the hand regimes",
        "",
        "The hand regimes are four calendar windows with *four* labels; the HMM has",
        "*two* statistical states. They cannot match one-to-one, so alignment is",
        "measured as (i) how pure each hand window is in HMM-state terms and (ii)",
        "how close HMM switch dates fall to the hand boundaries.",
        "",
        "| hand regime | days | share turbulent | majority HMM state | purity |",
        "|---|---|---|---|---|",
    ]
    for _, r in align.iterrows():
        lines.append(f"| {r.regime} | {int(r.days)} | {r.share_turbulent:.3f} "
                     f"| {r.majority_state} | {r.purity:.3f} |")
    lines += [
        "",
        f"**Key metric: day-weighted purity over the four hand windows = {wp:.3f}.**",
        f"A hand-regime day sits in that window's majority HMM state {wp:.1%} of the",
        "time. Adjusted Rand index between the 5-way hand labelling (four windows +",
        f"gaps) and the 2-way HMM path: {ari:.3f}.",
        "",
        "### Hand boundaries vs nearest HMM switch",
        "",
        "Distance in trading days; positive = the HMM switched after the hand date.",
        "",
        "| regime | boundary | hand date | nearest HMM switch | switches to | distance |",
        "|---|---|---|---|---|---|",
    ]
    for _, r in bps.iterrows():
        lines.append(f"| {r.regime} | {r.boundary} | {r.hand_date} | {r.nearest_hmm_switch} "
                     f"| {r.switch_to} | {r.distance_days:+d} |")
    lines += [
        "",
        "## Reading",
        "",
        "The HMM is a volatility-regime model, so the honest expectation is that",
        "crisis_2020 is dominated by the turbulent state and the other three windows",
        "by the calm state. The tables above are the measurement; the paper should",
        "cite them instead of the earlier \"visual inspection\" remark.",
        "",
        "## Limitations",
        "",
        "- Two Gaussian states on returns only. Hamilton's original switching model",
        "  has autoregressive terms; a 3-state or Student-t variant may split the",
        "  calm state further. Not explored here (kept to the $0, NumPy-only path).",
        "- The smoothed (two-sided) posterior uses the whole sample, so HMM dates are",
        "  ex-post, like the hand dates. This is a regime-*definition* check, not a",
        "  real-time detector.",
        "- The hand regimes encode macro narratives (SELIC, election, vaccine rally)",
        "  that a return-volatility HMM cannot see. Low purity on a calm window is",
        "  not evidence the hand label is wrong, only that volatility alone does not",
        "  separate it.",
        "- No strategy results are re-run under HMM regimes here; this artifact only",
        "  documents label alignment. No LLM results are involved or implied.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    px = load_prices("^BVSP")["Close"].astype(float)
    logret = np.log(px).diff().dropna()
    fit = fit_hmm(logret.to_numpy())
    state = smooth_states(fit["gamma"].argmax(axis=1))
    df = pd.DataFrame({
        "close": px.loc[logret.index].round(2),
        "logret": logret.round(6),
        "p_turbulent": np.round(fit["gamma"][:, 1], 6),
        "state": state,
        "hand_regime": hand_labels(logret.index),
    })
    df.index.name = "date"
    align = alignment_table(df)
    bps = breakpoint_table(df)
    ari = adjusted_rand(df.hand_regime.to_numpy(), df.state.to_numpy())
    n_switches = int((np.diff(state) != 0).sum())
    fit["first"], fit["last"] = df.index[0].date(), df.index[-1].date()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "states.csv", date_format="%Y-%m-%d", lineterminator="\n")
    align.to_csv(OUT_DIR / "alignment.csv", index=False, lineterminator="\n")
    bps.to_csv(OUT_DIR / "breakpoints.csv", index=False, lineterminator="\n")
    write_note(OUT_DIR / "README.md", fit, align, bps, ari, n_switches)

    print(align.to_string(index=False))
    print(f"\nday-weighted purity (4 hand windows): {weighted_purity(align):.3f}   "
          f"ARI: {ari:.3f}   switches: {n_switches}   EM iters: {fit['iters']}   "
          f"loglik: {fit['loglik']:.2f}")
    print(f"wrote {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
