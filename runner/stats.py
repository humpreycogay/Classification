"""Statistical comparisons for the report: multiple-comparison correction, paired differences
(Tango score intervals), paired bootstrap intervals, bootstrap rank intervals and the leading tier,
and the Pareto frontier.

All apps are scored on the same articles, so comparisons are paired: the bootstrap draws one set
of resampled articles per replicate and scores every app on that same set. The bootstrap is
deterministic (fixed seed) and is computed with per-article resampling weights, so every metric
is O(replicates × articles) per app.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

ALPHA = 0.05
Z = 1.959963984540054  # two-sided 95% normal quantile
BOOTSTRAP_SEED = 20260927
BOOTSTRAP_REPLICATES = 2000
ECE_BINS = 10


# ---------------------------------------------------------------- tests and paired differences

def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for k successes out of n."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return center - half, center + half



def mcnemar_p(b: int, c: int) -> float:
    """Exact two-sided McNemar test on the discordant pairs (b, c)."""
    m = b + c
    if m == 0:
        return 1.0
    tail = sum(math.comb(m, k) for k in range(min(b, c) + 1)) / 2**m
    return min(1.0, 2 * tail)


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm step-down adjusted p-values (family-wise error control, no independence assumption)."""
    order = sorted(pvalues, key=lambda k: pvalues[k])
    m, running, adjusted = len(order), 0.0, {}
    for i, key in enumerate(order):
        running = max(running, min(1.0, (m - i) * pvalues[key]))
        adjusted[key] = running
    return adjusted


def tango_restricted_p21(only_a: int, only_b: int, n: int, d: float) -> float:
    """Maximum-likelihood P(only A right) given the difference d = P(only B right) - P(only A right).

    Solves 2n p^2 + W p - only_a d (1 - d) = 0 with W = -only_b - only_a + (2n - only_b + only_a) d
    (Tango 1998), from maximizing the likelihood of the paired table under that constraint.
    """
    w = -only_b - only_a + (2 * n - only_b + only_a) * d
    return (math.sqrt(max(w * w + 8 * n * only_a * d * (1 - d), 0.0)) - w) / (4 * n)


def tango_statistic(only_a: int, only_b: int, n: int, d: float) -> float:
    """Score statistic for H0: accuracy difference = d; decreasing in d."""
    var = n * (2 * tango_restricted_p21(only_a, only_b, n, d) + d * (1 - d))
    diff = only_b - only_a - n * d
    if var <= 0:
        return math.copysign(math.inf, diff) if diff else 0.0
    return diff / math.sqrt(var)


def tango_interval(only_a: int, only_b: int, n: int, z: float = Z) -> tuple[float, float]:
    """Tango (1998) score interval for a difference of paired proportions.

    Unlike the Wald interval it stays valid with few discordant pairs: two apps that always agree get
    an interval around 0 whose width reflects n, not an interval of zero width.
    """
    d_hat = (only_b - only_a) / n

    def solve(target: float, lo: float, hi: float) -> float:
        for _ in range(200):  # bisection on the decreasing statistic
            mid = (lo + hi) / 2
            if tango_statistic(only_a, only_b, n, mid) > target:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    return solve(z, -1.0, d_hat), solve(-z, d_hat, 1.0)


def paired_difference(a_ok: np.ndarray, b_ok: np.ndarray) -> dict:
    """Accuracy of B minus accuracy of A on the same articles, with a Tango score 95% interval."""
    a_ok, b_ok = np.asarray(a_ok, bool), np.asarray(b_ok, bool)
    n = len(a_ok)
    only_a, only_b = int((a_ok & ~b_ok).sum()), int((~a_ok & b_ok).sum())
    low, high = tango_interval(only_a, only_b, n)
    return {"diff": (only_b - only_a) / n, "low": low, "high": high, "only_a": only_a, "only_b": only_b,
            "p": mcnemar_p(only_a, only_b)}


# ---------------------------------------------------------------- paired bootstrap

def resample_weights(n: int, replicates: int = BOOTSTRAP_REPLICATES, seed: int = BOOTSTRAP_SEED) -> np.ndarray:
    """(replicates × n) counts: how often each article appears in each bootstrap resample."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(replicates, n))
    flat = (idx + np.arange(replicates)[:, None] * n).ravel()
    return np.bincount(flat, minlength=replicates * n).reshape(replicates, n).astype(np.float64)


def weighted_auroc(W: np.ndarray, conf: np.ndarray, correct: np.ndarray) -> np.ndarray:
    """AUROC of confidence for correct vs wrong under each row of weights (ties count half)."""
    order = np.argsort(conf, kind="stable")
    c, pos = conf[order], correct[order].astype(float)
    Wp, Wn = W[:, order] * pos, W[:, order] * (1 - pos)
    cum_neg = np.cumsum(Wn, axis=1)
    _, start, counts = np.unique(c, return_index=True, return_counts=True)
    group = np.repeat(np.arange(len(start)), counts)
    end = start + counts - 1
    below = np.where(start[group] > 0, cum_neg[:, np.maximum(start[group] - 1, 0)], 0.0)
    equal = cum_neg[:, end[group]] - below
    numerator = (Wp * (below + 0.5 * equal)).sum(axis=1)
    denominator = Wp.sum(axis=1) * Wn.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denominator > 0, numerator / denominator, np.nan)


def bootstrap_metrics(df: pd.DataFrame, W: np.ndarray, labels: list[str]) -> dict[str, np.ndarray]:
    """Per-replicate accuracy, macro F1, AUROC, ECE and Brier for one app's predictions."""
    n = len(df)
    label, pred = df["label"].to_numpy(), df["pred"].to_numpy()
    correct = label == pred
    out = {"accuracy": W @ correct / n}

    f1s = []
    for lab in labels:
        tp = W @ ((pred == lab) & (label == lab))
        predicted, actual = W @ (pred == lab), W @ (label == lab)
        with np.errstate(invalid="ignore", divide="ignore"):
            precision = np.where(predicted > 0, tp / predicted, 0.0)
            recall = np.where(actual > 0, tp / actual, 0.0)
            f1s.append(np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0))
    out["macro_f1"] = np.mean(f1s, axis=0)

    conf = df["confidence"].astype(float).to_numpy()
    out["auroc"] = weighted_auroc(W, conf, correct)
    bins = np.minimum((np.clip(conf, 0, 1) * ECE_BINS).astype(int), ECE_BINS - 1)
    onehot = np.eye(ECE_BINS)[bins]
    out["ece"] = np.abs(W @ (onehot * correct[:, None]) - W @ (onehot * conf[:, None])).sum(axis=1) / n

    prob_cols = [f"p_{lab}" for lab in labels]
    if all(c in df.columns for c in prob_cols):
        probs = df[prob_cols].astype(float).clip(lower=0).to_numpy()
        sums = probs.sum(axis=1, keepdims=True)
        probs = probs / np.where(sums == 0, 1, sums)
        truth = (label[:, None] == np.array(labels)[None, :]).astype(float)
        out["brier"] = W @ ((probs - truth) ** 2).sum(axis=1) / n
    return out


def percentile_ci(samples: np.ndarray) -> tuple[float, float]:
    s = samples[~np.isnan(samples)]
    if len(s) == 0:
        return float("nan"), float("nan")
    return float(np.quantile(s, ALPHA / 2)), float(np.quantile(s, 1 - ALPHA / 2))


# ---------------------------------------------------------------- ranking presentation

def rank_intervals(accuracy: dict[str, np.ndarray]) -> dict[str, dict]:
    """Per app, from paired bootstrap accuracies (same resamples for every app): the 95% interval of its
    rank (1 = best; ties share the better rank) and the probability of being best (ties split evenly)."""
    names = list(accuracy)
    acc = np.stack([accuracy[n] for n in names])  # apps x replicates
    ranks = 1 + (acc[None, :, :] > acc[:, None, :]).sum(axis=1)  # [i, b] = 1 + apps strictly better than i
    top = ranks == 1
    share = np.where(top, 1.0 / top.sum(axis=0, keepdims=True), 0.0)
    return {n: {"rank_low": int(np.percentile(ranks[i], 100 * ALPHA / 2, method="lower")),
                "rank_high": int(np.percentile(ranks[i], 100 * (1 - ALPHA / 2), method="higher")),
                "p_best": float(share[i].mean())} for i, n in enumerate(names)}


def leading_tier(ranks: dict[str, dict]) -> list[str]:
    """Apps whose percentile bootstrap rank interval includes rank 1 (the bootstrap leading tier).

    An uncertainty summary under this resampling rule, not a test that the apps are equivalent or statistically tied.
    """
    return [n for n, r in ranks.items() if r["rank_low"] == 1]


def pareto_front(points: dict[str, tuple[float, float]]) -> list[str]:
    """Apps not dominated on (accuracy: higher is better, cost: lower is better).

    A dominates B when A is at least as accurate and at least as cheap, and strictly better on one.
    Returned in order of increasing cost.
    """
    front = []
    for name, (acc, cost) in points.items():
        dominated = any(a >= acc and c <= cost and (a > acc or c < cost)
                        for other, (a, c) in points.items() if other != name)
        if not dominated:
            front.append(name)
    return sorted(front, key=lambda k: (points[k][1], -points[k][0]))
