"""Selective prediction and calibration metrics, all tie-aware.

Selective prediction accepts the most confident predictions and routes the rest to review. A real
threshold ("accept when confidence >= t") can only accept or reject a group of equal confidences as a
whole, and sorting inside such a group is arbitrary (several apps give 60-80% of articles the same
confidence). So:

- achievable operating points are the boundaries between distinct confidence values;
- a fixed coverage that falls inside a tied group uses the expected accuracy under random order
  within the group (and is flagged as not reachable by a threshold);
- AURC averages the expected risk at every coverage, so it does not depend on row order.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from stats import wilson

TARGETS = [0.90, 0.95, 0.99]
COVERAGES = [0.25, 0.50, 0.75, 0.90]
ECE_BINS = 10
MCE_MIN_BIN = 10  # bins with fewer predictions are left out of MCE (too noisy)
SATURATED = 0.999  # max p at or above this counts as a one-hot / saturated prediction
ONE_HOT_SHARE = 0.95  # share of saturated predictions that makes an app "one-hot"
SATURATED_SHARE = 0.50  # share that earns a "saturated" warning
MIN_N_FOR_99 = 1000  # the 99% target is only reported from the common tier up


@dataclass(frozen=True)
class Groups:
    """Predictions grouped by equal confidence, most confident first."""
    value: np.ndarray  # confidence of each group
    size: np.ndarray  # predictions in each group
    correct: np.ndarray  # correct predictions in each group
    n: int

    @property
    def accepted(self) -> np.ndarray:  # predictions accepted with threshold = value[g]
        return np.cumsum(self.size)

    @property
    def accepted_correct(self) -> np.ndarray:
        return np.cumsum(self.correct)


def groups(conf, correct) -> Groups:
    conf = np.round(np.asarray(conf, dtype=float), 12)
    correct = np.asarray(correct, dtype=bool)
    values, inverse = np.unique(-conf, return_inverse=True)  # ascending in -conf = descending conf
    size = np.bincount(inverse, minlength=len(values))
    right = np.bincount(inverse, weights=correct, minlength=len(values))
    return Groups(-values, size, right, len(conf))


def expected_correct(g: Groups, m: np.ndarray) -> np.ndarray:
    """Expected correct predictions among the m most confident, ties broken at random."""
    m = np.asarray(m, dtype=float)
    ends = g.accepted
    idx = np.searchsorted(ends, m, side="left")  # group containing position m
    idx = np.minimum(idx, len(ends) - 1)
    before = np.where(idx > 0, ends[idx - 1], 0)
    correct_before = np.where(idx > 0, g.accepted_correct[idx - 1], 0)
    return correct_before + (m - before) * g.correct[idx] / g.size[idx]


def curve(g: Groups) -> list[tuple[float, float]]:
    """(coverage, expected accuracy of the most confident share) for every coverage k/n."""
    m = np.arange(1, g.n + 1)
    acc = expected_correct(g, m) / m
    return [(float(k / g.n), float(a)) for k, a in zip(m, acc)]


def aurc(g: Groups) -> tuple[float, float]:
    """(AURC, E-AURC): mean selective risk over coverages 1/n..1, and its excess over the ideal
    ordering (every correct prediction ranked above every wrong one). Lower is better for both."""
    m = np.arange(1, g.n + 1)
    risk = 1 - expected_correct(g, m) / m
    k = int(g.correct.sum())
    ideal = np.where(m <= k, 0.0, (m - k) / m)
    value = float(risk.mean())
    return value, value - float(ideal.mean())


def accuracy_at_coverage(g: Groups, coverage: float) -> dict:
    """Expected accuracy of the most confident `coverage` share; reachable = a threshold gives exactly it."""
    m = max(1, round(coverage * g.n))
    acc = float(expected_correct(g, np.array([m]))[0] / m)
    return {"coverage": coverage, "accepted": m, "accuracy": acc, "reachable": bool(m in set(g.accepted.tolist()))}


def coverage_at_target(g: Groups, target: float, certified: bool = False) -> dict:
    """Largest coverage over achievable thresholds with accepted accuracy >= target.

    certified: require the Wilson 95% lower bound (not the point estimate) to reach the target. The threshold is still
    selected on this sample, so the bound is not a guarantee for future data (the key name is kept for stored metrics).
    The curve is not monotone, so every threshold is checked, not only the first crossing.
    """
    best = {"target": target, "certified": certified, "coverage": 0.0, "threshold": None, "accepted": 0,
            "accuracy": float("nan")}
    for value, n_acc, k_acc in zip(g.value, g.accepted, g.accepted_correct):
        score = wilson(int(k_acc), int(n_acc))[0] if certified else k_acc / n_acc
        if score >= target - 1e-12 and n_acc / g.n > best["coverage"]:
            best.update(coverage=float(n_acc / g.n), threshold=float(value), accepted=int(n_acc),
                        accuracy=float(k_acc / n_acc))
    return best


# ---------------------------------------------------------------- held-out thresholds
#
# coverage_at_target picks the threshold on the items it then scores, which overstates what a threshold fixed in
# advance achieves. The functions below separate the two steps: a threshold is fitted on one set of items (a
# calibration sample, or the other fold in cross-fitting) and only then applied to the evaluation items.

CROSS_FIT_FOLDS = 2
CROSS_FIT_SEED = 0


def fit_threshold(conf, correct, target: float, certified: bool = True) -> float | None:
    """The threshold coverage_at_target selects on these items (None when no threshold reaches the target)."""
    return coverage_at_target(groups(conf, correct), target, certified)["threshold"]


def apply_threshold(conf, correct, threshold: float | None) -> dict:
    """Coverage and accuracy of a fixed threshold ("accept when confidence >= threshold") on new items."""
    conf = np.round(np.asarray(conf, dtype=float), 12)
    correct = np.asarray(correct, dtype=bool)
    accepted = conf >= threshold if threshold is not None else np.zeros(len(conf), dtype=bool)
    k, m = int(correct[accepted].sum()), int(accepted.sum())
    return {"threshold": threshold, "n": len(conf), "accepted": m, "correct": k,
            "coverage": m / len(conf) if len(conf) else float("nan"),
            "accuracy": k / m if m else float("nan"), "ci": wilson(k, m) if m else (float("nan"), float("nan"))}


def _summarize(target: float, certified: bool, method: str, n: int, accepted: int, k: int, **extra) -> dict:
    acc = k / accepted if accepted else float("nan")
    return {"target": target, "certified": certified, "method": method, "n": n, "accepted": accepted,
            "coverage": accepted / n if n else float("nan"), "accuracy": acc,
            "ci": wilson(k, accepted) if accepted else (float("nan"), float("nan")),
            "met": bool(accepted and acc >= target - 1e-12), **extra}


def held_out_at_target(cal_conf, cal_correct, conf, correct, target: float, certified: bool = True) -> dict:
    """Fit the threshold on a calibration sample, then measure coverage and accuracy on the evaluation items."""
    t = fit_threshold(cal_conf, cal_correct, target, certified)
    r = apply_threshold(conf, correct, t)
    return _summarize(target, certified, "held-out", r["n"], r["accepted"], r["correct"], threshold=t,
                      n_calibration=len(np.asarray(cal_conf)))


def cross_fit_at_target(conf, correct, target: float, certified: bool = True, folds: int = CROSS_FIT_FOLDS,
                        seed: int = CROSS_FIT_SEED) -> dict:
    """K-fold cross-fitting: each fold is scored with the threshold fitted on the other folds; counts are pooled.

    Used when no separate calibration sample exists. Every item is scored by a threshold that never saw it, so the
    result is not optimistic the way coverage_at_target is (at the cost of fitting on fewer items).
    """
    conf = np.asarray(conf, dtype=float)
    correct = np.asarray(correct, dtype=bool)
    fold = np.random.default_rng(seed).permutation(len(conf)) % folds
    accepted = k = 0
    thresholds = []
    for f in range(folds):
        fit, score = fold != f, fold == f
        t = fit_threshold(conf[fit], correct[fit], target, certified)
        r = apply_threshold(conf[score], correct[score], t)
        accepted, k = accepted + r["accepted"], k + r["correct"]
        thresholds.append(t)
    return _summarize(target, certified, "cross-fitted", len(conf), accepted, k, thresholds=thresholds, folds=folds)


# ---------------------------------------------------------------- calibration

def reliability(conf, correct, bins: int = ECE_BINS) -> list[dict]:
    """Equal-width bins: count, mean confidence and accuracy (empty bins included with count 0)."""
    conf = np.asarray(conf, dtype=float)
    correct = np.asarray(correct, dtype=float)
    idx = np.minimum((np.clip(conf, 0, 1) * bins).astype(int), bins - 1)
    out = []
    for b in range(bins):
        sel = idx == b
        count = int(sel.sum())
        out.append({"low": b / bins, "high": (b + 1) / bins, "count": count,
                    "confidence": float(conf[sel].mean()) if count else float("nan"),
                    "accuracy": float(correct[sel].mean()) if count else float("nan")})
    return out


def mce(bins: list[dict], min_count: int = MCE_MIN_BIN) -> tuple[float, int]:
    """Maximum |accuracy - confidence| over bins with at least min_count predictions, and bins used."""
    gaps = [abs(b["accuracy"] - b["confidence"]) for b in bins if b["count"] >= min_count]
    return (max(gaps) if gaps else float("nan")), len(gaps)


def normalized_probs(df: pd.DataFrame, labels: list[str]) -> np.ndarray | None:
    cols = [f"p_{label}" for label in labels]
    if not all(c in df.columns for c in cols):
        return None
    probs = df[cols].astype(float).clip(lower=0).to_numpy()
    sums = probs.sum(axis=1, keepdims=True)
    return probs / np.where(sums == 0, 1, sums)


def classwise_ece(probs: np.ndarray, label_idx: np.ndarray, bins: int = ECE_BINS) -> float:
    """Mean over labels of the ECE of p_label against 'true label is this label', over all predictions."""
    n, k = probs.shape
    total = 0.0
    for j in range(k):
        p, y = probs[:, j], (label_idx == j).astype(float)
        idx = np.minimum((np.clip(p, 0, 1) * bins).astype(int), bins - 1)
        counts = np.bincount(idx, minlength=bins)
        gap = np.abs(np.bincount(idx, weights=y, minlength=bins) - np.bincount(idx, weights=p, minlength=bins))
        total += gap[counts > 0].sum() / n
    return total / k


def top2_accuracy(probs: np.ndarray, label_idx: np.ndarray) -> float:
    """Share of articles whose true label is among the two highest p_* (ties broken at random, in expectation)."""
    p_true = probs[np.arange(len(probs)), label_idx][:, None]
    higher = (probs > p_true + 1e-12).sum(axis=1)
    equal = (np.abs(probs - p_true) <= 1e-12).sum(axis=1) - 1  # other labels tied with the true one
    slots = 2 - higher
    return float(np.where(slots <= 0, 0.0, np.minimum(1.0, slots / (equal + 1))).mean())


def calibration_warnings(df: pd.DataFrame, probs: np.ndarray | None, *, confidence_rule: str,
                         probabilities: str | None) -> list[str]:
    """Why ECE / MCE / reliability for this app may not measure probability calibration."""
    warnings = []
    conf = df["confidence"].astype(float)
    if confidence_rule == "own":
        warnings.append("confidence is the app's own score, not a probability")
    if probabilities == "uncalibrated":
        warnings.append("probabilities are an uncalibrated score (softmax of margins)")
    if conf.nunique() == 1:
        warnings.append("constant confidence: it cannot rank predictions")
    if probs is not None:
        saturated = float((probs.max(axis=1) >= SATURATED).mean())
        if saturated >= ONE_HOT_SHARE:
            warnings.append(f"one-hot outputs ({saturated:.0%} of predictions have max p ≥ {SATURATED})")
        elif saturated >= SATURATED_SHARE:
            warnings.append(f"saturated: {saturated:.0%} of predictions have max p ≥ {SATURATED}")
    else:
        warnings.append("no p_* columns")
    return warnings


def selective_metrics(df: pd.DataFrame, labels: list[str]) -> dict:
    """Everything the report shows for one app beyond the basic metrics."""
    correct = (df["label"] == df["pred"]).to_numpy()
    conf = df["confidence"].astype(float).to_numpy()
    g = groups(conf, correct)
    value, excess = aurc(g)
    probs = normalized_probs(df, labels)
    label_idx = df["label"].map({label: i for i, label in enumerate(labels)}).to_numpy()
    bins = reliability(conf, correct)
    mce_value, mce_bins = mce(bins)
    return {
        "groups": g, "curve": curve(g), "aurc": value, "eaurc": excess,
        "largest_tie": int(g.size.max()),
        "at_coverage": [accuracy_at_coverage(g, c) for c in COVERAGES],
        "at_target": {t: {"optimistic": coverage_at_target(g, t), "certified": coverage_at_target(g, t, True)}
                      for t in TARGETS},
        # Not fitted on the items it scores; report.py replaces it with a held-out result when a calibration
        # sample (another seed's run) exists.
        "held_out": {t: cross_fit_at_target(conf, correct, t) for t in TARGETS},
        "reliability": bins, "mce": mce_value, "mce_bins": mce_bins,
        "classwise_ece": classwise_ece(probs, label_idx) if probs is not None else float("nan"),
        "top2": top2_accuracy(probs, label_idx) if probs is not None else float("nan"),
        "probs": probs,
    }
