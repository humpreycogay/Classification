"""Strict checks on one app result (predictions CSV + run metadata) before it may enter the report.

`validate_result` returns a list of problems; an empty list means the result is valid. The runner
marks results with problems as "invalid" (reported like a failure, never ranked).

Checks:
    - required columns present; row count equals the expected sample size
    - labels and predictions are known labels; sample_id (when present) unique
    - texts, labels (and sample_id) identical to the expected sample, in order
    - p_* finite, in [0, 1], each row summing to ~1
    - pred is a label with the largest p_*
    - confidence finite in [0, 1]; equal to max p_* for apps whose rule is "max_p"
    - metadata n, seed and sample_sha1 agree with the expected sample and with the CSV
    - (runner runs only) both files written by this run: newer than the start time and carrying the run id
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import shared_path  # noqa: F401  (puts shared/ on sys.path)
from task import load_task, sample_sha1

# Default (AG News) labels; every check takes the run's task labels via `labels=`.
LABELS = load_task().label_ids
PROB_COLS = [f"p_{label}" for label in LABELS]
REQUIRED_COLUMNS = ["text", "label", "pred", "confidence", *PROB_COLS]

PROB_TOL = 1e-6  # p_* outside [0, 1] by more than this is invalid
SUM_TOL = 1e-3  # |sum(p_*) - 1| allowed per row (apps round their probabilities)
ROUNDING_PER_LABEL = 1e-4  # rounding each p to 4 decimals adds up over many labels: tolerance >= k x this
CONF_TOL = 1e-3  # |confidence - max p_*| allowed for max_p apps
MAX_LISTED = 3  # example rows quoted per problem


def _rows(mask: np.ndarray) -> str:
    idx = np.flatnonzero(mask)
    shown = ", ".join(str(i) for i in idx[:MAX_LISTED])
    return f"{len(idx)} row(s), e.g. {shown}"


def check_frame(df: pd.DataFrame, *, confidence_rule: str = "max_p", labels: list[str] | None = None,
                sum_tol: float = SUM_TOL) -> list[str]:
    """Checks that need only the predictions CSV. labels: the task's label ids (default AG News);
    sum_tol: allowed |sum(p_*) - 1| (larger for apps that round their probabilities)."""
    labels = labels or LABELS
    prob_cols = [f"p_{label}" for label in labels]
    missing = [c for c in ["text", "label", "pred", "confidence", *prob_cols] if c not in df.columns]
    if missing:
        return [f"missing column(s): {', '.join(missing)}"]
    problems = []
    for col in ("label", "pred"):
        bad = ~df[col].isin(labels).to_numpy()
        if bad.any():
            values = sorted({str(v) for v in df.loc[bad, col]})[:MAX_LISTED]
            problems.append(f"unknown {col} value(s) {values} in {_rows(bad)}")
    if "sample_id" in df.columns and df["sample_id"].duplicated().any():
        problems.append(f"duplicate sample_id in {_rows(df['sample_id'].duplicated().to_numpy())}")

    probs = df[prob_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(probs).all(axis=1)
    if not finite.all():
        problems.append(f"non-numeric or non-finite p_* in {_rows(~finite)}")
    out_of_range = finite & ((probs < -PROB_TOL) | (probs > 1 + PROB_TOL)).any(axis=1)
    if out_of_range.any():
        problems.append(f"p_* outside [0, 1] in {_rows(out_of_range)}")
    sum_tol = max(sum_tol, ROUNDING_PER_LABEL * len(labels))
    bad_sum = finite & (np.abs(np.where(finite[:, None], probs, 0).sum(axis=1) - 1) > sum_tol)
    if bad_sum.any():
        problems.append(f"p_* rows not summing to 1 (±{sum_tol:g}) in {_rows(bad_sum)}")

    row_max = np.where(finite[:, None], probs, -np.inf).max(axis=1)
    pred_idx = df["pred"].map({label: i for i, label in enumerate(labels)})
    known = pred_idx.notna().to_numpy() & finite
    p_pred = np.full(len(df), np.nan)
    p_pred[known] = probs[np.flatnonzero(known), pred_idx[known].astype(int).to_numpy()]
    not_argmax = known & (p_pred < row_max - PROB_TOL)
    if not_argmax.any():
        problems.append(f"pred is not argmax(p_*) in {_rows(not_argmax)}")

    conf = pd.to_numeric(df["confidence"], errors="coerce").to_numpy(dtype=float)
    bad_conf = ~np.isfinite(conf) | (conf < -PROB_TOL) | (conf > 1 + PROB_TOL)
    if bad_conf.any():
        problems.append(f"confidence not a finite value in [0, 1] in {_rows(bad_conf)}")
    if confidence_rule == "max_p":
        off = ~bad_conf & finite & (np.abs(conf - row_max) > CONF_TOL)
        if off.any():
            problems.append(f"confidence differs from max p_* (rule max_p) in {_rows(off)}")
    return problems


def check_sample(df: pd.DataFrame, expected: pd.DataFrame) -> list[str]:
    """Same texts and labels (and sample_id when both have it) as the expected sample, in order."""
    if len(df) != len(expected):
        return [f"{len(df)} rows, expected {len(expected)}"]
    problems = []
    for col in ("text", "label", "sample_id"):
        if col in df.columns and col in expected.columns:
            differs = df[col].to_numpy() != expected[col].to_numpy()
            if differs.any():
                problems.append(f"{col} differs from the expected sample in {_rows(differs)}")
    return problems


def validate_result(csv_path: Path, meta_path: Path | None, *, expected: pd.DataFrame, seed: int,
                    confidence_rule: str = "max_p", meta_required: bool = True,
                    not_before: float | None = None, run_id: str | None = None,
                    labels: list[str] | None = None, sum_tol: float = SUM_TOL) -> list[str]:
    """All problems with one result; [] if it may be compared with the others.

    expected: the sample (sample_id, text, label) the app had to classify.
    not_before / run_id: set by the runner for results it just produced, to reject stale files.
    """
    if not csv_path.exists():
        return ["no predictions CSV"]
    problems = []
    meta = None
    if meta_path is not None and meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
        except json.JSONDecodeError as e:
            problems.append(f"run metadata is not valid JSON ({e})")
    elif meta_required:
        problems.append("no run metadata")

    if not_before is not None:
        for label, path in (("predictions CSV", csv_path), ("run metadata", meta_path)):
            if path is not None and path.exists() and path.stat().st_mtime < not_before:
                problems.append(f"{label} is older than this run (stale file)")
    if run_id is not None and meta is not None and meta.get("run_id") != run_id:
        problems.append(f"run metadata has run_id {meta.get('run_id')!r}, not this run's {run_id!r}")

    try:
        df = pd.read_csv(csv_path)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as e:
        return problems + [f"predictions CSV unreadable ({e})"]
    frame_problems = check_frame(df, confidence_rule=confidence_rule, labels=labels, sum_tol=sum_tol)
    problems += frame_problems
    if any(p.startswith("missing column") for p in frame_problems):
        return problems
    problems += check_sample(df, expected)

    if meta is not None:
        expected_sha, csv_sha = sample_sha1(expected), sample_sha1(df)
        if meta.get("n") != len(expected):
            problems.append(f"metadata n = {meta.get('n')}, expected {len(expected)}")
        if meta.get("seed") != seed:
            problems.append(f"metadata seed = {meta.get('seed')}, expected {seed}")
        if meta.get("sample_sha1") != expected_sha:
            problems.append("metadata sample_sha1 differs from the expected sample")
        if meta.get("sample_sha1") != csv_sha:
            problems.append("metadata sample_sha1 differs from the CSV's texts and labels")
        for key in ("classify_seconds", "load_seconds", "warmup_seconds"):
            value = meta.get(key)
            if not isinstance(value, (int, float)) or math.isnan(value) or value < 0:
                problems.append(f"metadata {key} missing or invalid ({value!r})")
    return problems
