"""Synthetic data shared by the tests: no model, dataset download or network needed."""

from __future__ import annotations

import numpy as np
import pandas as pd

from validate import LABELS, PROB_COLS


def fake_test_set(per_label: int = 10) -> pd.DataFrame:
    """A balanced labeled set shaped like load_test_set() output (label + text)."""
    rows = [{"label": label, "text": f"{label} article {i}"} for label in LABELS for i in range(per_label)]
    return pd.DataFrame(rows)


def fake_predictions(sample: pd.DataFrame, accuracy_mask: list[bool] | None = None, seed: int = 0) -> pd.DataFrame:
    """A valid predictions CSV for `sample`: correct where accuracy_mask is True, else the next label."""
    rng = np.random.default_rng(seed)
    df = sample[[c for c in ("sample_id", "text", "label") if c in sample.columns]].reset_index(drop=True).copy()
    mask = accuracy_mask if accuracy_mask is not None else [True] * len(df)
    df["pred"] = [lab if ok else LABELS[(LABELS.index(lab) + 1) % len(LABELS)] for lab, ok in zip(df["label"], mask)]
    probs = rng.dirichlet(np.ones(len(LABELS)), size=len(df)) * 0.4
    for i, pred in enumerate(df["pred"]):
        probs[i, LABELS.index(pred)] += 0.6  # the prediction is the argmax
    for j, col in enumerate(PROB_COLS):
        df[col] = probs[:, j]
    df["confidence"] = probs.max(axis=1)
    return df
