"""Setup-time checks for Plumb-4B (run by setup.sh; exit 1 on failure). Never used in a timed run.

    python checks.py probes    # three AG News probes in the right class (MLX)
    python checks.py parity    # MLX vs jevk5's own PyTorch path (CPU, float32) on 20 fixed texts

Parity: same argmax on every text and max |dp| <= 0.01 (probabilities at the release temperature). The MLX run
uses the shared prompt start when classifier.py would; the reference is jevk5.JevK5.probabilities, unchanged.
"""

from __future__ import annotations

import gc
import sys

from classifier import TASK, TASK_LABELS, PlumbClassifier, model_dir, question
from probes import PARITY_TEXTS, check_probes

TOL = 0.01


def probes() -> None:
    check_probes(PlumbClassifier(TASK_LABELS, TASK.instruction).predict_many, name="Plumb probe check")


def parity() -> None:
    import mlx.core as mx
    import torch
    from jevk5 import JevK5

    clf = PlumbClassifier(TASK_LABELS, TASK.instruction)
    ours = clf.predict_many(PARITY_TEXTS)
    scoring = clf.scoring
    del clf
    gc.collect()
    mx.clear_cache()
    ref = JevK5(model_dir(), device="cpu", dtype=torch.float32, graphs=False)
    q = question(TASK.instruction, TASK_LABELS)
    by_name = {s["name"]: s["id"] for s in TASK.labels}
    flips, gap = [], 0.0
    for text, pred in zip(PARITY_TEXTS, ours):
        probs, _ = ref.probabilities(text, q)
        probs = {by_name[k]: v for k, v in probs.items()}
        if max(probs, key=probs.get) != pred.label:
            flips.append(text[:50])
        gap = max(gap, max(abs(probs[l] - pred.probabilities[l]) for l in TASK_LABELS))
    print(f"parity: MLX BF16 ({scoring}) vs jevk5 PyTorch CPU float32 on {len(ours)} texts: "
          f"{len(flips)} argmax changes, max |dp| {gap:.2e}")
    if flips or gap > TOL:
        sys.exit(f"Plumb parity check failed (flips {flips}, max |dp| {gap:.3f} > {TOL})")
    print("parity check passed")


if __name__ == "__main__":
    {"probes": probes, "parity": parity}[sys.argv[1]]()
