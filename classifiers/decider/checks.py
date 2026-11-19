"""Setup-time checks for Decider-4B v3 (run by setup.sh; exit 1 on failure). Never used in a timed run.

    python checks.py parity    # MPS float16 (the runtime) vs CPU float32 on 20 fixed texts: no argmax change
    python checks.py probes    # three AG News probes in the right class
    python checks.py overlap   # the public train splits Decider trained on vs our test texts -> results/overlap.json

Overlap. Decider's training mixture (github.com/Mapika/decider, decider/data/core.py) takes the train split of every
registered task, including ag_news (fancyzhx/ag_news) and banking77 (mteb/banking77); their test splits are its
evaluation halves. Our test sets are those test splits, so the model saw only the datasets' own train/test
duplicates, which shared/overlap.py reports without counting (the supervised baselines see them too).
"""

from __future__ import annotations

import gc
import sys
from pathlib import Path

import pandas as pd
import torch
from huggingface_hub import hf_hub_download

from classifier import TASK, TASK_LABELS, DeciderClassifier

APP_DIR = Path(__file__).resolve().parent

# (dataset, pinned revision, text column) per task, as decider/data/core.py loads them (train split).
TRAIN_SPLITS = {
    "ag_news": ("fancyzhx/ag_news", "eb185aade064a813bc0b7f42de02595523103ca4", "text"),
    "banking77": ("mteb/banking77", "18072d2685ea682290f7b8924d94c62acc19c0b2", "text"),
}


def parity() -> None:
    from probes import PARITY_TEXTS

    gpu = DeciderClassifier(TASK_LABELS, TASK.instruction)
    if gpu.device == "cpu":
        print("parity check skipped: no GPU")
        return
    a = gpu.predict_many(PARITY_TEXTS)
    runtime = f"{gpu.device} {gpu.dtype}"
    del gpu
    gc.collect()
    torch.mps.empty_cache()
    cpu = DeciderClassifier(TASK_LABELS, TASK.instruction, device="cpu")
    b = cpu.predict_many(PARITY_TEXTS)
    flips = [t[:50] for t, x, y in zip(PARITY_TEXTS, a, b) if x.label != y.label]
    gap = max(abs(x.probabilities[l] - y.probabilities[l]) for x, y in zip(a, b) for l in TASK_LABELS)
    print(f"parity: {runtime} vs CPU {cpu.dtype} on {len(a)} texts: {len(flips)} argmax changes, "
          f"max |dp| {gap:.2e}")
    if flips:
        sys.exit(f"Decider parity check failed: argmax changes on {flips}")
    print("parity check passed")


def probes() -> None:
    from probes import check_probes

    check_probes(DeciderClassifier(TASK_LABELS, TASK.instruction).predict_many, name="Decider probe check")


def overlap() -> None:
    import overlap as shared_overlap

    entries = {}
    for task, (repo, revision, column) in TRAIN_SPLITS.items():
        path = hf_hub_download(repo, "data/train-00000-of-00001.parquet", repo_type="dataset", revision=revision)
        seen = pd.read_parquet(path)[column].tolist()
        entries[task] = shared_overlap.check(task, {"train": seen}, note=f"{repo}@{revision[:7]} train split")
    shared_overlap.write(APP_DIR, entries)


if __name__ == "__main__":
    {"parity": parity, "probes": probes, "overlap": overlap}[sys.argv[1]]()
