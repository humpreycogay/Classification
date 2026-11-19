"""Setup-time checks for Strands Decider (run by setup.sh; exit 1 on failure). Never used in a timed run.

    python checks.py manifest  # the downloaded weights against the repository's MANIFEST.sha256
    python checks.py probes    # three AG News probes in the right class
    python checks.py parity    # the runtime (MLX bfloat16) vs CPU float32 on 20 fixed texts: no argmax change
    python checks.py overlap   # the public train splits the model trained on vs our test texts -> results/overlap.json

Overlap. The model's corpus (github.com/strands-labs/strands-decider, data/sources.md and
src/strands_decider/data/recipes.py) takes the train split of fancyzhx/ag_news (up to 6,000 examples) and of
legacy-datasets/banking77 (up to 8,000); no recipe or evaluation of the repository reads their test splits. Our
test sets are those test splits, so the model saw only the datasets' own train/test duplicates, which
shared/overlap.py reports without counting (the supervised baselines see them too).
"""

from __future__ import annotations

import gc
import hashlib
import sys
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

from classifier import TASK, TASK_LABELS, StrandsDeciderClassifier, model_dir

APP_DIR = Path(__file__).resolve().parent

# (dataset, pinned revision, text column) per task, as data/recipes.py loads them (train split).
TRAIN_SPLITS = {
    "ag_news": ("fancyzhx/ag_news", "eb185aade064a813bc0b7f42de02595523103ca4", "text"),
    "banking77": ("legacy-datasets/banking77", "f54121560de48f2852f90be299010d1d6dc612ec", "text"),
}


def manifest() -> None:
    root = Path(model_dir())
    lines = (root / "MANIFEST.sha256").read_text().splitlines()
    bad = []
    for digest, name in (line.split(maxsplit=1) for line in lines if line.strip()):
        path = root / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            bad.append(name)
    if bad:
        sys.exit(f"Strands Decider manifest check failed: {bad} do not match MANIFEST.sha256")
    print(f"manifest check passed: {len(lines)} files match MANIFEST.sha256")


def parity() -> None:
    from probes import PARITY_TEXTS

    gpu = StrandsDeciderClassifier(TASK_LABELS, TASK.instruction)
    if gpu.device == "cpu":
        print("parity check skipped: no GPU")
        return
    a = gpu.predict_many(PARITY_TEXTS)
    runtime = f"{gpu.device} {gpu.dtype}"
    del gpu
    gc.collect()
    cpu = StrandsDeciderClassifier(TASK_LABELS, TASK.instruction, device="cpu")
    b = cpu.predict_many(PARITY_TEXTS)
    flips = [t[:50] for t, x, y in zip(PARITY_TEXTS, a, b) if x.label != y.label]
    gap = max(abs(x.probabilities[l] - y.probabilities[l]) for x, y in zip(a, b) for l in TASK_LABELS)
    print(f"parity: {runtime} vs CPU {cpu.dtype} on {len(a)} texts: {len(flips)} argmax changes, "
          f"max |dp| {gap:.2e}")
    if flips:
        sys.exit(f"Strands Decider parity check failed: argmax changes on {flips}")
    print("parity check passed")


def probes() -> None:
    from probes import check_probes

    check_probes(StrandsDeciderClassifier(TASK_LABELS, TASK.instruction).predict_many,
                 name="Strands Decider probe check")


def overlap() -> None:
    import overlap as shared_overlap

    entries = {}
    for task, (repo, revision, column) in TRAIN_SPLITS.items():
        path = hf_hub_download(repo, "data/train-00000-of-00001.parquet", repo_type="dataset", revision=revision)
        seen = pd.read_parquet(path)[column].tolist()
        entries[task] = shared_overlap.check(task, {"train": seen}, note=f"{repo}@{revision[:7]} train split")
    shared_overlap.write(APP_DIR, entries)


if __name__ == "__main__":
    {"manifest": manifest, "parity": parity, "probes": probes, "overlap": overlap}[sys.argv[1]]()
