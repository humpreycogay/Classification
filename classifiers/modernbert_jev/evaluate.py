"""Evaluate ModernBertJevClassifier on a benchmark task (default AG News; --task NAME, see data/tasks/).

Dataset: Zhang, Zhao & LeCun (2015), "Character-level Convolutional Networks for
Text Classification". Classes: 1=World, 2=Sports, 3=Business, 4=Sci/Tech.

Usage:
    python evaluate.py              # stratified sample of 200 articles
    python evaluate.py --n 1000     # larger sample
    python evaluate.py --n 0        # full test set (7,600 articles)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
# The shared benchmark library (task.py, timing.py, harness.py, ...): BENCHMARK_SHARED_DIR (the runner sets
# it), else the repo's shared/ folder (classifiers/<app>/ -> two levels up).
SHARED_DIR = Path(os.environ.get("BENCHMARK_SHARED_DIR") or APP_DIR.parents[1] / "shared")
if not (SHARED_DIR / "harness.py").is_file():
    sys.exit(f"Shared benchmark library not found in {SHARED_DIR}; set BENCHMARK_SHARED_DIR to the repo's shared/ folder")
sys.path.insert(0, str(SHARED_DIR))

from task import select_task_from_argv

select_task_from_argv()  # --task NAME -> BENCHMARK_TASK; must run before classifier.py reads the task
from classifier import (ADAPTER_REPO, ADAPTER_REVISION, BACKBONE, REPO_COMMIT, TASK_LABELS,
                        ModernBertJevClassifier)
from harness import Setup, main

# Versions recorded in the run metadata (only those installed in this app's .venv).
KEY_PACKAGES = ["torch", "transformers", "onnxruntime", "tokenizers", "httpx", "pandas"]


def add_arguments(parser) -> None:
    # Default 1: fastest in a --sweep (2026-09-30, AG News: 28.1 items/s vs 26.6 at 8, 17.5 at 32; all agreed with 1).
    parser.add_argument("--batch-size", type=int, default=1,
                        help="texts per step; their (text, option) pairs are scored in length-sorted chunks")


def adapter(args, ctx) -> Setup:
    clf = ModernBertJevClassifier(TASK_LABELS, ctx.task.instruction)
    return Setup(model=f"{ADAPTER_REPO}@{ADAPTER_REVISION[:7]} on {BACKBONE} (T={clf.temperature:.4f}, "
                       f"{clf.backend}) on {clf.device}", classifier=clf,
                 axis="batch_size", value=args.batch_size,
                 predict=lambda texts, v=args.batch_size: clf.predict_many(texts, batch_size=v),
                 model_repo=ADAPTER_REPO,
                 metadata={"modernbert_jev": {
                     "adapter": ADAPTER_REPO, "adapter_revision": ADAPTER_REVISION, "backbone": BACKBONE,
                     "backbone_revision": "8949b909ec900327062f0ebf497f51aef5e6f0c8", "code_commit": REPO_COMMIT,
                     "temperature": clf.temperature, "temperature_source": "calibration.json",
                     "backend": clf.backend, "dtype": "float32", "lora_folded": clf.folded,
                     "scoring": "one logit per (text, option) pair, softmax over options",
                     "candidate_template": "Question: <instruction>\\nCandidate: <description>",
                     "batching": f"flattened (text, option) pairs, length-sorted, chunks of {clf.chunk}"}})


if __name__ == "__main__":
    main(adapter, app_dir=APP_DIR, doc=__doc__, key_packages=KEY_PACKAGES, add_arguments=add_arguments)
