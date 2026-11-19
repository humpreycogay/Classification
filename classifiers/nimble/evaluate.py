"""Evaluate NimbleClassifier on a benchmark task (default AG News; --task NAME, see data/tasks/).

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
from classifier import BASE_MODEL, MODEL, REVISION, TASK_LABELS, NimbleClassifier
from harness import Setup, main

# Versions recorded in the run metadata (only those installed in this app's .venv).
KEY_PACKAGES = ["typesafe-sdk", "laya", "semif-phase1", "gliclass", "torch", "transformers", "peft", "mlx", "mlx-lm",
                "httpx", "pandas"]


def adapter(args, ctx) -> Setup:
    clf = NimbleClassifier(TASK_LABELS, ctx.task.instruction)
    return Setup(model=f"Nimble ({MODEL}@{REVISION[:7]}, merged BF16, MLX)", classifier=clf, sequential=True,
                 predict=lambda texts, v=None: clf.predict_many(texts),
                 model_repo=BASE_MODEL,  # the merged folder is the base's size (BF16, same shapes)
                 metadata={"nimble": {"model": MODEL, "revision": REVISION, "base_model": BASE_MODEL,
                                      "model_path": str(clf.model_path), "temperature": clf.temperature,
                                      "scoring": clf.scoring}})


if __name__ == "__main__":
    main(adapter, app_dir=APP_DIR, doc=__doc__, key_packages=KEY_PACKAGES)
