"""Evaluate StrandsDeciderClassifier on a benchmark task (default AG News; --task NAME, see data/tasks/).

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
from classifier import (BASE_REPO, BASE_REVISION, CODE_COMMIT, MODEL_REPO, MODEL_REVISION, TASK_LABELS,
                        StrandsDeciderClassifier)
from harness import Setup, main

# Versions recorded in the run metadata (only those installed in this app's .venv).
KEY_PACKAGES = ["strands-decider", "torch", "transformers", "peft", "mlx", "mlx-lm", "httpx", "pandas"]


def adapter(args, ctx) -> Setup:
    clf = StrandsDeciderClassifier(TASK_LABELS, ctx.task.instruction)
    return Setup(model=f"{MODEL_REPO}@{MODEL_REVISION[:7]} (T={clf.temperature}) on {clf.device}",
                 classifier=clf, sequential=True, predict=lambda texts, v=None: clf.predict_many(texts),
                 model_repo=BASE_REPO,
                 metadata={"strands_decider": {
                     "model": MODEL_REPO, "revision": MODEL_REVISION, "base_model": BASE_REPO,
                     "base_revision": BASE_REVISION, "code_commit": CODE_COMMIT,
                     "device": clf.device, "dtype": clf.dtype, "head": "float32 on CPU",
                     "temperature": clf.temperature, "temperature_source": "hobson_config.json temperature_by_kind.choice",
                     "context_window": clf.max_length, "strict_window": True,
                     "scoring": "one choice question per text; pointer-head option scores / T, softmax over the options",
                     "option_template": "<name> — <description>",
                     "probabilities": "rounded to 4 decimals by the engine",
                     "confidence": "max p; the engine's own value is (N * max p - 1) / (N - 1), a monotone "
                                   "function of max p for a fixed number of options N"}})


if __name__ == "__main__":
    main(adapter, app_dir=APP_DIR, doc=__doc__, key_packages=KEY_PACKAGES)
