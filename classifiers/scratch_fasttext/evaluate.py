"""Evaluate the from-scratch fastText-style classifier on a benchmark task (default AG News; --task NAME).

Trains on the task's train split (a stratified 90% to fit, 10% for early stopping and the temperature), or loads
the identical earlier model from models/<task>/. Test labels are used only for scoring. Training time is recorded
in the run's "training" metadata and never counted as load or classification time.

Usage:
    python evaluate.py              # stratified sample of 200 texts
    python evaluate.py --n 1000     # the common tier
    python evaluate.py --n 0        # the whole test split
    python evaluate.py --epochs 3   # a different recipe = a different saved model
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
# The shared benchmark library (task.py, harness.py, textnn.py, ...): BENCHMARK_SHARED_DIR (the runner sets it),
# else the repo's shared/ folder (classifiers/<app>/ -> two levels up).
SHARED_DIR = Path(os.environ.get("BENCHMARK_SHARED_DIR") or APP_DIR.parents[1] / "shared")
if not (SHARED_DIR / "harness.py").is_file():
    sys.exit(f"Shared benchmark library not found in {SHARED_DIR}; set BENCHMARK_SHARED_DIR to the repo's shared/ folder")
sys.path.insert(0, str(SHARED_DIR))

from task import select_task_from_argv

select_task_from_argv()  # --task NAME -> BENCHMARK_TASK; must run before classifier.py reads the task
from classifier import CONFIGS, ScratchFastTextClassifier
from harness import Setup, main
from task import load_split
from textnn import add_training_arguments, config_from_args

# Versions recorded in the run metadata (only those installed in this app's .venv).
KEY_PACKAGES = ["torch", "numpy", "pandas", "httpx"]


def adapter(args, ctx) -> Setup:
    cfg = config_from_args(CONFIGS[ctx.task.name], args)
    clf = ScratchFastTextClassifier(cfg, args.device)
    with ctx.exclude():  # training is not load time
        train = load_split(ctx.task, "train")
        training = clf.fit_cached(train, APP_DIR / "models" / ctx.task.name)
        training["train_split"] = f"{ctx.task.name} train (all; stratified 90% fit, 10% validation)"
        training["n_train"] = len(train)
        # Test texts that also appear verbatim in the train split (the corpus has a few duplicates).
        training["test_texts_in_train"] = int(ctx.sample["text"].isin(set(train["text"])).sum())
    ctx.add_load(training.pop("_load_seconds"))  # loading a saved model is load time
    return Setup(model=clf.model_description, classifier=clf, axis="batch_size", value=args.batch_size,
                 predict=lambda texts, v=args.batch_size: clf.predict_many(texts, batch_size=v),
                 metadata={"training": training})


if __name__ == "__main__":
    main(adapter, app_dir=APP_DIR, doc=__doc__, key_packages=KEY_PACKAGES, add_arguments=add_training_arguments)
