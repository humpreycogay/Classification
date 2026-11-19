"""Evaluate TfidfLogRegClassifier on a benchmark task (default AG News; --task NAME, see data/tasks/).

Dataset: Zhang, Zhao & LeCun (2015), "Character-level Convolutional Networks for
Text Classification". Classes: 1=World, 2=Sports, 3=Business, 4=Sci/Tech.

The model is trained on the task's train split (120,000 articles for AG News); test labels are
used only for scoring. C is chosen by 3-fold cross-validation on the train split (cached in
hyperparameters.json) unless --C is given, and the fitted model is cached in results/cache/, so
repeated runs load it instead of refitting. Training time is recorded separately from inference.

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
from classifier import GRID, build, TfidfLogRegClassifier, load_train_set, load_train_sample
from harness import Setup, main
from training import fit_baseline

# Versions recorded in the run metadata (only those installed in this app's .venv).
KEY_PACKAGES = ["scikit-learn", "scipy", "numpy", "httpx", "pandas"]


def add_arguments(parser) -> None:
    parser.add_argument("--C", type=float, default=None,
                        help=f"inverse regularization strength (default: chosen by CV on the train split from {GRID})")
    parser.add_argument("--train-per-class", default="0",
                        help="train on this many articles per label (embedsup's sample); 'task' = the task's "
                             "supervised_sample_per_label (the embedding baselines' size); 0 = the whole train split")
    parser.add_argument("--train-seed", type=int, default=0, help="seed of the train sample (not the test sample)")


def adapter(args, ctx) -> Setup:
    # Load = reading the train split (+ loading a cached fitted model); selection and training are timed separately.
    matched = args.train_per_class == "task"  # the same train sample as the embedding baselines
    per_class = ctx.task.supervised_sample_per_label if matched else int(args.train_per_class)
    if per_class:
        train = load_train_sample(per_class, args.train_seed)
        variant = f"train{len(train)}_seed{args.train_seed}"
        train_split = f"{ctx.task.name} train, stratified {per_class} per label, seed {args.train_seed}"
    else:
        train = load_train_set()
        variant, train_split = "full", f"{ctx.task.name} train (all)"
    # A reduced-train variant is a separate result set (results/train<N>/), so it never overwrites the full one.
    results = "matched_train" if matched else f"train{len(train)}" if per_class else ""
    clf, training = fit_baseline(ctx, lambda C: TfidfLogRegClassifier(C=C), build, GRID, train, param="C",
                                 override=args.C, variant=variant, train_split=train_split, probabilities="native")
    # Classification is one vectorized call for the whole sample.
    return Setup(model=clf.model, classifier=clf, predict=lambda texts, v=None: clf.predict_many(texts),
                 results=results, metadata={"training": training})


if __name__ == "__main__":
    main(adapter, app_dir=APP_DIR, doc=__doc__, key_packages=KEY_PACKAGES, add_arguments=add_arguments)
