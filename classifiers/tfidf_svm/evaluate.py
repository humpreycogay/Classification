"""Evaluate TfidfSVMClassifier on a benchmark task (default AG News; --task NAME, see data/tasks/).

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
    python evaluate.py --calibration none   # raw LinearSVC margins; results in results/uncalibrated/
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
from classifier import GRID, build, CALIBRATIONS, TfidfSVMClassifier, load_train_set
from harness import Setup, main
from training import fit_baseline

# Versions recorded in the run metadata (only those installed in this app's .venv).
KEY_PACKAGES = ["scikit-learn", "scipy", "numpy", "httpx", "pandas"]


def add_arguments(parser) -> None:
    parser.add_argument("--C", type=float, default=None,
                        help=f"inverse regularization strength (default: chosen by CV on the train split from {GRID})")
    parser.add_argument("--calibration", choices=list(CALIBRATIONS), default="sigmoid",
                        help="sigmoid = calibrated probabilities (default); none = raw SVM margins")


def adapter(args, ctx) -> Setup:
    # Load = reading the train split (+ loading a cached fitted model); selection and training are timed separately.
    train = load_train_set()
    make_classifier = lambda C: TfidfSVMClassifier(C=C, calibration=args.calibration)  # noqa: E731
    clf, training = fit_baseline(ctx, make_classifier, build, GRID, train, param="C", override=args.C, variant="linearsvc",
                                 train_split=f"{ctx.task.name} train (all)",
                                 probabilities="calibrated" if args.calibration == "sigmoid" else "uncalibrated")
    # Classification is one vectorized call for the whole sample.
    return Setup(model=clf.model, classifier=clf, predict=lambda texts, v=None: clf.predict_many(texts),
                 # The uncalibrated variant is a separate result set, so it never overwrites the calibrated one.
                 results="" if args.calibration == "sigmoid" else "uncalibrated", metadata={"training": training})


if __name__ == "__main__":
    main(adapter, app_dir=APP_DIR, doc=__doc__, key_packages=KEY_PACKAGES, add_arguments=add_arguments)
