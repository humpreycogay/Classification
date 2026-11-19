"""Evaluate CLMClassifier on a benchmark task (default AG News; --task NAME, see data/tasks/).

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

from task import calibration_split, select_task_from_argv

select_task_from_argv()  # --task NAME -> BENCHMARK_TASK; must run before classifier.py reads the task
from classifier import (ACTION_TEMPLATE, ENCODER, ENCODER_REVISION, HEADS, HEADS_REVISION, MAX_TOKENS, TASK_LABELS,
                        CLMClassifier)
from harness import Setup, main

# Versions recorded in the run metadata (only those installed in this app's .venv).
KEY_PACKAGES = ["contrastive-lm", "torch", "transformers", "httpx", "pandas"]


def add_arguments(parser) -> None:
    # Default 1: batching changes BF16 padding numerics (scores move by up to ~0.5, as MPS vs CPU), which flips
    # near-ties: 97-98% agreement with batch 1 at 8 and 16 (2026-09-30), though batch 8 is 1.8x faster.
    parser.add_argument("--batch-size", type=int, default=1, help="texts per encoder pass")


def adapter(args, ctx) -> Setup:
    clf = CLMClassifier(TASK_LABELS, ctx.task.instruction)
    # Temperature from the task's calibration split (train only), cached; part of load time.
    calibration = calibration_split(ctx.task)
    temperature = clf.calibrate(calibration["text"].tolist(), calibration["label"].tolist(), args.batch_size)
    print(f"Temperature {temperature['temperature']} ({temperature['source']})")
    return Setup(model=f"CLM ({HEADS}@{HEADS_REVISION[:7]} heads on {ENCODER} BF16) on {clf.device}, "
                       f"T={clf.temperature}", classifier=clf, axis="batch_size", value=args.batch_size,
                 predict=lambda texts, v=args.batch_size: clf.predict_many(texts, batch_size=v), model_repo=ENCODER,
                 metadata={"calibration": temperature, "clm": {
                     "heads": HEADS, "heads_revision": HEADS_REVISION, "encoder": ENCODER,
                     "encoder_revision": ENCODER_REVISION, "dtype": str(clf.dtype), "device": clf.device,
                     "logit_scale_exp": clf.scale, "temperature": clf.temperature,
                     "state_template": "<text>\\n\\n<instruction>", "action_template": ACTION_TEMPLATE,
                     "pooling": "last token, L2-normalized", "max_tokens": MAX_TOKENS,
                     "label_embeddings": "computed once at load", "batching": "length-sorted"}})


if __name__ == "__main__":
    main(adapter, app_dir=APP_DIR, doc=__doc__, key_packages=KEY_PACKAGES, add_arguments=add_arguments)
