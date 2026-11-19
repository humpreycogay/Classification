"""Evaluate the debug copy of JEVClassifier (verbose logging on stderr) on a benchmark task
(default AG News; --task NAME, see data/tasks/).

Same run as classifiers/jev/evaluate.py; every step and request is logged (see classifier.py for
JEV_DEBUG_LEVEL). Each item is a paid API request, so start small.

Usage:
    python evaluate.py --n 8 --latency-items 0      # 8 articles, no latency run
    python evaluate.py --n 8 --concurrency 2        # watch items wait for a permit
    python evaluate.py 2> debug.log                 # default 200 articles, log to a file
    JEV_DEBUG_LEVEL=info python evaluate.py --n 8   # one line per item, no request / response bodies
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
from classifier import TASK_LABELS, JEVClassifier
from harness import Setup, main

# Versions recorded in the run metadata (only those installed in this app's .venv).
KEY_PACKAGES = ["typesafe-sdk", "laya", "semif-phase1", "gliclass", "torch", "transformers", "mlx", "mlx-lm", "httpx", "pandas"]


def add_arguments(parser) -> None:
    parser.add_argument("--concurrency", type=int, default=16)


def adapter(args, ctx) -> Setup:
    clf = JEVClassifier(TASK_LABELS, ctx.task.instruction)
    return Setup(model=f"{clf.model}", classifier=clf, axis="concurrency", value=args.concurrency,
                 predict=lambda texts, v=args.concurrency: clf.predict_many(texts, concurrency=v))


if __name__ == "__main__":
    main(adapter, app_dir=APP_DIR, doc=__doc__, key_packages=KEY_PACKAGES, add_arguments=add_arguments)
