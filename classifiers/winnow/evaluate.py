"""Evaluate WinnowClassifier on a benchmark task (default AG News; --task NAME, see data/tasks/).

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
from classifier import (DEFAULT_CONCURRENCY, MODEL_FILE, MODEL_REPO, MODEL_REVISION, TASK_LABELS,
                        WinnowClassifier)
from harness import Setup, main
import shortlist

# Versions recorded in the run metadata (only those installed in this app's .venv).
KEY_PACKAGES = ["httpx", "pandas"]
SERVER_COMMIT = "77d14580c6732ca2f3745750c1dc1fd446d8bcee"  # keep in step with server/setup.sh


def add_arguments(parser) -> None:
    shortlist.add_argument(parser)
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY, help="requests in flight")


def adapter(args, ctx) -> Setup:
    clf = WinnowClassifier(shortlist.initial_labels(TASK_LABELS, args.shortlist), ctx.task.instruction)
    clf.check_ready()
    setup = Setup(model=f"{MODEL_REPO}@{MODEL_REVISION[:7]} Q8_0 (winnow-inference, Metal) at {clf.url}",
                  classifier=clf, axis="concurrency", value=args.concurrency,
                  predict=lambda texts, v=args.concurrency: clf.predict_many(texts, concurrency=v),
                  model_repo=MODEL_REPO,
                  metadata={"winnow": {
                      "model": MODEL_REPO, "revision": MODEL_REVISION, "file": MODEL_FILE, "quantization": "Q8_0",
                      "server": "winnow-inference", "server_commit": SERVER_COMMIT, "profile": "apple-silicon",
                      "text_only": True, "context": int(os.getenv("WINNOW_CONTEXT") or 8192), "temperature": 1.0,
                      "scoring": "one choice question per text; server softmax over option-letter logits",
                      "option_template": "<name>: <description>"}})
    return shortlist.apply(setup, clf, ctx, TASK_LABELS, args.shortlist)


if __name__ == "__main__":
    main(adapter, app_dir=APP_DIR, doc=__doc__, key_packages=KEY_PACKAGES, add_arguments=add_arguments)
