"""Evaluate LLMClassifier on a benchmark task (default AG News; --task NAME, see data/tasks/).

Dataset: Zhang, Zhao & LeCun (2015), "Character-level Convolutional Networks for
Text Classification". Classes: 1=World, 2=Sports, 3=Business, 4=Sci/Tech.

Usage:
    python evaluate.py              # stratified sample of 200 articles
    python evaluate.py --n 1000     # larger sample
    python evaluate.py --n 0        # full test set (7,600 articles)
    python evaluate.py --model qwen3.5:9b-q4_K_M --results qwen   # another model, its own results/qwen/
    python evaluate.py --host http://gpu-box:11434                 # another Ollama server

Results go to results/<--results>/ (default: the model tag, e.g. results/gemma4_e4b/), so the models
never overwrite each other's results.
"""

from __future__ import annotations

import os
import re
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
from classifier import TASK_LABELS, DEFAULT_CONCURRENCY, DEFAULT_HOST, DEFAULT_MODEL, LLMClassifier
from harness import Setup, main
import shortlist

# Versions recorded in the run metadata (only those installed in this app's .venv).
KEY_PACKAGES = ["typesafe-sdk", "laya", "semif-phase1", "gliclass", "torch", "transformers", "mlx", "mlx-lm", "httpx", "pandas"]


def add_arguments(parser) -> None:
    parser.add_argument("--model", help=f"Ollama model tag (default: $LLM_MODEL or {DEFAULT_MODEL})")
    parser.add_argument("--host", help=f"Ollama base URL (default: $LLM_HOST, $OLLAMA_HOST or {DEFAULT_HOST})")
    parser.add_argument("--results", metavar="NAME", help="results subfolder (default: the model tag)")
    shortlist.add_argument(parser)
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY, help="requests in flight")


def adapter(args, ctx) -> Setup:
    clf = LLMClassifier(shortlist.initial_labels(TASK_LABELS, args.shortlist), ctx.task.instruction,
                        model=args.model, host=args.host)
    clf.check_ready()
    setup = Setup(model=f"{clf.model} (num_ctx {clf.num_ctx}) via Ollama at {clf.host}", classifier=clf, axis="concurrency", value=args.concurrency,
                  predict=lambda texts, v=args.concurrency: clf.predict_many(texts, concurrency=v),
                  on_main=lambda: count_answer_overrides(clf))
    setup = shortlist.apply(setup, clf, ctx, TASK_LABELS, args.shortlist)
    # One results folder per model (a shortlist variant nests inside it: results/<name>/shortlist16/).
    name = args.results or re.sub(r"[^A-Za-z0-9.-]+", "_", clf.model)
    setup.results = f"{name}/{setup.results}" if setup.results else name
    return setup


def count_answer_overrides(clf):
    """Texts (main run, after warm-up) whose prediction = most probable option, not the written letter."""
    before = clf.answer_overrides
    return lambda: {"answer_overrides": clf.answer_overrides - before}


if __name__ == "__main__":
    main(adapter, app_dir=APP_DIR, doc=__doc__, key_packages=KEY_PACKAGES, add_arguments=add_arguments)
