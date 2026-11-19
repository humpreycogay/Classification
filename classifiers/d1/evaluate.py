"""Evaluate LiquidAI d1-3B on a benchmark task (default AG News; --task NAME).

Usage:
    python evaluate.py
    python evaluate.py --task banking77 --n 1000 --seed 42
    python evaluate.py --batch-size 1
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
SHARED_DIR = Path(os.environ.get("BENCHMARK_SHARED_DIR") or APP_DIR.parents[1] / "shared")
if not (SHARED_DIR / "harness.py").is_file():
    sys.exit(f"Shared benchmark library not found in {SHARED_DIR}; set BENCHMARK_SHARED_DIR")
sys.path.insert(0, str(SHARED_DIR))

from task import select_task_from_argv

select_task_from_argv()
from classifier import MAX_CONTEXT_TOKENS, MODEL_REPO, MODEL_REVISION, TASK_LABELS, D1Classifier
from harness import Setup, main

KEY_PACKAGES = ["torch", "transformers", "torchvision", "pillow", "httpx", "pandas"]


def add_arguments(parser) -> None:
    parser.add_argument("--batch-size", type=int, default=1, help="states packed into each forward pass")


def adapter(args, ctx) -> Setup:
    clf = D1Classifier(TASK_LABELS, ctx.task.instruction)
    return Setup(
        model=f"{MODEL_REPO}@{MODEL_REVISION[:7]} on {clf.device}", classifier=clf,
        axis="batch_size", value=args.batch_size,
        predict=lambda texts, v=args.batch_size: clf.predict_many(texts, batch_size=v), model_repo=MODEL_REPO,
        metadata={"d1": {
            "model": MODEL_REPO, "revision": MODEL_REVISION, "device": clf.device, "dtype": str(clf.dtype),
            "context_window": MAX_CONTEXT_TOKENS, "trust_remote_code": True,
            "scoring": "native system_one_batch choice scoring; one forward pass and zero output tokens",
            "option_template": "choice name mapped to task label id; description supplied as criteria",
            "batching": "native packed requests with no padding", "confidence": "max native probability",
            "training_provenance": "post-trained for decisions; no published AG News or Banking77 training",
        }})


if __name__ == "__main__":
    main(adapter, app_dir=APP_DIR, doc=__doc__, key_packages=KEY_PACKAGES, add_arguments=add_arguments)