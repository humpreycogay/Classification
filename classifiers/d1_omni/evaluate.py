"""Evaluate LiquidAI d1-omni-600M (default AG News; --task NAME)."""
import os, sys
from pathlib import Path
APP_DIR = Path(__file__).resolve().parent
SHARED_DIR = Path(os.environ.get("BENCHMARK_SHARED_DIR") or APP_DIR.parents[1] / "shared")
sys.path.insert(0, str(SHARED_DIR))
from task import select_task_from_argv
select_task_from_argv()
from classifier import D1OmniClassifier, MAX_CONTEXT_TOKENS, MODEL_REPO, MODEL_REVISION, TASK_LABELS
from harness import Setup, main

def add_arguments(parser):
    parser.add_argument("--batch-size", type=int, default=16)

def adapter(args, ctx):
    clf = D1OmniClassifier(TASK_LABELS, ctx.task.instruction)
    return Setup(model=f"{MODEL_REPO}@{MODEL_REVISION[:7]} on {clf.device}", classifier=clf,
                 axis="batch_size", value=args.batch_size,
                 predict=lambda texts, v=args.batch_size: clf.predict_many(texts, v), model_repo=MODEL_REPO,
                 metadata={"d1_omni": {"model": MODEL_REPO, "revision": MODEL_REVISION,
                 "device": clf.device, "dtype": str(clf.dtype), "context_window": MAX_CONTEXT_TOKENS,
                 "scoring": "native calibrated choice head; zero output tokens"}})

if __name__ == "__main__":
    main(adapter, app_dir=APP_DIR, doc=__doc__, key_packages=["torch", "transformers", "pandas", "httpx"], add_arguments=add_arguments)