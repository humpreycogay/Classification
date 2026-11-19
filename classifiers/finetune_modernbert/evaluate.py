"""Evaluate ModernBERT-base fine-tuned on a benchmark task (default AG News; --task NAME, see data/tasks/).

Fine-tunes on the task's train split (a stratified 90% to fit, 10% for early stopping and the temperature), or
loads the identical earlier model from models/<task>/. Test labels are used only for scoring. Fine-tuning time is
recorded in the run's "training" metadata and never counted as load or classification time.

Usage:
    python evaluate.py --task banking77 --n 1000
    python evaluate.py --train-limit 40000     # a quick AG News pilot: its own model and results/train40000/
    python evaluate.py --train-seed 1          # another seed: its own model and results/seed1/
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
# The shared benchmark library (task.py, harness.py, finetune.py, ...): BENCHMARK_SHARED_DIR (the runner sets it),
# else the repo's shared/ folder (classifiers/<app>/ -> two levels up).
SHARED_DIR = Path(os.environ.get("BENCHMARK_SHARED_DIR") or APP_DIR.parents[1] / "shared")
if not (SHARED_DIR / "harness.py").is_file():
    sys.exit(f"Shared benchmark library not found in {SHARED_DIR}; set BENCHMARK_SHARED_DIR to the repo's shared/ folder")
sys.path.insert(0, str(SHARED_DIR))

from task import select_task_from_argv

select_task_from_argv()  # --task NAME -> BENCHMARK_TASK; must run before classifier.py reads the task
from classifier import CONFIGS, FineTunedEncoder
from finetune import add_finetune_arguments, config_from_args
from harness import Setup, main
from task import load_split

# Versions recorded in the run metadata (only those installed in this app's .venv).
KEY_PACKAGES = ["torch", "transformers", "tokenizers", "huggingface_hub", "numpy", "pandas"]


def adapter(args, ctx) -> Setup:
    cfg = config_from_args(CONFIGS[ctx.task.name], args)
    clf = FineTunedEncoder(cfg, args.device)
    with ctx.exclude():  # fine-tuning is not load time
        train = load_split(ctx.task, "train")
        training = clf.fit_cached(train, APP_DIR / "models" / ctx.task.name)
        training["train_split"] = f"{ctx.task.name} train (all; stratified 90% fit, 10% validation)"
        training["n_train"] = len(train)
        # Test texts that also appear verbatim in the train split (the corpus has a few duplicates).
        training["test_texts_in_train"] = int(ctx.sample["text"].isin(set(train["text"])).sum())
    ctx.add_load(training.pop("_load_seconds"))  # loading a saved model is load time
    # Non-default training choices get their own results folder, so they never overwrite the registered result.
    variant = "/".join(([f"seed{cfg.seed}"] if cfg.seed else []) + ([f"train{cfg.train_limit}"] if cfg.train_limit else []))
    return Setup(model=clf.model_description, classifier=clf, axis="batch_size", value=args.batch_size,
                 predict=lambda texts, v=args.batch_size: clf.predict_many(texts, batch_size=v),
                 results=variant, metadata={"training": training})


if __name__ == "__main__":
    main(adapter, app_dir=APP_DIR, doc=__doc__, key_packages=KEY_PACKAGES, add_arguments=add_finetune_arguments)
