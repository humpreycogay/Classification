"""Evaluate EmbedSupClassifier on a benchmark task (default AG News; --task NAME, see data/tasks/).

Dataset: Zhang, Zhao & LeCun (2015), "Character-level Convolutional Networks for
Text Classification". Classes: 1=World, 2=Sports, 3=Business, 4=Sci/Tech.

The classifier is trained on embeddings of a stratified sample of the task's train
split; test labels are used only for scoring. Embedding the train sample, fitting the classifier
and classifying the test articles are timed separately.

Usage:
    python evaluate.py                          # centroid method, 200 test articles
    python evaluate.py --method logreg          # logistic regression on the embeddings
    python evaluate.py --train-per-class 2000   # more training articles per label
    python evaluate.py --n 0                    # full test set (7,600 articles)

Results go to results/<method>/, so the two methods never overwrite each other.
"""

from __future__ import annotations

import os
import sys
import time
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
from classifier import GRID_C, METHODS, EmbedSupClassifier, load_train_sample, make_logreg
from harness import Setup, main
from training import select

# Versions recorded in the run metadata (only those installed in this app's .venv).
KEY_PACKAGES = ["scikit-learn", "numpy", "httpx", "pandas"]


def add_arguments(parser) -> None:
    parser.add_argument("--method", choices=list(METHODS), default="centroid")
    parser.add_argument("--train-per-class", default="task",
                        help="train articles embedded per label ('task' = the manifest's supervised_sample_per_label)")
    parser.add_argument("--train-seed", type=int, default=0, help="seed of the train sample (not the test sample)")
    parser.add_argument("--batch-size", type=int, default=32, help="texts per embedding request")


def adapter(args, ctx) -> Setup:
    # Load = client setup and reading the train sample; embedding and fitting are timed separately.
    clf = EmbedSupClassifier(args.method)
    clf.check_ready()
    per_class = ctx.task.supervised_sample_per_label if args.train_per_class == "task" else int(args.train_per_class)
    train = load_train_sample(per_class, args.train_seed)
    labels = train["label"].tolist()

    # Warm-up on one full batch, timed separately: loads the model in Ollama before any timed embedding.
    warm = ctx.sample["text"].head(args.batch_size).tolist()
    ctx.warm_up(lambda: clf.embed(warm), len(warm))

    with ctx.exclude():
        print(f"Embedding {len(train)} {ctx.task.name} train articles with {clf.model} ...")
        vectors, train_embed_seconds, cached = clf.embed_train(train, args.train_seed, args.batch_size)
        print(f"Train embeddings {'loaded from cache' if cached else 'computed'} ({train_embed_seconds:.1f}s to embed)")

        selection = None
        C = 10.0
        if args.method == "logreg":  # C chosen by CV on the train embeddings only (cached in hyperparameters.json)
            variant = f"logreg_{clf.model}_{clf.digest[:12]}_train{len(train)}_seed{args.train_seed}"
            selection = select(make_logreg, GRID_C, vectors, labels, param="C",
                               store=ctx.app_dir / "hyperparameters.json", variant=f"{ctx.task.name}:{variant}")
            C = selection["chosen"]

        train_start = time.perf_counter()
        clf.fit(vectors, labels, seed=args.train_seed, C=C)
        train_seconds = time.perf_counter() - train_start
        print(f"Fitted {args.method} in {train_seconds:.2f}s")

    training = {
        "supervision": "supervised",
        "train_split": f"{ctx.task.name} train, stratified {per_class} per label, seed {args.train_seed}",
        "n_train": len(train),
        "train_embed_seconds": round(train_embed_seconds, 3),
        "train_embed_cached": cached,
        "train_seconds": round(train_seconds, 3),
        "hyperparameter_selection": selection,
        # Articles whose text also appears verbatim in the train split (the corpus has a few duplicates).
        "test_texts_in_train": int(ctx.sample["text"].isin(set(train["text"])).sum()),
        "probabilities": "temperature-scaled on held-out train" if args.method == "centroid" else "native",
        "calibration": METHODS[args.method],
        "hyperparameters": clf.hyperparameters,
    }
    # Classification = embedding the test articles + applying the fitted classifier.
    return Setup(model=f"{clf.model} embeddings via Ollama at {clf.host} + {METHODS[args.method]}", classifier=clf,
                 axis="batch_size", value=args.batch_size,
                 predict=lambda texts, v=args.batch_size: clf.predict_many(texts, batch_size=v),
                 results=args.method, metadata={"training": training})


if __name__ == "__main__":
    main(adapter, app_dir=APP_DIR, doc=__doc__, key_packages=KEY_PACKAGES, add_arguments=add_arguments)
