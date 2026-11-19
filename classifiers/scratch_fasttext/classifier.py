"""fastText-style classifier trained from scratch on the task's train split (supervised; no pre-trained weights).

The text vector is the mean of learned word vectors and hashed word-pair (bigram) vectors; one linear layer scores
the labels. Training, calibration and the model cache are in shared/textnn.py. The trained model is saved in
models/<task>/ and loaded on later runs.

Usage (from this folder):
    PYTHONPATH=../../shared .venv/bin/python classifier.py "Stocks rallied after the Fed held rates steady."
    BENCHMARK_TASK=banking77 PYTHONPATH=../../shared .venv/bin/python classifier.py "my card has not arrived yet"
The first call per task trains (a few minutes for AG News) and saves the model; later calls load it.

Settings (root .env):
    SCRATCH_FASTTEXT_DEVICE    cpu | mps | cuda | auto    (default: cpu, faster than MPS for this tiny model)
"""

from __future__ import annotations

import sys
from pathlib import Path

from model import FastText
from task import current_task, load_split
from textnn import Config, ScratchClassifier, pick_device

APP = "scratch_fasttext"
APP_DIR = Path(__file__).resolve().parent
TASK = current_task()

# One fixed recipe per task, set before any test run. Early stopping on a validation split picks the epoch.
CONFIGS = {
    "ag_news": Config(max_len=100, min_freq=2, bigrams=200_000, batch_size=128, lr=5e-3, max_epochs=8, patience=2,
                      model={"dim": 64, "dropout": 0.0}),
    "banking77": Config(max_len=48, min_freq=1, bigrams=50_000, batch_size=32, lr=5e-3, max_epochs=60, patience=6,
                        model={"dim": 64, "dropout": 0.0}),
}


class ScratchFastTextClassifier(ScratchClassifier):
    def __init__(self, cfg: Config | None = None, device: str | None = None) -> None:
        super().__init__(APP, FastText, cfg or CONFIGS[TASK.name], TASK.label_ids,
                         device or pick_device("SCRATCH_FASTTEXT_DEVICE", default="cpu"),
                         description="fastText-style mean of word + bigram embeddings")


def trained(cfg: Config | None = None, device: str | None = None) -> ScratchFastTextClassifier:
    """The classifier for the current task, trained on its train split or loaded from models/<task>/."""
    clf = ScratchFastTextClassifier(cfg, device)
    clf.fit_cached(load_split(TASK, "train"), APP_DIR / "models" / TASK.name)
    return clf


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    pred = trained().predict(" ".join(sys.argv[1:]))
    print(f"label:      {pred.label}")
    print(f"confidence: {pred.confidence:.2f}")
    for label, p in sorted(pred.probabilities.items(), key=lambda kv: -kv[1])[:5]:
        print(f"  {label:<24} {p:.3f}")


if __name__ == "__main__":
    main()
