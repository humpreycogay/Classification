"""BiLSTM trained from scratch on the task's train split (supervised; no pre-trained weights).

Reads the text forwards and backwards with an LSTM; the final states of both directions plus a max over every step
feed one linear layer that scores the labels. Training, calibration and the model cache are in shared/textnn.py.
The trained model is saved in models/<task>/ and loaded on later runs.

Usage (from this folder):
    PYTHONPATH=../../shared .venv/bin/python classifier.py "Stocks rallied after the Fed held rates steady."
    BENCHMARK_TASK=banking77 PYTHONPATH=../../shared .venv/bin/python classifier.py "my card has not arrived yet"
The first call per task trains (minutes for AG News) and saves the model; later calls load it.

Settings (root .env):
    SCRATCH_BILSTM_DEVICE    cpu | mps | cuda | auto    (default: cpu; measured: one AG News epoch 102 s on CPU,
                                                        109 s on MPS, where packed LSTMs gain nothing)
"""

from __future__ import annotations

import sys
from pathlib import Path

from model import BiLSTM
from task import current_task, load_split
from textnn import Config, ScratchClassifier, pick_device

APP = "scratch_bilstm"
APP_DIR = Path(__file__).resolve().parent
TASK = current_task()

# One fixed recipe per task, set before any test run. Early stopping on a validation split picks the epoch.
CONFIGS = {
    "ag_news": Config(max_len=100, min_freq=2, batch_size=64, lr=1e-3, max_epochs=6, patience=2,
                      model={"dim": 128, "hidden": 128, "layers": 1, "dropout": 0.3}),
    "banking77": Config(max_len=48, min_freq=1, batch_size=32, lr=2e-3, max_epochs=40, patience=5,
                        model={"dim": 128, "hidden": 128, "layers": 1, "dropout": 0.3}),
}


class ScratchBiLSTMClassifier(ScratchClassifier):
    def __init__(self, cfg: Config | None = None, device: str | None = None) -> None:
        super().__init__(APP, BiLSTM, cfg or CONFIGS[TASK.name], TASK.label_ids,
                         device or pick_device("SCRATCH_BILSTM_DEVICE", default="cpu"),
                         description="BiLSTM (1 layer, 128 per direction, last states + max-pooling)")


def trained(cfg: Config | None = None, device: str | None = None) -> ScratchBiLSTMClassifier:
    """The classifier for the current task, trained on its train split or loaded from models/<task>/."""
    clf = ScratchBiLSTMClassifier(cfg, device)
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
