"""DeBERTa-v3-base fine-tuned on the task's train split (supervised; a pre-trained encoder, not trained from scratch).

The pinned pre-trained backbone (microsoft/deberta-v3-base at de19fe7) gets a new linear head with one output per label, and every
weight is then trained gently on the task's train split (shared/finetune.py: a stratified 90% to fit, 10% for early
stopping and the temperature). The backbone is downloaded once into models/base/; fine-tuned models are saved in
models/<task>/<digest>/ and loaded on later runs.

Usage (from this folder):
    PYTHONPATH=../../shared .venv/bin/python classifier.py "Stocks rallied after the Fed held rates steady."
    BENCHMARK_TASK=banking77 PYTHONPATH=../../shared .venv/bin/python classifier.py "my card has not arrived yet"
The first call per task fine-tunes (minutes for Banking77, an hour or more for AG News on an M5); later calls load.

Settings (root .env):
    FINETUNE_DEBERTA_DEVICE    mps | cuda | cpu    (default: mps when available, else cpu)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

from finetune import FineTuneClassifier, FTConfig
from task import current_task, load_split
from textnn import pick_device

APP = "finetune_deberta"
APP_DIR = Path(__file__).resolve().parent
TASK = current_task()

# The pinned backbone. refs/pr/14: Hugging Face's SFconvertbot safetensors conversion of the
# original weights (model.safetensors sha256 57cbd0ca...; the same file as in refs/pr/4). The main revision
# (8ccc9b6) ships only a pickle (pytorch_model.bin), which this app never downloads.
REPO, REVISION, NAME = "microsoft/deberta-v3-base", "de19fe7db5162df5f3d8f0b41321c0267288fd74", "DeBERTa-v3-base"
FILES = ["config.json", "model.safetensors", "spm.model", "tokenizer_config.json"]
PARAMETERS = (180e6, 190e6)  # expected parameter count with a head (checks.py)

# One recipe per task. Early stopping picks the epoch; --epochs/--lr/... override (and train a separate model).
CONFIGS = {
    "ag_news": FTConfig(max_len=160, batch_size=32, lr=2e-5, max_epochs=3, patience=1),
    "banking77": FTConfig(max_len=64, batch_size=32, lr=4e-5, max_epochs=15, patience=3),
}


def base_dir() -> Path:
    """The pinned backbone in models/base/<name>/, downloaded on first use (safetensors only, never pickles)."""
    target = APP_DIR / "models" / "base" / NAME
    if not all((target / f).exists() for f in FILES):
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        snapshot_download(REPO, revision=REVISION, local_dir=str(target), allow_patterns=FILES)
    return target


class FineTunedEncoder(FineTuneClassifier):
    def __init__(self, cfg: FTConfig | None = None, device: str | None = None) -> None:
        super().__init__(APP, base_dir(), f"{REPO}@{REVISION}", cfg or CONFIGS[TASK.name], TASK.label_ids,
                         device or pick_device("FINETUNE_DEBERTA_DEVICE"), description=NAME)


def trained(cfg: FTConfig | None = None, device: str | None = None) -> FineTunedEncoder:
    """The classifier for the current task, fine-tuned on its train split or loaded from models/<task>/."""
    clf = FineTunedEncoder(cfg, device)
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
