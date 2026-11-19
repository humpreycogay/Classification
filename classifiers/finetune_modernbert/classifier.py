"""ModernBERT-base fine-tuned on the task's train split (supervised; a pre-trained encoder, not trained from scratch).

The pinned pre-trained backbone (answerdotai/ModernBERT-base at 8949b90) gets a new linear head with one output per label, and every
weight is then trained gently on the task's train split (shared/finetune.py: a stratified 90% to fit, 10% for early
stopping and the temperature). The backbone is downloaded once into models/base/; fine-tuned models are saved in
models/<task>/<digest>/ and loaded on later runs.

Usage (from this folder):
    PYTHONPATH=../../shared .venv/bin/python classifier.py "Stocks rallied after the Fed held rates steady."
    BENCHMARK_TASK=banking77 PYTHONPATH=../../shared .venv/bin/python classifier.py "my card has not arrived yet"
The first call per task fine-tunes (minutes for Banking77, an hour or more for AG News on an M5); later calls load.

Settings (root .env):
    FINETUNE_MODERNBERT_DEVICE    mps | cuda | cpu    (default: mps when available, else cpu)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

from finetune import FineTuneClassifier, FTConfig
from task import current_task, load_split
from textnn import pick_device

APP = "finetune_modernbert"
APP_DIR = Path(__file__).resolve().parent
TASK = current_task()

# The pinned backbone. The same revision modernbert_jev pins.
REPO, REVISION, NAME = "answerdotai/ModernBERT-base", "8949b909ec900327062f0ebf497f51aef5e6f0c8", "ModernBERT-base"
FILES = ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]
PARAMETERS = (140e6, 160e6)  # expected parameter count with a head (checks.py)

# One recipe per task. Early stopping picks the epoch; --epochs/--lr/... override (and train a separate model).
CONFIGS = {
    "ag_news": FTConfig(max_len=160, batch_size=32, lr=3e-5, max_epochs=3, patience=1),
    "banking77": FTConfig(max_len=64, batch_size=32, lr=5e-5, max_epochs=15, patience=3),
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
                         device or pick_device("FINETUNE_MODERNBERT_DEVICE"), description=NAME)


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
