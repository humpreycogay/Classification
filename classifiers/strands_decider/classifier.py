"""Strands Decider Classifier: classification with Strands Labs' strands-decider-2B-hobson-v19, through strands-decider.

Not the `decider` app (Mapika's Decider-4B): a different model, code and weights. Strands Decider is a one-pass
decision model: Qwen3.5-2B-Base with a rank-16 LoRA adapter and a pointer head in place of the LM head. The prompt
is "<state>text</state>", then the question with its numbered options "1. <name> — <description>" and an
"<answer>" marker; the head scores each option from the hidden state at the option's last token, the scores are
divided by the temperature and softmaxed over the options. Nothing is generated.

One `choice` question per text, with criteria {label name: label description} (the form of the model's own AG News
training recipe). Probabilities are read for every option (the engine rounds them to 4 decimals); confidence = max p,
not the engine's rescaled (N * max p - 1) / (N - 1). The temperature is the release's value for choice answers
(hobson_config.json temperature_by_kind["choice"] = 0.734); the engine applies it.

The model was trained on the train splits of AG News and Banking77 (data/sources.md of its repository), so on
those tasks it is in-domain trained, not zero-shot.

On a Mac the torso runs on MLX (mlx-lm, bfloat16, the adapter merged into the weights at load), the fastest of the
package's devices there; the head runs in float32 on the CPU. `mps` and `cpu` are the package's PyTorch paths; CPU
runs float32 and is the reference of the setup-time parity check. A prompt longer than the 4,096-token window
raises instead of being cut (strict_window).

Usage:
    python classifier.py "Stocks rallied after the Fed held rates steady."

Environment (optional, read from the root .env):
    STRANDS_DECIDER_DEVICE   mlx | mps | cuda | cpu   (default: mlx on Apple silicon, else cuda, else mps, else cpu)
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, replace
from pathlib import Path

# The shared benchmark library (task.py, ...), so this file also runs on its own: BENCHMARK_SHARED_DIR (the
# runner sets it), else the repo's shared/ folder (classifiers/<app>/ -> two levels up).
sys.path.insert(0, os.environ.get("BENCHMARK_SHARED_DIR") or str(Path(__file__).resolve().parents[2] / "shared"))

# Xet transfers stall on some networks; plain HTTPS downloads are reliable.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import torch  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from huggingface_hub import HfApi, snapshot_download  # noqa: E402

from task import current_task  # noqa: E402

# .env lives in the repo root (classifiers/<app>/ -> two levels up).
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

MODEL_REPO = "StrandsAgents/strands-decider-2B-hobson-v19"
MODEL_REVISION = "bb282d786bc251fd4e3068de3ada9ddbb38127cd"  # 2026-10-01
BASE_REPO = "Qwen/Qwen3.5-2B-Base"
BASE_REVISION = "b1485b2fa6dfa1287294f269f5fb618e03d52d7c"  # the revision in the model's provenance.json
CODE_COMMIT = "890947e7ccd44c3de4115e26a7f46cc5c3147b44"  # strands-decider, as requirements.txt (2026-10-02)
MAX_OPTIONS = 255

# Labels, descriptions and instruction come from the task manifest (data/tasks/<task>.json, see task.py),
# chosen with --task on evaluate.py (default: ag_news).
TASK = current_task()
_LABEL_SPECS = TASK.labels
CLASS_TO_LABEL = {spec["raw"]: spec["id"] for spec in _LABEL_SPECS}
TASK_LABELS: dict[str, str] = {spec["id"]: spec["description"] for spec in _LABEL_SPECS}
LABEL_NAMES = {spec["id"]: spec["name"] for spec in _LABEL_SPECS}


def model_dir() -> str:
    return snapshot_download(MODEL_REPO, revision=MODEL_REVISION)


def base_dir() -> str:
    """The base model as strands-decider's loader gets it. The loader passes no revision, so it loads the Hub's
    current `main`: refuse to go on unless that is the pinned commit, so a change to the base weights cannot
    change our results unnoticed."""
    try:
        main = HfApi().model_info(BASE_REPO).sha
    except Exception:  # offline: the cached `main` is checked below
        main = BASE_REVISION
    path = snapshot_download(BASE_REPO) if main == BASE_REVISION else None
    if path is None or Path(path).name != BASE_REVISION:
        raise RuntimeError(f"{BASE_REPO} main is {main if path is None else Path(path).name}, not the pinned "
                           f"{BASE_REVISION}: check the change, then re-pin BASE_REVISION and rerun setup.sh")
    return path


def default_device() -> str:
    from strands_decider.infer import mlx_available

    if mlx_available():
        return "mlx"
    return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


@dataclass
class Prediction:
    label: str
    confidence: float
    probabilities: dict[str, float]


class StrandsDeciderClassifier:
    """Single-label text classifier: one Strands Decider choice question per text."""

    def __init__(self, labels: dict[str, str], instruction: str, device: str | None = None) -> None:
        if not 2 <= len(labels) <= MAX_OPTIONS:
            raise ValueError(f"Strands Decider supports 2-{MAX_OPTIONS} options")
        from strands_decider.infer import load_engine
        from strands_decider.schema import ChoiceQuestion

        self.labels = labels
        # Options are keyed by the label's name; the answer is mapped back to the label id.
        self.label_of = {LABEL_NAMES.get(l, l): l for l in labels}
        if len(self.label_of) != len(labels):
            raise ValueError("label names must be unique: the model's options are keyed by name")
        self.question = {"label": ChoiceQuestion(
            instructions=instruction, criteria={LABEL_NAMES.get(l, l): d for l, d in labels.items()})}
        self.device = device or os.getenv("STRANDS_DECIDER_DEVICE") or default_device()
        self.path = model_dir()
        self.base_path = base_dir()
        self.engine = load_engine(self.path, device=self.device)
        self.engine.cfg = replace(self.engine.cfg, strict_window=True)
        config = self.engine.model.config
        # The torch CPU path runs the torso in float32 (the package upcasts it); MLX and MPS keep the checkpoint's.
        self.dtype = "float32" if self.device == "cpu" else config.torch_dtype
        self.max_length = config.max_length
        self.temperature = float((config.temperature_by_kind or {}).get("choice", config.temperature))

    def predict(self, text: str) -> Prediction:
        answer = self.engine.ask(text, self.question).answers["label"]
        probabilities = {self.label_of[name]: p for name, p in answer.probabilities.items()}
        label = max(probabilities, key=probabilities.get)
        # Confidence = max p, the probability of the chosen label.
        return Prediction(label, probabilities[label], probabilities)

    def predict_many(self, texts: list[str]) -> list[Prediction]:
        # The engine takes one state per call (its batching is across the questions of one state).
        return [self.predict(t) for t in texts]


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    text = " ".join(sys.argv[1:])
    pred = StrandsDeciderClassifier(TASK_LABELS, TASK.instruction).predict(text)
    print(f"label:      {pred.label}")
    print(f"confidence: {pred.confidence:.2f}")
    for label, p in sorted(pred.probabilities.items(), key=lambda kv: -kv[1]):
        print(f"  {label:<10} {p:.3f}")


if __name__ == "__main__":
    main()
