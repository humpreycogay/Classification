"""LiquidAI d1-3B classifier using the checkpoint's native System One API.

d1 is a decision model, not a chat model. Each text and one choice question are
scored in one forward pass with no generated tokens. The returned native
probability distribution is used unchanged; confidence is its maximum value.

Environment:
    D1_DEVICE  cuda | mps | cpu (default: cuda, then mps, then cpu)
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, os.environ.get("BENCHMARK_SHARED_DIR") or str(Path(__file__).resolve().parents[2] / "shared"))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import torch  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from task import current_task  # noqa: E402

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

MODEL_REPO = "LiquidAI/d1-3B"
MODEL_REVISION = "05b80faf55536582ad5ffb2da7456f6a653acf71"  # 2026-11-12 model-card revision
MAX_CONTEXT_TOKENS = 32768

TASK = current_task()
TASK_LABELS: dict[str, str] = {spec["id"]: spec["description"] for spec in TASK.labels}
LABEL_NAMES: dict[str, str] = {spec["id"]: spec["name"] for spec in TASK.labels}


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


@dataclass
class Prediction:
    label: str
    confidence: float
    probabilities: dict[str, float]


class D1Classifier:
    """Single-label classifier backed by d1's native packed choice scorer."""

    def __init__(self, labels: dict[str, str], instruction: str, device: str | None = None, model=None) -> None:
        if len(labels) < 2:
            raise ValueError("d1 requires at least two options")
        self.labels = labels
        self.device = device or os.getenv("D1_DEVICE") or default_device()
        if self.device not in {"cuda", "mps", "cpu"}:
            raise ValueError("D1_DEVICE must be cuda, mps or cpu")
        self.dtype = torch.float32 if self.device == "cpu" else torch.bfloat16

        # Choice keys are returned verbatim by the native API. Use human-readable
        # names in the prompt and map them back to stable task ids.
        self.label_of = {LABEL_NAMES.get(label, label): label for label in labels}
        if len(self.label_of) != len(labels):
            raise ValueError("label names must be unique: d1 choice options are keyed by name")
        self.question = {"label": {
            "type": "choice",
            "instructions": instruction,
            "criteria": {LABEL_NAMES.get(label, label): description for label, description in labels.items()},
        }}

        if model is None:
            from transformers import AutoModel

            model = AutoModel.from_pretrained(
                MODEL_REPO, revision=MODEL_REVISION, trust_remote_code=True, dtype=self.dtype
            ).to(self.device)
        self.model = model

    def _prediction(self, response: dict) -> Prediction:
        answer = response["answers"]["label"]
        native = answer["probabilities"]
        if set(native) != set(self.label_of):
            raise RuntimeError(f"d1 returned unexpected choice names: {sorted(native)}")
        probabilities = {self.label_of[name]: float(value) for name, value in native.items()}
        label = max(probabilities, key=probabilities.get)
        return Prediction(label, probabilities[label], probabilities)

    def predict(self, text: str) -> Prediction:
        return self._prediction(self.model.system_one(text, self.question))

    def predict_many(self, texts: list[str], batch_size: int = 1) -> list[Prediction]:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        predictions: list[Prediction] = []
        for start in range(0, len(texts), batch_size):
            requests = [(text, self.question) for text in texts[start:start + batch_size]]
            predictions.extend(self._prediction(response) for response in self.model.system_one_batch(requests))
        return predictions


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    pred = D1Classifier(TASK_LABELS, TASK.instruction).predict(" ".join(sys.argv[1:]))
    print(f"label:      {pred.label}")
    print(f"confidence: {pred.confidence:.3f}")
    for label, probability in sorted(pred.probabilities.items(), key=lambda item: -item[1]):
        print(f"  {label:<24} {probability:.4f}")


if __name__ == "__main__":
    main()