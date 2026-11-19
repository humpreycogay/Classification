"""LiquidAI d1-omni-600M text classifier using its native System One API."""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, os.environ.get("BENCHMARK_SHARED_DIR") or str(Path(__file__).resolve().parents[2] / "shared"))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import torch  # noqa: E402
from task import current_task  # noqa: E402

MODEL_REPO = "LiquidAI/d1-omni-600M"
MODEL_REVISION = "02b55d7076f15129e59ab3f94783f32c4b088674"
MAX_CONTEXT_TOKENS = 16384
TASK = current_task()
TASK_LABELS = {s["id"]: s["description"] for s in TASK.labels}
LABEL_NAMES = {s["id"]: s["name"] for s in TASK.labels}


@dataclass
class Prediction:
    label: str
    confidence: float
    probabilities: dict[str, float]


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


class D1OmniClassifier:
    def __init__(self, labels: dict[str, str], instruction: str, device: str | None = None, model=None) -> None:
        if len(labels) < 2:
            raise ValueError("d1-omni requires at least two options")
        self.device = device or os.getenv("D1_OMNI_DEVICE") or default_device()
        if self.device not in {"cuda", "mps", "cpu"}:
            raise ValueError("D1_OMNI_DEVICE must be cuda, mps or cpu")
        self.dtype = torch.float32  # the released checkpoint is FP32
        self.label_of = {LABEL_NAMES.get(label, label): label for label in labels}
        if len(self.label_of) != len(labels):
            raise ValueError("label names must be unique")
        self.question = {"label": {"type": "choice", "instructions": instruction,
                                    "criteria": {LABEL_NAMES.get(k, k): v for k, v in labels.items()}}}
        if model is None:
            from transformers import AutoModel
            model = AutoModel.from_pretrained(MODEL_REPO, revision=MODEL_REVISION,
                                              trust_remote_code=True, dtype=self.dtype).to(self.device)
        self.model = model

    def _prediction(self, response: dict) -> Prediction:
        native = response["answers"]["label"]["probabilities"]
        if set(native) != set(self.label_of):
            raise RuntimeError(f"d1-omni returned unexpected choice names: {sorted(native)}")
        probabilities = {self.label_of[k]: float(v) for k, v in native.items()}
        label = max(probabilities, key=probabilities.get)
        return Prediction(label, probabilities[label], probabilities)

    def predict(self, text: str) -> Prediction:
        return self._prediction(self.model.system_one(text, self.question))

    def predict_many(self, texts: list[str], batch_size: int = 16) -> list[Prediction]:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        out = []
        for start in range(0, len(texts), batch_size):
            requests = [(text, self.question) for text in texts[start:start + batch_size]]
            out.extend(map(self._prediction, self.model.system_one_batch(requests)))
        return out


if __name__ == "__main__":
    p = D1OmniClassifier(TASK_LABELS, TASK.instruction).predict(" ".join(sys.argv[1:]))
    print(p)