"""GLiClass Classifier: zero-shot text classification with Knowledgator's GLiClass.

GLiClass encodes the text and all candidate labels together in one pass and scores
every label at once. Model: knowledgator/gliclass-base-v3.0 (187M, Apache-2.0).
Runs locally on the Mac GPU via PyTorch; no API key needed.

Probabilities are GLiClass's single-label distribution (softmax over label logits).
The pipeline's single-label mode returns only the top label, so labels are scored in
multi-label mode (sigmoid of each logit, threshold 0), the logits are recovered from
the sigmoids, and the softmax is applied here.

Usage:
    python classifier.py "Stocks rallied after the Fed held rates steady."

Environment (optional, read from the root .env):
    GLICLASS_MODEL    Hugging Face model id   (default: knowledgator/gliclass-base-v3.0)
    GLICLASS_DEVICE   mps | cuda | cpu        (default: auto-detect)
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

# Xet transfers stall on some networks; plain HTTPS downloads are reliable.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import torch  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from gliclass import GLiClassModel, ZeroShotClassificationPipeline  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from task import current_task, load_split

# .env lives in the repo root (classifiers/<app>/ -> two levels up).
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

DEFAULT_MODEL = "knowledgator/gliclass-base-v3.0"

# Labels, descriptions and instruction come from the task manifest (data/tasks/<task>.json, see task.py),
# chosen with --task on evaluate.py (default: ag_news).
TASK = current_task()
_LABEL_SPECS = TASK.labels
CLASS_TO_LABEL = {spec["raw"]: spec["id"] for spec in _LABEL_SPECS}
TASK_LABELS: dict[str, str] = {spec["id"]: spec["description"] for spec in _LABEL_SPECS}
LABEL_NAMES = {spec["id"]: spec["name"] for spec in _LABEL_SPECS}


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def logit(p: float) -> float:
    p = min(max(p, 1e-7), 1 - 1e-7)
    return math.log(p / (1 - p))


def softmax(values: list[float]) -> list[float]:
    top = max(values)
    exps = [math.exp(v - top) for v in values]
    total = sum(exps)
    return [e / total for e in exps]


@dataclass
class Prediction:
    label: str
    confidence: float
    probabilities: dict[str, float]


class GLiClassClassifier:
    """Single-label text classifier backed by a GLiClass zero-shot model."""

    def __init__(
        self,
        labels: dict[str, str],
        model: str | None = None,
        device: str | None = None,
    ) -> None:
        self.labels = labels
        # Candidate text per label, e.g. "Business (companies, markets, ...)".
        self.candidates = {
            f"{LABEL_NAMES.get(label, label)} ({description.lower()})": label for label, description in labels.items()
        }
        self.model = model or os.getenv("GLICLASS_MODEL") or DEFAULT_MODEL
        self.device = device or os.getenv("GLICLASS_DEVICE") or default_device()
        self.pipe = ZeroShotClassificationPipeline(
            GLiClassModel.from_pretrained(self.model),
            AutoTokenizer.from_pretrained(self.model, add_prefix_space=True),
            classification_type="multi-label",
            device=self.device,
            progress_bar=False,
        )

    def _to_prediction(self, scores: list[dict]) -> Prediction:
        sigmoid = {self.candidates[s["label"]]: s["score"] for s in scores}
        labels = list(self.labels)
        probs = softmax([logit(sigmoid[l]) for l in labels])
        probabilities = dict(zip(labels, probs))
        label = max(probabilities, key=probabilities.get)
        # Confidence = max p, the probability of the chosen label.
        return Prediction(label, probabilities[label], probabilities)

    def predict_many(self, texts: list[str], batch_size: int = 16) -> list[Prediction]:
        results = self.pipe(texts, list(self.candidates), threshold=0.0, batch_size=batch_size)
        return [self._to_prediction(r) for r in results]

    def predict(self, text: str) -> Prediction:
        return self.predict_many([text])[0]


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    text = " ".join(sys.argv[1:])
    pred = GLiClassClassifier(TASK_LABELS).predict(text)
    print(f"label:      {pred.label}")
    print(f"confidence: {pred.confidence:.2f}")
    for label, p in sorted(pred.probabilities.items(), key=lambda kv: -kv[1]):
        print(f"  {label:<10} {p:.3f}")


if __name__ == "__main__":
    main()
