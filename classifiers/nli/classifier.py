"""NLI Classifier: zero-shot text classification with a natural-language-inference model.

Each label becomes a hypothesis ("The topic of this news article is Business (...)"),
and an NLI model (default MoritzLaurer/deberta-v3-large-zeroshot-v2.0; the nli_modernbert entry runs
MoritzLaurer/ModernBERT-large-zeroshot-v2.0) scores how strongly the text entails it.
The entailment scores are softmaxed across labels (single-label mode). Runs locally on
the Mac GPU via PyTorch; no API key needed.

Texts are classified longest first in length-sorted batches (shared/batching.py): less padding, the same
scores, results in input order.

Usage:
    python classifier.py "Stocks rallied after the Fed held rates steady."

Environment (optional, read from the root .env):
    NLI_MODEL    Hugging Face model id   (default: MoritzLaurer/deberta-v3-large-zeroshot-v2.0; --model wins)
    NLI_DEVICE   mps | cuda | cpu        (default: auto-detect)
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

# Xet transfers stall on some networks; plain HTTPS downloads are reliable.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import torch  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from transformers import pipeline  # noqa: E402

from batching import sorted_batches
from task import current_task, load_split

# .env lives in the repo root (classifiers/<app>/ -> two levels up).
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

DEFAULT_MODEL = "MoritzLaurer/deberta-v3-large-zeroshot-v2.0"
# Known models: pinned revision (setup.sh downloads it), results subfolder, and dtype. "auto" = the dtype stored
# in the checkpoint (DeBERTa: float16, as this entry has always run); ModernBERT is stored in bfloat16, which
# moves its probabilities by up to 0.04 against float32, so it runs in float32 (an exact upcast; MPS then matches
# CPU within 1e-4 in logits). Another model runs at its latest revision, "auto" dtype, in results/<its name>/.
MODELS = {
    "MoritzLaurer/deberta-v3-large-zeroshot-v2.0": ("cf44676c28ba7312e5c5f8f8d2c22b3e0c9cdae2", "", "auto"),
    "MoritzLaurer/ModernBERT-large-zeroshot-v2.0": ("a51e07b524299e309dd2b88d48b0cfa2bd9ec598", "modernbert-large",
                                                    "float32"),
}


def results_folder(model: str) -> str:
    """Results subfolder for a model: none for the default (the plain nli entry), else its own."""
    return MODELS[model][1] if model in MODELS else model.split("/")[-1]


# Labels, descriptions and instruction come from the task manifest (data/tasks/<task>.json, see task.py),
# chosen with --task on evaluate.py (default: ag_news).
TASK = current_task()
_LABEL_SPECS = TASK.labels
HYPOTHESIS_TEMPLATE = TASK.hypothesis_template  # e.g. "The topic of this news article is {}."
CLASS_TO_LABEL = {spec["raw"]: spec["id"] for spec in _LABEL_SPECS}
TASK_LABELS: dict[str, str] = {spec["id"]: spec["description"] for spec in _LABEL_SPECS}
LABEL_NAMES = {spec["id"]: spec["name"] for spec in _LABEL_SPECS}


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class Prediction:
    label: str
    confidence: float
    probabilities: dict[str, float]


class NLIClassifier:
    """Single-label text classifier backed by an NLI zero-shot model."""

    def __init__(
        self,
        labels: dict[str, str],
        model: str | None = None,
        device: str | None = None,
        dtype: str | None = None,  # "auto" | "float32" | ...; default: the model's entry in MODELS
    ) -> None:
        self.labels = labels
        # Candidate text per label, e.g. "Business (Companies, markets, ...)".
        self.candidates = {
            f"{LABEL_NAMES.get(label, label)} ({description.lower()})": label for label, description in labels.items()
        }
        self.model = model or os.getenv("NLI_MODEL") or DEFAULT_MODEL
        self.revision, _, default_dtype = MODELS.get(self.model, (None, "", "auto"))
        self.dtype = dtype or default_dtype
        self.device = device or os.getenv("NLI_DEVICE") or default_device()
        self.pipe = pipeline("zero-shot-classification", model=self.model, revision=self.revision, device=self.device,
                             dtype=self.dtype if self.dtype == "auto" else getattr(torch, self.dtype))

    def _to_prediction(self, result: dict) -> Prediction:
        probabilities = {self.candidates[c]: s for c, s in zip(result["labels"], result["scores"])}
        label = max(probabilities, key=probabilities.get)
        # Confidence = max p, the probability of the chosen label.
        return Prediction(label, probabilities[label], probabilities)

    def _classify(self, texts: list[str], batch_size: int) -> list[Prediction]:
        results = self.pipe(
            texts,
            candidate_labels=list(self.candidates),
            hypothesis_template=HYPOTHESIS_TEMPLATE,
            multi_label=False,
            batch_size=batch_size,
        )
        return [self._to_prediction(r) for r in results]

    def predict_many(self, texts: list[str], batch_size: int = 16) -> list[Prediction]:
        # Length-sorted chunks of batch_size texts; the pipeline batches their (text, hypothesis) pairs.
        return sorted_batches(texts, batch_size, lambda chunk: self._classify(chunk, batch_size))

    def predict(self, text: str) -> Prediction:
        return self.predict_many([text])[0]


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    text = " ".join(sys.argv[1:])
    pred = NLIClassifier(TASK_LABELS).predict(text)
    print(f"label:      {pred.label}")
    print(f"confidence: {pred.confidence:.2f}")
    for label, p in sorted(pred.probabilities.items(), key=lambda kv: -kv[1]):
        print(f"  {label:<10} {p:.3f}")


if __name__ == "__main__":
    main()
