"""SemIf Classifier: local text classification with SemIf (Qwen3.5-4B on Apple Silicon MLX).

SemIf (https://github.com/TheoLeeCJ/SemIf-OpenJev) reads option probabilities straight
from a frozen open model's next-token logits: options are lettered A, B, C, ... and the
model's logits for those letters are softmaxed. No text is generated. Runs locally on
the Mac GPU via MLX; no API key needed.

Each text is sent as the `state` with one question whose options are the class labels.
SemIf shows the model only option descriptions (not ids), so each description starts
with the label's name.

Usage:
    python classifier.py "Stocks rallied after the Fed held rates steady."

Environment (optional, read from the root .env):
    SEMIF_BITS   4 | 8   in-memory quantization (default: unquantized BF16, ~9 GB)
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

# Xet transfers stall on some networks; plain HTTPS downloads are reliable.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from dotenv import load_dotenv  # noqa: E402
from semif_phase1 import mlx_backend  # noqa: E402

from task import current_task, load_split

# .env lives in the repo root (classifiers/<app>/ -> two levels up).
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

# The checkpoint SemIf's own docs and benchmarks pin for direct option scoring.
MODEL = "Qwen/Qwen3.5-4B"
REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"

# Labels, descriptions and instruction come from the task manifest (data/tasks/<task>.json, see task.py),
# chosen with --task on evaluate.py (default: ag_news).
TASK = current_task()
_LABEL_SPECS = TASK.labels
CLASS_TO_LABEL = {spec["raw"]: spec["id"] for spec in _LABEL_SPECS}
TASK_LABELS: dict[str, str] = {spec["id"]: spec["description"] for spec in _LABEL_SPECS}
LABEL_NAMES = {spec["id"]: spec["name"] for spec in _LABEL_SPECS}


@dataclass
class Prediction:
    label: str
    confidence: float
    probabilities: dict[str, float]


class SemIfClassifier:
    """Single-label text classifier backed by SemIf direct option scoring."""

    def __init__(
        self,
        labels: dict[str, str],
        instructions: str = "Which category does this text belong to?",
        bits: int | None = None,
    ) -> None:
        self.labels = labels
        self.instructions = instructions
        env_bits = os.getenv("SEMIF_BITS")
        self.bits = bits or (int(env_bits) if env_bits else None)
        self.model, self.tokenizer, self.metadata = mlx_backend.load_model(MODEL, REVISION, self.bits)

    def _row(self, text: str, row_id: str) -> dict:
        options = [
            {"id": label, "description": f"{LABEL_NAMES.get(label, label)}: {description}"}
            for label, description in self.labels.items()
        ]
        return {"id": row_id, "state": text, "question": self.instructions, "options": options}

    def predict(self, text: str, row_id: str = "0") -> Prediction:
        result = mlx_backend.score(self.model, self.tokenizer, self._row(text, row_id), self.metadata)
        probabilities = dict(zip(result["option_ids"], result["probabilities"]))
        label = max(probabilities, key=probabilities.get)
        # Confidence = max p, the probability of the chosen label. SemIf reports no confidence
        # of its own (and labels its probabilities uncalibrated).
        return Prediction(label, probabilities[label], probabilities)

    def predict_shortlisted(self, texts: list[str], label_sets: list[dict[str, str]], _=None) -> list[Prediction]:
        """Classify each text among its own labels (a shortlist); other labels get probability 0."""
        full, preds = self.labels, []
        try:
            for i, (text, labels) in enumerate(zip(texts, label_sets)):
                self.labels = labels
                preds.append(self.predict(text, str(i)))
        finally:
            self.labels = full
        return preds

    def predict_many(self, texts: list[str]) -> list[Prediction]:
        """Score texts one after another: each has a different state, so there is no prefix to share."""
        return [self.predict(text, str(i)) for i, text in enumerate(texts)]


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    text = " ".join(sys.argv[1:])
    pred = SemIfClassifier(TASK_LABELS, TASK.instruction).predict(text)
    print(f"label:      {pred.label}")
    print(f"confidence: {pred.confidence:.2f}")
    for label, p in sorted(pred.probabilities.items(), key=lambda kv: -kv[1]):
        print(f"  {label:<10} {p:.3f}")


if __name__ == "__main__":
    main()
