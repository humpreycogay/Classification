"""Verdict Classifier: zero-shot classification with Verdict (heman10x/rlcd-modernbert-151m).

Verdict is a 151M Jev-style decision model (ModernBERT-base + GLiClass head, Apache-2.0)
trained for calibrated choices. It is loaded here with the `gliclass` package from PyPI.
The prompt contract and calibration follow Verdict's v1.4 inference engine
(github.com/Heman10x-NGU/Verdict-open-jev), as reproduced by OpenJev:

- labels are prefixed "<<LABEL>>It is <option>", then "<<SEP>>", then the question and text;
- an "insufficient evidence" option is always appended as the last label;
- logits are divided by the checkpoint's per-option-count temperature (calibrator.json)
  and softmaxed; the abstention mass is dropped and the rest renormalized.

Usage:
    python classifier.py "Stocks rallied after the Fed held rates steady."

Environment (optional, read from the root .env):
    VERDICT_MODEL         Hugging Face model id   (default: heman10x/rlcd-modernbert-151m)
    VERDICT_DEVICE        mps | cuda | cpu        (default: auto-detect)
    VERDICT_TEMPERATURE   override the shipped calibration temperature (default: unset)

The shipped calibrator (T=3.056 for 5 options) was fitted on banking/synthetic data and is
too cautious on news: almost nothing reaches 0.9 confidence. VERDICT_TEMPERATURE=1.68 was
fitted on 400 AG News *train* articles (never the test set) and is well calibrated on news.
Temperature only changes confidence, never which label wins.
"""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

# Xet transfers stall on some networks; plain HTTPS downloads are reliable.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import torch  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from gliclass import GLiClassModel  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from task import current_task, load_split

# .env lives in the repo root (classifiers/<app>/ -> two levels up).
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

DEFAULT_MODEL = "heman10x/rlcd-modernbert-151m"
LABEL_MARKER, SEP_MARKER = "<<LABEL>>", "<<SEP>>"
ABSTAIN = "insufficient evidence"
MAX_LENGTH = 512  # Verdict v1.4 input limit
MAX_OPTIONS = 24  # 25 logits, the last reserved for "insufficient evidence"

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


@dataclass
class Prediction:
    label: str
    confidence: float
    probabilities: dict[str, float]


class VerdictClassifier:
    """Single-label text classifier backed by Verdict's calibrated choice head."""

    def __init__(
        self,
        labels: dict[str, str],
        instructions: str = "Which category does this text belong to?",
        model: str | None = None,
        device: str | None = None,
    ) -> None:
        if not 2 <= len(labels) <= MAX_OPTIONS:
            raise ValueError(f"Verdict supports 2-{MAX_OPTIONS} labels")
        self.labels = labels
        self.instructions = instructions
        self.model_id = model or os.getenv("VERDICT_MODEL") or DEFAULT_MODEL
        self.device = device or os.getenv("VERDICT_DEVICE") or default_device()
        self.model = GLiClassModel.from_pretrained(self.model_id).to(self.device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        with open(hf_hub_download(self.model_id, "calibrator.json")) as f:
            calibrator = json.load(f)
        k = len(labels) + 1  # plus the abstention option
        self.temperature = float(
            os.getenv("VERDICT_TEMPERATURE") or calibrator.get("per_k", {}).get(str(k), calibrator["temperature"])
        )
        self.label_prefix = self._prefix(labels)

    @staticmethod
    def _prefix(labels: dict[str, str]) -> str:
        options = [f"It is {LABEL_NAMES.get(l, l)} ({d.lower()})" for l, d in labels.items()] + [ABSTAIN]
        return "".join(LABEL_MARKER + o for o in options) + SEP_MARKER

    def _prompt(self, text: str, prefix: str | None = None) -> str:
        return f"{prefix or self.label_prefix}Question: {self.instructions}\n\nContext:\n{text}"

    def predict_shortlisted(self, texts: list[str], label_sets: list[dict[str, str]], batch_size: int = 16) -> list[Prediction]:
        """Classify each text among its own labels (a shortlist, all of the constructor's size, so the
        calibration temperature for that option count applies); other labels get probability 0."""
        if any(len(s) != len(self.labels) for s in label_sets):
            raise ValueError("every shortlist must have as many labels as the classifier was built with")
        return self.predict_many(texts, batch_size, label_sets)

    def predict_many(self, texts: list[str], batch_size: int = 16,
                     label_sets: list[dict[str, str]] | None = None) -> list[Prediction]:
        sets = label_sets or [self.labels] * len(texts)
        k = len(self.labels) + 1
        preds = []
        for i in range(0, len(texts), batch_size):
            chunk, chunk_sets = texts[i : i + batch_size], sets[i : i + batch_size]
            batch = self.tokenizer(
                [self._prompt(t, self._prefix(s)) for t, s in zip(chunk, chunk_sets)],
                padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt",
            ).to(self.device)
            with torch.inference_mode():
                logits = self.model(**batch).logits.float().cpu()
            for row, row_labels in zip(logits, chunk_sets):
                labels = list(row_labels)
                p = torch.softmax(row[:k] / self.temperature, -1)[:-1]  # drop abstention mass
                p = (p / p.sum()).tolist()
                if not math.isfinite(sum(p)):
                    p = [1.0 / len(labels)] * len(labels)
                probabilities = dict(zip(labels, p))
                label = max(probabilities, key=probabilities.get)
                # Confidence = max p, the probability of the chosen label.
                preds.append(Prediction(label, probabilities[label], probabilities))
        return preds

    def predict(self, text: str) -> Prediction:
        return self.predict_many([text])[0]


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    text = " ".join(sys.argv[1:])
    pred = VerdictClassifier(TASK_LABELS, TASK.instruction).predict(text)
    print(f"label:      {pred.label}")
    print(f"confidence: {pred.confidence:.2f}")
    for label, p in sorted(pred.probabilities.items(), key=lambda kv: -kv[1]):
        print(f"  {label:<10} {p:.3f}")


if __name__ == "__main__":
    main()
