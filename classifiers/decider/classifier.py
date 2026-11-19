"""Decider Classifier: classification with Decider-4B v3 (Mapika/decider-4b), through decider-ai.

Decider is a one-pass decision model (Qwen3.5-4B-Base fine-tuned): the prompt is "Context:\\n<text>", the question,
the lettered options "(A) ... (B) ..." and an answer slot "Answer: ("; the letter logits at the slot are divided by
the temperature and softmaxed over the options. Nothing is generated.

One `choice` question per text; each option is "<name>: <description>" (the "named" format of Decider's
training mixture). Probabilities are read for every option; confidence = max p. The temperature is the release's
per-type value for Choice answers (decider_config.json temperature_by_type["choice"] = 2.098, decider-ai >= 1.4.0).
The schema cache stays off: v3 was not trained for it (schema_first_trained: false).

On a Mac, decider-ai runs the model in float16 on MPS with its Metal kernels for the gated-delta attention layers
(decider.mps_ops, via MLX); CPU runs float32 (the reference of the setup-time parity check).

Usage:
    python classifier.py "Stocks rallied after the Fed held rates steady."

Environment (optional, read from the root .env):
    DECIDER_DEVICE   mps | cuda | cpu   (default: auto-detect)
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

# Xet transfers stall on some networks; plain HTTPS downloads are reliable.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import torch  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402

from batching import sorted_batches
from task import current_task

# .env lives in the repo root (classifiers/<app>/ -> two levels up).
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

MODEL_REPO = "Mapika/decider-4b"
MODEL_REVISION = "585473dec4fa93f0325c2bf02dda7273b61f44c1"  # the v3 release commit (2026-10-20)
MODEL_RELEASE = "v3"
MAX_OPTIONS = 255
MAX_CTX_TOKENS = 1536  # decider-ai's default context cut, in tokens

# Labels, descriptions and instruction come from the task manifest (data/tasks/<task>.json, see task.py),
# chosen with --task on evaluate.py (default: ag_news).
TASK = current_task()
_LABEL_SPECS = TASK.labels
CLASS_TO_LABEL = {spec["raw"]: spec["id"] for spec in _LABEL_SPECS}
TASK_LABELS: dict[str, str] = {spec["id"]: spec["description"] for spec in _LABEL_SPECS}
LABEL_NAMES = {spec["id"]: spec["name"] for spec in _LABEL_SPECS}


def model_dir() -> str:
    return snapshot_download(MODEL_REPO, revision=MODEL_REVISION)


@dataclass
class Prediction:
    label: str
    confidence: float
    probabilities: dict[str, float]


class DeciderClassifier:
    """Single-label text classifier: one Decider choice question per text."""

    def __init__(self, labels: dict[str, str], instruction: str, device: str | None = None,
                 dtype: torch.dtype | None = None) -> None:
        if not 2 <= len(labels) <= MAX_OPTIONS:
            raise ValueError(f"Decider supports 2-{MAX_OPTIONS} options")
        for var in ("DECIDER_TEMPERATURE", "DECIDER_SCHEMA_CACHE"):
            if os.getenv(var):
                raise ValueError(f"{var} is set; this app runs the release's per-type temperatures without the schema "
                                 "cache (unset it)")
        from decider.infer import Decider

        self.labels = labels
        self.question = {"question": instruction,
                         "options": [f"{LABEL_NAMES.get(l, l)}: {d}" for l, d in labels.items()]}
        self.device = device or os.getenv("DECIDER_DEVICE") or None
        self.path = model_dir()
        self.decider = Decider(self.path, device=self.device, dtype=dtype, use_graphs=False)
        self.device = str(self.decider.dev)
        self.dtype = str(next(self.decider.m.parameters()).dtype).removeprefix("torch.")
        config = json.loads((Path(self.path) / "decider_config.json").read_text())
        # v3's upstream config retained the parent model's `version: 4b-v2.1`; report both values rather
        # than presenting that stale internal field as the release loaded by MODEL_REVISION.
        self.release = MODEL_RELEASE
        self.config_version = config.get("version")
        self.temperature = float((self.decider.T_by_type or {}).get("choice", self.decider.T))

    def _classify(self, texts: list[str]) -> list[Prediction]:
        answers = self.decider.decide_batch([(t, [self.question]) for t in texts], max_ctx_tokens=MAX_CTX_TOKENS)
        preds = []
        for (answer,) in answers:
            probabilities = dict(zip(self.labels, answer["probs_list"]))
            label = max(probabilities, key=probabilities.get)
            # Confidence = max p, the probability of the chosen label.
            preds.append(Prediction(label, probabilities[label], probabilities))
        return preds

    def predict_many(self, texts: list[str], batch_size: int = 8) -> list[Prediction]:
        return sorted_batches(texts, batch_size, self._classify)

    def predict(self, text: str) -> Prediction:
        return self.predict_many([text])[0]


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    text = " ".join(sys.argv[1:])
    pred = DeciderClassifier(TASK_LABELS, TASK.instruction).predict(text)
    print(f"label:      {pred.label}")
    print(f"confidence: {pred.confidence:.2f}")
    for label, p in sorted(pred.probabilities.items(), key=lambda kv: -kv[1]):
        print(f"  {label:<10} {p:.3f}")


if __name__ == "__main__":
    main()
