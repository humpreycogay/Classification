"""Laya Classifier: local text classification with convaiinnovations/laya.

Laya is an open-weights (Apache 2.0) System One model that runs on this machine.
Each text is sent as the `state` with a single Choice question holding the class labels. No API key needed.

Usage:
    python classifier.py "Stocks rallied after the Fed held rates steady."

Environment (optional, read from the root .env):
    LAYA_DEVICE   mps | cuda | cpu   (default: auto-detect)
    LAYA_MODEL    english | multilingual | typed-decisions   (default: auto-route by language)
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

# transformers probes for TensorFlow at import, which can deadlock Laya model construction.
os.environ.setdefault("USE_TF", "0")
# Xet transfers stall on some networks; plain HTTPS downloads are reliable.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import torch  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from huggingface_hub import try_to_load_from_cache  # noqa: E402
from laya import Router  # noqa: E402

from task import current_task, load_split

# .env lives in the repo root (classifiers/<app>/ -> two levels up).
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

# The three checkpoints are subfolders of one repo, pinned here (keep in step with setup.sh).
REPO = "convaiinnovations/laya"
REVISION = "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851"
SUBFOLDERS = {"english": None, "multilingual": "multilingual", "typed-decisions": "typed-decisions"}
MODELS = list(SUBFOLDERS)
MAX_LEN = 1024
HEAD_MAX_LEN = 512

# Labels, descriptions and instruction come from the task manifest (data/tasks/<task>.json, see task.py),
# chosen with --task on evaluate.py (default: ag_news).
TASK = current_task()
_LABEL_SPECS = TASK.labels
CLASS_TO_LABEL = {spec["raw"]: spec["id"] for spec in _LABEL_SPECS}
TASK_LABELS: dict[str, str] = {spec["id"]: spec["description"] for spec in _LABEL_SPECS}


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


class LayaClassifier:
    """Single-label text classifier backed by a Laya Choice question."""

    def __init__(
        self,
        labels: dict[str, str | None],
        instructions: str = "Which category does this text belong to?",
        model: str | None = None,
        device: str | None = None,
    ) -> None:
        self.labels = labels
        self.instructions = instructions
        self.model = model or os.getenv("LAYA_MODEL") or None  # None = route by language
        self.device = device or os.getenv("LAYA_DEVICE") or default_device()
        # Auto-routing keeps English and multilingual resident (the Router's default of 2); a pinned model
        # only ever uses one checkpoint, so only that one is loaded and kept.
        # The pinned snapshot setup.sh downloaded (the Router's own download would follow main). Only some of the
        # repo's files are downloaded, so the folder is found from one of them.
        config = try_to_load_from_cache(REPO, "rl_agent_config.json", revision=REVISION)
        if not isinstance(config, str):
            raise FileNotFoundError(f"Laya {REPO}@{REVISION[:7]} is not downloaded: run classifiers/laya/setup.sh")
        snapshot = os.path.dirname(config)
        self.router = Router(models={name: (snapshot, sub) for name, sub in SUBFOLDERS.items()},
                             device=self.device, max_loaded=1 if self.model else 2)

    def _questions(self) -> dict:
        return {"label": {"type": "choice", "instructions": self.instructions, "criteria": self.labels}}

    @staticmethod
    def _to_prediction(result: dict) -> Prediction:
        answer = result["answers"]["label"]
        # `answer_confidence` is max(p) as Laya returns it, without refitting. The upstream model card
        # warns the shipped checkpoints are over-confident and recommends refitting one temperature per
        # (question type, option count) before trusting the probabilities. Laya's `confidence` field
        # is 1 - normalized entropy, on a different scale.
        return Prediction(answer["choice"], answer["answer_confidence"], dict(answer["probabilities"]))

    def predict(self, text: str) -> Prediction:
        return self._to_prediction(self.router.predict(text, self._questions(), model=self.model,
                                                       max_len=MAX_LEN, head_max_len=HEAD_MAX_LEN))

    def predict_many(self, texts: list[str], batch_size: int = 32) -> list[Prediction]:
        """Classify many texts, sharing forward passes across batches of states."""
        questions = self._questions()
        requests = [{"state": t, "questions": questions} for t in texts]
        for request in requests:
            request.update(max_len=MAX_LEN, head_max_len=HEAD_MAX_LEN)
        if self.model:
            for r in requests:
                r["model"] = self.model
        results = self.router.predict_batch(requests, batch_size=batch_size)
        return [self._to_prediction(r) for r in results]


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    text = " ".join(sys.argv[1:])
    pred = LayaClassifier(TASK_LABELS, TASK.instruction).predict(text)
    print(f"label:      {pred.label}")
    print(f"confidence: {pred.confidence:.2f}")
    for label, p in sorted(pred.probabilities.items(), key=lambda kv: -kv[1]):
        print(f"  {label:<10} {p:.3f}")


if __name__ == "__main__":
    main()
