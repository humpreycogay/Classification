"""JEV Classifier: text classification with TypeSafe's Jev model (System One API).

Each text is sent as the `state` with a single Choice question whose criteria are
the class labels. Jev returns the chosen label, per-label probabilities, and a
confidence score that can be used to route low-certainty predictions elsewhere.

Usage:
    python classifier.py "Stocks rallied after the Fed held rates steady."
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from typesafe_sdk import AsyncTypeSafeClient, Choice, TypeSafeClient

from timing import RequestStats

from task import current_task, load_split

# .env lives in the repo root (classifiers/<app>/ -> two levels up).
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

DEFAULT_MODEL = "jev-latest"

# Labels, descriptions and instruction come from the task manifest (data/tasks/<task>.json, see task.py),
# chosen with --task on evaluate.py (default: ag_news).
TASK = current_task()
_LABEL_SPECS = TASK.labels
CLASS_TO_LABEL = {spec["raw"]: spec["id"] for spec in _LABEL_SPECS}
TASK_LABELS: dict[str, str] = {spec["id"]: spec["description"] for spec in _LABEL_SPECS}


@dataclass
class Prediction:
    label: str
    confidence: float
    probabilities: dict[str, float]


class JEVClassifier:
    """Single-label text classifier backed by a Jev Choice question."""

    def __init__(
        self,
        labels: dict[str, str | None],
        instructions: str = "Which category does this text belong to?",
        model: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self.labels = labels
        # The SDK retries 429s internally, so retry counts are not observable here.
        self.stats = RequestStats(retries_observable=False)
        self.instructions = instructions
        self.model = model or os.getenv("JEV_MODEL", DEFAULT_MODEL)
        self.api_key = api_key or os.getenv("jev_api_key") or os.getenv("TYPESAFE_API_KEY")
        if not self.api_key:
            raise RuntimeError("No API key found: set jev_api_key in .env")

    def _questions(self) -> dict[str, Choice]:
        return {"label": Choice(instructions=self.instructions, criteria=self.labels)}

    @staticmethod
    def _to_prediction(response) -> Prediction:
        answer = response.choices["label"]
        return Prediction(answer.choice, answer.confidence, dict(answer.probabilities))

    def predict(self, text: str) -> Prediction:
        with TypeSafeClient(api_key=self.api_key, model=self.model) as client:
            response = client.system_one(state=text, questions=self._questions())
        return self._to_prediction(response)

    async def apredict_many(self, texts: list[str], concurrency: int = 16) -> list[Prediction]:
        """Classify many texts concurrently. The SDK retries 429s with backoff."""
        semaphore = asyncio.Semaphore(concurrency)
        questions = self._questions()

        async with AsyncTypeSafeClient(api_key=self.api_key, model=self.model) as client:

            async def one(text: str) -> Prediction:
                async with semaphore:
                    start = time.perf_counter()
                    self.stats.requests += 1
                    try:
                        response = await client.system_one(state=text, questions=questions)
                    except Exception:
                        self.stats.failures += 1
                        raise
                    self.stats.request_seconds.append(time.perf_counter() - start)
                usage = getattr(response, "usage", None)
                self.stats.add_tokens(getattr(usage, "input_tokens", None), getattr(usage, "output_tokens", None))
                return self._to_prediction(response)

            return await asyncio.gather(*(one(t) for t in texts))

    def predict_many(self, texts: list[str], concurrency: int = 16) -> list[Prediction]:
        return asyncio.run(self.apredict_many(texts, concurrency))


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    text = " ".join(sys.argv[1:])
    pred = JEVClassifier(TASK_LABELS, TASK.instruction).predict(text)
    print(f"label:      {pred.label}")
    print(f"confidence: {pred.confidence:.2f}")
    for label, p in sorted(pred.probabilities.items(), key=lambda kv: -kv[1]):
        print(f"  {label:<10} {p:.3f}")


if __name__ == "__main__":
    main()
