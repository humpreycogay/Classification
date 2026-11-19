"""JEV Classifier, debug copy: classifiers/jev with verbose logging of every step and request.

Each text is sent as the `state` with a single Choice question whose criteria are
the class labels. Jev returns the chosen label, per-label probabilities, and a
confidence score that can be used to route low-certainty predictions elsewhere.

Logging goes to stderr, so the harness's stdout ("progress:" lines, the report) is unchanged.
JEV_DEBUG_LEVEL sets the detail (default debug):
    debug   every item's wait / start / result, plus the SDK's wire log: URL, headers (Authorization
            redacted by the SDK) and the full request and response bodies
    info    one line per item (start, result) and the SDK's one line per HTTP response and retry
    warning failures only

The logging costs time, so this copy's timings are not comparable with jev's.

Usage:
    python classifier.py "Stocks rallied after the Fed held rates steady."
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# The shared benchmark library, as in evaluate.py, so this file also runs on its own from the console.
SHARED_DIR = Path(os.environ.get("BENCHMARK_SHARED_DIR") or Path(__file__).resolve().parents[2] / "shared")
if str(SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_DIR))

from dotenv import load_dotenv
from typesafe_sdk import AsyncTypeSafeClient, Choice, TypeSafeClient

from timing import RequestStats

from task import current_task, load_split

# .env lives in the repo root (classifiers/<app>/ -> two levels up).
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

DEFAULT_MODEL = "jev-latest"
TEXT_PREVIEW = 80  # characters of each text shown in its log lines

log = logging.getLogger("jev_debug")


def setup_logging() -> None:
    """Send this app's and the SDK's log records to stderr at JEV_DEBUG_LEVEL (default debug)."""
    level = getattr(logging, os.getenv("JEV_DEBUG_LEVEL", "debug").strip().upper(), logging.DEBUG)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s.%(msecs)03d %(levelname)-7s %(name)s: %(message)s", "%H:%M:%S"))
    for name in ("jev_debug", "typesafe_sdk"):
        logger = logging.getLogger(name)
        logger.handlers = [handler]
        logger.setLevel(level)
        logger.propagate = False


setup_logging()

# Labels, descriptions and instruction come from the task manifest (data/tasks/<task>.json, see task.py),
# chosen with --task on evaluate.py (default: ag_news).
TASK = current_task()
_LABEL_SPECS = TASK.labels
CLASS_TO_LABEL = {spec["raw"]: spec["id"] for spec in _LABEL_SPECS}
TASK_LABELS: dict[str, str] = {spec["id"]: spec["description"] for spec in _LABEL_SPECS}
log.info("task %s: %d labels %s", TASK.name, len(TASK_LABELS), list(TASK_LABELS))


@dataclass
class Prediction:
    label: str
    confidence: float
    probabilities: dict[str, float]


def preview(text: str) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= TEXT_PREVIEW else flat[:TEXT_PREVIEW] + "..."


def describe(pred: Prediction) -> str:
    ranked = sorted(pred.probabilities.items(), key=lambda kv: -kv[1])
    return f"{pred.label} (confidence {pred.confidence:.2f}; " + ", ".join(f"{k}={v:.2f}" for k, v in ranked[:5]) + ")"


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
        log.info("classifier ready: model %s, %d labels, api key set (%d characters), instruction %r",
                 self.model, len(labels), len(self.api_key), instructions)

    def _questions(self) -> dict[str, Choice]:
        return {"label": Choice(instructions=self.instructions, criteria=self.labels)}

    @staticmethod
    def _to_prediction(response) -> Prediction:
        answer = response.choices["label"]
        return Prediction(answer.choice, answer.confidence, dict(answer.probabilities))

    def predict(self, text: str) -> Prediction:
        log.info("predict: 1 text, synchronous: %r", preview(text))
        start = time.perf_counter()
        with TypeSafeClient(api_key=self.api_key, model=self.model) as client:
            response = client.system_one(state=text, questions=self._questions())
        pred = self._to_prediction(response)
        log.info("predict: done in %.0f ms -> %s", (time.perf_counter() - start) * 1000, describe(pred))
        return pred

    async def apredict_many(self, texts: list[str], concurrency: int = 16) -> list[Prediction]:
        """Classify many texts concurrently. The SDK retries 429s with backoff."""
        semaphore = asyncio.Semaphore(concurrency)
        questions = self._questions()
        total = len(texts)
        in_flight = 0
        done = 0
        batch_start = time.perf_counter()
        log.info("predict_many: %d texts, concurrency %d (one request per text)", total, concurrency)

        async with AsyncTypeSafeClient(api_key=self.api_key, model=self.model) as client:

            async def one(index: int, text: str) -> Prediction:
                nonlocal in_flight, done
                tag = f"[{index + 1}/{total}]"
                queued = time.perf_counter()
                log.debug("%s waiting for a permit", tag)
                async with semaphore:
                    start = time.perf_counter()
                    in_flight += 1
                    log.info("%s start (waited %.0f ms, %d in flight): %r", tag, (start - queued) * 1000, in_flight,
                             preview(text))
                    self.stats.requests += 1
                    try:
                        response = await client.system_one(state=text, questions=questions)
                    except Exception as e:
                        self.stats.failures += 1
                        in_flight -= 1
                        log.error("%s FAILED after %.0f ms: %s: %s", tag, (time.perf_counter() - start) * 1000,
                                  type(e).__name__, e)
                        raise
                    seconds = time.perf_counter() - start
                    self.stats.request_seconds.append(seconds)
                    in_flight -= 1
                usage = getattr(response, "usage", None)
                input_tokens, output_tokens = getattr(usage, "input_tokens", None), getattr(usage, "output_tokens", None)
                self.stats.add_tokens(input_tokens, output_tokens)
                pred = self._to_prediction(response)
                done += 1
                log.info("%s done in %.0f ms (%d/%d finished, tokens in/out %s/%s) -> %s", tag, seconds * 1000, done,
                         total, input_tokens, output_tokens, describe(pred))
                return pred

            preds = await asyncio.gather(*(one(i, t) for i, t in enumerate(texts)))
        elapsed = time.perf_counter() - batch_start
        log.info("predict_many: %d texts in %.2f s (%.1f texts/s)", total, elapsed, total / elapsed if elapsed else 0.0)
        return preds

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
