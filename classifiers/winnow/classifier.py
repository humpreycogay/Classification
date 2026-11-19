"""Winnow Classifier: text classification with Winnow-12B (Gemma 4 12B decision fine-tune), via its local server.

Winnow (https://huggingface.co/EldanRing/Winnow-12B) is served by winnow-inference, a patched llama.cpp server
(Metal on a Mac), with the Q8_0 weights; start it with server/serve.sh (the runner does this itself). Each text
is sent to POST /v1/systemone as the `state` with one Choice question; each option is "<name>: <description>"
(the server renders a criterion as "key: description", so the key is the label's name). The server reads the
option letters' logits and returns softmax(logit / T) with T = 1 (the release has no fitted temperature).
Confidence here = max p (the server's own confidence is 1 - normalized entropy; it is kept in the metadata only
as the definition, not used).

The server takes 2-64 options per question; Banking77 (77 labels) runs through the shortlist variant.

Usage:
    python classifier.py "Stocks rallied after the Fed held rates steady."

Environment (optional, read from the root .env):
    WINNOW_URL       server base URL      (default: http://127.0.0.1:8091)
    WINNOW_API_KEY   Bearer key, only if the server was started with --api-key-file
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
from dotenv import load_dotenv

from task import current_task
from timing import RequestStats

# .env lives in the repo root (classifiers/<app>/ -> two levels up).
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

DEFAULT_URL = "http://127.0.0.1:8091"
MODEL = "Winnow-12B"
MODEL_REPO = "EldanRing/Winnow-12B"
MODEL_REVISION = "b2b14213dfa252e6d6b543c8b334762e51772d29"  # keep in step with server/setup.sh
MODEL_FILE = "gguf/Winnow-12B-Q8_0.gguf"
MAX_OPTIONS = 64  # the server's limit per question (docs/API.md)
# The Apple Silicon profile has one decision branch: requests are served one at a time.
DEFAULT_CONCURRENCY = 1
RETRY_STATUSES = {429, 503}
MAX_ATTEMPTS = 6

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


class WinnowClassifier:
    """Single-label text classifier backed by one Winnow Choice question."""

    def __init__(self, labels: dict[str, str], instructions: str, url: str | None = None,
                 api_key: str | None = None, timeout: float = 300.0) -> None:
        if not 2 <= len(labels) <= MAX_OPTIONS:
            raise ValueError(f"Winnow takes 2-{MAX_OPTIONS} options per question (use --shortlist)")
        self.labels = labels
        self.instructions = instructions
        self.stats = RequestStats()  # per-request latency, retries, tokens
        self.url = (url or os.getenv("WINNOW_URL") or DEFAULT_URL).rstrip("/")
        api_key = api_key or os.getenv("WINNOW_API_KEY")
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.timeout = timeout

    def body(self, text: str, labels: dict[str, str] | None = None) -> dict:
        labels = labels or self.labels
        return {"model": MODEL, "state": text, "questions": {"label": {
            "type": "choice", "instructions": self.instructions,
            "criteria": {LABEL_NAMES.get(l, l): d for l, d in labels.items()}}}}

    @staticmethod
    def _to_prediction(data: dict, labels: dict[str, str]) -> Prediction:
        by_name = {LABEL_NAMES.get(l, l): l for l in labels}
        answer = data["answers"]["label"]
        probabilities = {by_name[k]: float(p) for k, p in answer["probabilities"].items()}
        label = max(probabilities, key=probabilities.get)
        # Confidence = max p, the probability of the chosen label.
        return Prediction(label, probabilities[label], probabilities)

    def check_ready(self) -> None:
        """Fail fast with a clear message if the server is not running."""
        try:
            r = httpx.get(f"{self.url}/health", headers=self.headers, timeout=30)
        except httpx.ConnectError:
            sys.exit(f"Cannot reach Winnow at {self.url}. Start it first: server/serve.sh")
        if r.status_code != 200:
            sys.exit(f"Winnow at {self.url} is not ready (HTTP {r.status_code}): {r.text[:200]}")

    async def _post(self, client: httpx.AsyncClient, text: str, labels: dict[str, str]) -> Prediction:
        start = time.perf_counter()
        self.stats.requests += 1
        for attempt in range(MAX_ATTEMPTS):
            try:
                r = await client.post(f"{self.url}/v1/systemone", json=self.body(text, labels))
            except httpx.TransportError as e:  # dropped connection or read timeout
                self.stats.timeouts += isinstance(e, httpx.TimeoutException)
                if attempt == MAX_ATTEMPTS - 1:
                    self.stats.failures += 1
                    raise
                self.stats.retries += 1
                await asyncio.sleep(2**attempt)
                continue
            if r.status_code not in RETRY_STATUSES or attempt == MAX_ATTEMPTS - 1:
                break
            self.stats.retries += 1
            await asyncio.sleep(min(float(r.headers.get("Retry-After", 2**attempt)), 30))
        if r.is_error:
            self.stats.failures += 1
        r.raise_for_status()
        data = r.json()
        self.stats.request_seconds.append(time.perf_counter() - start)
        usage = data.get("usage") or {}
        self.stats.add_tokens(usage.get("input_tokens"), usage.get("output_tokens"))
        return self._to_prediction(data, labels)

    async def _apredict(self, texts: list[str], label_sets: list[dict[str, str]], concurrency: int) -> list[Prediction]:
        semaphore = asyncio.Semaphore(concurrency)
        limits = httpx.Limits(max_keepalive_connections=concurrency, max_connections=concurrency)
        async with httpx.AsyncClient(headers=self.headers, timeout=self.timeout, limits=limits) as client:

            async def one(text: str, labels: dict[str, str]) -> Prediction:
                async with semaphore:
                    return await self._post(client, text, labels)

            return await asyncio.gather(*(one(t, s) for t, s in zip(texts, label_sets)))

    def predict_many(self, texts: list[str], concurrency: int = DEFAULT_CONCURRENCY) -> list[Prediction]:
        return asyncio.run(self._apredict(texts, [self.labels] * len(texts), concurrency))

    def predict_shortlisted(self, texts: list[str], label_sets: list[dict[str, str]],
                            concurrency: int = DEFAULT_CONCURRENCY) -> list[Prediction]:
        """Classify each text among its own labels (a shortlist); other labels get probability 0."""
        return asyncio.run(self._apredict(texts, label_sets, concurrency))

    def predict(self, text: str) -> Prediction:
        return self.predict_many([text])[0]


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    text = " ".join(sys.argv[1:])
    clf = WinnowClassifier(TASK_LABELS, TASK.instruction)
    clf.check_ready()
    pred = clf.predict(text)
    print(f"label:      {pred.label}")
    print(f"confidence: {pred.confidence:.2f}")
    for label, p in sorted(pred.probabilities.items(), key=lambda kv: -kv[1]):
        print(f"  {label:<10} {p:.3f}")


if __name__ == "__main__":
    main()
