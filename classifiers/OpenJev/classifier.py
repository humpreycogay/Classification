"""OpenJev Classifier: text classification with OpenJev (DiffusionGemma 26B-A4B).

OpenJev (https://github.com/razorback16/openjev) is an open decision server that reads
typed answers from DiffusionGemma's probabilities. On a Mac it runs in-process on MLX
with 4-bit weights; start it with server/serve.sh, then use this client.

Each text is sent as the `state` with a single Choice question holding the class labels.

Usage:
    python classifier.py "Stocks rallied after the Fed held rates steady."

Environment (optional, read from the root .env):
    OPENJEV_URL       server base URL          (default: http://127.0.0.1:8080)
    OPENJEV_API_KEY   Bearer key, only if the server sets OPENJEV_API_KEY
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

from timing import RequestStats

from task import current_task, load_split

# .env lives in the repo root (classifiers/<app>/ -> two levels up).
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

DEFAULT_URL = "http://127.0.0.1:8080"
MODEL = "openjev-latest"
# The MLX backend runs one read at a time; a few requests in flight keep it busy.
DEFAULT_CONCURRENCY = 4
RETRY_STATUSES = {429, 503, 529}
MAX_ATTEMPTS = 6

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


class OpenJevClassifier:
    """Single-label text classifier backed by an OpenJev Choice question."""

    def __init__(
        self,
        labels: dict[str, str | None],
        instructions: str = "Which category does this text belong to?",
        url: str | None = None,
        api_key: str | None = None,
        timeout: float = 300.0,
    ) -> None:
        self.labels = labels
        self.stats = RequestStats()  # per-request latency, retries, tokens
        self.instructions = instructions
        self.url = (url or os.getenv("OPENJEV_URL") or DEFAULT_URL).rstrip("/")
        api_key = api_key or os.getenv("OPENJEV_API_KEY")
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.timeout = timeout

    def _body(self, text: str) -> dict:
        return {
            "state": text,
            "model": MODEL,
            "questions": {"label": {"type": "choice", "instructions": self.instructions, "criteria": self.labels}},
        }

    @staticmethod
    def _to_prediction(data: dict) -> Prediction:
        answer = data["answers"]["label"]
        return Prediction(answer["choice"], answer["confidence"], dict(answer["probabilities"]))

    def check_ready(self) -> None:
        """Fail fast with a clear message if the server is not running."""
        try:
            r = httpx.get(f"{self.url}/v1/models", headers=self.headers, timeout=30)
        except httpx.ConnectError:
            sys.exit(f"Cannot reach OpenJev at {self.url}. Start it first: server/serve.sh")
        if r.status_code != 200:
            sys.exit(f"OpenJev at {self.url} is not ready (HTTP {r.status_code}): {r.text[:200]}")

    async def _post(self, client: httpx.AsyncClient, text: str) -> Prediction:
        start = time.perf_counter()
        self.stats.requests += 1
        for attempt in range(MAX_ATTEMPTS):
            try:
                r = await client.post(f"{self.url}/v1/systemone", json=self._body(text))
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
            delay = float(r.headers.get("Retry-After", 2**attempt))
            await asyncio.sleep(min(delay, 30))
        if r.is_error:
            self.stats.failures += 1
        r.raise_for_status()
        data = r.json()
        self.stats.request_seconds.append(time.perf_counter() - start)
        usage = data.get("usage") or {}  # token counts, if the server reports them
        self.stats.add_tokens(usage.get("input_tokens", usage.get("prompt_tokens")),
                              usage.get("output_tokens", usage.get("completion_tokens")))
        return self._to_prediction(data)

    def predict(self, text: str) -> Prediction:
        return self.predict_many([text])[0]

    async def apredict_many(self, texts: list[str], concurrency: int = DEFAULT_CONCURRENCY) -> list[Prediction]:
        semaphore = asyncio.Semaphore(concurrency)
        async with httpx.AsyncClient(headers=self.headers, timeout=self.timeout) as client:

            async def one(text: str) -> Prediction:
                async with semaphore:
                    return await self._post(client, text)

            return await asyncio.gather(*(one(t) for t in texts))

    def predict_many(self, texts: list[str], concurrency: int = DEFAULT_CONCURRENCY) -> list[Prediction]:
        return asyncio.run(self.apredict_many(texts, concurrency))


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    text = " ".join(sys.argv[1:])
    clf = OpenJevClassifier(TASK_LABELS, TASK.instruction)
    clf.check_ready()
    pred = clf.predict(text)
    print(f"label:      {pred.label}")
    print(f"confidence: {pred.confidence:.2f}")
    for label, p in sorted(pred.probabilities.items(), key=lambda kv: -kv[1]):
        print(f"  {label:<10} {p:.3f}")


if __name__ == "__main__":
    main()
