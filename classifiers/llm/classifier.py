"""LLM Classifier: text classification with a local chat model served by Ollama (Gemma, Qwen, ...).

Each text goes to Ollama's /api/chat with the class labels as lettered options (A, B, ...).
The reply is constrained to {"answer": "<letter>"} by a JSON schema, so it is always a
valid label. Probabilities come from the model's log-probabilities for each option
letter at the answer position, renormalized over the options.

The model and the Ollama server are parameters (model=, host=; --model, --host on the command line),
so one app serves every model; the runner's registry (Runner/apps.py) names the models it benchmarks.

Usage:
    python classifier.py "Stocks rallied after the Fed held rates steady."
    python classifier.py --model qwen3.5:9b-q4_K_M --host http://gpu-box:11434 "Stocks rallied ..."

Environment (optional, read from the root .env; the parameters take precedence):
    LLM_MODEL       Ollama model tag   (default: qwen3.8:27b-q4_K_M)
    LLM_NUM_CTX     context window in tokens, Ollama's num_ctx (default: 64000)
    LLM_HOST        Ollama base URL for this app, e.g. a GPU box on the network (default: OLLAMA_HOST)
    OLLAMA_HOST     Ollama base URL    (default: http://127.0.0.1:11434)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
from dotenv import load_dotenv

# The shared benchmark library (timing.py, task.py): BENCHMARK_SHARED_DIR (set by the runner and evaluate.py),
# else the repo's shared/ folder, so `python classifier.py "..."` also works on its own.
SHARED_DIR = Path(os.environ.get("BENCHMARK_SHARED_DIR") or Path(__file__).resolve().parents[2] / "shared")
if str(SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_DIR))

from timing import RequestStats

from task import current_task, load_split

# .env lives in the repo root (classifiers/<app>/ -> two levels up).
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

DEFAULT_MODEL = "qwen3.8:27b-q4_K_M"
DEFAULT_NUM_CTX = 64000  # Ollama context window (num_ctx), in tokens
DEFAULT_HOST = "http://127.0.0.1:11434"
# Ollama answers a few requests in parallel per loaded model (OLLAMA_NUM_PARALLEL).
DEFAULT_CONCURRENCY = 4
MAX_ATTEMPTS = 4  # retries cover dropped connections and read timeouts
LETTERS = "ABCDEFGHIJKLMNOP"
SYSTEM_PROMPT = (
    "You are a text classifier. Choose exactly one of the listed options. "
    'Reply with JSON {"answer": "<letter>"} and nothing else.'
)

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


class LLMClassifier:
    """Single-label text classifier backed by a chat model in Ollama."""

    def __init__(
        self,
        labels: dict[str, str],
        instructions: str = "Which category does this text belong to?",
        model: str | None = None,
        host: str | None = None,
        timeout: float = 300.0,
        num_ctx: int | None = None,
    ) -> None:
        if not 2 <= len(labels) <= len(LETTERS):
            raise ValueError(f"Need 2-{len(LETTERS)} labels")
        self.labels = labels
        self.stats = RequestStats()  # per-request latency, retries, tokens
        self.answer_overrides = 0  # texts where the most probable option differed from the written letter
        self.letters = LETTERS[: len(labels)]
        self.instructions = instructions
        self.model = model or os.getenv("LLM_MODEL") or DEFAULT_MODEL
        self.host = (host or os.getenv("LLM_HOST") or os.getenv("OLLAMA_HOST") or DEFAULT_HOST).rstrip("/")
        if not self.host.startswith("http"):
            self.host = f"http://{self.host}"
        self.timeout = timeout
        self.num_ctx = num_ctx or int(os.getenv("LLM_NUM_CTX") or DEFAULT_NUM_CTX)

    def _body(self, text: str, labels: dict[str, str] | None = None) -> dict:
        labels = labels or self.labels
        letters = LETTERS[: len(labels)]
        options = "\n".join(
            f"{letter}) {LABEL_NAMES.get(label, label)}: {description}"
            for letter, (label, description) in zip(letters, labels.items())
        )
        return {
            "model": self.model,
            "stream": False,
            "think": False,
            "logprobs": True,
            "top_logprobs": 20,
            "options": {"temperature": 0, "seed": 0, "num_ctx": self.num_ctx},
            "format": {
                "type": "object",
                "properties": {"answer": {"type": "string", "enum": list(letters)}},
                "required": ["answer"],
            },
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"{self.instructions}\n\nText: {text}\n\nOptions:\n{options}"},
            ],
        }

    def _to_prediction(self, data: dict, labels: dict[str, str] | None = None) -> Prediction:
        labels_map = labels or self.labels
        letters = LETTERS[: len(labels_map)]
        answer = json.loads(data["message"]["content"])["answer"]
        # Log-probabilities of each letter at the answer token, before the schema constraint.
        letter_logprobs: dict[str, float] = {}
        for token in data.get("logprobs") or []:
            if token["token"].strip().strip('"') == answer:
                for candidate in token.get("top_logprobs", []):
                    letter = candidate["token"].strip().strip('"')
                    if letter in set(letters) and letter not in letter_logprobs:
                        letter_logprobs[letter] = candidate["logprob"]
                break
        if answer not in letter_logprobs:  # no usable log-probabilities: fall back to the chosen answer
            letter_logprobs = {answer: 0.0}
        weights = [math.exp(letter_logprobs[l]) if l in letter_logprobs else 0.0 for l in letters]
        total = sum(weights)
        probs = [w / total for w in weights]
        labels = list(labels_map)
        # Predict the most probable option. Under Ollama's JSON-schema constraint the written letter can
        # differ from the model's own most probable option (seen on 7 of 200 banking77 texts with 16
        # options; never with 4), so the written letter is only the fallback when log-probabilities are missing.
        best = max(range(len(letters)), key=lambda i: (probs[i], -i))
        if letters[best] != answer:
            self.answer_overrides += 1
        label = labels[best]
        probabilities = dict(zip(labels, probs))
        # Confidence = max p, the probability of the chosen label.
        return Prediction(label, probabilities[label], probabilities)

    def check_ready(self) -> None:
        """Fail fast if Ollama is not running or the model is not pulled."""
        try:
            r = httpx.get(f"{self.host}/api/tags", timeout=10)
        except httpx.ConnectError:
            sys.exit(f"Cannot reach Ollama at {self.host}. Start the Ollama app or run: ollama serve")
        names = {m["name"] for m in r.json().get("models", [])}
        if self.model not in names and f"{self.model}:latest" not in names:
            sys.exit(f"Model {self.model} is not pulled. Run: ollama pull {self.model}")

    async def _post(self, client: httpx.AsyncClient, text: str, labels: dict[str, str] | None = None) -> Prediction:
        start = time.perf_counter()
        self.stats.requests += 1
        for attempt in range(MAX_ATTEMPTS):
            try:
                r = await client.post(f"{self.host}/api/chat", json=self._body(text, labels))
                break
            except httpx.TransportError as e:
                self.stats.timeouts += isinstance(e, httpx.TimeoutException)
                if attempt == MAX_ATTEMPTS - 1:
                    self.stats.failures += 1
                    raise
                self.stats.retries += 1
                await asyncio.sleep(2**attempt)
        if r.is_error:
            self.stats.failures += 1
        r.raise_for_status()
        data = r.json()
        self.stats.request_seconds.append(time.perf_counter() - start)
        eval_ns = data.get("eval_duration")
        self.stats.add_tokens(data.get("prompt_eval_count"), data.get("eval_count"), eval_ns / 1e9 if eval_ns else None)
        return self._to_prediction(data, labels)

    def predict(self, text: str) -> Prediction:
        return self.predict_many([text])[0]

    async def apredict_many(self, texts: list[str], concurrency: int = DEFAULT_CONCURRENCY,
                            label_sets: list[dict[str, str]] | None = None) -> list[Prediction]:
        semaphore = asyncio.Semaphore(concurrency)
        async with httpx.AsyncClient(timeout=self.timeout) as client:

            async def one(text: str, labels: dict[str, str] | None) -> Prediction:
                async with semaphore:
                    return await self._post(client, text, labels)

            sets = label_sets or [None] * len(texts)
            return await asyncio.gather(*(one(t, s) for t, s in zip(texts, sets)))

    def predict_shortlisted(self, texts: list[str], label_sets: list[dict[str, str]],
                            concurrency: int = DEFAULT_CONCURRENCY) -> list[Prediction]:
        """Classify each text among its own labels (a shortlist); other labels get probability 0."""
        return asyncio.run(self.apredict_many(texts, concurrency, label_sets))

    def predict_many(self, texts: list[str], concurrency: int = DEFAULT_CONCURRENCY) -> list[Prediction]:
        return asyncio.run(self.apredict_many(texts, concurrency))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", help=f"Ollama model tag (default: $LLM_MODEL or {DEFAULT_MODEL})")
    parser.add_argument("--host", help=f"Ollama base URL (default: $LLM_HOST, $OLLAMA_HOST or {DEFAULT_HOST})")
    parser.add_argument("text", nargs="+")
    args = parser.parse_args()
    text = " ".join(args.text)
    clf = LLMClassifier(TASK_LABELS, TASK.instruction, model=args.model, host=args.host)
    clf.check_ready()
    pred = clf.predict(text)
    print(f"label:      {pred.label}")
    print(f"confidence: {pred.confidence:.2f}")
    for label, p in sorted(pred.probabilities.items(), key=lambda kv: -kv[1]):
        print(f"  {label:<10} {p:.3f}")


if __name__ == "__main__":
    main()
