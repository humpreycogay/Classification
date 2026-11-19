"""Embedding Similarity Classifier: zero-shot classification with bge-m3 embeddings via Ollama.

Each label's text and each article are embedded with bge-m3 (served by the local Ollama
install). The article gets the label whose embedding is most similar (cosine). No text
is generated and no prompt engineering is involved.

Probabilities are softmax(cosine / T). T is fitted (minimum negative log-likelihood) on the task's
calibration split (carved from train by task.py, never the test set) and cached per task, model
digest and label texts in results/cache/. It only changes confidence, never which label wins.

Usage:
    python classifier.py "Stocks rallied after the Fed held rates steady."

Environment (optional, read from the root .env):
    EMBEDSIM_MODEL         Ollama embedding model tag   (default: bge-m3:567m)
    EMBEDSIM_TEMPERATURE   fixed softmax temperature    (default: unset = fitted on the calibration split)
    OLLAMA_HOST            Ollama base URL              (default: http://127.0.0.1:11434)
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import httpx
from dotenv import load_dotenv

from task import current_task, load_split

# .env lives in the repo root (classifiers/<app>/ -> two levels up).
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

DEFAULT_MODEL = "bge-m3:567m"
DEFAULT_HOST = "http://127.0.0.1:11434"
# Temperature fallback before calibration (e.g. the classifier.py demo); cosine gaps between labels are small.
DEFAULT_TEMPERATURE = 0.022
TEMPERATURE_GRID = [0.002 * (1.0 / 0.002) ** (i / 199) for i in range(200)]  # geometric, 0.002 to 1
CACHE_DIR = Path(__file__).resolve().parent / "results" / "cache"

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


def softmax(values: list[float]) -> list[float]:
    top = max(values)
    exps = [math.exp(v - top) for v in values]
    total = sum(exps)
    return [e / total for e in exps]


class EmbedSimClassifier:
    """Single-label text classifier: nearest label by embedding cosine similarity."""

    def __init__(
        self,
        labels: dict[str, str],
        model: str | None = None,
        host: str | None = None,
        timeout: float = 300.0,
    ) -> None:
        self.labels = labels
        self.model = model or os.getenv("EMBEDSIM_MODEL") or DEFAULT_MODEL
        self.host = (host or os.getenv("OLLAMA_HOST") or DEFAULT_HOST).rstrip("/")
        if not self.host.startswith("http"):
            self.host = f"http://{self.host}"
        self.client = httpx.Client(timeout=timeout)
        self._label_vectors: list[list[float]] | None = None
        self.temperature = float(os.getenv("EMBEDSIM_TEMPERATURE") or DEFAULT_TEMPERATURE)
        self.temperature_source = "EMBEDSIM_TEMPERATURE" if os.getenv("EMBEDSIM_TEMPERATURE") else "default (not calibrated)"
        self.digest = ""

    def _embed(self, texts: list[str]) -> list[list[float]]:
        r = self.client.post(f"{self.host}/api/embed", json={"model": self.model, "input": texts})
        r.raise_for_status()
        return r.json()["embeddings"]  # Ollama returns unit-length vectors

    def label_texts(self) -> list[str]:
        template = TASK.manifest.get("embedding_label_template", "{name}: {description}")
        return [template.format(name=LABEL_NAMES.get(l, l), description=d) for l, d in self.labels.items()]

    @property
    def label_vectors(self) -> list[list[float]]:
        if self._label_vectors is None:
            self._label_vectors = self._embed(self.label_texts())
        return self._label_vectors

    def check_ready(self) -> None:
        """Fail fast if Ollama is not running or the model is not pulled."""
        try:
            r = self.client.get(f"{self.host}/api/tags", timeout=10)
        except httpx.ConnectError:
            sys.exit(f"Cannot reach Ollama at {self.host}. Start the Ollama app or run: ollama serve")
        digests = {m["name"]: m.get("digest", "") for m in r.json().get("models", [])}
        if self.model not in digests and f"{self.model}:latest" not in digests:
            sys.exit(f"Model {self.model} is not pulled. Run: ollama pull {self.model}")
        self.digest = digests.get(self.model) or digests.get(f"{self.model}:latest", "")

    def calibrate(self, texts: list[str], labels: list[str], batch_size: int = 32) -> dict:
        """Fit the temperature on labeled calibration texts (minimum log loss), cached per task, model digest
        and label texts; an EMBEDSIM_TEMPERATURE setting wins. Returns where the temperature came from."""
        if os.getenv("EMBEDSIM_TEMPERATURE"):
            return {"temperature": self.temperature, "source": "EMBEDSIM_TEMPERATURE"}
        key = hashlib.sha1(json.dumps([TASK.name, self.model, self.digest, self.label_texts(), texts, labels]).encode()).hexdigest()[:16]
        path = CACHE_DIR / f"temperature_{TASK.name}_{key}.json"
        if path.exists():
            info = json.loads(path.read_text())
        else:
            sims = []
            for i in range(0, len(texts), batch_size):
                for v in self._embed(texts[i : i + batch_size]):
                    sims.append([sum(a * b for a, b in zip(v, lv)) for lv in self.label_vectors])
            order = list(self.labels)
            true_idx = [order.index(label) for label in labels]
            best_t, best_loss = DEFAULT_TEMPERATURE, float("inf")
            for t in TEMPERATURE_GRID:
                loss = -sum(math.log(max(softmax([s / t for s in row])[j], 1e-12)) for row, j in zip(sims, true_idx)) / len(sims)
                if loss < best_loss:
                    best_t, best_loss = t, loss
            info = {"temperature": round(best_t, 5), "log_loss": round(best_loss, 5), "n_calibration": len(texts),
                    "source": f"fitted on the {TASK.name} calibration split ({len(texts)} train articles)"}
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(info, indent=2) + "\n")
        self.temperature, self.temperature_source = info["temperature"], info["source"]
        return info

    def _to_prediction(self, vector: list[float]) -> Prediction:
        sims = [sum(a * b for a, b in zip(vector, lv)) for lv in self.label_vectors]
        probs = softmax([s / self.temperature for s in sims])
        probabilities = dict(zip(self.labels, probs))
        label = max(probabilities, key=probabilities.get)
        # Confidence = max p, the probability of the chosen label (temperature calibrated; see module doc).
        return Prediction(label, probabilities[label], probabilities)

    def predict_many(self, texts: list[str], batch_size: int = 32) -> list[Prediction]:
        preds = []
        for i in range(0, len(texts), batch_size):
            preds += [self._to_prediction(v) for v in self._embed(texts[i : i + batch_size])]
        return preds

    def predict(self, text: str) -> Prediction:
        return self.predict_many([text])[0]


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    text = " ".join(sys.argv[1:])
    clf = EmbedSimClassifier(TASK_LABELS)
    clf.check_ready()
    pred = clf.predict(text)
    print(f"label:      {pred.label}")
    print(f"confidence: {pred.confidence:.2f}")
    for label, p in sorted(pred.probabilities.items(), key=lambda kv: -kv[1]):
        print(f"  {label:<10} {p:.3f}")


if __name__ == "__main__":
    main()
