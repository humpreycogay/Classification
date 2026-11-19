"""Supervised embedding baseline: bge-m3 embeddings (via Ollama) + a classifier trained on the task's train split.

Uses the same embedding model as embedsim, but instead of comparing articles with label
descriptions (zero-shot), it embeds a stratified sample of the official AG News *train* split
and learns the classes from it. This isolates the embedding model's quality from embedsim's
label-description strategy. Two methods:

    centroid  Nearest class centroid (mean of the train embeddings per class), cosine similarity.
              p_* = softmax(cosine / T); T is fitted by minimum log loss on a held-out 20% of the
              train sample (centroids from the other 80%), then centroids are refit on all of it.
    logreg    Multinomial logistic regression on the embeddings; native predict_proba. C is chosen
              from GRID_C by 3-fold cross-validation on the train embeddings (training.py).

Neither method sees the test set. Train embeddings are cached in results/cache/, keyed by
model tag and Ollama digest (a re-pulled model is re-embedded), sample size and seed, together
with the time it originally took to embed them.

Usage:
    python classifier.py "Stocks rallied after the Fed held rates steady."   # centroid method

Environment (optional, read from the root .env):
    EMBEDSUP_MODEL   Ollama embedding model tag   (default: bge-m3:567m)
    OLLAMA_HOST      Ollama base URL              (default: http://127.0.0.1:11434)
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.linear_model import LogisticRegression

from task import current_task, load_split

# .env lives in the repo root (classifiers/<app>/ -> two levels up).
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

DEFAULT_MODEL = "bge-m3:567m"
DEFAULT_HOST = "http://127.0.0.1:11434"
CACHE_DIR = Path(__file__).resolve().parent / "results" / "cache"
# Labels, descriptions and instruction come from the task manifest (data/tasks/<task>.json, see task.py),
# chosen with --task on evaluate.py (default: ag_news).
TASK = current_task()
_LABEL_SPECS = TASK.labels
CLASS_TO_LABEL = {spec["raw"]: spec["id"] for spec in _LABEL_SPECS}
TASK_LABELS = list(CLASS_TO_LABEL.values())
METHODS = {
    "centroid": "nearest class centroid, softmax(cosine / T) with T fitted on held-out train articles",
    "logreg": "LogisticRegression (C chosen by CV on the train embeddings) on embeddings, native predict_proba",
}


@dataclass
class Prediction:
    label: str
    confidence: float
    probabilities: dict[str, float]


GRID_C = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]  # candidate C values for the logreg method


def make_logreg(C: float) -> LogisticRegression:
    return LogisticRegression(C=C, max_iter=2000)


def load_train_sample(per_class: int, seed: int) -> pd.DataFrame:
    """Stratified sample of the train split: per_class articles of each label, shuffled."""
    train = load_split(TASK, "train")
    return (train.groupby("label").sample(per_class, random_state=seed)
            .sample(frac=1, random_state=seed).reset_index(drop=True))


def softmax(rows: np.ndarray) -> np.ndarray:
    exps = np.exp(rows - rows.max(axis=1, keepdims=True))
    return exps / exps.sum(axis=1, keepdims=True)


def centroids(vectors: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Unit-length mean embedding per label, in TASK_LABELS order."""
    means = np.stack([vectors[labels == label].mean(axis=0) for label in TASK_LABELS])
    return means / np.linalg.norm(means, axis=1, keepdims=True)


def fit_temperature(sims: np.ndarray, labels: np.ndarray) -> float:
    """Temperature with the lowest log loss of softmax(sims / T) on labeled held-out rows."""
    true_idx = np.array([TASK_LABELS.index(label) for label in labels])
    best_t, best_loss = 1.0, float("inf")
    for t in np.geomspace(0.002, 1.0, 200):
        p = softmax(sims / t)[np.arange(len(true_idx)), true_idx]
        loss = -np.log(np.clip(p, 1e-12, 1)).mean()
        if loss < best_loss:
            best_t, best_loss = float(t), loss
    return best_t


class EmbedSupClassifier:
    """Single-label text classifier trained on embeddings of labeled train articles."""

    def __init__(self, method: str = "centroid", model: str | None = None, host: str | None = None,
                 timeout: float = 300.0) -> None:
        if method not in METHODS:
            raise ValueError(f"method must be one of {list(METHODS)}")
        self.method = method
        self.model = model or os.getenv("EMBEDSUP_MODEL") or DEFAULT_MODEL
        self.host = (host or os.getenv("OLLAMA_HOST") or DEFAULT_HOST).rstrip("/")
        if not self.host.startswith("http"):
            self.host = f"http://{self.host}"
        self.client = httpx.Client(timeout=timeout)
        self.hyperparameters: dict = {"method": method, "embedding_model": self.model}
        self.digest = ""  # Ollama model digest, set by check_ready()
        self._centroids: np.ndarray | None = None
        self._temperature: float | None = None
        self._logreg: LogisticRegression | None = None

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
        self.hyperparameters["embedding_model_digest"] = self.digest[:12]

    def embed(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        vectors = []
        for i in range(0, len(texts), batch_size):
            r = self.client.post(f"{self.host}/api/embed", json={"model": self.model, "input": texts[i : i + batch_size]})
            r.raise_for_status()
            vectors += r.json()["embeddings"]  # Ollama returns unit-length vectors
        return np.asarray(vectors, dtype=np.float32)

    def embed_train(self, train: pd.DataFrame, seed: int, batch_size: int = 32) -> tuple[np.ndarray, float, bool]:
        """Train embeddings, from the cache when available: (vectors, seconds it took to embed, cache hit)."""
        slug = re.sub(r"[^A-Za-z0-9.]+", "-", self.model)
        stem = CACHE_DIR / f"train_{TASK.name}_{slug}_{self.digest[:12] or 'nodigest'}_n{len(train)}_seed{seed}"
        vec_path, meta_path = stem.with_suffix(".npy"), stem.with_suffix(".json")
        texts_sha = hashlib.sha1("\n".join(train["text"]).encode()).hexdigest()
        if vec_path.exists() and meta_path.exists():
            meta = json.loads(meta_path.read_text())
            if meta.get("texts_hash") == texts_sha and meta.get("digest") == self.digest:
                return np.load(vec_path), meta["embed_seconds"], True
        start = time.perf_counter()
        vectors = self.embed(train["text"].tolist(), batch_size)
        seconds = time.perf_counter() - start
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.save(vec_path, vectors)
        meta_path.write_text(json.dumps({"model": self.model, "digest": self.digest, "n": len(train), "seed": seed,
                                         "texts_hash": texts_sha, "embed_seconds": round(seconds, 3)}) + "\n")
        return vectors, seconds, False

    def fit(self, vectors: np.ndarray, labels: list[str], seed: int = 0, C: float = 10.0) -> "EmbedSupClassifier":
        labels = np.asarray(labels)
        if self.method == "centroid":
            # Temperature from a held-out 20% of the train sample, so it only changes confidence, never the label.
            order = np.random.default_rng(seed).permutation(len(labels))
            cut = int(len(order) * 0.8)
            fit_idx, held_idx = order[:cut], order[cut:]
            sims = vectors[held_idx] @ centroids(vectors[fit_idx], labels[fit_idx]).T
            self._temperature = fit_temperature(sims, labels[held_idx])
            self._centroids = centroids(vectors, labels)
            self.hyperparameters.update(temperature=round(self._temperature, 5), temperature_holdout=len(held_idx))
        else:
            self._logreg = make_logreg(C).fit(vectors, labels)
            self.hyperparameters.update({k: v for k, v in self._logreg.get_params().items()
                                         if v is None or isinstance(v, (bool, int, float, str))})
        return self

    def predict_vectors(self, vectors: np.ndarray) -> list[Prediction]:
        if self.method == "centroid":
            rows, classes = softmax((vectors @ self._centroids.T) / self._temperature), TASK_LABELS
        else:
            rows, classes = self._logreg.predict_proba(vectors), list(self._logreg.classes_)
        preds = []
        for row in rows:
            probabilities = {label: float(row[classes.index(label)]) for label in TASK_LABELS}
            label = max(probabilities, key=probabilities.get)
            # Confidence = max p, the probability of the chosen label.
            preds.append(Prediction(label, probabilities[label], probabilities))
        return preds

    def predict_many(self, texts: list[str], batch_size: int = 32) -> list[Prediction]:
        return self.predict_vectors(self.embed(texts, batch_size))

    def predict(self, text: str) -> Prediction:
        return self.predict_many([text])[0]


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    text = " ".join(sys.argv[1:])
    clf = EmbedSupClassifier()
    clf.check_ready()
    train = load_train_sample(per_class=1000, seed=0)
    vectors, _, _ = clf.embed_train(train, seed=0)
    pred = clf.fit(vectors, train["label"].tolist()).predict(text)
    print(f"label:      {pred.label}")
    print(f"confidence: {pred.confidence:.2f}")
    for label, p in sorted(pred.probabilities.items(), key=lambda kv: -kv[1]):
        print(f"  {label:<10} {p:.3f}")


if __name__ == "__main__":
    main()
