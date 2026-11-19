"""Zero-shot label shortlisting for tasks with more labels than an app can score at once.

Two-stage classification: a cheap first stage keeps the K most likely labels for each text, and the
app (an LLM, SemIf, Verdict) chooses among those K only. The first stage here is zero-shot, the same
method as embedsim: cosine similarity between the text's bge-m3 embedding (via Ollama) and each label's
text; it uses no labeled data, so an app stays zero-shot with it. Labels outside the shortlist get
probability 0, so the app's accuracy can never exceed the shortlist's recall (share of texts whose
true label is kept), which is measured on the task's calibration split (train data) and recorded.

An app's evaluate.py adds --shortlist with add_argument(), builds its classifier for
initial_labels(), and passes its harness Setup through apply(), which turns it into the two-stage variant.

Part of the shared benchmark library (shared/); used by SemIf, Verdict and the two LLM apps.
"""

from __future__ import annotations

import json
import os
import urllib.request

import numpy as np

from task import calibration_split

DEFAULT_MODEL = "bge-m3:567m"
DEFAULT_HOST = "http://127.0.0.1:11434"
METHOD = "zero-shot: bge-m3 cosine similarity to label texts (as embedsim), top K labels kept"


class Shortlister:
    def __init__(self, task, k: int, model: str | None = None, host: str | None = None, batch_size: int = 32) -> None:
        self.task, self.k, self.batch_size = task, k, batch_size
        self.model = model or os.getenv("SHORTLIST_MODEL") or DEFAULT_MODEL
        host = (host or os.getenv("OLLAMA_HOST") or DEFAULT_HOST).rstrip("/")
        self.host = host if host.startswith("http") else f"http://{host}"
        self.label_ids = task.label_ids
        template = task.manifest.get("embedding_label_template", "{name}: {description}")
        names, descriptions = task.label_names, task.label_descriptions
        texts = [template.format(name=names[i], description=descriptions[i]) for i in self.label_ids]
        self.label_vectors = self._embed(texts)

    def _embed(self, texts: list[str]) -> np.ndarray:
        vectors = []
        for i in range(0, len(texts), self.batch_size):
            body = json.dumps({"model": self.model, "input": texts[i:i + self.batch_size]}).encode()
            req = urllib.request.Request(f"{self.host}/api/embed", data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=300) as r:
                vectors += json.loads(r.read())["embeddings"]  # unit length
        return np.asarray(vectors, dtype=np.float32)

    def ranked(self, texts: list[str]) -> np.ndarray:
        """Label indices per text, most similar first."""
        return np.argsort(-(self._embed(texts) @ self.label_vectors.T), axis=1, kind="stable")

    def top_k(self, texts: list[str]) -> list[list[str]]:
        """The K most similar label ids per text (most similar first)."""
        return [[self.label_ids[j] for j in row[:self.k]] for row in self.ranked(texts)]

    def recall(self, texts: list[str], labels: list[str]) -> float:
        """Share of texts whose true label is in their shortlist."""
        kept = self.top_k(texts)
        return float(np.mean([label in keep for label, keep in zip(labels, kept)]))

    def describe(self, calibration_recall: float) -> dict:
        return {"k": self.k, "method": METHOD, "model": self.model,
                "recall_calibration": round(calibration_recall, 4)}


def add_argument(parser) -> None:
    parser.add_argument("--shortlist", type=int, default=0, metavar="K",
                        help="two-stage: keep each text's K most similar labels (zero-shot, shortlist.py) and "
                             "classify among those only; for tasks with more labels than this app takes")


def initial_labels(labels: dict, k: int) -> dict:
    """With a shortlist the classifier is built for K labels; each text then gets its own K (see apply)."""
    return dict(list(labels.items())[:k]) if k else labels


def apply(setup, clf, ctx, labels: dict, k: int):
    """Make a harness Setup two-stage: every call (warm-up, main, latency, sweep) is shortlisted first, via
    clf.predict_shortlisted(texts, label_sets, v). No-op when k is 0 (metadata "shortlist" is then None)."""
    setup.metadata["shortlist"] = None
    if not k:
        return setup
    shortlister = Shortlister(ctx.task, k)
    calibration = calibration_split(ctx.task)
    info = shortlister.describe(shortlister.recall(calibration["text"].tolist(), calibration["label"].tolist()))
    print(f"Shortlist: top {k} labels, recall on the calibration split "
          f"{info['recall_calibration']:.1%} (the ceiling for accuracy)")

    def predict(texts, v=setup.value):
        sets = [{label: labels[label] for label in keep} for keep in shortlister.top_k(texts)]
        return clf.predict_shortlisted(texts, sets, v)

    setup.predict = predict
    setup.model += f" + zero-shot shortlist of {k} labels"
    setup.results = f"shortlist{k}"  # a separate result set, so it never overwrites the plain app
    # The ceiling (share of texts whose true label was kept) on calibration and on this sample; not load time.
    with ctx.exclude():
        info["recall_test_sample"] = round(shortlister.recall(ctx.sample["text"].tolist(), ctx.sample["label"].tolist()), 4)
    setup.metadata["shortlist"] = info
    return setup
