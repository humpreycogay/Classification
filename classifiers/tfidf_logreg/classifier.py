"""TF-IDF + Logistic Regression: supervised baseline trained on the task's train split (AG News by default).

Word (1-2 gram) and character (3-5 gram, within word boundaries) TF-IDF features feed a
multinomial logistic regression. It is fitted only on AG News *train* articles, never the
test set. C is chosen from GRID by 3-fold cross-validation on the same train articles (training.py).
Probabilities are the model's native predict_proba.

Usage:
    python classifier.py "Stocks rallied after the Fed held rates steady."   # trains first (~1 min)
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from training import fit_cached

from task import current_task, load_split

# Labels, descriptions and instruction come from the task manifest (data/tasks/<task>.json, see task.py),
# chosen with --task on evaluate.py (default: ag_news).
TASK = current_task()
_LABEL_SPECS = TASK.labels
CLASS_TO_LABEL = {spec["raw"]: spec["id"] for spec in _LABEL_SPECS}
TASK_LABELS = list(CLASS_TO_LABEL.values())
GRID = [0.3, 1.0, 3.0, 10.0, 30.0, 100.0]  # candidate C values, chosen by CV on the train split
DEFAULT_C = 10.0  # used only when no selection is available (classifier.py demo)


@dataclass
class Prediction:
    label: str
    confidence: float
    probabilities: dict[str, float]


def load_train_set() -> pd.DataFrame:
    """The task's train split (text + label id), downloaded and checksum-checked by task.py."""
    return load_split(TASK, "train")


def load_train_sample(per_class: int, seed: int) -> pd.DataFrame:
    """Stratified sample of the train split, identical to embedsup's (same articles, same order)."""
    train = load_train_set()
    return (train.groupby("label").sample(per_class, random_state=seed)
            .sample(frac=1, random_state=seed).reset_index(drop=True))


def build(C: float) -> Pipeline:
    return Pipeline([
        ("tfidf", FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, strip_accents="unicode")),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True,
                                     max_features=300_000)),
        ])),
        ("clf", LogisticRegression(C=C, max_iter=1000)),
    ])


def json_params(estimator) -> dict:
    """Hyperparameters of a scikit-learn estimator, keeping only JSON-friendly values."""
    params = {}
    for key, value in estimator.get_params(deep=True).items():
        try:
            json.dumps(value)  # tuples become lists; nested estimators (steps, transformer_list) are skipped
        except TypeError:
            continue
        params[key] = value
    return params


class TfidfLogRegClassifier:
    """Single-label text classifier: TF-IDF word + char n-grams, multinomial logistic regression."""

    def __init__(self, C: float = DEFAULT_C) -> None:
        self.pipeline = build(C)
        self.model = f"TF-IDF (word 1-2 + char_wb 3-5) + LogisticRegression(C={C})"
        self.calibration = "native predict_proba (multinomial logistic regression)"

    def fit(self, texts: list[str], labels: list[str]) -> "TfidfLogRegClassifier":
        self.pipeline.fit(texts, labels)
        return self

    def fit_cached(self, texts: list[str], labels: list[str], cache_dir: Path) -> tuple[float, bool, float]:
        """Fit, or load the same fit from cache_dir: (fit seconds, cached?, cache load seconds)."""
        key = {"model": self.model, "hyperparameters": self.hyperparameters()}
        self.pipeline, fit_seconds, cached, load_seconds = fit_cached(self.pipeline, key, texts, labels, cache_dir)
        return fit_seconds, cached, load_seconds

    def hyperparameters(self) -> dict:
        return json_params(self.pipeline)

    def predict_many(self, texts: list[str]) -> list[Prediction]:
        classes = list(self.pipeline.classes_)
        preds = []
        for row in self.pipeline.predict_proba(texts):
            probabilities = {label: float(row[classes.index(label)]) for label in TASK_LABELS}
            label = max(probabilities, key=probabilities.get)
            # Confidence = max p, the probability of the chosen label.
            preds.append(Prediction(label, probabilities[label], probabilities))
        return preds

    def predict(self, text: str) -> Prediction:
        return self.predict_many([text])[0]


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    text = " ".join(sys.argv[1:])
    train = load_train_set()
    clf = TfidfLogRegClassifier().fit(train["text"].tolist(), train["label"].tolist())
    pred = clf.predict(text)
    print(f"label:      {pred.label}")
    print(f"confidence: {pred.confidence:.2f}")
    for label, p in sorted(pred.probabilities.items(), key=lambda kv: -kv[1]):
        print(f"  {label:<10} {p:.3f}")


if __name__ == "__main__":
    main()
