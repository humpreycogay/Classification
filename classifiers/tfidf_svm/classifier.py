"""TF-IDF + Linear SVM: supervised baseline trained on the task's train split (AG News by default).

Same word (1-2 gram) + character (3-5 gram) TF-IDF features as tfidf_logreg, with a
LinearSVC. An SVM gives margins, not probabilities, so two variants are reported:

    calibration="sigmoid"  CalibratedClassifierCV (Platt scaling, 5-fold CV on the *train* split).
                           Probabilities are calibrated; the label is their argmax.
    calibration="none"     Raw LinearSVC: the label is the largest margin. p_* = softmax(margins),
                           an UNCALIBRATED score that only ranks labels.

Neither variant ever sees the test set. C is chosen from GRID by 3-fold cross-validation of the
raw LinearSVC on the train split (training.py) and used by both variants.

Usage:
    python classifier.py "Stocks rallied after the Fed held rates steady."   # trains first
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from training import fit_cached

from task import current_task, load_split

# Labels, descriptions and instruction come from the task manifest (data/tasks/<task>.json, see task.py),
# chosen with --task on evaluate.py (default: ag_news).
TASK = current_task()
_LABEL_SPECS = TASK.labels
CLASS_TO_LABEL = {spec["raw"]: spec["id"] for spec in _LABEL_SPECS}
TASK_LABELS = list(CLASS_TO_LABEL.values())
GRID = [0.1, 0.3, 1.0, 3.0]  # candidate C values, chosen by CV on the train split
DEFAULT_C = 0.5  # used only when no selection is available (classifier.py demo)
CALIBRATIONS = {
    "sigmoid": "sigmoid (Platt) via CalibratedClassifierCV, 5-fold CV on the train split",
    "none": "softmax of LinearSVC margins (ranks labels, not a probability)",
}


@dataclass
class Prediction:
    label: str
    confidence: float
    probabilities: dict[str, float]


def load_train_set() -> pd.DataFrame:
    """The task's train split (text + label id), downloaded and checksum-checked by task.py."""
    return load_split(TASK, "train")


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


def build(C: float, calibration: str = "none") -> Pipeline:
    svm = LinearSVC(C=C)
    clf = CalibratedClassifierCV(svm, method="sigmoid", cv=5) if calibration == "sigmoid" else svm
    return Pipeline([
        ("tfidf", FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, strip_accents="unicode")),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True,
                                     max_features=300_000)),
        ])),
        ("clf", clf),
    ])


def softmax(rows: np.ndarray) -> np.ndarray:
    exps = np.exp(rows - rows.max(axis=1, keepdims=True))
    return exps / exps.sum(axis=1, keepdims=True)


class TfidfSVMClassifier:
    """Single-label text classifier: TF-IDF word + char n-grams, LinearSVC (optionally calibrated)."""

    def __init__(self, C: float = DEFAULT_C, calibration: str = "sigmoid") -> None:
        if calibration not in CALIBRATIONS:
            raise ValueError(f"calibration must be one of {list(CALIBRATIONS)}")
        self.pipeline = build(C, calibration)
        self.calibration_method = calibration
        self.calibration = CALIBRATIONS[calibration]
        suffix = ", sigmoid-calibrated" if calibration == "sigmoid" else ", uncalibrated"
        self.model = f"TF-IDF (word 1-2 + char_wb 3-5) + LinearSVC(C={C}){suffix}"

    def fit(self, texts: list[str], labels: list[str]) -> "TfidfSVMClassifier":
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
        if self.calibration_method == "sigmoid":
            rows = self.pipeline.predict_proba(texts)
        else:
            rows = softmax(self.pipeline.decision_function(texts))  # argmax = LinearSVC's own prediction
        preds = []
        for row in rows:
            probabilities = {label: float(row[classes.index(label)]) for label in TASK_LABELS}
            label = max(probabilities, key=probabilities.get)
            # Confidence = max p (calibrated), or the max softmax margin score (uncalibrated).
            preds.append(Prediction(label, probabilities[label], probabilities))
        return preds

    def predict(self, text: str) -> Prediction:
        return self.predict_many([text])[0]


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    text = " ".join(sys.argv[1:])
    train = load_train_set()
    clf = TfidfSVMClassifier().fit(train["text"].tolist(), train["label"].tolist())
    pred = clf.predict(text)
    print(f"label:      {pred.label}")
    print(f"confidence: {pred.confidence:.2f}")
    for label, p in sorted(pred.probabilities.items(), key=lambda kv: -kv[1]):
        print(f"  {label:<10} {p:.3f}")


if __name__ == "__main__":
    main()
