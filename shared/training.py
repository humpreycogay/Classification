"""Hyperparameter selection by cross-validation on the train split, and a cache of fitted models.

Part of the shared benchmark library (shared/); used by the supervised baselines (tfidf_logreg,
tfidf_svm, tfidf_cnb and embedsup). Only training data is ever used: the test set
plays no part in choosing hyperparameters.

    select(...)      k-fold stratified CV accuracy for each grid value; the best mean wins (ties: the
                     first value in grid order). Stored in <app>/hyperparameters.json per variant and
                     reused while the train data and grid are unchanged.
    fit_cached(...)  fit a pipeline, or load the fitted one from <app>/results/cache/ when the train
                     data, hyperparameters and scikit-learn version are the same.
    fit_baseline(...) both, for a TF-IDF baseline's evaluate.py (harness adapter), with the run's
                     "training" metadata.
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import sklearn
from sklearn.model_selection import StratifiedKFold, cross_val_score

CV_FOLDS = 3
CV_SEED = 0


def data_sha1(texts, labels) -> str:
    """Identity of a training set (texts or embedding rows, plus labels, in order)."""
    h = hashlib.sha1()
    if isinstance(texts, np.ndarray):
        h.update(np.ascontiguousarray(texts).tobytes())
    else:
        h.update("\n".join(texts).encode())
    h.update("\n".join(labels).encode())
    return h.hexdigest()


def select(make: Callable[[float], object], grid: list[float], X, y, *, param: str, store: Path, variant: str,
           n_jobs: int = 3) -> dict:
    """Choose `param` from `grid` by stratified CV accuracy on (X, y); cached in `store` under `variant`."""
    sha = data_sha1(X, y)
    saved = json.loads(store.read_text()) if store.exists() else {}
    entry = saved.get(variant)
    if entry and entry.get("train_sha1") == sha and entry.get("grid") == grid and entry.get("folds") == CV_FOLDS:
        return {**entry, "reused": True, **edge_check(entry, grid)}
    print(f"Selecting {param} from {grid} by {CV_FOLDS}-fold CV on {len(y)} train articles ...", flush=True)
    folds = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=CV_SEED)
    start = time.perf_counter()
    scores = {}
    for value in grid:
        s = cross_val_score(make(value), X, y, cv=folds, scoring="accuracy", n_jobs=n_jobs)
        scores[value] = (float(s.mean()), float(s.std(ddof=1)))
        print(f"  {param}={value}: CV accuracy {s.mean():.4f} ± {s.std(ddof=1):.4f}", flush=True)
    chosen = max(grid, key=lambda v: (scores[v][0], -grid.index(v)))
    entry = {
        "param": param, "grid": grid, "chosen": chosen, "method": f"{CV_FOLDS}-fold stratified CV accuracy on the train split",
        "folds": CV_FOLDS, "cv_seed": CV_SEED, "cv_accuracy": {str(v): round(m, 5) for v, (m, _) in scores.items()},
        "cv_sd": {str(v): round(sd, 5) for v, (_, sd) in scores.items()}, "n_train": len(y), "train_sha1": sha,
        "sklearn": sklearn.__version__, "seconds": round(time.perf_counter() - start, 1),
        "selected_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    store.write_text(json.dumps({**saved, variant: entry}, indent=2) + "\n")
    edge = edge_check(entry, grid)
    if edge["edge_still_improving"]:
        print(f"  WARNING: {param}={chosen} is at the edge of the grid and still improving by more than one CV SD; "
              "the optimum may lie outside it", flush=True)
    elif edge["at_grid_edge"]:
        print(f"  note: {param}={chosen} is at the edge of the grid, but the curve is flat there (gain < one CV SD)", flush=True)
    return {**entry, "reused": False, **edge}


def edge_check(entry: dict, grid: list[float]) -> dict:
    """Whether the choice is the smallest or largest candidate, and if so whether CV accuracy is still rising
    toward that edge by more than one CV standard deviation (then the grid is too narrow; otherwise flat)."""
    chosen = entry["chosen"]
    at_edge = chosen in (min(grid), max(grid))
    improving = False
    if at_edge and len(grid) > 1:
        ordered = sorted(grid)
        neighbour = ordered[1] if chosen == ordered[0] else ordered[-2]
        gain = entry["cv_accuracy"][str(chosen)] - entry["cv_accuracy"][str(neighbour)]
        improving = gain > entry["cv_sd"][str(chosen)]
    return {"at_grid_edge": at_edge, "edge_still_improving": improving}


def fit_cached(pipeline, key: dict, texts, labels, cache_dir: Path) -> tuple[object, float, bool, float]:
    """(fitted pipeline, seconds the fit took, loaded from cache?, seconds to load it from cache)."""
    ident = {**key, "train_sha1": data_sha1(texts, labels), "sklearn": sklearn.__version__}
    digest = hashlib.sha1(json.dumps(ident, sort_keys=True).encode()).hexdigest()[:16]
    model_path, meta_path = cache_dir / f"model_{digest}.joblib", cache_dir / f"model_{digest}.json"
    if model_path.exists() and meta_path.exists():
        start = time.perf_counter()
        fitted = joblib.load(model_path)
        return fitted, json.loads(meta_path.read_text())["fit_seconds"], True, time.perf_counter() - start
    start = time.perf_counter()
    pipeline.fit(texts, labels)
    fit_seconds = time.perf_counter() - start
    cache_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    meta_path.write_text(json.dumps({**ident, "fit_seconds": round(fit_seconds, 3)}, indent=2) + "\n")
    return pipeline, fit_seconds, False, 0.0


def fit_baseline(ctx, make_classifier: Callable[[float], object], make: Callable[[float], object], grid: list[float],
                 train, *, param: str, override: float | None, variant: str, train_split: str,
                 probabilities: str) -> tuple[object, dict]:
    """Choose `param` by CV on the train split (unless overridden), then fit the classifier or load its cached
    fit. Selection and training are not load time (ctx: the harness Context); loading a cached fit is.
    Returns (fitted classifier, the run's "training" metadata)."""
    with ctx.exclude():
        texts, labels = train["text"].tolist(), train["label"].tolist()
        if override is None:  # reused while train data and grid are unchanged
            selection = select(make, grid, texts, labels, param=param, store=ctx.app_dir / "hyperparameters.json",
                               variant=f"{ctx.task.name}:{variant}")
            value = selection["chosen"]
        else:
            value = override
            selection = {"param": param, "chosen": value, "method": f"manual override (--{param})"}
        clf = make_classifier(value)
        print(f"Training {clf.model} on {len(train)} {ctx.task.name} train articles ...")
        train_seconds, train_cached, cache_load_seconds = clf.fit_cached(texts, labels, ctx.app_dir / "results" / "cache")
        # Articles whose text also appears verbatim in the train split (the corpus has a few duplicates).
        test_texts_in_train = int(ctx.sample["text"].isin(set(train["text"])).sum())
    ctx.add_load(cache_load_seconds)
    print("Loaded the cached fit" if train_cached else f"Trained in {train_seconds:.1f}s")
    return clf, {
        "supervision": "supervised",
        "train_split": train_split,
        "n_train": len(train),
        "train_seconds": round(train_seconds, 3),
        # A cached fit is loaded (its load time is in load_seconds); train_seconds is then the original fit time.
        "train_cached": train_cached,
        "hyperparameter_selection": selection,
        "test_texts_in_train": test_texts_in_train,
        "probabilities": probabilities,
        "calibration": clf.calibration,
        "hyperparameters": clf.hyperparameters(),
    }
