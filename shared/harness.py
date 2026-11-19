"""Shared evaluation harness: everything an app's evaluate.py does apart from building its classifier.

Part of the shared benchmark library (shared/), imported by every app from its own .venv. Bump
HARNESS_VERSION whenever a run's behaviour or output changes; it is recorded in the run metadata, with
shared_sha1, a fingerprint of all the shared library's code the run used.

Apps find the library with the bootstrap at the top of evaluate.py: BENCHMARK_SHARED_DIR if set (the runner
sets it), else the shared/ folder next to the app folders.

An app's evaluate.py selects the task, imports its classifier and hands the harness an adapter: a
function that receives the parsed arguments and a Context, builds the classifier and returns a Setup.

    ...  # bootstrap: APP_DIR, and shared/ on sys.path
    from task import select_task_from_argv
    select_task_from_argv()  # --task NAME -> BENCHMARK_TASK; must run before classifier.py reads the task
    from classifier import TASK_LABELS, NLIClassifier
    from harness import Setup, main

    def adapter(args, ctx):
        clf = NLIClassifier(TASK_LABELS)
        return Setup(model=clf.model, classifier=clf, axis="batch_size", value=args.batch_size,
                     predict=lambda texts, v=args.batch_size: clf.predict_many(texts, batch_size=v))

    main(adapter, app_dir=APP_DIR, doc=__doc__, key_packages=[...], add_arguments=lambda p: p.add_argument("--batch-size", ...))

The harness owns the rules every app must follow the same way: the shared arguments, the task's
stratified sample, load / warm-up / classification timing (warm-up is never part of the timed run),
interactive latency and the optional sweep (timing.py), a sanity check of the predictions, the
console report, and the predictions CSV and run metadata JSON (names from task.py).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Callable

import pandas as pd

from task import Task, current_task, load_split, result_stem, sample, sample_sha1, tier_name
from timing import (SWEEP_ITEMS, SWEEP_TEXTS, GpuProbe, Progress, hf_model_bytes, interactive, percentiles, sweep,
                    timed_calls, unseen_texts, write_status)

HARNESS_VERSION = "3"  # 2: imported from shared/ instead of copied into each app; records shared_sha1
# 3: prints "progress:" lines (items done, %, time left) during the main run and the latency run
SHARED_DIR = Path(__file__).resolve().parent
# One status file per running app (./run status reads them): BENCHMARK_PROGRESS_DIR, else reports/progress/.
PROGRESS_DIR = Path(os.environ.get("BENCHMARK_PROGRESS_DIR") or SHARED_DIR.parent / "reports" / "progress")
THRESHOLDS = [0.0, 0.5, 0.7, 0.9]  # confidence gates shown in the console report

PER_CALL_UNITS = {
    "batch_size": "batch",
    "concurrency": "request (under the run's concurrency, so it includes queueing)",
}


@dataclass
class Setup:
    """What an app's adapter returns: its classifier and how to call it.

    predict(texts, v) classifies texts; v is the batch size or concurrency on `axis` (the sweep varies
    it; interactive latency passes 1), and predict(texts) must use the run's own `value`.
    """

    model: str  # description recorded in the run metadata
    predict: Callable[..., list]  # -> objects with .label, .confidence, .probabilities
    classifier: object = None  # its `stats` (RequestStats), if any, give per-request latency, retries, tokens
    axis: str | None = None  # "batch_size" | "concurrency" | None (one call for the whole sample)
    value: int | None = None  # the run's batch size or concurrency
    sequential: bool = False  # axis None, but one article per call (per-call timing per article)
    model_repo: str | None = None  # Hugging Face repo, for the model's size on disk
    results: str = ""  # subfolder of results/ for a variant, so it never overwrites the plain app
    metadata: dict = field(default_factory=dict)  # app-specific run metadata (training, calibration, ...)
    # Called just before the timed run; the function it returns is called right after, and its dict is
    # added to the run metadata (counters that must cover the main run only).
    on_main: Callable[[], Callable[[], dict]] | None = None

    @property
    def step(self) -> int | None:
        """Texts per timed call (None = one call for the whole sample)."""
        return 1 if self.sequential else self.value if self.axis == "batch_size" else None

    @property
    def per_call_unit(self) -> str:
        if self.axis:
            return PER_CALL_UNITS[self.axis]
        return "article (sequential)" if self.sequential else "whole sample (one vectorized call)"

    @property
    def warmup_items(self) -> int:
        # One full batch for batched apps, so GPU shape compilation for full batches is not timed;
        # otherwise one article (model load, first connection).
        return self.value if self.axis == "batch_size" else 1


@dataclass
class Context:
    """Given to the adapter; everything it does counts as load time unless excluded."""

    task: Task
    args: argparse.Namespace
    sample: pd.DataFrame  # the texts this run classifies (sample_id, text, label)
    app_dir: Path  # the app folder (results/, caches, hyperparameters.json)
    _start: float = field(default_factory=time.perf_counter)
    _excluded: float = 0.0
    _added: float = 0.0
    warmup: tuple[int, float] | None = None  # (items, seconds), if the app warmed up itself

    @contextmanager
    def exclude(self):
        """Time spent inside is not load time (training, hyperparameter selection: recorded separately)."""
        start = time.perf_counter()
        try:
            yield
        finally:
            self._excluded += time.perf_counter() - start

    def add_load(self, seconds: float) -> None:
        """Count part of an excluded step as load time after all (loading a cached fit)."""
        self._added += seconds

    def warm_up(self, run: Callable[[], object], items: int) -> None:
        """Warm up now, inside the adapter, when later build steps need a loaded model (embedding the train set)."""
        with self.exclude():
            start = time.perf_counter()
            run()
            self.warmup = (items, time.perf_counter() - start)

    def load_seconds(self) -> float:
        return time.perf_counter() - self._start - self._excluded + self._added


def shared_sha1() -> str:
    """Fingerprint of the shared library's code (every .py file in shared/, by name and content)."""
    h = hashlib.sha1()
    for path in sorted(SHARED_DIR.glob("*.py")):
        h.update(path.name.encode() + b"\0" + path.read_bytes() + b"\0")
    return h.hexdigest()


def package_versions(names: list[str]) -> dict[str, str]:
    """Versions of the named packages installed in this app's .venv (missing ones are left out)."""
    versions = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    return versions


def check_predictions(df: pd.DataFrame, labels: list[str], prob_cols: list[str]) -> None:
    """Fail before writing anything if the classifier returned labels or scores outside the contract.
    (The runner's validator applies the full rules; this catches a broken app at the source.)"""
    problems = []
    bad = ~df["pred"].isin(labels)
    if bad.any():
        problems.append(f"{bad.sum()} predictions are not task labels (e.g. {df.loc[bad, 'pred'].iloc[0]!r})")
    for col in ["confidence", *prob_cols]:
        values = pd.to_numeric(df[col], errors="coerce")
        if not values.map(lambda v: math.isfinite(v) and -1e-9 <= v <= 1 + 1e-9).all():
            problems.append(f"{col} has values that are missing, not finite or outside [0, 1]")
    if problems:
        raise SystemExit("Invalid predictions: " + "; ".join(problems))


def report(df: pd.DataFrame, labels: list[str], thresholds: list[float] = THRESHOLDS) -> None:
    accuracy = (df["label"] == df["pred"]).mean()
    print(f"\nAccuracy: {accuracy:.1%}  ({(df['label'] == df['pred']).sum()}/{len(df)})\n")

    rows = []
    for label in labels:
        tp = ((df["pred"] == label) & (df["label"] == label)).sum()
        precision = tp / max((df["pred"] == label).sum(), 1)
        recall = tp / max((df["label"] == label).sum(), 1)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append({"label": label, "precision": precision, "recall": recall, "f1": f1,
                     "support": (df["label"] == label).sum()})
    per_class = pd.DataFrame(rows).set_index("label")
    print(per_class.to_string(float_format=lambda x: f"{x:.3f}"))
    print(f"\nMacro F1: {per_class['f1'].mean():.3f}")

    print("\nConfusion matrix (rows = true, cols = predicted):")
    confusion = pd.crosstab(df["label"], df["pred"]).reindex(index=labels, columns=labels, fill_value=0)
    print(confusion.to_string())

    # Confidence-gated routing: auto-accept only confident predictions, send the rest to review.
    print("\nConfidence gating (auto-accept when confidence >= threshold):")
    print(f"  {'threshold':>9}  {'coverage':>8}  {'accuracy':>8}")
    for t in thresholds:
        kept = df[df["confidence"] >= t]
        acc = (kept["label"] == kept["pred"]).mean() if len(kept) else float("nan")
        print(f"  {t:>9.2f}  {len(kept) / len(df):>8.1%}  {acc:>8.1%}")


def parse_args(doc: str, task: Task, add_arguments: Callable[[argparse.ArgumentParser], None] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=200, help="sample size (0 = full test set)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task", default=task.name, help="task manifest in data/tasks/ (default: ag_news)")
    parser.add_argument("--latency-items", type=int, default=100,
                        help="articles timed one at a time for interactive latency (0 = skip)")
    parser.add_argument("--sweep", action="store_true", help="also time several batch sizes / concurrency levels")
    if add_arguments:
        add_arguments(parser)
    return parser.parse_args()


def main(adapter: Callable[[argparse.Namespace, Context], Setup], *, app_dir: Path, doc: str | None,
         key_packages: list[str], add_arguments: Callable[[argparse.ArgumentParser], None] | None = None) -> None:
    """Run one evaluation: sample, build the classifier (adapter), time it, check, report and write the results."""
    task = current_task()
    labels = task.label_ids
    args = parse_args(doc or "", task, add_arguments)

    df = sample(task, args.n, args.seed)  # stratified as the task manifest says; sample_id = row in the test file
    # Status file for ./run status, from loading to the written results (removed then; a killed run leaves it).
    # BENCHMARK_APP_NAME: the runner's name for the entry (variants share a folder).
    app_name = os.environ.get("BENCHMARK_APP_NAME") or app_dir.name
    status_path = PROGRESS_DIR / f"{app_name}-{os.getpid()}.json"
    info = {"app": app_name, "folder": app_dir.name, "task": task.name, "n": len(df), "seed": args.seed,
            "run_id": os.environ.get("BENCHMARK_RUN_ID"), "started": time.time()}
    write_status(status_path, **info, phase="loading")
    ctx = Context(task, args, df, app_dir)
    setup = adapter(args, ctx)
    load_seconds = ctx.load_seconds()
    print(f"Classifying {len(df)} articles with {setup.model} ...")

    # Timing (timing.py): per-call samples, interactive latency and an optional batch/concurrency sweep.
    run = setup.predict
    write_status(status_path, **info, phase="warm-up")
    if ctx.warmup:
        warmup_items, warmup_seconds = ctx.warmup
    else:
        warm = df["text"].head(setup.warmup_items).tolist()
        warmup_start = time.perf_counter()
        run(warm)
        warmup_items, warmup_seconds = len(warm), time.perf_counter() - warmup_start
    stats = getattr(setup.classifier, "stats", None)  # HTTP apps: per-request latency, retries, tokens
    if stats:
        stats.reset()  # warm-up requests are not counted

    finish_main = setup.on_main() if setup.on_main else None
    probe = GpuProbe()
    start = time.perf_counter()
    # Progress lines: after every call, and for HTTP apps (one call for the whole sample) from their completed
    # requests, which the watcher polls. They print to stdout only, so timings are unchanged.
    completed = (lambda: len(stats.request_seconds)) if stats else None
    preds, call_seconds = timed_calls(run, df["text"].tolist(), setup.step, probe,
                                      Progress("classifying", len(df), status=status_path, info=info), completed)
    elapsed = time.perf_counter() - start
    main_metadata = finish_main() if finish_main else {}
    requests = stats.summary() if stats else None

    # Latency runs use test articles outside the sample, never ones the main run already sent.
    needed = args.latency_items + (SWEEP_TEXTS if args.sweep else 0)
    extra, reused = unseen_texts(load_split(task, "test"), df, needed, args.seed)
    latency_texts = extra[: args.latency_items]
    if latency_texts:
        print(f"Timing {len(latency_texts)} articles one at a time (interactive latency) ...")
    one_seconds = interactive(lambda t: run([t], 1), latency_texts,
                              Progress("latency", len(latency_texts), status=status_path, info=info)
                              if latency_texts else None)
    sweep_rows = None
    if args.sweep and setup.axis:
        print(f"Sweeping {setup.axis}: {SWEEP_ITEMS} fresh articles per setting ...")
        write_status(status_path, **info, phase="sweep")
        sweep_rows = sweep(run, setup.axis, extra[args.latency_items:])
    timing = {
        "axis": setup.axis, "value": setup.value, "warmup_items": warmup_items,
        "per_call": {"unit": setup.per_call_unit,
                     **(requests["per_request"] if requests else percentiles(call_seconds))},
        "interactive": {"unit": "article, one per call (batch 1 / concurrency 1)", **percentiles(one_seconds)},
        "sweep": sweep_rows,
        # False: latency / sweep articles are outside the sample; True: some repeat sample articles (full tier).
        "latency_texts_reused": reused,
        "gpu_memory": probe.result(),
        "model_bytes": hf_model_bytes(setup.model_repo),
        "model_bytes_source": f"Hugging Face cache: {setup.model_repo}" if setup.model_repo else None,
    }

    df["pred"] = [p.label for p in preds]
    df["confidence"] = [p.confidence for p in preds]
    prob_cols = [f"p_{label}" for label in labels]
    for label, col in zip(labels, prob_cols):
        df[col] = [p.probabilities.get(label, 0.0) for p in preds]
    print(f"Done in {elapsed:.2f}s ({len(df) / elapsed:.1f} articles/s)")
    check_predictions(df, labels, prob_cols)

    report(df, labels)

    stem = result_stem(task, len(df), args.seed)
    out = app_dir / "results" / setup.results / f"predictions_{stem}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df[["sample_id", "text", "label", "pred", "confidence", *prob_cols]].to_csv(out, index=False)
    print(f"\nPredictions saved to {out}")

    run_meta = {
        "app": app_dir.name,
        "model": setup.model,
        "n": len(df),
        "seed": args.seed,
        "n_requested": args.n,
        "task": task.name,
        "tier": tier_name(task, len(df)),
        # Set by the runner so it can tell this run's output from stale files.
        "run_id": os.environ.get("BENCHMARK_RUN_ID"),
        "sample_sha1": sample_sha1(df),
        "load_seconds": round(load_seconds, 3),
        "warmup_seconds": round(warmup_seconds, 3),
        "classify_seconds": round(elapsed, 6),
        "seconds_per_item": round(elapsed / len(df), 7),
        "items_per_second": round(len(df) / elapsed, 3),
        "timing": timing,
        "requests": requests,
        **setup.metadata,
        **main_metadata,
        "args": vars(args),
        "python": platform.python_version(),
        "packages": package_versions(key_packages),
        "harness_version": HARNESS_VERSION,
        "shared_sha1": shared_sha1(),
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    run_path = out.with_name(f"run_{stem}.json")
    run_path.write_text(json.dumps(run_meta, indent=2) + "\n")
    print(f"Run metadata saved to {run_path}")
    status_path.unlink(missing_ok=True)

    errors = df[df["label"] != df["pred"]].head(5)
    if len(errors):
        print("\nSample errors:")
        for _, r in errors.iterrows():
            print(f"  [true={r['label']}, pred={r['pred']}, conf={r['confidence']:.2f}] {r['text'][:110]}")
