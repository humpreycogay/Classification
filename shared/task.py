"""Task definitions: which dataset, labels, instruction and sampling a benchmark run uses.

A task is a manifest in data/tasks/<name>.json (see ag_news.json). Part of the shared benchmark library
(shared/), imported by the runner and by every app's evaluate.py and classifier.py.

Apps select the task before importing their classifier:

    from task import select_task_from_argv
    select_task_from_argv()          # --task NAME on the command line -> BENCHMARK_TASK
    ...
    TASK = current_task()            # in classifier.py: labels, instruction, splits

Only single-label classification is supported; other task types are rejected. Data files are
downloaded on first use and checked against the manifest's SHA-256. The calibration split is carved
from train (never from test) and is the only data zero-shot apps may fit anything on.
"""

from __future__ import annotations

import hashlib
import json
import os
import ssl
import sys
import urllib.request
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
TASKS_DIR = DATA_DIR / "tasks"
DEFAULT_TASK = "ag_news"
SUPPORTED_TASK_TYPES = {"single_label"}
SAMPLING_STRATEGIES = {"balanced", "proportional"}


@dataclass(frozen=True)
class Task:
    name: str
    manifest: dict = field(repr=False)

    @property
    def labels(self) -> list[dict]:
        return self.manifest["labels"]

    @property
    def label_ids(self) -> list[str]:
        return [spec["id"] for spec in self.labels]

    @property
    def label_names(self) -> dict[str, str]:
        return {spec["id"]: spec.get("name", spec["id"]) for spec in self.labels}

    @property
    def label_descriptions(self) -> dict[str, str]:
        return {spec["id"]: spec.get("description", spec.get("name", spec["id"])) for spec in self.labels}

    @property
    def raw_to_label(self) -> dict:
        return {spec["raw"]: spec["id"] for spec in self.labels}

    def __getattr__(self, key: str):  # manifest fields as attributes: instruction, sampling, tiers, ...
        try:
            return self.manifest[key]
        except KeyError:
            raise AttributeError(key) from None


def load_task(name: str | None = None) -> Task:
    name = name or DEFAULT_TASK
    path = TASKS_DIR / f"{name}.json"
    if not path.exists():
        known = ", ".join(sorted(p.stem for p in TASKS_DIR.glob("*.json")))
        raise ValueError(f"unknown task {name!r} (manifests in data/tasks/: {known})")
    manifest = json.loads(path.read_text())
    if manifest.get("task_type") not in SUPPORTED_TASK_TYPES:
        raise ValueError(f"task {name}: task_type {manifest.get('task_type')!r} is not supported "
                         f"(supported: {', '.join(sorted(SUPPORTED_TASK_TYPES))})")
    if manifest.get("sampling", "balanced") not in SAMPLING_STRATEGIES:
        raise ValueError(f"task {name}: sampling must be one of {sorted(SAMPLING_STRATEGIES)}")
    ids = [spec["id"] for spec in manifest["labels"]]
    if len(set(ids)) != len(ids):
        raise ValueError(f"task {name}: duplicate label ids")
    return Task(name, manifest)


def select_task_from_argv(argv: list[str] | None = None) -> str:
    """Put `--task NAME` from the command line into BENCHMARK_TASK, before classifier modules import it."""
    argv = sys.argv if argv is None else argv
    for i, arg in enumerate(argv):
        if arg == "--task" and i + 1 < len(argv):
            os.environ["BENCHMARK_TASK"] = argv[i + 1]
        elif arg.startswith("--task="):
            os.environ["BENCHMARK_TASK"] = arg.split("=", 1)[1]
    return os.environ.get("BENCHMARK_TASK", DEFAULT_TASK)


def current_task() -> Task:
    return load_task(os.environ.get("BENCHMARK_TASK") or DEFAULT_TASK)


# ---------------------------------------------------------------- data files

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _ssl_context() -> ssl.SSLContext:
    """certifi's CA bundle when installed: the python.org macOS build ships without root certificates."""
    try:
        import certifi
    except ImportError:
        return ssl.create_default_context()
    return ssl.create_default_context(cafile=certifi.where())


def split_path(task: Task, split: str) -> Path:
    """Local file of a split, downloaded on first use and checked against the manifest checksum."""
    spec = task.files.get(split)
    if spec is None:
        raise ValueError(f"task {task.name} has no {split!r} split")
    path = DATA_DIR / spec["path"]
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {task.name} {split} split -> {path}", flush=True)
        with urllib.request.urlopen(spec["url"], timeout=120, context=_ssl_context()) as r:
            path.write_bytes(r.read())
    if spec.get("sha256") and sha256(path) != spec["sha256"]:
        raise ValueError(f"{path} does not match the checksum in data/tasks/{task.name}.json "
                         "(file changed or a different version was downloaded)")
    return path


@lru_cache(maxsize=8)
def _load(task_name: str, split: str) -> pd.DataFrame:
    task = load_task(task_name)
    fmt = task.format
    path = split_path(task, split)
    if fmt["type"] != "csv":
        raise ValueError(f"task {task.name}: format {fmt['type']!r} is not supported")
    if fmt.get("header"):
        df = pd.read_csv(path)
    else:
        df = pd.read_csv(path, header=None, names=fmt["columns"])
    raw = df[fmt["label_column"]]
    unknown = sorted(set(raw) - set(task.raw_to_label))
    if unknown:
        raise ValueError(f"task {task.name} {split}: labels not in the manifest: {unknown[:5]}")
    df["label"] = raw.map(task.raw_to_label)
    text = pd.Series([fmt["text_template"].format(**row) for row in df[fmt["columns"]].to_dict("records")],
                     index=df.index, dtype=object)
    for old, new in fmt.get("replace", []):
        text = text.str.replace(old, new, regex=False)
    df["text"] = text
    return df


def load_split(task: Task, split: str) -> pd.DataFrame:
    """A split as a DataFrame with `label` (label id) and `text`; the row index is the row in the file."""
    return _load(task.name, split).copy()


def calibration_split(task: Task) -> pd.DataFrame:
    """Calibration articles: a stratified sample of train (per_label per label, seed from the manifest).
    Never drawn from test; the only data zero-shot apps may fit anything (e.g. a temperature) on."""
    spec = task.calibration
    if spec.get("from") != "train":
        return load_split(task, spec["from"])
    train = load_split(task, "train")
    per = spec["per_label"]
    parts = [group.sample(min(per, len(group)), random_state=spec["seed"]) for _, group in train.groupby("label")]
    return pd.concat(parts).sample(frac=1, random_state=spec["seed"])


# ---------------------------------------------------------------- sampling

def stratified_sample(df: pd.DataFrame, n: int, seed: int, strategy: str = "balanced") -> pd.DataFrame:
    """Exactly n rows, stratified by label (n = 0 or n = len(df): the whole set, unshuffled).

    balanced:     each label gets n // k rows; the n % k leftover rows go one each to the first labels
                  in sorted order. When k divides n this equals groupby("label").sample(n // k).
    proportional: each label gets n x its share of df, rounded down; the leftover rows go to the labels
                  with the largest remainders (ties: sorted label order), so rare labels stay rare.
    Rows are drawn per label (sorted) from one generator seeded with `seed`, then shuffled with `seed`.
    Sizes that cannot be produced exactly raise ValueError.
    """
    if n < 0 or n > len(df):
        raise ValueError(f"sample size must be between 0 (whole set) and {len(df)}, got {n}")
    if n == 0 or n == len(df):
        return df
    labels = sorted(df["label"].unique())
    available = df["label"].value_counts()
    if strategy == "balanced":
        per_class, extra = divmod(n, len(labels))
        counts = {label: per_class + (i < extra) for i, label in enumerate(labels)}
    elif strategy == "proportional":
        exact = {label: n * available[label] / len(df) for label in labels}
        counts = {label: int(exact[label]) for label in labels}
        leftover = n - sum(counts.values())
        by_remainder = sorted(labels, key=lambda label: (-(exact[label] - counts[label]), labels.index(label)))
        for label in by_remainder[:leftover]:
            counts[label] += 1
    else:
        raise ValueError(f"unknown sampling strategy {strategy!r}")
    short = [f"{label} ({counts[label]} > {available[label]})" for label in labels if counts[label] > available[label]]
    if short:
        raise ValueError(f"not enough articles for a stratified sample of {n}: {', '.join(short)}")
    rng = np.random.RandomState(seed)
    parts = [group.sample(counts[label], random_state=rng) for label, group in df.groupby("label")]
    return pd.concat(parts).sample(frac=1, random_state=seed)


def sample(task: Task, n: int, seed: int) -> pd.DataFrame:
    """The test sample every app classifies for (task, n, seed), with sample_id = row in the test file."""
    test = load_split(task, "test")
    return stratified_sample(test, n, seed, task.manifest.get("sampling", "balanced")).rename_axis("sample_id").reset_index()


def test_size(task: Task) -> int:
    return len(load_split(task, "test"))


def tiers(task: Task) -> dict[str, int]:
    """Named sample sizes: the manifest's tiers plus full = the whole test split."""
    size = test_size(task)
    named = {k: v for k, v in task.manifest.get("tiers", {}).items() if v < size}
    return {**named, "full": size}


def tier_name(task: Task, n: int) -> str:
    return next((name for name, size in tiers(task).items() if size == n), "custom")


def sample_sha1(df: pd.DataFrame) -> str:
    """Identity of a sample: labels and texts in order."""
    return hashlib.sha1("\n".join(df["label"] + "\t" + df["text"]).encode()).hexdigest()


def result_stem(task: Task | str, n: int, seed: int) -> str:
    """Result identity: predictions_<stem>.csv and run_<stem>.json."""
    name = task if isinstance(task, str) else task.name
    return f"{name}_n{n}_seed{seed}"


def overlap(task: Task, texts: pd.Series) -> dict:
    """Exact-text overlap of test texts with train and calibration (known duplicates in some corpora)."""
    out = {}
    for split in ("train",):
        if split in task.files:
            out[f"in_{split}"] = int(texts.isin(set(load_split(task, split)["text"])).sum())
    out["in_calibration"] = int(texts.isin(set(calibration_split(task)["text"])).sum()) if "train" in task.files else 0
    return out
