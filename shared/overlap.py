"""Training-overlap check for models trained on a task's data (App.trained_on_tasks in runner/apps.py).

An app's setup.sh compares the texts its model saw in training with the task's test texts and writes
results/overlap.json. The runner skips the task when `overlap` > 0. Overlap counts only test texts the model saw
that are not also in the task's own train split: those (the dataset's own train/test duplicates, e.g. 22 AG News
and 29 Banking77 test texts) are seen by every supervised baseline too, so they are reported
(`dataset_duplicates`), not counted. Texts match when their letters and digits agree (case-folded; spacing and
punctuation ignored), so a source that joins fields differently still matches: AG News is "title. description" here
and "title description" on the Hugging Face Hub.

Part of the shared benchmark library (shared/); used only at setup, never in a timed run.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

from task import load_split, load_task


def normalize_text(text) -> str:
    text = text if isinstance(text, str) else json.dumps(text, ensure_ascii=False, sort_keys=True)
    return re.sub(r"[\W_]+", "", text.casefold())


def check(task_name: str, seen: dict[str, Iterable], note: str = "") -> dict:
    """Overlap entry for one task. `seen` maps a name for each training source (e.g. "train", "validation") to the
    texts the model saw there."""
    task = load_task(task_name)
    test = set(load_split(task, "test")["text"].map(normalize_text))
    train = set(load_split(task, "train")["text"].map(normalize_text))
    by_source = {}
    extra: set[str] = set()
    duplicates: set[str] = set()
    for source, texts in seen.items():
        hits = {normalize_text(t) for t in texts} & test
        by_source[source] = {"test_texts": len(hits), "not_in_train_split": len(hits - train)}
        extra |= hits - train
        duplicates |= hits & train
    parts = [f"{s} {v['test_texts']}" for s, v in by_source.items()]
    detail = (f"test texts seen in training: {', '.join(parts)}; {len(duplicates)} of them are the dataset's own "
              f"train/test duplicates (also seen by the supervised baselines), {len(extra)} are not"
              + (f"; {note}" if note else ""))
    return {"overlap": len(extra), "dataset_duplicates": len(duplicates), "by_source": by_source,
            "our_test_texts": len(test), "detail": detail}


def write(app_dir: Path, entries: dict[str, dict]) -> Path:
    """Merge entries ({task: check(...)}) into app_dir/results/overlap.json and print them."""
    path = app_dir / "results" / "overlap.json"
    path.parent.mkdir(exist_ok=True)
    try:
        current = json.loads(path.read_text())
    except (OSError, ValueError):
        current = {}
    current.update(entries)
    path.write_text(json.dumps(current, indent=2) + "\n")
    for task, info in entries.items():
        print(f"overlap check, {task}: {info['overlap']} -> {'skipped' if info['overlap'] else 'runs'} "
              f"({info['detail']})")
    return path
