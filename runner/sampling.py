"""The shared test sample for the runner: sampling, sample identity, tiers and result names.

A thin layer over task.py (the task manifest loader shared with every app). The runner computes
the expected sample itself, before any app runs, so it can reject unsupported sizes up front and
check every app's output against the exact texts, labels and sample hash. Functions take an
optional task name; the default is AG News.
"""

from __future__ import annotations

import pandas as pd

import shared_path  # noqa: F401  (puts shared/ on sys.path)
import task as tasks
from task import DATA_DIR, DEFAULT_TASK, ROOT, sample_sha1, stratified_sample  # noqa: F401 (re-exported)

DATA_PATH = DATA_DIR / "ag_news_test.csv"  # the default task's test file (tests skip when it is absent)


def load_test_set(task: str | None = None) -> pd.DataFrame:
    return tasks.load_split(tasks.load_task(task), "test")


def tiers(task: str | None = None) -> dict[str, int]:
    return tasks.tiers(tasks.load_task(task))


def tier_name(n: int, task: str | None = None) -> str:
    return tasks.tier_name(tasks.load_task(task), n)


def expected_sample(n: int, seed: int, task: str | None = None) -> pd.DataFrame:
    """The sample every app must produce for (task, n, seed), with sample_id = row in the test file."""
    return tasks.sample(tasks.load_task(task), n, seed)


def result_stem(n: int, seed: int, task: str | None = None) -> str:
    """Result identity: predictions_<stem>.csv and run_<stem>.json (the task name comes first)."""
    return tasks.result_stem(task or DEFAULT_TASK, n, seed)
