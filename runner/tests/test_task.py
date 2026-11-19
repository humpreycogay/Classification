"""Task manifests (task.py): validation, files, sampling strategies, calibration, app limits."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

import task as tasks
from apps import APPS_BY_NAME, App
from run_all import task_limits
from tests.helpers import fake_test_set


def manifest(**overrides) -> dict:
    base = {"name": "fake", "task_type": "single_label", "labels": [{"raw": "a", "id": "a"}, {"raw": "b", "id": "b"}],
            "files": {"test": {"path": "fake_test.csv", "url": "http://invalid"}},
            "format": {"type": "csv", "header": True, "columns": ["text", "category"], "label_column": "category",
                       "text_template": "{text}", "replace": []},
            "sampling": "balanced"}
    return {**base, **overrides}


class TaskFilesTest(unittest.TestCase):
    """Manifests and data files in a temporary data/ folder."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.data = Path(self.tmp.name)
        (self.data / "tasks").mkdir()
        self.patches = [mock.patch.object(tasks, "DATA_DIR", self.data), mock.patch.object(tasks, "TASKS_DIR", self.data / "tasks")]
        for p in self.patches:
            p.start()
        tasks._load.cache_clear()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        tasks._load.cache_clear()
        self.tmp.cleanup()

    def write(self, m: dict, rows: list[tuple[str, str]]):
        (self.data / "tasks" / f"{m['name']}.json").write_text(json.dumps(m))
        pd.DataFrame(rows, columns=["text", "category"]).to_csv(self.data / "fake_test.csv", index=False)

    def test_loads_texts_and_label_ids(self):
        self.write(manifest(labels=[{"raw": "x", "id": "a"}, {"raw": "y", "id": "b"}],
                            format={**manifest()["format"], "replace": [["!", ""]]}), [("hi!", "x"), ("yo", "y")])
        df = tasks.load_split(tasks.load_task("fake"), "test")
        self.assertEqual(df["text"].tolist(), ["hi", "yo"])
        self.assertEqual(df["label"].tolist(), ["a", "b"])

    def test_rejects_unsupported_task_type(self):
        self.write(manifest(task_type="multi_label"), [("t", "a")])
        with self.assertRaisesRegex(ValueError, "not supported"):
            tasks.load_task("fake")

    def test_rejects_unknown_labels_and_bad_checksum(self):
        self.write(manifest(), [("t", "a"), ("u", "zzz")])
        with self.assertRaisesRegex(ValueError, "labels not in the manifest"):
            tasks.load_split(tasks.load_task("fake"), "test")
        tasks._load.cache_clear()
        m = manifest()
        m["files"]["test"]["sha256"] = "0" * 64
        self.write(m, [("t", "a")])
        with self.assertRaisesRegex(ValueError, "checksum"):
            tasks.load_split(tasks.load_task("fake"), "test")

    def test_unknown_task_lists_known(self):
        self.write(manifest(), [("t", "a")])
        with self.assertRaisesRegex(ValueError, "fake"):
            tasks.load_task("nope")


class SamplingStrategyTest(unittest.TestCase):
    def test_proportional_keeps_label_shares(self):
        df = pd.DataFrame({"label": ["a"] * 70 + ["b"] * 20 + ["c"] * 10, "text": [str(i) for i in range(100)]})
        counts = tasks.stratified_sample(df, 21, 3, "proportional")["label"].value_counts()
        # exact shares 14.7 / 4.2 / 2.1 -> 14 / 4 / 2, and the leftover row goes to the largest remainder (a: 0.7)
        self.assertEqual(counts.to_dict(), {"a": 15, "b": 4, "c": 2})

    def test_balanced_is_unchanged(self):
        df = fake_test_set(10)
        pd.testing.assert_frame_equal(tasks.stratified_sample(df, 8, 1, "balanced"), tasks.stratified_sample(df, 8, 1))

    def test_unknown_strategy(self):
        with self.assertRaises(ValueError):
            tasks.stratified_sample(fake_test_set(10), 8, 1, "random")


class SelectionTest(unittest.TestCase):
    def test_argv_sets_the_task_for_classifier_modules(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("BENCHMARK_TASK", None)
            self.assertEqual(tasks.select_task_from_argv(["evaluate.py", "--n", "5"]), "ag_news")
            self.assertEqual(tasks.select_task_from_argv(["evaluate.py", "--task", "banking77"]), "banking77")
            self.assertEqual(os.environ["BENCHMARK_TASK"], "banking77")
            self.assertEqual(tasks.select_task_from_argv(["evaluate.py", "--task=ag_news"]), "ag_news")

    def test_result_names_carry_the_task(self):
        self.assertEqual(tasks.result_stem("banking77", 200, 42), "banking77_n200_seed42")


@unittest.skipUnless((tasks.DATA_DIR / "ag_news_train.csv").exists(), "AG News train CSV not downloaded")
class CalibrationTest(unittest.TestCase):
    def test_calibration_comes_from_train_only(self):
        task = tasks.load_task("ag_news")
        cal = tasks.calibration_split(task)
        self.assertEqual(len(cal), 400)
        self.assertTrue(cal["label"].value_counts().eq(100).all())
        test_texts = set(tasks.load_split(task, "test")["text"])
        self.assertFalse(set(cal["text"]) & test_texts)
        pd.testing.assert_frame_equal(cal, tasks.calibration_split(task))  # deterministic


class AppLimitTest(unittest.TestCase):
    def fake_task(self, n_labels: int, remote_allowed: bool = True, train: bool = True):
        m = manifest(labels=[{"raw": i, "id": f"l{i}"} for i in range(n_labels)], remote_allowed=remote_allowed)
        if train:
            m["files"]["train"] = {"path": "x.csv", "url": "http://invalid"}
        return tasks.Task("fake", m)

    def test_label_limit(self):
        llm = APPS_BY_NAME["llmqwen"]
        self.assertIsNone(task_limits(llm, self.fake_task(16)))
        self.assertIn("at most 16", task_limits(llm, self.fake_task(77)))

    def test_remote_apps_need_permission(self):
        self.assertIn("remote", task_limits(APPS_BY_NAME["jev"], self.fake_task(4, remote_allowed=False)))
        self.assertIsNone(task_limits(APPS_BY_NAME["laya"], self.fake_task(4, remote_allowed=False)))

    def test_supervised_apps_need_train(self):
        self.assertIn("no train split", task_limits(APPS_BY_NAME["tfidf_cnb"], self.fake_task(4, train=False)))


if __name__ == "__main__":
    unittest.main()
