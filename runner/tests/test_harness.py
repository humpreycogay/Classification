"""Priority 8: the shared benchmark library (shared/: harness.py and friends) and the app adapters."""

import importlib.util
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from unittest import mock

import pandas as pd

import task as tasks
from apps import CLASSIFIERS_DIR
from sampling import ROOT

APP_FOLDERS = sorted(p.parent.name for p in CLASSIFIERS_DIR.glob("*/evaluate.py"))


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses look the module up while the class is created
    spec.loader.exec_module(module)
    return module


SHARED = ROOT / "shared"
if "timing" not in sys.modules:
    load("timing", SHARED / "timing.py")
harness = load("harness", SHARED / "harness.py")
SHARED_MODULES = sorted(p.name for p in SHARED.glob("*.py"))

LABELS = ["a", "b", "c"]
TASK = tasks.Task("fake", {"labels": [{"raw": i, "id": label} for i, label in enumerate(LABELS)]})


@dataclass
class Pred:
    label: str
    confidence: float
    probabilities: dict


class FakeClassifier:
    """Predicts the label named in the text ("a 3" -> a); counts calls and batch sizes."""

    def __init__(self, probability: float = 0.8):
        self.probability, self.calls = probability, []

    def predict_many(self, texts, batch_size=None):
        self.calls.append((len(texts), batch_size))
        rest = (1 - self.probability) / (len(LABELS) - 1)
        return [Pred(t[0], self.probability, {label: self.probability if label == t[0] else rest for label in LABELS})
                for t in texts]


def fake_split(n_per_label: int = 6) -> pd.DataFrame:
    return pd.DataFrame([{"label": label, "text": f"{label} {i}"} for i in range(n_per_label) for label in LABELS])


class SharedLibraryTest(unittest.TestCase):
    def test_no_folder_keeps_its_own_copy(self):
        """A stale copy next to evaluate.py (or in runner/) would drift from shared/; the library has one home."""
        self.assertEqual(SHARED_MODULES, ["batching.py", "finetune.py", "harness.py", "mlx_prefix.py", "overlap.py", "probes.py",
                                          "shortlist.py", "task.py", "textnn.py", "timing.py", "training.py"])
        self.assertEqual(len(APP_FOLDERS), 33)
        for folder in [*(CLASSIFIERS_DIR / a for a in APP_FOLDERS), ROOT / "runner"]:
            with self.subTest(folder=folder.name):
                self.assertEqual([m for m in SHARED_MODULES if (folder / m).exists()], [])

    def test_every_app_imports_the_shared_library(self):
        for app in APP_FOLDERS:
            with self.subTest(app=app):
                source = (CLASSIFIERS_DIR / app / "evaluate.py").read_text()
                self.assertIn('os.environ.get("BENCHMARK_SHARED_DIR") or APP_DIR.parents[1] / "shared"', source)
                self.assertLess(source.index("sys.path.insert(0, str(SHARED_DIR))"), source.index("from task import"))

    def test_shared_sha1_changes_with_the_code(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(harness, "SHARED_DIR", Path(tmp)):
            (Path(tmp) / "a.py").write_text("x = 1\n")
            first = harness.shared_sha1()
            (Path(tmp) / "a.py").write_text("x = 2\n")
            self.assertNotEqual(harness.shared_sha1(), first)

    def test_apps_leave_the_shared_rules_to_the_harness(self):
        """evaluate.py is an adapter: sampling, timing, reporting and output live in harness.py only."""
        for app in APP_FOLDERS:
            with self.subTest(app=app):
                source = (CLASSIFIERS_DIR / app / "evaluate.py").read_text()
                self.assertIn("main(adapter, app_dir=APP_DIR,", source)
                for duplicated in ("def report(", "to_csv(", "timed_calls(", "sample(TASK", "perf_counter() - start",
                                   "argparse.ArgumentParser("):
                    self.assertNotIn(duplicated, source)


class SetupTest(unittest.TestCase):
    def setup(self, **kw):
        return harness.Setup(model="m", predict=lambda texts, v=None: [], **kw)

    def test_batched_apps_time_one_batch_per_call_and_warm_up_on_a_full_batch(self):
        s = self.setup(axis="batch_size", value=16)
        self.assertEqual((s.step, s.warmup_items, s.per_call_unit), (16, 16, "batch"))

    def test_concurrent_apps_time_requests(self):
        s = self.setup(axis="concurrency", value=8)
        self.assertEqual((s.step, s.warmup_items), (None, 1))
        self.assertIn("request", s.per_call_unit)

    def test_sequential_and_vectorized_apps(self):
        self.assertEqual((self.setup(sequential=True).step, self.setup(sequential=True).per_call_unit), (1, "article (sequential)"))
        self.assertEqual((self.setup().step, self.setup().per_call_unit), (None, "whole sample (one vectorized call)"))


class ContextTest(unittest.TestCase):
    def test_excluded_time_is_not_load_time_and_cache_loads_are(self):
        ctx = harness.Context(TASK, None, fake_split(), Path("."))
        with mock.patch.object(harness.time, "perf_counter", side_effect=[11.0, 15.0, 20.0]):
            ctx._start = 10.0
            with ctx.exclude():  # 11 -> 15: training
                pass
            ctx.add_load(0.5)  # a cached fit's load
            self.assertAlmostEqual(ctx.load_seconds(), 20.0 - 10.0 - 4.0 + 0.5)

    def test_warm_up_inside_the_adapter_is_recorded_and_excluded(self):
        ctx = harness.Context(TASK, None, fake_split(), Path("."))
        ctx.warm_up(lambda: None, 4)
        self.assertEqual(ctx.warmup[0], 4)
        self.assertAlmostEqual(ctx._excluded, ctx.warmup[1], delta=1e-3)


class CheckPredictionsTest(unittest.TestCase):
    def frame(self, **overrides):
        df = pd.DataFrame({"pred": ["a", "b"], "confidence": [0.9, 0.6], "p_a": [0.9, 0.4], "p_b": [0.1, 0.6], "p_c": [0.0, 0.0]})
        for col, values in overrides.items():
            df[col] = values
        return df

    def test_valid_predictions_pass(self):
        harness.check_predictions(self.frame(), LABELS, ["p_a", "p_b", "p_c"])

    def test_unknown_labels_and_bad_scores_fail(self):
        for bad in ({"pred": ["a", "zz"]}, {"confidence": [0.9, float("nan")]}, {"p_a": [1.2, 0.4]}, {"p_c": [0.0, -0.1]}):
            with self.subTest(bad=bad), self.assertRaises(SystemExit):
                harness.check_predictions(self.frame(**bad), LABELS, ["p_a", "p_b", "p_c"])


class MainTest(unittest.TestCase):
    """An end-to-end run of main() with a fake classifier: no model, dataset or network."""

    def run_main(self, argv, adapter):
        test = fake_split(10)
        sample = test.sample(9, random_state=0).reset_index().rename(columns={"index": "sample_id"})
        self.progress_files = None  # status files left in reports/progress/ (checked even when main() fails)
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(harness, "PROGRESS_DIR", Path(tmp) / "progress"), \
                mock.patch.object(harness, "current_task", return_value=TASK), \
                mock.patch.object(harness, "sample", return_value=sample), \
                mock.patch.object(harness, "load_split", return_value=test), \
                mock.patch.object(harness, "tier_name", return_value="custom"), \
                mock.patch.object(sys, "argv", ["evaluate.py", *argv]), redirect_stdout(StringIO()):
            try:
                harness.main(adapter, app_dir=Path(tmp) / "fakeapp", doc="fake app", key_packages=["pandas"],
                             add_arguments=lambda p: p.add_argument("--batch-size", type=int, default=4))
            finally:
                self.progress_files = [json.loads(p.read_text()) for p in (Path(tmp) / "progress").glob("*.json")]
            results = Path(tmp) / "fakeapp" / "results"
            files = sorted(str(p.relative_to(results)) for p in results.rglob("*.*"))
            csv = pd.read_csv(next(results.rglob("predictions_*.csv")))
            meta = json.loads(next(results.rglob("run_*.json")).read_text())
        return sample, files, csv, meta

    def test_writes_the_standard_outputs(self):
        clf = FakeClassifier()

        def adapter(args, ctx):
            return harness.Setup(model="fake", classifier=clf, axis="batch_size", value=args.batch_size,
                                 predict=lambda texts, v=args.batch_size: clf.predict_many(texts, batch_size=v),
                                 results="variant", metadata={"extra": 1})

        sample, files, csv, meta = self.run_main(["--n", "9", "--seed", "3", "--latency-items", "2"], adapter)
        self.assertEqual(files, ["variant/predictions_fake_n9_seed3.csv", "variant/run_fake_n9_seed3.json"])
        self.assertEqual(list(csv.columns), ["sample_id", "text", "label", "pred", "confidence", "p_a", "p_b", "p_c"])
        self.assertEqual(csv["sample_id"].tolist(), sample["sample_id"].tolist())
        self.assertTrue((csv["pred"] == csv["label"]).all())
        self.assertEqual(meta["sample_sha1"], tasks.sample_sha1(sample))
        self.assertEqual((meta["app"], meta["n"], meta["seed"], meta["task"], meta["extra"]), ("fakeapp", 9, 3, "fake", 1))
        self.assertEqual(meta["harness_version"], harness.HARNESS_VERSION)
        self.assertEqual(meta["shared_sha1"], harness.shared_sha1())
        self.assertEqual(meta["timing"]["warmup_items"], 4)
        self.assertFalse(meta["timing"]["latency_texts_reused"])
        # Warm-up (one full batch), then the main run in batches of 4 (4 + 4 + 1), then 2 single-article calls.
        self.assertEqual(clf.calls, [(4, 4), (4, 4), (4, 4), (1, 4), (1, 1), (1, 1)])
        self.assertEqual(self.progress_files, [])  # a finished run removes its status file

    def test_main_run_hook_sees_only_the_main_run(self):
        clf = FakeClassifier()

        def on_main():
            before = len(clf.calls)
            return lambda: {"main_calls": len(clf.calls) - before}

        def adapter(args, ctx):
            return harness.Setup(model="fake", classifier=clf, predict=lambda texts, v=None: clf.predict_many(texts),
                                 on_main=on_main)

        _, _, _, meta = self.run_main(["--n", "9", "--latency-items", "3"], adapter)
        self.assertEqual(meta["main_calls"], 1)  # one vectorized call; not the warm-up or the 3 latency calls

    def test_invalid_predictions_write_nothing(self):
        clf = FakeClassifier(probability=1.5)

        def adapter(args, ctx):
            return harness.Setup(model="fake", classifier=clf, predict=lambda texts, v=None: clf.predict_many(texts))

        with self.assertRaises(SystemExit):
            self.run_main(["--n", "9", "--latency-items", "0"], adapter)
        # A run that did not finish leaves its status file, so ./run status can say where it stopped.
        [left] = self.progress_files
        self.assertEqual((left["app"], left["phase"], left["done"], left["total"]), ("fakeapp", "classifying", 9, 9))


if __name__ == "__main__":
    unittest.main()
