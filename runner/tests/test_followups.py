"""Review follow-ups: runner guards, the shared training helper, and equal training data."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from apps import App, APPS_BY_NAME, CLASSIFIERS_DIR
from run_all import estimated_seconds, preflight
from sampling import DATA_PATH, ROOT

RUNNER = ROOT / "runner"


class RunnerGuardTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.app = App("fake", "Fake", "Local", "max p", folder=self.tmp.name, cost="paid")
        self.app.results_dir.mkdir()
        (self.app.path / ".venv" / "bin").mkdir(parents=True)
        (self.app.path / ".venv" / "bin" / "python").write_text("")

    def tearDown(self):
        self.tmp.cleanup()

    def write_meta(self, **training):
        meta = {"seconds_per_item": 0.01, "load_seconds": 5.0, "warmup_seconds": 1.0, "training": training}
        (self.app.results_dir / "run_n200_seed42.json").write_text(json.dumps(meta))

    def test_paid_cap_counts_every_run(self):
        self.assertIsNone(preflight(self.app, 200, 5, 3600, allow_paid=False))  # 1,000 requests
        self.assertIn("1200 requests", preflight(self.app, 200, 6, 3600, allow_paid=False))
        self.assertIsNone(preflight(self.app, 200, 6, 3600, allow_paid=True))

    def test_estimate_includes_setup_and_uncached_training(self):
        self.write_meta(train_seconds=80.0, train_cached=False)
        self.assertAlmostEqual(estimated_seconds(self.app, 1000), 10 + 5 + 1 + 80)
        self.write_meta(train_seconds=80.0, train_cached=True)
        self.assertAlmostEqual(estimated_seconds(self.app, 1000), 10 + 5 + 1)

    def test_timeout_uses_the_estimate(self):
        self.write_meta(train_seconds=80.0, train_cached=False)
        self.assertIn("exceeds", preflight(self.app, 200, 1, 60, allow_paid=True))

    @unittest.skipUnless(DATA_PATH.exists(), "AG News test CSV not downloaded")
    def test_full_tier_rejects_several_seeds(self):
        proc = subprocess.run([sys.executable, "run_all.py", "--tier", "full", "--seeds", "1", "2"],
                              cwd=RUNNER, capture_output=True, text=True, timeout=120)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("use --repeat", proc.stderr)


class TrainingHelperTest(unittest.TestCase):
    def test_same_train_sample_as_embedding_baselines(self):
        """tfidf_logreg --train-per-class uses embedsup's sampling code, so both see the same 4,000 articles."""
        def body(path: Path) -> str:
            text = path.read_text()
            start = text.index("def load_train_sample(")
            return text[text.index("    return", start):text.index("\n\n", start)]
        self.assertEqual(body(CLASSIFIERS_DIR / "tfidf_logreg" / "classifier.py"), body(CLASSIFIERS_DIR / "embedsup" / "classifier.py"))
        app = APPS_BY_NAME["tfidf_logreg_4k"]
        self.assertEqual(app.args, ("--train-per-class", "task"))  # the task's embedding-baseline sample size


    @unittest.skipUnless((CLASSIFIERS_DIR / "tfidf_cnb" / ".venv" / "bin" / "python").exists(), "tfidf_cnb not set up")
    def test_grid_edge_flat_vs_improving(self):
        """training.py needs scikit-learn, so it is checked in an app's own environment."""
        code = (
            "from training import edge_check\n"
            "g = [1.0, 3.0, 10.0]\n"
            "flat = {'chosen': 10.0, 'cv_accuracy': {'1.0': .80, '3.0': .85, '10.0': .851}, 'cv_sd': {'10.0': .01}}\n"
            "rising = {**flat, 'cv_accuracy': {'1.0': .80, '3.0': .82, '10.0': .85}}\n"
            "inside = {**flat, 'chosen': 3.0}\n"
            "print(edge_check(flat, g), edge_check(rising, g), edge_check(inside, g))\n")
        out = subprocess.run([str(CLASSIFIERS_DIR / "tfidf_cnb" / ".venv" / "bin" / "python"), "-c", code], cwd=ROOT / "shared",
                             capture_output=True, text=True, timeout=120).stdout
        self.assertIn("{'at_grid_edge': True, 'edge_still_improving': False} "
                      "{'at_grid_edge': True, 'edge_still_improving': True} "
                      "{'at_grid_edge': False, 'edge_still_improving': False}", out)


if __name__ == "__main__":
    unittest.main()
