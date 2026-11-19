"""Two-stage shortlist variants: shared shortlist.py, top-K and recall, runner rules."""

import importlib.util
import os
import sys
import unittest
from unittest import mock

import numpy as np

import task as tasks
from apps import APPS_BY_NAME, CLASSIFIERS_DIR
from run_all import task_limits
from sampling import ROOT

SHORTLIST_APPS = ["llm", "SemIf", "verdict"]
spec = importlib.util.spec_from_file_location("shortlist", ROOT / "shared" / "shortlist.py")
shortlist = importlib.util.module_from_spec(spec)
sys.modules["shortlist"] = shortlist
spec.loader.exec_module(shortlist)


def fake_shortlister(k: int, label_vectors: np.ndarray, text_vectors: dict[str, list[float]]):
    """A Shortlister whose embeddings come from a dict instead of Ollama."""
    s = shortlist.Shortlister.__new__(shortlist.Shortlister)
    s.k, s.label_ids = k, ["a", "b", "c", "d"]
    s.label_vectors = label_vectors.astype(np.float32)
    s._embed = lambda texts: np.asarray([text_vectors[t] for t in texts], dtype=np.float32)
    return s


class ShortlistTest(unittest.TestCase):
    def setUp(self):
        self.s = fake_shortlister(2, np.eye(4), {"x": [0.9, 0.5, 0.1, 0.0], "y": [0.0, 0.2, 0.3, 0.95]})

    def test_apps_use_the_shared_shortlist(self):
        for app in SHORTLIST_APPS:
            with self.subTest(app=app):
                self.assertIn("import shortlist", (CLASSIFIERS_DIR / app / "evaluate.py").read_text())

    def test_top_k_most_similar_first(self):
        self.assertEqual(self.s.top_k(["x", "y"]), [["a", "b"], ["d", "c"]])

    def test_recall_is_the_share_kept(self):
        self.assertEqual(self.s.recall(["x", "y"], ["b", "a"]), 0.5)  # b kept for x; a not kept for y


class RunnerRuleTest(unittest.TestCase):
    def task_with(self, n_labels: int) -> tasks.Task:
        return tasks.Task("fake", {"labels": [{"raw": i, "id": f"l{i}"} for i in range(n_labels)],
                                   "files": {"test": {}, "train": {}}, "remote_allowed": True})

    def test_shortlist_only_when_the_task_needs_it(self):
        app = APPS_BY_NAME["llmqwen_shortlist"]
        self.assertIn("no shortlist needed", task_limits(app, self.task_with(4)))
        self.assertIsNone(task_limits(app, self.task_with(77)))
        self.assertIsNone(task_limits(APPS_BY_NAME["verdict_shortlist"], self.task_with(77)))

    def test_variants_keep_their_own_results(self):
        for name in ("llmqwen_shortlist", "SemIf_shortlist", "verdict_shortlist"):
            app = APPS_BY_NAME[name]
            self.assertTrue(app.results.endswith(f"/shortlist{app.shortlist}"), name)
            self.assertEqual(app.args[-2:], ("--shortlist", str(app.shortlist)))
            plain = APPS_BY_NAME[name.removesuffix("_shortlist")]
            self.assertEqual(app.results, f"{plain.results}/shortlist{app.shortlist}")  # nested in the plain app's

    def test_llm_models_share_one_folder_but_not_results(self):
        llm_apps = [a for a in APPS_BY_NAME.values() if a.folder == "llm"]
        self.assertEqual({a.name for a in llm_apps},
                         {"llmqwen", "llmqwen_shortlist"})
        self.assertTrue(all(a.model_arg and a.ollama_model for a in llm_apps))
        # evaluate.py writes to results/<model tag>/ (its default --results), which the registry must match.
        with mock.patch.dict(os.environ, {"LLMQWEN_MODEL": "qwen3.8:27b-q4_K_M"}):
            self.assertEqual(APPS_BY_NAME["llmqwen"].results_dir, CLASSIFIERS_DIR / "llm" / "results" / "qwen3.8_27b-q4_K_M")
            self.assertEqual(APPS_BY_NAME["llmqwen_shortlist"].results_dir,
                             CLASSIFIERS_DIR / "llm" / "results" / "qwen3.8_27b-q4_K_M" / "shortlist16")
            self.assertEqual(APPS_BY_NAME["llmqwen"].display_title, "LLM Qwen (qwen3.8:27b-q4_K_M)")
            self.assertEqual(len({a.results_dir for a in llm_apps}), len(llm_apps))
        # Switching the model in .env gives it a fresh results folder.
        with mock.patch.dict(os.environ, {"LLMQWEN_MODEL": "x:1"}):
            self.assertEqual(APPS_BY_NAME["llmqwen"].results_dir, CLASSIFIERS_DIR / "llm" / "results" / "x_1")


if __name__ == "__main__":
    unittest.main()
