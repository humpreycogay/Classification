"""./run (run_one.py): pick one classifier by its registry name and run it like the runner does."""

import io
import os
import unittest
from contextlib import redirect_stdout
from unittest import mock

import run_one
from apps import APPS_BY_NAME
from run_all import app_command


class RunOneTest(unittest.TestCase):
    def test_names_are_the_registry_names(self):
        self.assertIs(run_one.find_app("tfidf_logreg_4k"), APPS_BY_NAME["tfidf_logreg_4k"])
        self.assertIs(run_one.find_app("LAYA"), APPS_BY_NAME["laya"])

    def test_unknown_name_suggests_close_ones(self):
        with self.assertRaises(SystemExit) as e:
            run_one.find_app("lya")
        self.assertIn("laya", str(e.exception.code))

    def test_variant_runs_with_its_own_arguments(self):
        cmd, env = app_command(APPS_BY_NAME["tfidf_logreg_4k"], 200, 42, "ag_news")
        self.assertEqual(cmd[1:], ["evaluate.py", "--task", "ag_news", "--n", "200", "--seed", "42",
                                   "--train-per-class", "task"])
        self.assertNotIn("BENCHMARK_RUN_ID", env)  # a single run is not part of a runner run

    def test_d1_uses_the_registered_adapter(self):
        cmd, _ = app_command(APPS_BY_NAME["d1"], 1000, 42, "banking77")
        self.assertEqual(cmd[1:], ["evaluate.py", "--task", "banking77", "--n", "1000", "--seed", "42"])

    def test_llm_gets_its_model_and_host(self):
        with mock.patch.dict(os.environ, {"LLMQWEN_MODEL": "m:1", "LLM_HOST": "http://box:11434"}):
            cmd, _ = app_command(APPS_BY_NAME["llmqwen"], 10, 1, "ag_news")
        self.assertEqual(cmd[-4:], ["--model", "m:1", "--host", "http://box:11434"])

    def test_dry_run_prints_the_command_and_runs_nothing(self):
        out = io.StringIO()
        with redirect_stdout(out), mock.patch("subprocess.call") as call, \
                mock.patch("sys.argv", ["run_one.py", "embedsup_logreg", "--n", "20", "--dry-run", "--", "--batch-size", "8"]):
            run_one.main()
        call.assert_not_called()
        self.assertIn("cd classifiers/embedsup && .venv/bin/python evaluate.py --task ag_news --n 20 --seed 42 --method logreg",
                      out.getvalue())
        self.assertIn("--batch-size 8", out.getvalue())


if __name__ == "__main__":
    unittest.main()
