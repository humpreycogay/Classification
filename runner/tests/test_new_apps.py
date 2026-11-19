"""Registry entries and runner helpers added with the JevBench classifiers: no models needed."""

from __future__ import annotations

import http.server
import json
import os
import stat
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

import shared_path  # noqa: F401  (puts shared/ on sys.path)
import task as tasks
from apps import APPS, APPS_BY_NAME, App
import overlap
from batching import length_order, restore, sorted_batches
from run_all import LocalServer, app_command, local_server, preflight, task_limits

NEW_ENTRIES = ["laya_typed", "laya_multilingual", "nli_modernbert", "naluri_279m", "naluri_308m",
               "d3_edge", "d3_lite", "d3_nano"]  # variants of existing folders
NEW_APPS = ["modernbert_jev", "decider", "strands_decider", "d1", "d1_omni", "kev", "dexy",
            "plumb", "winnow", "clm"]  # new folders


class RegistryTest(unittest.TestCase):
    def test_new_entries_resolve_to_existing_folders(self):
        for name in NEW_ENTRIES:
            app = APPS_BY_NAME[name]
            self.assertTrue((app.path / "evaluate.py").is_file(), name)
            self.assertTrue((app.path / "setup.sh").is_file(), name)
            self.assertEqual(app.results_dir.parent.parent, app.path, name)  # results/<variant>/ in the app folder
        for name in NEW_APPS:
            app = APPS_BY_NAME[name]
            for f in ("evaluate.py", "classifier.py", "setup.sh", "requirements.txt"):
                self.assertTrue((app.path / f).is_file(), f"{name}/{f}")
            self.assertEqual(app.results_dir, app.path / "results", name)

    def test_variants_write_to_their_own_results_folder(self):
        folders = [a.results_dir for a in APPS]
        self.assertEqual(len(folders), len(set(folders)), "two registry entries share a results folder")

    def test_shortlist_variants_fit_the_model_limit(self):
        for name, base in (("plumb_shortlist", "plumb"), ("winnow_shortlist", "winnow")):
            app, plain = APPS_BY_NAME[name], APPS_BY_NAME[base]
            self.assertEqual(app.shortlist, plain.max_labels, name)
            self.assertEqual(app.args, ("--shortlist", str(plain.max_labels)))
            self.assertEqual(app.path, plain.path)

    def test_in_domain_labels(self):
        self.assertEqual(APPS_BY_NAME["decider"].supervision_on("ag_news"), "in-domain trained")
        for task in ("ag_news", "banking77"):
            self.assertEqual(APPS_BY_NAME["strands_decider"].supervision_on(task), "in-domain trained")
            self.assertEqual(APPS_BY_NAME["kev"].supervision_on(task), "in-domain trained")
            self.assertEqual(APPS_BY_NAME["d1"].supervision_on(task), "zero-shot")
        self.assertEqual(APPS_BY_NAME["modernbert_jev"].supervision_on("ag_news"), "zero-shot")
        self.assertEqual(APPS_BY_NAME["modernbert_jev"].supervision_on("banking77"), "in-domain trained")

    def test_variant_arguments(self):
        cmd, _ = app_command(APPS_BY_NAME["laya_typed"], 200, 42, "ag_news")
        self.assertEqual(cmd[-4:], ["--model", "typed-decisions", "--batch-size", "16"])
        cmd, _ = app_command(APPS_BY_NAME["nli_modernbert"], 200, 42, "ag_news")
        self.assertEqual(cmd[-4:], ["--model", "MoritzLaurer/ModernBERT-large-zeroshot-v2.0", "--batch-size", "8"])


class OverlapRuleTest(unittest.TestCase):
    """shared/overlap.py: only test texts outside the task's own train split count as overlap."""

    def test_dataset_duplicates_are_reported_not_counted(self):
        task = tasks.load_task("banking77")
        test, train = tasks.load_split(task, "test")["text"], tasks.load_split(task, "train")["text"]
        norm = lambda t: overlap.normalize_text(t)
        in_train = set(train.map(norm))
        duplicate = next(t for t in test if norm(t) in in_train)
        fresh = next(t for t in test if norm(t) not in in_train)
        entry = overlap.check("banking77", {"train": ["  " + duplicate.upper() + " ", "not a test text"]})
        self.assertEqual((entry["overlap"], entry["dataset_duplicates"]), (0, 1))
        entry = overlap.check("banking77", {"train": [duplicate], "validation": [fresh]})
        self.assertEqual((entry["overlap"], entry["dataset_duplicates"]), (1, 1))
        self.assertEqual(entry["by_source"]["validation"], {"test_texts": 1, "not_in_train_split": 1})

    def test_write_merges_tasks(self):
        with tempfile.TemporaryDirectory() as tmp:
            overlap.write(Path(tmp), {"a": {"overlap": 0, "detail": "x"}})
            overlap.write(Path(tmp), {"b": {"overlap": 2, "detail": "y"}})
            self.assertEqual(set(json.loads((Path(tmp) / "results" / "overlap.json").read_text())), {"a", "b"})


class TrainedOnTasksTest(unittest.TestCase):
    def app(self, folder: Path) -> App:
        return App("x", "X", "Local", "max p", folder=str(folder), trained_on_tasks=("banking77",))

    def write_overlap(self, folder: Path, info: dict) -> None:
        (folder / "results").mkdir(parents=True, exist_ok=True)
        (folder / "results" / "overlap.json").write_text(json.dumps(info))

    def test_labelled_in_domain_only_on_its_training_tasks(self):
        app = self.app(Path("unused"))
        self.assertEqual(app.supervision_on("banking77"), "in-domain trained")
        self.assertEqual(app.supervision_on("ag_news"), "zero-shot")

    def test_skipped_when_overlap_found_or_check_missing(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch("apps.CLASSIFIERS_DIR", Path(tmp)):
            folder = Path(tmp) / "app"
            app = self.app(folder)
            banking = tasks.load_task("banking77")
            self.assertIn("overlap check has not run", task_limits(app, banking))
            self.write_overlap(folder, {"banking77": {"overlap": 4, "detail": "train 3, validation 1"}})
            self.assertIn("contains 4 of the banking77 test texts", task_limits(app, banking))
            self.write_overlap(folder, {"banking77": {"overlap": 0}})
            self.assertIsNone(task_limits(app, banking))
            self.assertIsNone(app.training_overlap("ag_news"))  # not a training task: never checked


class _Health(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()

    def log_message(self, *args):
        pass


class LocalServerTest(unittest.TestCase):
    def test_already_running_server_is_left_alone(self):
        httpd = http.server.HTTPServer(("127.0.0.1", 0), _Health)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        try:
            url = f"http://127.0.0.1:{httpd.server_port}/health"
            with tempfile.TemporaryDirectory() as tmp:
                with LocalServer("Fake", Path(tmp) / "never-run.sh", url, Path(tmp) / "log") as server:
                    self.assertIsNone(server.proc)
                    self.assertIsNone(server.start_seconds)
                    self.assertIsNone(server.memory())
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_starts_and_stops_a_server_it_started(self):
        with tempfile.TemporaryDirectory() as tmp:
            port_file = Path(tmp) / "port"
            script = Path(tmp) / "serve.sh"
            # A fake server: answers 200 on any GET until killed (started in its own process group).
            script.write_text(f"""#!/bin/sh
exec {sys.executable} -c '
import http.server
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200); self.end_headers()
    def log_message(self, *a): pass
s = http.server.HTTPServer(("127.0.0.1", 0), H)
open("{port_file}", "w").write(str(s.server_port))
s.serve_forever()'
""")
            script.chmod(script.stat().st_mode | stat.S_IEXEC)
            server = LocalServer("Fake", script, "", Path(tmp) / "log", ready_timeout=30)
            # The health URL is only known once the fake server picked its port.
            server.ready = lambda: port_file.exists() and bool(port_file.read_text()) and \
                LocalServer.ready(_with_url(server, f"http://127.0.0.1:{port_file.read_text()}/"))
            with mock.patch("run_all.time.sleep", lambda s: None):
                with server:
                    self.assertIsNotNone(server.proc)
                    self.assertIsNone(server.proc.poll())
                    server.sample()
                    self.assertEqual(server.memory()["kind"], "fake_server")
                    pid = server.proc.pid
            self.assertIsNotNone(server.proc.poll())  # stopped on exit
            with self.assertRaises(ProcessLookupError):
                os.killpg(pid, 0)

    def test_server_that_exits_early_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "serve.sh"
            script.write_text("#!/bin/sh\nexit 3\n")
            script.chmod(script.stat().st_mode | stat.S_IEXEC)
            server = LocalServer("Fake", script, "http://127.0.0.1:9/", Path(tmp) / "log", ready_timeout=30)
            with mock.patch("run_all.time.sleep", lambda s: None), self.assertRaisesRegex(RuntimeError, "exited early"):
                with server:
                    pass

    def test_only_server_apps_get_a_managed_server(self):
        self.assertIsNone(local_server(APPS_BY_NAME["laya"], Path("/tmp")))
        openjev = local_server(APPS_BY_NAME["OpenJev"], Path("/tmp"))
        self.assertEqual(openjev.kind, "openjev_server")
        self.assertEqual(openjev.log_path, Path("/tmp/OpenJev-server.log"))  # the same log name as before
        # Both Winnow entries share one server (one log per folder), health on /health.
        for name in ("winnow", "winnow_shortlist"):
            winnow = local_server(APPS_BY_NAME[name], Path("/tmp"))
            self.assertEqual((winnow.kind, winnow.log_path), ("winnow_server", Path("/tmp/winnow-server.log")))
            self.assertTrue(winnow.health_url.endswith("/health"))
            self.assertEqual(winnow.start_script, APPS_BY_NAME["winnow"].path / "server" / "serve.sh")

    def test_extra_ollama_checked_for_server_apps(self):
        """winnow_shortlist needs its server and bge-m3 on Ollama: an unreachable Ollama skips it."""
        app = APPS_BY_NAME["winnow_shortlist"]
        with mock.patch("run_all.ollama_models", return_value=None), \
                mock.patch("run_all.estimated_seconds", return_value=None), \
                mock.patch.object(Path, "exists", return_value=True):
            self.assertIn("Ollama not reachable", preflight(app, 200, 1, 3600, False))


def _with_url(server: LocalServer, url: str) -> LocalServer:
    server.health_url = url
    return server


class LengthSortTest(unittest.TestCase):
    def test_restores_input_order(self):
        texts = ["bb", "a", "dddd", "ccc", "", "ee"]
        seen = []

        def run(batch):
            seen.append(list(batch))
            return [t.upper() for t in batch]

        self.assertEqual(sorted_batches(texts, 2, run), [t.upper() for t in texts])
        self.assertEqual(seen, [["dddd", "ccc"], ["bb", "ee"], ["a", ""]])  # longest first; ties keep input order

    def test_order_and_restore_are_inverse(self):
        items = ["xx", "x", "xxx"]
        order = length_order(items)
        self.assertEqual(order, [2, 0, 1])
        self.assertEqual(restore(order, [items[i] for i in order]), items)

    def test_empty(self):
        self.assertEqual(sorted_batches([], 4, lambda b: b), [])


if __name__ == "__main__":
    unittest.main()
