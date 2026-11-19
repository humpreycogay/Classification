"""Priority 6: timing helpers (shared/timing.py), perf.py figures and runner measurement."""

import importlib.util
import json
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock

import perf
from apps import App
from run_all import latest_progress, parse_time_l, preflight
from sampling import ROOT

APP_FOLDERS = ["laya", "nli", "gliclass", "verdict", "embedsim", "embedsup", "SemIf", "nimble", "llm",
               "OpenJev", "djev", "jev", "tfidf_logreg", "tfidf_svm", "tfidf_cnb"]

spec = importlib.util.spec_from_file_location("timing", ROOT / "shared" / "timing.py")
timing = importlib.util.module_from_spec(spec)
sys.modules["timing"] = timing  # dataclasses look the module up while the class is created
spec.loader.exec_module(timing)


class Pred:
    def __init__(self, label):
        self.label = label


class TimingHelperTest(unittest.TestCase):
    def test_percentiles_need_enough_samples(self):
        few = timing.percentiles([0.1] * 19)
        self.assertEqual((few["p50"], few["p90"], few["p95"], few["p99"]), (0.1, None, None, None))
        some = timing.percentiles([i / 100 for i in range(100)])
        self.assertAlmostEqual(some["p50"], 0.495)
        self.assertAlmostEqual(some["p95"], 0.9405)
        self.assertIsNone(some["p99"])
        self.assertIsNotNone(timing.percentiles([1.0] * 1000)["p99"])
        self.assertEqual(timing.percentiles([])["count"], 0)

    def test_timed_calls_chunks(self):
        seen = []
        preds, seconds = timing.timed_calls(lambda chunk: seen.append(len(chunk)) or [Pred(t) for t in chunk],
                                            [str(i) for i in range(10)], 4)
        self.assertEqual(seen, [4, 4, 2])
        self.assertEqual(len(seconds), 3)
        self.assertEqual([p.label for p in preds], [str(i) for i in range(10)])
        _, one = timing.timed_calls(lambda chunk: [Pred(t) for t in chunk], ["a", "b"], None)
        self.assertEqual(len(one), 1)

    def test_sweep_reports_agreement(self):
        texts = [f"t{i}" for i in range(timing.SWEEP_TEXTS)]
        seen = []

        def run(chunk, v):
            seen.append((v, tuple(chunk)))
            # batch size 32 flips the first shared article: 1 of 32 differs from the batch-1 reference
            return [Pred("x" if (v == 32 and t == "t0") else t) for t in chunk]

        rows = timing.sweep(run, "batch_size", texts)
        self.assertEqual([r["value"] for r in rows], [1, 8, 16, 32])
        self.assertEqual([r["agreement_with_first"] for r in rows], [1.0, 1.0, 1.0, round(31 / 32, 4)])
        # every setting is timed on its own 128 articles; only the untimed agreement call repeats the shared 32
        shared = tuple(texts[:32])
        for i, value in enumerate([1, 8, 16, 32]):
            received = {x for v, chunk in seen if v == value and chunk != shared for x in chunk}
            self.assertEqual(received, set(texts[i * 128:(i + 1) * 128]))
        self.assertEqual([v for v, chunk in seen if chunk == shared], [8, 16, 32])
        self.assertEqual(rows[1]["items"], 128)

    def test_unseen_texts(self):
        import pandas as pd
        full = pd.DataFrame({"text": [f"t{i}" for i in range(10)]})
        sample = full.loc[[1, 3, 5]].rename_axis("sample_id").reset_index()
        texts, reused = timing.unseen_texts(full, sample, 4, seed=0)
        self.assertFalse(reused)
        self.assertEqual(len(texts), 4)
        self.assertFalse(set(texts) & {"t1", "t3", "t5"})
        texts, reused = timing.unseen_texts(full, sample, 9, seed=0)
        self.assertTrue(reused)
        self.assertEqual(len(texts), 9)

    def test_progress_lines_and_status_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            status = Path(tmp) / "app-1.json"
            out = StringIO()
            with redirect_stdout(out):
                progress = timing.Progress("classifying", 4, every=0, status=status, info={"app": "x"})
                self.assertEqual(json.loads(status.read_text())["done"], 0)  # written as soon as it starts
                progress.update(1)
                progress.update(1)  # no change: no line
                progress.update(0)  # never goes backwards
                progress.update(9)  # capped at the total
            lines = out.getvalue().splitlines()
            self.assertEqual(len(lines), 1)  # every=0: only the final line
            self.assertTrue(lines[0].startswith("progress: classifying 4/4 (100.0%) · "))
            saved = json.loads(status.read_text())
            self.assertEqual((saved["app"], saved["done"], saved["total"], saved["left_seconds"]), ("x", 4, 4, None))
            self.assertEqual(saved["line"], lines[0])

        p = timing.Progress("latency", 10, every=3600)
        p.done = 5
        self.assertIn("5/10 (50.0%)", p.line())
        self.assertIn("left", p.line())
        self.assertEqual([timing.duration(s) for s in (42, 723, 3840)], ["42s", "12m 03s", "1h 04m"])

    def test_timed_calls_reports_progress(self):
        def run(chunk):
            return [Pred(t) for t in chunk]

        seen = []
        progress = timing.Progress("classifying", 10, every=3600)
        progress.update = seen.append  # record every update
        timing.timed_calls(run, [str(i) for i in range(10)], 4, progress=progress)
        self.assertEqual(seen, [4, 8, 10])  # after every call

        # One call for the whole sample (HTTP apps): the watcher polls the completed-requests count meanwhile.
        done = []

        def slow(chunk):
            for t in chunk:
                time.sleep(0.03)
                done.append(t)
            return [Pred(t) for t in chunk]

        polled = []
        progress = timing.Progress("classifying", 10, every=3600)
        progress.update = polled.append
        watch = timing.Progress.watch  # poll every 10 ms instead of every second
        with mock.patch.object(timing.Progress, "watch", lambda self, completed: watch(self, completed, interval=0.01)):
            timing.timed_calls(slow, [str(i) for i in range(10)], None, progress=progress, completed=lambda: len(done))
        self.assertEqual(polled[-1], 10)
        self.assertTrue(any(0 < n < 10 for n in polled))  # progress showed during the one call

    def test_runner_reads_the_latest_progress_line_from_the_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "app.log"
            self.assertIsNone(latest_progress(log))  # no log yet
            log.write_text("Loading...\nprogress: classifying 10/100 (10.0%) · 5s elapsed\n"
                           "a warning\nprogress: classifying 20/100 (20.0%) · 10s elapsed\n")
            self.assertEqual(latest_progress(log), "classifying 20/100 (20.0%) · 10s elapsed")

    def test_request_stats(self):
        s = timing.RequestStats()
        s.requests, s.retries = 3, 1
        s.request_seconds += [0.1, 0.2, 0.3]
        s.add_tokens(100, 5, 0.5)
        s.add_tokens(None, None)  # a response without token counts
        out = s.summary()
        self.assertEqual((out["input_tokens"], out["output_tokens"], out["output_tokens_per_second"]), (100, 5, 10.0))
        self.assertEqual(out["tokens_reported_by"], "1/3 requests")
        self.assertEqual(out["per_request"]["count"], 3)
        hidden = timing.RequestStats(retries_observable=False)
        hidden.retries = 2
        self.assertIsNone(hidden.summary()["retries"])
        s.reset()
        self.assertEqual((s.requests, s.request_seconds), (0, []))

    def test_hf_model_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            blobs = Path(tmp) / "models--org--model" / "blobs"
            blobs.mkdir(parents=True)
            (blobs / "a").write_bytes(b"x" * 10)
            (blobs / "b").write_bytes(b"x" * 5)
            import os
            old = os.environ.get("HF_HUB_CACHE")
            os.environ["HF_HUB_CACHE"] = tmp
            try:
                self.assertEqual(timing.hf_model_bytes("org/model"), 15)
                self.assertIsNone(timing.hf_model_bytes("org/missing"))
                self.assertIsNone(timing.hf_model_bytes(None))
            finally:
                if old is None:
                    del os.environ["HF_HUB_CACHE"]
                else:
                    os.environ["HF_HUB_CACHE"] = old


class PerfTest(unittest.TestCase):
    def test_cost_bases(self):
        meta = {"n": 100, "requests": {"requests": 100, "input_tokens": 20000, "output_tokens": 1000}}
        self.assertEqual(perf.cost("local", "free", meta, {})["basis"], "none")
        unknown = perf.cost("jev", "paid", meta, {"jev": {"currency": "USD", "per_request": None}})
        self.assertEqual((unknown["basis"], unknown["per_1000"]), ("unknown", None))
        tokens = perf.cost("jev", "paid", meta, {"jev": {"currency": "USD", "per_1k_input_tokens": 0.5,
                                                        "per_1k_output_tokens": 2.0}})
        # (20000 * 0.5 + 1000 * 2.0) / 1000 / 100 items = 0.12 per item
        self.assertEqual(tokens["basis"], "estimated")
        self.assertAlmostEqual(tokens["per_item"], 0.12)
        self.assertAlmostEqual(tokens["per_1000"], 120.0)
        per_request = perf.cost("jev", "paid", meta, {"jev": {"per_request": 0.002}})
        self.assertAlmostEqual(per_request["per_item"], 0.002)
        billed = perf.cost("jev", "paid", {**meta, "requests": {**meta["requests"], "billed": 1.5}}, {})
        self.assertEqual((billed["basis"], billed["per_item"]), ("measured", 0.015))

    def test_memory_adds_server_model(self):
        meta = {"runner": {"peak_footprint_bytes": 1000, "server_memory": {"kind": "ollama", "loaded_bytes": 5000,
                                                                          "gpu_bytes": 4000}},
                "timing": {"model_bytes": None}}
        m = perf.memory(meta)
        self.assertEqual((m["total"], m["server"], m["server_gpu"]), (6000, 5000, 4000))
        self.assertIsNone(perf.memory({})["total"])

    def test_sweep_latency(self):
        meta = {"timing": {"sweep": [
            {"axis": "concurrency", "value": 4, "items_per_second": 8.0, "per_call": None},
            {"axis": "batch_size", "value": 8, "items_per_second": 10.0, "per_call": {"p50": 0.7}}]}}
        pts = perf.sweep_points(meta)
        self.assertAlmostEqual(pts[0]["latency"], 0.5)  # Little's law: 4 in flight / 8 per second
        self.assertEqual(pts[1]["latency"], 0.7)

    def test_config_text(self):
        self.assertEqual(perf.config_text({"timing": {"axis": "batch_size", "value": 16}}), "batch size 16")
        self.assertEqual(perf.config_text({"args": {"concurrency": 4}}), "concurrency 4")


class RunnerMeasurementTest(unittest.TestCase):
    def test_parse_time_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "t.txt"
            path.write_text("        0.43 real  0.29 user  0.06 sys\n   214859776  maximum resident set size\n"
                            "   206226032  peak memory footprint\n")
            self.assertEqual(parse_time_l(path), {"peak_rss_bytes": 214859776, "peak_footprint_bytes": 206226032})
            self.assertEqual(parse_time_l(Path(tmp) / "missing.txt"), {})

    def test_paid_cap_counts_latency_and_sweep_requests(self):
        with tempfile.TemporaryDirectory() as tmp:
            app = App("fake", "Fake", "Local", "max p", folder=tmp, cost="paid")
            (app.path / ".venv" / "bin").mkdir(parents=True)
            (app.path / ".venv" / "bin" / "python").write_text("")
            self.assertIsNone(preflight(app, 200, 1, 3600, False, extra_requests=100))
            self.assertIn("requests is above", preflight(app, 200, 1, 3600, False, extra_requests=100 + 512 + 200))


if __name__ == "__main__":
    unittest.main()
