import json
import re
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from apps import App
from report import build_report, prediction_stability, spread
from run_all import result_files
from sampling import sample_sha1, stratified_sample
from tests.helpers import fake_predictions, fake_test_set
from validate import LABELS


def sample(seed: int = 42, n: int = 20):
    return stratified_sample(fake_test_set(), n, seed).rename_axis("sample_id").reset_index()


class ResultFilesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.app = App("fake", "Fake", "Local", "max p", folder=self.tmp.name)
        self.app.results_dir.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def test_names_include_the_task(self):
        csv, meta = result_files(self.app, 200, 42)
        self.assertEqual((csv.name, meta.name), ("predictions_ag_news_n200_seed42.csv", "run_ag_news_n200_seed42.json"))
        csv, _ = result_files(self.app, 200, 42, "banking77")
        self.assertEqual(csv.name, "predictions_banking77_n200_seed42.csv")

    def test_older_ag_news_names_still_read(self):
        (self.app.results_dir / "predictions_n200_seed42.csv").write_text("x")
        self.assertEqual(result_files(self.app, 200, 42)[0].name, "predictions_n200_seed42.csv")
        # never for another task: an AG News file must not stand in for its results
        self.assertEqual(result_files(self.app, 200, 42, "banking77")[0].name, "predictions_banking77_n200_seed42.csv")

    def test_legacy_file_used_only_for_its_seed(self):
        (self.app.results_dir / "predictions_n200.csv").write_text("x")
        (self.app.results_dir / "run_n200.json").write_text(json.dumps({"seed": 42}))
        self.assertEqual(result_files(self.app, 200, 42)[0].name, "predictions_n200.csv")
        self.assertEqual(result_files(self.app, 200, 43)[0].name, "predictions_ag_news_n200_seed43.csv")

    def test_task_named_file_wins(self):
        (self.app.results_dir / "predictions_n200.csv").write_text("x")
        (self.app.results_dir / "predictions_n200_seed42.csv").write_text("x")
        (self.app.results_dir / "predictions_ag_news_n200_seed42.csv").write_text("x")
        self.assertEqual(result_files(self.app, 200, 42)[0].name, "predictions_ag_news_n200_seed42.csv")


class StatsTest(unittest.TestCase):
    def test_spread(self):
        s = spread([1.0, 2.0, 3.0, 4.0])
        self.assertEqual((s["k"], s["mean"], s["median"]), (4, 2.5, 2.5))
        self.assertAlmostEqual(s["sd"], 1.2909944, places=6)
        self.assertAlmostEqual(s["iqr"], 1.5)
        self.assertTrue(spread([]) ["k"] == 0)

    def test_prediction_stability(self):
        import pandas as pd
        a, b, c = pd.Series(list("aaaa")), pd.Series(list("aaab")), pd.Series(list("aabb"))
        self.assertEqual(prediction_stability([a, a]), (1.0, 1.0))
        unanimous, pairwise = prediction_stability([a, b, c])
        self.assertEqual(unanimous, 0.5)
        self.assertAlmostEqual(pairwise, (0.75 + 0.5 + 0.75) / 3)


class ReportTest(unittest.TestCase):
    """Synthetic run folder: a good app, one on another sample, an invalid one, a virtual row, repeats."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.run = Path(self.tmp.name)
        ref, other = sample(42), sample(43)
        mask = [i % 5 != 0 for i in range(len(ref))]  # 80% accurate
        apps = {}

        def add(name, df, status="ok", meta_sha=None, runs=None, **extra):
            folder = self.run / "apps" / name
            folder.mkdir(parents=True)
            df.to_csv(folder / "p.csv", index=False)
            meta = {"n": len(df), "seed": 42, "sample_sha1": meta_sha or sample_sha1(df), "model": name,
                    "load_seconds": 0.1, "warmup_seconds": 0.1, "classify_seconds": 1.0, "seconds_per_item": 0.05,
                    "items_per_second": 20.0, "python": "3", "packages": {}}
            (folder / "r.json").write_text(json.dumps(meta))
            apps[name] = {"title": f"Title {name}", "runs_on": "Local", "confidence": "max p", "supervision": "zero-shot",
                          "status": status, "csv": f"apps/{name}/p.csv", "run_json": f"apps/{name}/r.json", **extra}
            if runs:
                apps[name]["runs"] = runs

        add("good", fake_predictions(ref, mask))
        add("perfect", fake_predictions(ref))
        add("othersample", fake_predictions(other))
        add("badsha", fake_predictions(ref), meta_sha="f" * 40)
        add("broken", fake_predictions(ref), status="invalid", reason="failed validation: pred is not argmax(p_*)")
        runs = []
        for rep, m in enumerate([mask, [True] * len(ref)], start=1):
            name = f"rep{rep}"
            add(name, fake_predictions(ref, m))
            runs.append({"seed": 42, "rep": rep, "status": "ok", "csv": f"apps/{name}/p.csv", "run_json": f"apps/{name}/r.json"})
        for name in ("rep1", "rep2"):
            del apps[name]
        add("repeated", fake_predictions(ref, mask), runs=runs)
        virtual = fake_predictions(ref)
        (self.run / "apps" / "virtual_majority").mkdir(parents=True)
        virtual.to_csv(self.run / "apps" / "virtual_majority" / "p.csv", index=False)
        apps["virtual_majority"] = {"title": "Title virtual", "runs_on": "Virtual", "confidence": "-", "supervision": "virtual",
                                    "virtual": True, "status": "virtual", "csv": "apps/virtual_majority/p.csv"}
        summary = {"started_at": "2026-09-27T00:00:00+00:00", "mode": "run", "n": len(ref), "seed": 42, "seeds": [42],
                   "repeat": 2, "tier": "custom", "sample_sha1": sample_sha1(ref), "environment": {}, "apps": apps}
        (self.run / "summary.json").write_text(json.dumps(summary))
        self.html = build_report(self.run).read_text()

    def tearDown(self):
        self.tmp.cleanup()

    def section(self, anchor: str) -> str:
        start = self.html.index(f"id='{anchor}'")
        end = self.html.find("<h2", start + 1)
        return self.html[start:end]

    def test_incompatible_results_left_out_of_comparisons(self):
        for anchor in ("overview", "accuracy", "stats", "rankings", "pareto", "confidence", "classes", "confusion",
                       "hardest"):
            text = self.section(anchor)
            for name in ("othersample", "badsha", "broken"):
                self.assertNotIn(f"Title {name}<", text, (anchor, name))
        tiles = self.html[self.html.index("class='tiles'"):self.html.index("id='overview'")]
        self.assertIn("<div class='v'>3 / 6</div>", tiles)  # good, perfect, repeated
        self.assertIn("no comparable results: othersample, badsha, broken", tiles)
        incompatible = self.section("incompatible")
        self.assertIn("Title othersample", incompatible)
        self.assertIn("Title badsha", incompatible)
        self.assertIn("sample_sha1", incompatible)

    def test_top_is_a_compatible_model(self):
        meta = json.loads((self.run / "report_meta.json").read_text())
        self.assertEqual(meta["top_point_estimate"], "perfect")
        self.assertNotIn("othersample", meta["multiple_comparisons"]["family"])
        first_row = re.search(r"<tr class='lead'><td>([^<]+)", self.section("overview")).group(1).strip()
        self.assertEqual(first_row, "Title perfect")

    def test_clear_winner_named(self):
        """100% vs 80% on 20 articles: 80% ranks first in ~1% of resamples, so the tier is the top app alone."""
        meta = json.loads((self.run / "report_meta.json").read_text())
        self.assertEqual(meta["leading_tier"], ["perfect"])
        self.assertEqual(meta["rank_intervals"]["good"]["rank_low"], 2)
        tiles = self.html[self.html.index("class='tiles'"):self.html.index("id='overview'")]
        self.assertIn("Most accurate", tiles)
        self.assertEqual(meta["bootstrap"]["seed"], 20260927)

    def test_no_single_winner_when_tied(self):
        """Make "good" as accurate as "perfect": both can rank first, so no single winner is named."""
        summary = json.loads((self.run / "summary.json").read_text())
        good = pd.read_csv(self.run / summary["apps"]["good"]["csv"])
        good["pred"] = good["label"]
        probs = good[[f"p_{label}" for label in LABELS]].to_numpy()
        probs[:] = 0.1
        for i, label in enumerate(good["label"]):
            probs[i, LABELS.index(label)] = 0.7
        good[[f"p_{label}" for label in LABELS]] = probs
        good["confidence"] = 0.7
        good.to_csv(self.run / summary["apps"]["good"]["csv"], index=False)
        html = build_report(self.run).read_text()
        meta = json.loads((self.run / "report_meta.json").read_text())
        self.assertEqual(sorted(meta["leading_tier"]), ["good", "perfect"])
        tiles = html[html.index("class='tiles'"):html.index("id='overview'")]
        self.assertIn("Bootstrap leading tier", tiles)
        self.assertNotIn("statistically tied)", tiles)
        self.assertNotIn("Most accurate", tiles)

    def test_wording_does_not_overclaim(self):
        """Errata E1-E5, E13: no pairwise claims from marginal CIs, no 'certified', no 'significantly ahead'."""
        for phrase in ("cannot separate", "unlikely to be chance", "(certified)", "significantly ahead",
                       "Leading tier (statistically tied)", "never see labeled"):
            self.assertNotIn(phrase, self.html)
        self.assertIn("Do not infer pairwise significance", self.section("accuracy"))
        self.assertIn("not evidence that the apps are equivalent", self.section("stats"))
        self.assertIn("Auto-accept coverage (Wilson LB", self.section("overview"))
        tiles = self.html[self.html.index("class='tiles'"):self.html.index("id='overview'")]
        self.assertIn("the only app whose bootstrap rank interval includes 1", tiles)

    def test_forest_plot_and_task_card(self):
        stats = self.section("stats")
        self.assertIn("class=\"chart forest\"", stats)
        self.assertIn("percentage points", stats)
        self.assertIn("class='card taskcard'", self.html)
        self.assertNotIn("{{MODELS_HREF}}", self.html)

    def test_statistics_section(self):
        stats = self.section("stats")
        self.assertIn("-20.0 pp", stats)  # good vs perfect
        self.assertIn("0.125", stats)  # raw McNemar p
        self.assertIn("0.250", stats)  # Holm-adjusted over 2 comparisons
        self.assertIn("virtual (not tested)", stats)

    def test_invalid_reported_as_failure(self):
        status = self.section("status")
        self.assertIn("pill invalid", status)
        self.assertIn("pred is not argmax", status)

    def test_repeated_runs_section(self):
        repeats = self.section("repeats")
        self.assertIn("Title repeated", repeats)
        self.assertIn("90.0%", repeats)  # mean of 80% and 100%
        self.assertIn("80.0%", repeats)  # 16 of 20 articles labeled the same in both runs

    def test_selective_and_calibration_sections(self):
        start = self.html.index("id='selective'")
        selective = self.html[start:self.html.index("id='calibration'")]
        self.assertIn("E-AURC", selective)
        self.assertIn("Cov. at 95%", selective)
        self.assertNotIn("Cov. at 99%", selective)  # n = 20 is below the common tier
        self.assertIn("The 99% target is omitted", selective)
        calibration = self.html[self.html.index("id='calibration'"):self.html.index("<h2", self.html.index("id='calibration'"))]
        self.assertIn("Classwise ECE", calibration)
        self.assertIn("Reliability diagrams", calibration)


if __name__ == "__main__":
    unittest.main()
