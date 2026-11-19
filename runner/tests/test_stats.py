import math
import unittest

import numpy as np
import pandas as pd

from report import app_metrics, auroc, ece
from stats import (bootstrap_metrics, holm, leading_tier, mcnemar_p, paired_difference, pareto_front,
                   percentile_ci, rank_intervals, resample_weights, tango_interval, tango_restricted_p21,
                   tango_statistic, weighted_auroc)
from tests.helpers import fake_predictions, fake_test_set
from validate import LABELS


class McNemarTest(unittest.TestCase):
    def test_exact_values(self):
        self.assertEqual(mcnemar_p(0, 0), 1.0)
        self.assertAlmostEqual(mcnemar_p(0, 5), 2 / 32)  # all 5 discordant pairs one way
        self.assertAlmostEqual(mcnemar_p(1, 9), 2 * 11 / 1024)
        self.assertEqual(mcnemar_p(4, 4), 1.0)


class HolmTest(unittest.TestCase):
    def test_step_down(self):
        adjusted = holm({"a": 0.01, "b": 0.04, "c": 0.03})
        # sorted 0.01, 0.03, 0.04 -> 3 * 0.01, max(0.03, 2 * 0.03), max(0.06, 1 * 0.04)
        self.assertAlmostEqual(adjusted["a"], 0.03)
        self.assertAlmostEqual(adjusted["c"], 0.06)
        self.assertAlmostEqual(adjusted["b"], 0.06)  # monotone: never below an earlier adjusted p

    def test_capped_at_one_and_empty(self):
        self.assertEqual(holm({"a": 0.6, "b": 0.7}), {"a": 1.0, "b": 1.0})
        self.assertEqual(holm({}), {})


class PairedDifferenceTest(unittest.TestCase):
    def test_difference_and_interval(self):
        a = np.array([1] * 90 + [0] * 10, bool)
        b = a.copy()
        b[:6] = False  # B loses 6 articles A gets right
        b[90:92] = True  # and wins 2 that A misses
        d = paired_difference(a, b)
        self.assertEqual((d["only_a"], d["only_b"]), (6, 2))
        self.assertAlmostEqual(d["diff"], -0.04)
        self.assertEqual((d["low"], d["high"]), tango_interval(6, 2, 100))
        self.assertLess(d["low"], -0.04)
        self.assertGreater(d["high"], -0.04)
        self.assertAlmostEqual(d["p"], mcnemar_p(6, 2))

    def test_identical_apps_get_a_real_interval(self):
        a = np.array([True] * 180 + [False] * 20)
        d = paired_difference(a, a)
        self.assertEqual((d["diff"], d["p"]), (0.0, 1.0))
        # Closed form for no discordant pairs: d / (1 - d) = z^2 / n at the bounds.
        z2 = 1.959963984540054 ** 2
        self.assertAlmostEqual(d["high"], z2 / (200 + z2), places=6)
        self.assertAlmostEqual(d["low"], -d["high"], places=9)


class TangoTest(unittest.TestCase):
    def test_restricted_mle_matches_numerical_maximum(self):
        for only_a, only_b, n, d in ((6, 2, 100, -0.01), (3, 9, 50, 0.05), (0, 4, 80, 0.02), (5, 0, 60, -0.1)):
            grid = np.linspace(max(0.0, -d) + 1e-9, (1 - d) / 2 - 1e-9, 200001)  # p21 keeping all cells valid
            with np.errstate(divide="ignore"):
                ll = (only_b * np.log(grid + d) + only_a * np.log(grid)
                      + (n - only_a - only_b) * np.log(1 - 2 * grid - d))
            self.assertAlmostEqual(tango_restricted_p21(only_a, only_b, n, d), grid[np.argmax(ll)], places=4)

    def test_interval_solves_the_score_equation(self):
        low, high = tango_interval(6, 2, 100)
        self.assertAlmostEqual(tango_statistic(6, 2, 100, low), 1.959963984540054, places=6)
        self.assertAlmostEqual(tango_statistic(6, 2, 100, high), -1.959963984540054, places=6)

    def test_coverage_with_few_discordant_pairs(self):
        """Simulated coverage stays near 95% where Wald undercovers (n = 60, 3% + 1% discordant)."""
        rng = np.random.default_rng(0)
        n, p_a, p_b, trials = 60, 0.03, 0.01, 3000
        true_d = p_b - p_a
        hits = wald_hits = 0
        for counts in rng.multinomial(n, [p_a, p_b, 1 - p_a - p_b], size=trials):
            only_a, only_b = int(counts[0]), int(counts[1])
            lo, hi = tango_interval(only_a, only_b, n)
            hits += lo <= true_d <= hi
            d = (only_b - only_a) / n
            se = math.sqrt(max(only_a + only_b - (only_b - only_a) ** 2 / n, 0)) / n
            wald_hits += d - 1.96 * se <= true_d <= d + 1.96 * se
        self.assertGreater(hits / trials, 0.93)
        self.assertLess(wald_hits / trials, hits / trials)


class BootstrapTest(unittest.TestCase):
    def setUp(self):
        sample = fake_test_set(per_label=25)
        mask = [i % 4 != 0 for i in range(len(sample))]
        self.df = fake_predictions(sample, mask, seed=3)
        # Coarse confidences create ties, which AUROC must count as half.
        self.df["confidence"] = self.df["confidence"].round(1)

    def test_unit_weights_reproduce_point_metrics(self):
        W = np.ones((1, len(self.df)))
        boot = bootstrap_metrics(self.df, W, LABELS)
        point = app_metrics(self.df)
        correct = self.df["label"] == self.df["pred"]
        conf = self.df["confidence"].astype(float)
        self.assertAlmostEqual(boot["accuracy"][0], point["accuracy"])
        self.assertAlmostEqual(boot["macro_f1"][0], point["macro_f1"])
        self.assertAlmostEqual(boot["auroc"][0], auroc(conf, correct))
        self.assertAlmostEqual(boot["ece"][0], ece(conf, correct))
        self.assertAlmostEqual(boot["brier"][0], point["brier"])

    def test_weighted_auroc_matches_duplicated_rows(self):
        conf = np.array([0.9, 0.9, 0.5, 0.3, 0.5, 0.7])
        correct = np.array([True, False, True, False, False, True])
        weights = np.array([[2, 0, 1, 3, 1, 1]], float)
        expanded = np.repeat(np.arange(len(conf)), weights[0].astype(int))
        expected = auroc(pd.Series(conf[expanded]), pd.Series(correct[expanded]))
        self.assertAlmostEqual(weighted_auroc(weights, conf, correct)[0], expected)

    def test_resamples_are_deterministic_and_complete(self):
        W1, W2 = resample_weights(50, 100, seed=7), resample_weights(50, 100, seed=7)
        np.testing.assert_array_equal(W1, W2)
        self.assertTrue((W1.sum(axis=1) == 50).all())
        self.assertFalse(np.array_equal(W1, resample_weights(50, 100, seed=8)))

    def test_interval_contains_estimate(self):
        W = resample_weights(len(self.df), 500, seed=1)
        lo, hi = percentile_ci(bootstrap_metrics(self.df, W, LABELS)["accuracy"])
        self.assertLess(lo, 0.75)
        self.assertGreater(hi, 0.75)

    def test_paired_difference_of_identical_apps_is_zero(self):
        W = resample_weights(len(self.df), 200, seed=1)
        acc = bootstrap_metrics(self.df, W, LABELS)["accuracy"]
        self.assertEqual(percentile_ci(acc - acc), (0.0, 0.0))


class RankingTest(unittest.TestCase):
    def test_rank_intervals(self):
        acc = {"a": np.array([0.9, 0.9, 0.8, 0.9]), "b": np.array([0.8, 0.9, 0.9, 0.7]), "c": np.array([0.5, 0.5, 0.5, 0.5])}
        r = rank_intervals(acc)
        # replicate ranks: a = 1, 1, 2, 1; b = 2, 1, 1, 2 (ties share rank 1); c always 3
        self.assertEqual((r["a"]["rank_low"], r["a"]["rank_high"]), (1, 2))
        self.assertEqual((r["c"]["rank_low"], r["c"]["rank_high"]), (3, 3))
        self.assertAlmostEqual(r["a"]["p_best"], (1 + 0.5 + 0 + 1) / 4)
        self.assertAlmostEqual(r["b"]["p_best"], (0 + 0.5 + 1 + 0) / 4)
        self.assertAlmostEqual(sum(x["p_best"] for x in r.values()), 1.0)

    def test_leading_tier(self):
        ranks = {"a": {"rank_low": 1}, "b": {"rank_low": 1}, "c": {"rank_low": 2}}
        self.assertEqual(leading_tier(ranks), ["a", "b"])

    def test_pareto_front(self):
        points = {"fast": (0.80, 0.001), "accurate": (0.95, 1.0), "middle": (0.90, 0.1),
                  "dominated": (0.85, 0.5), "tie_slower": (0.95, 2.0)}
        self.assertEqual(pareto_front(points), ["fast", "middle", "accurate"])

    def test_equal_points_both_kept(self):
        self.assertEqual(sorted(pareto_front({"a": (0.9, 0.1), "b": (0.9, 0.1)})), ["a", "b"])


if __name__ == "__main__":
    unittest.main()
