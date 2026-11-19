import itertools
import unittest

import numpy as np
import pandas as pd

from report import ece
from selective import (accuracy_at_coverage, apply_threshold, aurc, calibration_warnings, classwise_ece,
                       coverage_at_target, cross_fit_at_target, curve, fit_threshold, held_out_at_target,
                       expected_correct, groups, mce, reliability, top2_accuracy)
from stats import wilson

# conf: 0.9 x3 (2 right), 0.5 x2 (1 right), 0.2 x1 (wrong)
CONF = [0.9, 0.5, 0.9, 0.2, 0.9, 0.5]
RIGHT = [True, True, False, False, True, False]


def brute_force_expected(conf, correct, m):
    """Mean over every tie-respecting ordering of correct predictions among the top m."""
    conf, correct = np.array(conf), np.array(correct)
    totals = []
    for perm in itertools.permutations(range(len(conf))):
        order = sorted(perm, key=lambda i: -conf[i])  # stable: ties keep the permuted order
        totals.append(correct[order[:m]].sum())
    return float(np.mean(totals))


class TieAwareTest(unittest.TestCase):
    def setUp(self):
        self.g = groups(CONF, RIGHT)

    def test_groups(self):
        np.testing.assert_array_equal(self.g.value, [0.9, 0.5, 0.2])
        np.testing.assert_array_equal(self.g.size, [3, 2, 1])
        np.testing.assert_array_equal(self.g.correct, [2, 1, 0])

    def test_expected_correct_matches_brute_force(self):
        for m in range(1, 7):
            self.assertAlmostEqual(expected_correct(self.g, np.array([m]))[0], brute_force_expected(CONF, RIGHT, m))

    def test_result_does_not_depend_on_row_order(self):
        perm = [5, 3, 1, 0, 4, 2]
        other = groups([CONF[i] for i in perm], [RIGHT[i] for i in perm])
        self.assertEqual(curve(self.g), curve(other))
        self.assertEqual(aurc(self.g), aurc(other))

    def test_aurc_and_excess(self):
        # risks at m = 1..6: 1/3, 1/3, 1/3, 1 - 2.5/4, 1 - 3/5, 1/2
        risks = [1 / 3, 1 / 3, 1 / 3, 0.375, 0.4, 0.5]
        ideal = [0, 0, 0, 0.25, 0.4, 0.5]  # 3 correct first
        value, excess = aurc(self.g)
        self.assertAlmostEqual(value, np.mean(risks))
        self.assertAlmostEqual(excess, np.mean(risks) - np.mean(ideal))

    def test_perfect_ranking_has_zero_excess(self):
        g = groups([0.9, 0.8, 0.7, 0.1], [True, True, True, False])
        self.assertAlmostEqual(aurc(g)[1], 0.0)

    def test_accuracy_at_coverage_flags_unreachable(self):
        half = accuracy_at_coverage(self.g, 0.5)  # 3 accepted = exactly the 0.9 group
        self.assertTrue(half["reachable"])
        self.assertAlmostEqual(half["accuracy"], 2 / 3)
        third = accuracy_at_coverage(self.g, 1 / 3)  # 2 accepted: inside the 0.9 group
        self.assertFalse(third["reachable"])
        self.assertAlmostEqual(third["accuracy"], 2 / 3)

    def test_coverage_at_target_checks_every_threshold(self):
        # Accuracy by threshold: 0.9 -> 2/3, 0.5 -> 3/5, 0.2 -> 3/6. The curve is not monotone in general,
        # so the largest coverage meeting the target wins, and only whole tie groups count.
        r = coverage_at_target(self.g, 0.6)
        self.assertEqual((r["accepted"], r["threshold"]), (5, 0.5))
        g = groups([0.9, 0.8, 0.7, 0.6], [False, True, True, True])  # accuracy rises after a bad first item
        self.assertEqual(coverage_at_target(g, 0.75)["accepted"], 4)
        self.assertEqual(coverage_at_target(groups([0.9], [False]), 0.5)["coverage"], 0.0)

    def test_certified_uses_wilson_lower_bound(self):
        g = groups(np.linspace(1, 0.5, 60), [True] * 60)
        self.assertEqual(coverage_at_target(g, 0.95)["coverage"], 1.0)
        self.assertLess(wilson(60, 60)[0], 0.95)
        self.assertEqual(coverage_at_target(g, 0.95, certified=True)["coverage"], 0.0)
        g = groups(np.linspace(1, 0.5, 100), [True] * 100)
        self.assertEqual(coverage_at_target(g, 0.95, certified=True)["accepted"], 100)  # LB 0.963


class CalibrationTest(unittest.TestCase):
    def test_reliability_bins_and_ece_agree(self):
        conf = pd.Series([0.95, 0.95, 0.85, 0.55, 0.15, 1.0])
        correct = pd.Series([True, False, True, True, False, True])
        bins = reliability(conf, correct)
        self.assertEqual(sum(b["count"] for b in bins), 6)
        self.assertEqual(bins[9]["count"], 3)  # 1.0 goes in the top bin
        from_bins = sum(abs(b["accuracy"] - b["confidence"]) * b["count"] for b in bins if b["count"]) / 6
        self.assertAlmostEqual(from_bins, ece(conf, correct))

    def test_mce_ignores_small_bins(self):
        bins = [{"count": 12, "accuracy": 0.9, "confidence": 0.95},
                {"count": 2, "accuracy": 0.0, "confidence": 0.95},
                {"count": 0, "accuracy": float("nan"), "confidence": float("nan")}]
        value, used = mce(bins, min_count=10)
        self.assertAlmostEqual(value, 0.05)
        self.assertEqual(used, 1)
        self.assertTrue(np.isnan(mce(bins, min_count=100)[0]))

    def test_classwise_ece(self):
        probs = np.array([[0.8, 0.2], [0.8, 0.2], [0.3, 0.7], [0.3, 0.7]])
        labels = np.array([0, 1, 1, 1])
        # label 0: bin 0.8 -> freq 0.5 vs 0.8 (2 items), bin 0.3 -> freq 0 vs 0.3 (2 items): (0.6 + 0.6) / 4 = 0.3
        # label 1: bin 0.2 -> 0.5 vs 0.2, bin 0.7 -> 1 vs 0.7: (0.6 + 0.6) / 4 = 0.3
        self.assertAlmostEqual(classwise_ece(probs, labels), 0.3)

    def test_top2_with_ties(self):
        probs = np.array([[0.5, 0.3, 0.2, 0.0],   # true 1: second -> in top 2
                          [0.5, 0.3, 0.2, 0.0],   # true 2: third -> out
                          [1.0, 0.0, 0.0, 0.0],   # one-hot, true 3: tied with 2 others for 1 slot -> 1/3
                          [0.25, 0.25, 0.25, 0.25]])  # uniform, true 0: 2 slots among 4 -> 1/2
        self.assertAlmostEqual(top2_accuracy(probs, np.array([1, 2, 3, 0])), (1 + 0 + 1 / 3 + 0.5) / 4)

    def test_warnings(self):
        df = pd.DataFrame({"confidence": [1.0, 1.0, 1.0]})
        onehot = np.array([[1.0, 0, 0, 0]] * 3)
        w = calibration_warnings(df, onehot, confidence_rule="own", probabilities="uncalibrated")
        self.assertEqual(len(w), 4)  # own score, uncalibrated, constant, one-hot
        df = pd.DataFrame({"confidence": [0.6, 0.7, 0.8]})
        soft = np.array([[0.6, 0.4, 0, 0], [0.7, 0.3, 0, 0], [0.8, 0.2, 0, 0]])
        self.assertEqual(calibration_warnings(df, soft, confidence_rule="max_p", probabilities="native"), [])


if __name__ == "__main__":
    unittest.main()


class HeldOutThresholdTest(unittest.TestCase):
    """Thresholds fitted on one set of items and applied to another (phase 2, item 2)."""

    def test_apply_threshold_counts(self):
        r = apply_threshold(CONF, RIGHT, 0.5)  # five predictions at >= 0.5, three of them right
        self.assertEqual((r["accepted"], r["correct"]), (5, 3))
        self.assertAlmostEqual(r["coverage"], 5 / 6)
        self.assertEqual(apply_threshold(CONF, RIGHT, None)["accepted"], 0)

    def test_fit_matches_in_sample_selection(self):
        g = groups(CONF, RIGHT)
        self.assertEqual(fit_threshold(CONF, RIGHT, 0.6, certified=False),
                         coverage_at_target(g, 0.6)["threshold"])

    def test_held_out_scores_only_evaluation_items(self):
        cal_conf, cal_right = [0.9] * 50 + [0.4] * 50, [True] * 50 + [False] * 50
        r = held_out_at_target(cal_conf, cal_right, CONF, RIGHT, 0.9, certified=False)
        self.assertEqual(r["threshold"], 0.9)
        self.assertEqual((r["accepted"], r["n"], r["n_calibration"]), (3, 6, 100))
        self.assertAlmostEqual(r["accuracy"], 2 / 3)
        self.assertFalse(r["met"])  # the calibration sample's threshold misses the target on new items

    def test_in_sample_threshold_overfits_and_cross_fitting_does_not(self):
        # Pure noise: confidence carries no information, accuracy is 80% everywhere. The in-sample search still
        # finds a "90%" threshold by luck; cross-fitting should not reproduce that coverage.
        rng = np.random.default_rng(1)
        conf, right = rng.random(400), rng.random(400) < 0.8
        in_sample = coverage_at_target(groups(conf, right), 0.9)
        crossed = cross_fit_at_target(conf, right, 0.9, certified=False)
        self.assertGreaterEqual(in_sample["accuracy"], 0.9)  # in-sample always "meets" the target it was fitted to
        self.assertLess(crossed["accuracy"], 0.9)  # held-out items show the real (lower) accuracy
        self.assertFalse(crossed["met"])
        self.assertEqual(crossed["method"], "cross-fitted")
        self.assertEqual(crossed["n"], 400)

    def test_cross_fit_is_deterministic(self):
        rng = np.random.default_rng(2)
        right = rng.random(400) < 0.85
        conf = 0.5 * right + 0.5 * rng.random(400)  # informative confidence: right answers score higher
        a, b = cross_fit_at_target(conf, right, 0.9), cross_fit_at_target(conf, right, 0.9)
        self.assertGreater(a["accepted"], 0)
        self.assertEqual((a["accepted"], a["thresholds"]), (b["accepted"], b["thresholds"]))


class CalibrationFrameTest(unittest.TestCase):
    """report.calibration_frame: another seed's run, minus items in the evaluation sample."""

    def test_overlap_removed_and_eval_seed_ignored(self):
        import tempfile
        from pathlib import Path

        from report import apply_held_out, calibration_frame
        from selective import TARGETS
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            ev = pd.DataFrame({"sample_id": [1, 2, 3], "text": list("abc"), "label": ["x"] * 3, "pred": ["x"] * 3,
                               "confidence": [0.9, 0.8, 0.7]})
            cal = pd.DataFrame({"sample_id": [3, 4, 5, 6], "text": list("cdef"), "label": ["x"] * 4,
                                "pred": ["x", "x", "y", "x"], "confidence": [0.9, 0.95, 0.2, 0.85]})
            ev.to_csv(run / "eval.csv", index=False)
            cal.to_csv(run / "cal.csv", index=False)
            entry = {"runs": [{"seed": 42, "status": "ok", "csv": "eval.csv"},
                              {"seed": 43, "status": "ok", "csv": "cal.csv"}]}
            df, seed = calibration_frame(run, entry, ev, 42)
            self.assertEqual(seed, 43)
            self.assertEqual(sorted(df["sample_id"]), [4, 5, 6])  # item 3 is in the evaluation sample
            self.assertIsNone(calibration_frame(run, {"runs": entry["runs"][:1]}, ev, 42))
            m = {}
            apply_held_out(m, ev, (df, seed))
            self.assertEqual(set(m["held_out"]), set(TARGETS))
            self.assertEqual(m["held_out"][0.9]["method"], "held-out")
            self.assertEqual(m["held_out"][0.9]["n_calibration"], 3)
