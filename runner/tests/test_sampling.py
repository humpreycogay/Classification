import re
import unittest
from pathlib import Path

import pandas as pd

import task as tasks
from apps import CLASSIFIERS_DIR
from sampling import DATA_PATH, ROOT, expected_sample, sample_sha1, stratified_sample, tier_name
from tests.helpers import fake_test_set


class StratifiedSampleTest(unittest.TestCase):
    def setUp(self):
        self.df = fake_test_set(per_label=10)  # 40 rows, 4 labels

    def test_n_below_number_of_labels(self):
        for n in (1, 2, 3):
            out = stratified_sample(self.df, n, seed=42)
            self.assertEqual(len(out), n)
            self.assertEqual(out["label"].nunique(), n)  # one article from each of the first n labels
            self.assertEqual(sorted(out["label"]), sorted(self.df["label"].unique())[:n])

    def test_remainder_allocated_exactly(self):
        out = stratified_sample(self.df, 23, seed=7)  # 23 = 4 * 5 + 3
        self.assertEqual(len(out), 23)
        counts = out["label"].value_counts()
        self.assertEqual(sorted(counts.tolist()), [5, 6, 6, 6])
        self.assertFalse(out.index.duplicated().any())

    def test_n_201_on_ag_news_shape(self):
        big = fake_test_set(per_label=1900)
        out = stratified_sample(big, 201, seed=42)
        self.assertEqual(len(out), 201)
        self.assertEqual(sorted(out["label"].value_counts().tolist()), [50, 50, 50, 51])

    def test_n_equal_dataset_size_is_whole_set(self):
        out = stratified_sample(self.df, len(self.df), seed=42)
        pd.testing.assert_frame_equal(out, self.df)

    def test_n_zero_is_whole_set(self):
        out = stratified_sample(self.df, 0, seed=42)
        pd.testing.assert_frame_equal(out, self.df)

    def test_rejects_unsupported_sizes(self):
        for n in (-1, len(self.df) + 1):
            with self.assertRaises(ValueError):
                stratified_sample(self.df, n, seed=42)

    def test_rejects_when_a_label_is_too_small(self):
        uneven = pd.concat([self.df, self.df[self.df["label"] == "world"]], ignore_index=True)  # world has 20
        with self.assertRaises(ValueError):
            stratified_sample(uneven, 45, seed=42)  # needs 12 of each other label, which only has 10

    def test_deterministic_and_seed_dependent(self):
        a, b = stratified_sample(self.df, 20, seed=1), stratified_sample(self.df, 20, seed=1)
        c = stratified_sample(self.df, 20, seed=2)
        pd.testing.assert_frame_equal(a, b)
        self.assertFalse(a.index.equals(c.index))

    def test_matches_previous_sampler_when_divisible(self):
        """Existing n=200 results must stay comparable: same rows, same order as the old groupby sampler."""
        for n, seed in ((20, 42), (8, 3)):
            old = self.df.groupby("label").sample(n // 4, random_state=seed).sample(frac=1, random_state=seed)
            pd.testing.assert_frame_equal(stratified_sample(self.df, n, seed), old)


class AppCopiesTest(unittest.TestCase):
    def test_every_app_uses_the_same_task_loader(self):
        """Sampling, splits and manifests live in shared/task.py, which the runner and every app import."""
        self.assertEqual(Path(tasks.__file__).resolve(), ROOT / "shared" / "task.py")
        apps = sorted(p.parent for p in CLASSIFIERS_DIR.glob("*/evaluate.py"))
        self.assertTrue(apps)
        for folder in apps:
            with self.subTest(app=folder.name):
                self.assertNotIn("def stratified_sample(", (folder / "evaluate.py").read_text())


class TierTest(unittest.TestCase):
    def test_tier_names(self):
        self.assertEqual(tier_name(200), "smoke")
        self.assertEqual(tier_name(1000), "common")
        self.assertEqual(tier_name(7600), "full")
        self.assertEqual(tier_name(201), "custom")


@unittest.skipUnless(DATA_PATH.exists(), "AG News test CSV not downloaded")
class RealSampleTest(unittest.TestCase):
    def test_known_sample_hash(self):
        """The n=200 seed-42 sample every stored result was produced on."""
        self.assertEqual(sample_sha1(expected_sample(200, 42)), "a41311eeeb9199533d93b305d2b6429ed76f7235")

    def test_full_set(self):
        self.assertEqual(len(expected_sample(0, 42)), 7600)
        self.assertEqual(len(expected_sample(7600, 42)), 7600)


if __name__ == "__main__":
    unittest.main()
