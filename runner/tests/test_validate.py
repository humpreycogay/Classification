import json
import os
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from sampling import sample_sha1, stratified_sample
from tests.helpers import fake_predictions, fake_test_set
from validate import check_frame, validate_result


def expected(n: int = 20, seed: int = 42):
    return stratified_sample(fake_test_set(), n, seed).rename_axis("sample_id").reset_index()


class CheckFrameTest(unittest.TestCase):
    def setUp(self):
        self.df = fake_predictions(expected())

    def assertProblem(self, df, fragment, **kwargs):
        problems = check_frame(df, **kwargs)
        self.assertTrue(any(fragment in p for p in problems), problems)

    def test_valid_frame(self):
        self.assertEqual(check_frame(self.df), [])

    def test_missing_column(self):
        self.assertProblem(self.df.drop(columns=["p_world"]), "missing column")

    def test_unknown_label_and_pred(self):
        df = self.df.copy()
        df.loc[0, "pred"] = "weather"
        df.loc[1, "label"] = "weather"
        problems = check_frame(df)
        self.assertTrue(any("unknown pred" in p for p in problems))
        self.assertTrue(any("unknown label" in p for p in problems))

    def test_duplicate_sample_id(self):
        df = self.df.copy()
        df.loc[1, "sample_id"] = df.loc[0, "sample_id"]
        self.assertProblem(df, "duplicate sample_id")

    def test_non_finite_probability(self):
        df = self.df.copy()
        df.loc[0, "p_world"] = np.nan
        self.assertProblem(df, "non-finite")

    def test_probability_out_of_range(self):
        df = self.df.copy()
        df.loc[0, ["p_world", "p_sports", "p_business", "p_sci_tech"]] = [1.5, -0.5, 0, 0]
        self.assertProblem(df, "outside [0, 1]")

    def test_rows_must_sum_to_one(self):
        df = self.df.copy()
        df["p_world"] = df["p_world"] + 0.1
        self.assertProblem(df, "not summing to 1")

    def test_sum_tolerance_grows_with_label_count(self):
        """Rounding each of 77 probabilities to 4 decimals can move the sum by up to ~0.004."""
        labels = [f"l{i}" for i in range(77)]
        probs = np.full((1, 77), 0.0129)  # 77 x 0.0129 = 0.9933
        probs[0, 0] = 1 - 0.0129 * 76 - 0.004  # sum = 0.996
        df = pd.DataFrame(probs, columns=[f"p_{l}" for l in labels])
        df["text"], df["label"], df["pred"], df["confidence"] = "t", "l0", "l0", probs[0].max()
        self.assertEqual(check_frame(df, labels=labels), [])  # 0.004 off: within 77 x 1e-4
        df["p_l1"] -= 0.01  # 0.014 off: too much even for 77 labels (l0 stays the top label)
        self.assertTrue(any("not summing" in p for p in check_frame(df, labels=labels)))
        self.assertEqual(check_frame(df, labels=labels, sum_tol=0.02), [])  # an app that rounds to 2 decimals

    def test_pred_must_be_argmax(self):
        df = self.df.copy()
        df.loc[0, "pred"] = next(label for label in ("world", "sports") if label != df.loc[0, "pred"])
        self.assertProblem(df, "not argmax")

    def test_confidence_rule(self):
        df = self.df.copy()
        df["confidence"] = 0.5
        self.assertProblem(df, "differs from max p", confidence_rule="max_p")
        self.assertEqual(check_frame(df, confidence_rule="own"), [])
        df.loc[0, "confidence"] = 1.2
        self.assertProblem(df, "confidence not a finite value", confidence_rule="own")


class ValidateResultTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.expected = expected()
        self.csv, self.meta = self.dir / "predictions_n20_seed42.csv", self.dir / "run_n20_seed42.json"

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, df=None, **meta_overrides):
        df = fake_predictions(self.expected) if df is None else df
        df.to_csv(self.csv, index=False)
        meta = {"n": len(df), "seed": 42, "sample_sha1": sample_sha1(self.expected), "run_id": "run-1",
                "load_seconds": 0.1, "warmup_seconds": 0.1, "classify_seconds": 1.0, **meta_overrides}
        self.meta.write_text(json.dumps(meta))

    def validate(self, **kwargs):
        return validate_result(self.csv, self.meta, expected=self.expected, seed=42, **kwargs)

    def test_valid(self):
        self.write()
        self.assertEqual(self.validate(run_id="run-1", not_before=time.time() - 60), [])

    def test_missing_files(self):
        self.assertEqual(self.validate(), ["no predictions CSV"])
        fake_predictions(self.expected).to_csv(self.csv, index=False)
        self.assertIn("no run metadata", self.validate())

    def test_wrong_row_count(self):
        self.write(fake_predictions(self.expected).head(19))
        self.assertTrue(any("19 rows, expected 20" in p for p in self.validate()))

    def test_different_sample(self):
        other = stratified_sample(fake_test_set(), 20, 43).rename_axis("sample_id").reset_index()
        self.write(fake_predictions(other))
        problems = self.validate()
        self.assertTrue(any("text differs" in p for p in problems), problems)
        self.assertTrue(any("sample_sha1 differs from the CSV" in p for p in problems), problems)

    def test_metadata_must_agree(self):
        self.write(seed=43, n=21, sample_sha1="0" * 40)
        problems = self.validate()
        for fragment in ("seed = 43", "n = 21", "differs from the expected sample"):
            self.assertTrue(any(fragment in p for p in problems), (fragment, problems))

    def test_stale_files_and_foreign_run_id(self):
        self.write()
        old = time.time() - 3600
        os.utime(self.csv, (old, old))
        problems = self.validate(not_before=time.time() - 60, run_id="run-2")
        self.assertTrue(any("predictions CSV is older than this run" in p for p in problems), problems)
        self.assertTrue(any("not this run's" in p for p in problems), problems)

    def test_virtual_baseline_without_metadata(self):
        fake_predictions(self.expected).to_csv(self.csv, index=False)
        self.assertEqual(validate_result(self.csv, None, expected=self.expected, seed=42, meta_required=False), [])


if __name__ == "__main__":
    unittest.main()
