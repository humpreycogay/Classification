"""shared/textnn.py (neural nets trained from scratch): no data and no real training needed.

Needs torch, which the runner's own .venv does not have, so the tests skip there. Run them with an app's venv:
    cd runner && ../classifiers/scratch_fasttext/.venv/bin/python -m unittest tests.test_textnn
"""

from __future__ import annotations

import importlib.util
import random
import unittest

import shared_path  # noqa: F401  (puts shared/ on sys.path)

HAS_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(HAS_TORCH, "torch is not installed in this venv")
class TextNNTest(unittest.TestCase):
    def setUp(self):
        global textnn, torch, pd
        import pandas as pd
        import torch
        import textnn

    def test_tokenize(self):
        self.assertEqual(textnn.tokenize("My card hasn't arrived!"), ["my", "card", "hasn't", "arrived", "!"])

    def test_vocab_reserves_pad_and_unknown(self):
        v = textnn.Vocab.build(["a b b", "b c"], min_freq=2, max_size=100)
        self.assertEqual(v.itos[:2], ["<pad>", "<unk>"])
        self.assertEqual(v.encode("b a zzz", 10), [2, textnn.UNK, textnn.UNK])  # only "b" is seen twice
        self.assertEqual(v.encode("", 10), [textnn.UNK])  # never empty
        self.assertEqual(len(v.encode("b " * 50, 7)), 7)  # cut to max_len

    def test_bigram_ids_are_stable_and_in_range(self):
        ids = textnn.bigram_ids([5, 9, 5, 9], buckets=100, offset=1000)
        self.assertEqual(len(ids), 3)
        self.assertEqual(ids[0], ids[2])  # the same pair gets the same bucket
        self.assertTrue(all(1000 <= i < 1100 for i in ids))
        self.assertEqual(ids, textnn.bigram_ids([5, 9, 5, 9], buckets=100, offset=1000))

    def test_batches_cover_every_row_once(self):
        seqs = [[1] * n for n in (3, 1, 5, 2, 4)]
        for rng in (None, random.Random(0)):
            seen = []
            for idx, x, y in textnn.batches(seqs, list(range(5)), 2, rng):
                seen += idx
                self.assertEqual(x.shape[1], max(len(seqs[i]) for i in idx))  # padded to the batch's longest
                self.assertEqual(y.tolist(), idx)
            self.assertEqual(sorted(seen), list(range(5)))

    def test_split_is_stratified_and_disjoint(self):
        df = pd.DataFrame({"text": [f"t{i}" for i in range(100)], "label": ["a"] * 50 + ["b"] * 50})
        fit, val = textnn.split_train_val(df, 0.1, seed=0)
        self.assertEqual(sorted(val["label"].value_counts().tolist()), [5, 5])
        self.assertEqual(len(fit) + len(val), 100)
        self.assertFalse(set(fit["text"]) & set(val["text"]))

    def test_temperature_is_positive_and_keeps_the_argmax(self):
        torch.manual_seed(0)
        logits = torch.randn(200, 4) * 5
        y = logits.argmax(1)
        y[:60] = torch.randint(0, 4, (60,))  # overconfident: many confident answers are wrong
        t = textnn.fit_temperature(logits, y)
        self.assertGreater(t, 1.0)  # overconfidence is softened
        self.assertTrue(torch.equal((logits / t).argmax(1), logits.argmax(1)))


if __name__ == "__main__":
    unittest.main()
