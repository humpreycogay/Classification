"""shared/finetune.py (pre-trained encoders fine-tuned locally): no model download and no training needed.

Needs torch and transformers, which the runner's own .venv does not have, so the tests skip there. Run them with a
fine-tune app's venv:
    cd runner && ../classifiers/finetune_modernbert/.venv/bin/python -m unittest tests.test_finetune
"""

from __future__ import annotations

import argparse
import importlib.util
import random
import unittest

import shared_path  # noqa: F401  (puts shared/ on sys.path)

HAS_DEPS = all(importlib.util.find_spec(m) is not None for m in ("torch", "transformers"))


@unittest.skipUnless(HAS_DEPS, "torch and transformers are not installed in this venv")
class FineTuneTest(unittest.TestCase):
    def setUp(self):
        global finetune
        import finetune

    def test_training_batches_cover_every_row_once_with_little_padding(self):
        rng = random.Random(0)
        seqs = [[7] * rng.randint(1, 60) for _ in range(500)]
        seen, padded = [], 0
        for idx, ids, mask, y in finetune.batches(seqs, list(range(500)), 32, pad_id=3, rng=random.Random(1)):
            seen += idx
            self.assertEqual(y.tolist(), idx)
            self.assertEqual(mask.sum().item(), sum(len(seqs[i]) for i in idx))  # mask marks the real tokens
            self.assertTrue(((ids == 3) == (mask == 0)).all())  # everything else is the pad id
            padded += (mask == 0).sum().item()
        self.assertEqual(sorted(seen), list(range(500)))
        self.assertLess(padded, 0.1 * sum(map(len, seqs)))  # length bucketing keeps padding small

    def test_classifying_batches_are_longest_first(self):
        seqs = [[1] * n for n in (3, 9, 1, 5)]
        order = [i for idx, *_ in finetune.batches(seqs, None, 2, 0, None) for i in idx]
        self.assertEqual(order, [1, 3, 0, 2])

    def test_config_overrides(self):
        cfg = finetune.FTConfig(lr=3e-5, max_epochs=3)
        args = argparse.Namespace(epochs=5, lr=None, train_batch_size=None, train_seed=2, train_limit=None,
                                  fp32=True)
        new = finetune.config_from_args(cfg, args)
        self.assertEqual((new.max_epochs, new.lr, new.seed, new.bf16), (5, 3e-5, 2, False))
        args.fp32 = False  # no --fp32: keep the recipe's bf16
        self.assertTrue(finetune.config_from_args(cfg, args).bf16)


if __name__ == "__main__":
    unittest.main()
