"""Setup-time checks for scratch_fasttext (run by setup.sh; exit 1 on failure). Never part of a timed run.

  1. learn:     the network can overfit 64 train texts (the loop, labels and gradients are wired right)
  2. invariant: batch size 1 and 64 give the same logits (padding is masked correctly)
  3. device:    the chosen device matches the CPU on the same weights (skipped when it is the CPU)
"""

from __future__ import annotations

import sys
from dataclasses import replace

import torch

from classifier import CONFIGS, TASK, ScratchFastTextClassifier
from task import load_split
from textnn import Vocab, encode_all, logits_for, train_loop


def main() -> int:
    per_label = max(1, 64 // len(TASK.label_ids))
    train = load_split(TASK, "train").groupby("label").head(per_label).head(64)
    cfg = replace(CONFIGS[TASK.name], max_epochs=60, patience=60, bigrams=min(CONFIGS[TASK.name].bigrams, 5_000))
    clf = ScratchFastTextClassifier(cfg, "cpu")
    clf.vocab = Vocab.build(train["text"], 1, 10_000)
    x = encode_all(clf.vocab, train["text"], cfg)
    y = [clf.index[label] for label in train["label"]]
    torch.manual_seed(0)
    clf.model = clf._new_model()
    train_loop(clf.model, x, y, x, y, cfg, "cpu", log=lambda *_: None)
    acc = (logits_for(clf.model, x, 64, "cpu").argmax(1) == torch.tensor(y)).float().mean().item()
    print(f"[{TASK.name}] learn: train accuracy on {len(x)} texts after overfitting = {acc:.3f}")
    ok = acc >= 0.98

    one = torch.cat([logits_for(clf.model, [s], 1, "cpu") for s in x])
    many = logits_for(clf.model, x, 64, "cpu")
    diff = (one - many).abs().max().item()
    print(f"[{TASK.name}] invariant: max |logit(batch 1) - logit(batch 64)| = {diff:.2e}")
    ok &= diff < 1e-4

    device = ScratchFastTextClassifier(cfg).device
    if device != "cpu":
        clf.model.to(device)
        dev = logits_for(clf.model, x, 64, device)
        clf.model.to("cpu")
        ddiff = (dev - many).abs().max().item()
        print(f"[{TASK.name}] device: max |logit({device}) - logit(cpu)| = {ddiff:.2e}")
        ok &= ddiff < 1e-3
    print(f"[{TASK.name}] checks passed" if ok else f"[{TASK.name}] CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
