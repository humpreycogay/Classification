"""Setup-time checks for finetune_modernbert (run by setup.sh; exit 1 on failure). Never part of a timed run.

  1. backbone:  the pinned backbone loads, with the expected number of parameters
  2. learn:     fine-tuning can fit 32 train texts in 15 short epochs (loop, labels and gradients are wired right).
               8 epochs were too few for DeBERTa on 32 Banking77 labels with one text each (0.66-0.75)
  3. invariant: batch size 1 and 16 give the same float32 logits (attention masks hide the padding)
  4. device:    the chosen device matches the CPU in float32 on the same weights
"""

from __future__ import annotations

import sys
from dataclasses import replace

import torch

from classifier import CONFIGS, PARAMETERS, TASK, FineTunedEncoder
from finetune import logits_for, tokenize_all, train_loop
from task import load_split


def main() -> int:
    clf = FineTunedEncoder(device="cpu")
    torch.manual_seed(0)
    clf.model = clf._new_model(clf.base_dir)
    n = sum(p.numel() for p in clf.model.parameters())
    print(f"[{TASK.name}] backbone: {clf.base_ref}, {n:,} parameters")
    ok = PARAMETERS[0] < n < PARAMETERS[1]

    per = max(1, 32 // len(TASK.label_ids))
    train = load_split(TASK, "train").groupby("label").head(per).head(32)
    x = tokenize_all(clf.tok, train["text"], 64)
    y = [clf.index[label] for label in train["label"]]
    device = FineTunedEncoder().device
    cfg = replace(CONFIGS[TASK.name], max_epochs=15, patience=15, batch_size=8, lr=1e-4, warmup=0.1)
    train_loop(clf.model, x, y, x, y, cfg, clf.pad_id, device, log=lambda *_: None)
    acc = (logits_for(clf.model, x, 16, clf.pad_id, device).argmax(1) == torch.tensor(y)).float().mean().item()
    print(f"[{TASK.name}] learn: train accuracy on {len(x)} texts after 15 epochs = {acc:.3f}")
    ok &= acc >= 0.9

    clf.model.to("cpu")
    one = torch.cat([logits_for(clf.model, [s], 1, clf.pad_id, "cpu") for s in x])
    many = logits_for(clf.model, x, 16, clf.pad_id, "cpu")
    diff = (one - many).abs().max().item()
    print(f"[{TASK.name}] invariant: max |logit(batch 1) - logit(batch 16)| = {diff:.2e}")
    ok &= diff < 1e-3
    if device != "cpu":
        clf.model.to(device)
        ddiff = (logits_for(clf.model, x, 16, clf.pad_id, device) - many).abs().max().item()
        print(f"[{TASK.name}] device: max |logit({device}) - logit(cpu)| = {ddiff:.2e}")
        ok &= ddiff < 1e-3
    print(f"[{TASK.name}] checks passed" if ok else f"[{TASK.name}] CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
