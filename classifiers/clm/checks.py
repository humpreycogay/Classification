"""Setup-time parity gate for CLM (run by setup.sh; exit 1 on failure). Never used in a timed run.

The encoder runs on the Mac GPU in BF16; the reference is the same code on CPU, also in BF16 (Qwen3-8B in float32
is ~33 GB, more than this 32 GB Mac holds). On 20 fixed texts (as states, with the question) and on the label texts:
every pooled embedding has cosine >= 0.999 with the reference, and every text ranks the labels in the same order,
except labels whose reference scores are within TIE_MARGIN of each other. Scores are exp(logit_scale) * cosine with
a scale of 100, so BF16 noise on either device moves them by up to ~0.5 (0.005 in cosine); measured 2026-09-30: min
cosine 0.9998, max |dscore| 0.48, one order change, between World and Sports at 20.93 vs 20.93 on CPU.
"""

from __future__ import annotations

import gc
import sys

import torch

from classifier import TASK, TASK_LABELS, CLMClassifier
from probes import PARITY_TEXTS

MIN_COSINE = 0.999
TIE_MARGIN = 0.5  # score units (0.005 in cosine)


def consistent(ours: torch.Tensor, ref: torch.Tensor) -> bool:
    """Every pair of labels in the same order, unless the reference scores them within TIE_MARGIN."""
    n = len(ref)
    return all((ours[i] - ours[j]) * (ref[i] - ref[j]) >= 0 or abs(ref[i] - ref[j]) < TIE_MARGIN
               for i in range(n) for j in range(i + 1, n))


def main() -> None:
    gpu = CLMClassifier(TASK_LABELS, TASK.instruction)
    if gpu.device == "cpu":
        print("parity check skipped: no GPU")
        return
    from clm.schema import state_text

    texts = [state_text(t, TASK.instruction) for t in PARITY_TEXTS] + gpu.action_texts
    a, sa = gpu.embed(texts), gpu.scores(PARITY_TEXTS)
    device = gpu.device
    del gpu
    gc.collect()
    torch.mps.empty_cache()
    cpu = CLMClassifier(TASK_LABELS, TASK.instruction, device="cpu", dtype=torch.bfloat16)
    b, sb = cpu.embed(texts), cpu.scores(PARITY_TEXTS)
    cosine = (a * b).sum(-1)
    same_rank = [consistent(x, y) for x, y in zip(sa, sb)]
    print(f"parity: {device} BF16 vs CPU BF16 on {len(texts)} texts: min cosine {cosine.min():.5f}, "
          f"label ranking consistent (ties < {TIE_MARGIN}) on {sum(same_rank)}/{len(same_rank)} texts, max |dscore| {(sa - sb).abs().max():.3f}")
    if cosine.min() < MIN_COSINE or not all(same_rank):
        sys.exit("CLM parity check failed")
    print("parity check passed")


if __name__ == "__main__":
    main()
