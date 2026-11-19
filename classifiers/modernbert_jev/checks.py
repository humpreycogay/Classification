"""Setup-time checks for ModernBERT-JEV (run by setup.sh; exit 1 on failure). Never used in a timed run.

    python checks.py parity    # our runtime vs the release's own scores, and MPS vs CPU float32
    python checks.py overlap   # jev-bench Banking77 training rows vs our test texts -> results/overlap.json
    python checks.py probes    # three AG News probes (informational: printed, never fails)

Parity. The release ships raw logits for its 9,599 jev-bench test rows (test_predictions.json), computed under
CUDA bfloat16 autocast, so they carry bfloat16 rounding (steps of 1/64 to 1/8 at these magnitudes) and an exact
match with float32 is impossible. The gate therefore has two parts:
  1. reference: our float32 logits on PARITY_ROWS fixed test rows are within LOGIT_TOL of the shipped ones, and
     the argmax agrees except on near-ties (shipped top-2 margin below TIE_MARGIN);
  2. runtime: MPS vs CPU float32 on the same rows, |dlogit| <= 1e-3 and no argmax change.
Measured 2026-09-30 on 60 rows: max |dlogit| 0.29 vs the release (2 near-tie flips, margins 0.03 and 0.09), MPS vs
CPU 6.4e-5.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

import bench_data  # src/: pinned jev-bench revision and its row normalization
from classifier import APP_DIR, TASK, TASK_LABELS, ModernBertJevClassifier, release_dir

PARITY_ROWS = 30
LOGIT_TOL = 0.5  # bfloat16-sized (see the module docstring)
TIE_MARGIN = 0.25
RUNTIME_TOL = 1e-3
PARITY_SOURCES = ("mnli", "banking77", "arc_challenge")  # 3, 77 and 4 options


def jev_bench(patterns: list[str]) -> Path:
    return Path(snapshot_download("Praveenrajus/jev-bench", repo_type="dataset", revision=bench_data.REVISION,
                                  allow_patterns=["manifest.json", *patterns]))


def read_rows(path: Path) -> list[dict]:
    return [bench_data.normalize(json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def argmax(values: list[float]) -> int:
    return max(range(len(values)), key=values.__getitem__)


def parity() -> None:
    root = jev_bench([f"data/{s}/test.jsonl" for s in PARITY_SOURCES])
    rows = {r["id"]: r for s in PARITY_SOURCES for r in read_rows(root / f"data/{s}/test.jsonl")}
    shipped = {r["id"]: r for r in json.loads((release_dir() / "test_predictions.json").read_text())}
    ids = random.Random(0).sample(sorted(i for i in shipped if i in rows), PARITY_ROWS)
    args = ([rows[i]["context"] for i in ids], [rows[i]["choices"] for i in ids], [rows[i]["question"] for i in ids])
    gpu = ModernBertJevClassifier(TASK_LABELS, TASK.instruction)
    ours = gpu.raw_logits(*args)
    failures = []
    worst = 0.0
    for i, got in zip(ids, ours):
        ref = shipped[i]["logits"]
        gap = max(abs(a - b) for a, b in zip(got, ref))
        worst = max(worst, gap)
        top = sorted(ref, reverse=True)
        if gap > LOGIT_TOL:
            failures.append(f"{i}: max |dlogit| {gap:.3f} > {LOGIT_TOL}")
        if argmax(got) != argmax(ref) and top[0] - top[1] >= TIE_MARGIN:
            failures.append(f"{i}: argmax {argmax(got)} vs shipped {argmax(ref)} (shipped margin {top[0] - top[1]:.3f})")
    print(f"reference: {len(ids)} jev-bench test rows vs the release's bfloat16 logits: max |dlogit| {worst:.3f}")
    if gpu.device != "cpu":
        cpu = ModernBertJevClassifier(TASK_LABELS, TASK.instruction, device="cpu").raw_logits(*args)
        gap = max(abs(a - b) for g, c in zip(ours, cpu) for a, b in zip(g, c))
        flips = sum(argmax(g) != argmax(c) for g, c in zip(ours, cpu))
        print(f"runtime: {gpu.device} vs CPU float32: max |dlogit| {gap:.2e}, {flips} argmax changes")
        if gap > RUNTIME_TOL or flips:
            failures.append(f"{gpu.device} vs CPU float32: max |dlogit| {gap:.2e}, {flips} argmax changes")
    if failures:
        sys.exit("ModernBERT-JEV parity check failed:\n  " + "\n  ".join(failures))
    print("parity check passed")


def overlap() -> None:
    """Our Banking77 test texts among jev-bench's Banking77 rows the model trained on: train (weights) and validation
    (checkpoint selection and the temperature). Its test split was only evaluated on, so it is not counted."""
    import overlap as shared_overlap

    root = jev_bench([f"data/banking77/{s}.jsonl" for s in ("train", "validation")])
    seen = {s: [r["context"] for r in read_rows(root / f"data/banking77/{s}.jsonl")] for s in ("train", "validation")}
    entry = shared_overlap.check("banking77", seen, note=f"jev-bench {bench_data.REVISION[:7]}")
    shared_overlap.write(APP_DIR, {"banking77": entry})


def probes() -> None:
    from probes import AG_NEWS_PROBES

    preds = ModernBertJevClassifier(TASK_LABELS, TASK.instruction).predict_many(list(AG_NEWS_PROBES))
    for (text, want), p in zip(AG_NEWS_PROBES.items(), preds):
        print(f"  {'ok  ' if p.label == want else 'miss'} {p.label:<9} (want {want:<9}) p={p.confidence:.3f}  {text[:60]}")
    print("probes (informational: a miss is the model's own answer; parity is the gate)")


if __name__ == "__main__":
    {"parity": parity, "overlap": overlap, "probes": probes}[sys.argv[1]]()
