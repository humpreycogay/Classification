#!/usr/bin/env bash
# One-time setup: virtualenv, dependencies, and the NLI models at pinned revisions (DeBERTa-v3-large ~870 MB,
# ModernBERT-large ~0.8 GB), then probe and MPS-vs-CPU parity checks (see below).
set -euo pipefail
cd "$(dirname "$0")"
. ../../shared/env.sh  # settings from the root .env
MODEL=${NLI_MODEL:-MoritzLaurer/deberta-v3-large-zeroshot-v2.0}

python3 -m venv .venv
.venv/bin/python -m pip install -q --upgrade pip
.venv/bin/python -m pip install -q -r requirements.txt

# Pre-download weights (pinned in classifier.py MODELS). Xet transfers can stall, so use plain HTTPS.
HF_HUB_DISABLE_XET=1 BENCHMARK_TASK=ag_news PYTHONPATH=../../shared MODEL="$MODEL" .venv/bin/python - <<'PY'
import os
from huggingface_hub import snapshot_download
from classifier import MODELS
for model in {os.environ["MODEL"], *MODELS}:
    snapshot_download(model, revision=MODELS.get(model, (None,))[0],
                      allow_patterns=['*.json', '*.safetensors', '*.model', '*.txt'])
PY

# Probe check (every model: three AG News probes in the right class) and, for the models pinned to float32
# (classifier.py MODELS), a parity check: scores on the Mac GPU must match CPU float32 on 20 fixed texts
# (same label, |dp| <= 0.01). Exit 1 otherwise.
BENCHMARK_TASK=ag_news PYTHONPATH=../../shared MODEL="$MODEL" .venv/bin/python - <<'PY'
import os, sys
from classifier import MODELS, TASK_LABELS, NLIClassifier
from probes import PARITY_TEXTS, check_probes
for model in sorted({os.environ["MODEL"], *MODELS}):
    gpu = NLIClassifier(TASK_LABELS, model=model)
    check_probes(gpu.predict_many, name=f"{model} probe check")
    if gpu.dtype != "float32" or gpu.device == "cpu":
        continue
    cpu = NLIClassifier(TASK_LABELS, model=model, device="cpu", dtype="float32")
    a, b = gpu.predict_many(PARITY_TEXTS), cpu.predict_many(PARITY_TEXTS)
    flips = sum(x.label != y.label for x, y in zip(a, b))
    gap = max(abs(x.probabilities[l] - y.probabilities[l]) for x, y in zip(a, b) for l in TASK_LABELS)
    print(f"{model}: {gpu.device} vs CPU float32 on {len(a)} texts: {flips} label flips, max |dp| {gap:.2e}")
    if flips or gap > 0.01:
        sys.exit(f"{model}: {gpu.device} scores disagree with CPU float32")
PY
.venv/bin/python -c "import torch; print('Ready. MPS:', torch.backends.mps.is_available())"
