#!/usr/bin/env bash
# One-time setup: virtualenv, dependencies, and the three Laya checkpoints at a pinned revision
# (English ~843 MB, multilingual ~0.7 GB, typed-decisions ~0.9 GB), then a probe check of each.
set -euo pipefail
cd "$(dirname "$0")"

python3 -m venv .venv
.venv/bin/python -m pip install -q --upgrade pip
.venv/bin/python -m pip install -q -r requirements.txt

# Pre-download weights so the first run doesn't block. Xet transfers can stall, so use plain HTTPS.
HF_HUB_DISABLE_XET=1 .venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
files = ["model.safetensors", "rl_agent_config.json", "encoder/*", "tokenizer/*"]
snapshot_download("convaiinnovations/laya", revision="55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851",  # keep in step with classifier.py
                  allow_patterns=[f"{sub}{f}" for sub in ("", "multilingual/", "typed-decisions/") for f in files])
PY

# Probe check: every checkpoint must put three fixed AG News probes in the expected class (exit 1 otherwise).
BENCHMARK_TASK=ag_news PYTHONPATH=../../shared .venv/bin/python - <<'PY'
from classifier import MODELS, TASK, TASK_LABELS, LayaClassifier
from probes import check_probes
for model in MODELS:
    clf = LayaClassifier(TASK_LABELS, TASK.instruction, model=model)
    check_probes(clf.predict_many, name=f"Laya {model} probe check")
PY

.venv/bin/python -c "import torch; print('Ready. MPS:', torch.backends.mps.is_available(), '| CUDA:', torch.cuda.is_available())"
