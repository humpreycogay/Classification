#!/usr/bin/env bash
# One-time setup: virtualenv, the ModernBERT-JEV code (github.com/ali-rehman-ML/modern-bert-jev, not a pip package)
# in src/ at a pinned commit, the LoRA release (ali-rehman-ML/modern-bert-jev, ~20 MB without ONNX) and
# answerdotai/ModernBERT-base (~0.6 GB), both pinned; then the parity gate and the Banking77 overlap check
# (checks.py). Exit 1 if parity fails.
set -euo pipefail
cd "$(dirname "$0")"
. ../../shared/env.sh  # settings from the root .env
REPO=https://github.com/ali-rehman-ML/modern-bert-jev
COMMIT=c74ba60a40f287085c77547078e374b6526fe645  # keep in step with classifier.py REPO_COMMIT

python3 -m venv .venv
.venv/bin/python -m pip install -q --upgrade pip
.venv/bin/python -m pip install -q -r requirements.txt

[ -d src/.git ] || git clone -q "$REPO" src
git -C src fetch -q origin "$COMMIT" 2>/dev/null || true
git -C src checkout -q "$COMMIT"

# Weights. Xet transfers can stall, so use plain HTTPS. The backbone goes where the repo's code expects it
# (src/models/ModernBERT-base, revision pinned in src/choice_model.py); the ONNX graph (~600 MB) only for
# MODERNBERT_JEV_BACKEND=onnx.
export HF_HUB_DISABLE_XET=1
(cd src && ../.venv/bin/python download_model.py answerdotai/ModernBERT-base)
ONNX=$([ "${MODERNBERT_JEV_BACKEND:-torch}" = onnx ] && echo 1 || echo 0)
BENCHMARK_TASK=ag_news PYTHONPATH=../../shared .venv/bin/python -c "from classifier import release_dir; print(release_dir(onnx=$ONNX == 1))"

# Parity gate (exit 1 on failure), the training-overlap check (results/overlap.json; the runner skips banking77
# when the model saw our test texts), and the probes (informational).
for check in parity overlap probes; do
  BENCHMARK_TASK=ag_news PYTHONPATH=../../shared:src .venv/bin/python checks.py "$check"
done
.venv/bin/python -c "import torch; print('Ready. MPS:', torch.backends.mps.is_available())"
