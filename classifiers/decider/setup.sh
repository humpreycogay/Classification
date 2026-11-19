#!/usr/bin/env bash
# One-time setup: virtualenv, decider-ai[metal] (pinned), and Decider-4B v3 at its release commit (~8.4 GB, BF16).
# Then the probe check, the parity gate (MPS float16 vs CPU float32: exit 1 on any argmax change) and the
# training-overlap check (results/overlap.json; Decider trained on the AG News and Banking77 train splits).
set -euo pipefail
cd "$(dirname "$0")"
. ../../shared/env.sh  # settings from the root .env

python3 -m venv .venv
.venv/bin/python -m pip install -q --upgrade pip
.venv/bin/python -m pip install -q -r requirements.txt

# Pre-download weights (pinned in classifier.py). Xet transfers can stall, so use plain HTTPS.
HF_HUB_DISABLE_XET=1 BENCHMARK_TASK=ag_news PYTHONPATH=../../shared .venv/bin/python -c "from classifier import model_dir; print(model_dir())"

for check in probes parity overlap; do
  BENCHMARK_TASK=ag_news PYTHONPATH=../../shared .venv/bin/python checks.py "$check"
done
.venv/bin/python -c "import torch; print('Ready. MPS:', torch.backends.mps.is_available())"
