#!/usr/bin/env bash
# One-time setup: Python 3.12 virtualenv, strands-decider (pinned git commit) with MLX, the
# strands-decider-2B-hobson-v19 weights (~92 MB) and their base model Qwen3.5-2B-Base (~4.5 GB), both at pinned
# commits. Then the manifest check, the probe check, the parity gate (MLX bfloat16 vs CPU float32: exit 1 on any
# argmax change) and the training-overlap check (results/overlap.json; the model trained on the AG News and
# Banking77 train splits).
set -euo pipefail
cd "$(dirname "$0")"
. ../../shared/env.sh  # settings from the root .env

PYTHON=${PYTHON:-python3.12}
"$PYTHON" -m venv .venv
.venv/bin/python -m pip install -q --upgrade pip
.venv/bin/python -m pip install -q -r requirements.txt

# Pre-download weights and base model (pinned in classifier.py; base_dir fails if the base model's main moved off
# its pin). Xet transfers can stall, so use plain HTTPS.
HF_HUB_DISABLE_XET=1 BENCHMARK_TASK=ag_news PYTHONPATH=../../shared .venv/bin/python -c "from classifier import base_dir, model_dir; print(model_dir()); print(base_dir())"

for check in manifest probes parity overlap; do
  BENCHMARK_TASK=ag_news PYTHONPATH=../../shared .venv/bin/python checks.py "$check"
done
.venv/bin/python -c "import mlx.core as mx; print('Ready. Metal:', mx.metal.is_available())"
