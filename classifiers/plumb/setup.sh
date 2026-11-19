#!/usr/bin/env bash
# One-time setup: Python 3.12 virtualenv, jevk5 (pinned, without CUDA extras) + MLX, and the official Plumb-4B
# weights at a pinned commit (~8.4 GB, BF16; not the community MLX 8-bit conversion). Then the probe check and the
# parity gate against jevk5's own PyTorch path on CPU (exit 1 on failure).
set -euo pipefail
cd "$(dirname "$0")"
. ../../shared/env.sh  # settings from the root .env

PYTHON=${PYTHON:-python3.12}
"$PYTHON" -m venv .venv
.venv/bin/python -m pip install -q --upgrade pip
.venv/bin/python -m pip install -q -r requirements.txt

# Pre-download weights (pinned in classifier.py). Xet transfers can stall, so use plain HTTPS.
HF_HUB_DISABLE_XET=1 BENCHMARK_TASK=ag_news PYTHONPATH=../../shared .venv/bin/python -c "from classifier import model_dir; print(model_dir())"

for check in probes parity; do
  BENCHMARK_TASK=ag_news PYTHONPATH=../../shared .venv/bin/python checks.py "$check"
done
.venv/bin/python -c "import mlx.core as mx; print('Ready. Metal:', mx.metal.is_available())"
