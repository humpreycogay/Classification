#!/usr/bin/env bash
# One-time setup: virtualenv, contrastive-lm (pinned, without vLLM), and the pinned weights: Qwen/Qwen3-8B
# (~16.4 GB, BF16) and the CLM-v0.1-8B heads (CLM_v0.1-8B.pt, ~75 MB). Then the parity gate (exit 1 on failure).
set -euo pipefail
cd "$(dirname "$0")"
. ../../shared/env.sh  # settings from the root .env
CLM_COMMIT=bb42c6c5bf914fd449bed2f6ca65be80602cb1f7  # contrastive-lm 0.1.0

python3 -m venv .venv
.venv/bin/python -m pip install -q --upgrade pip
.venv/bin/python -m pip install -q -r requirements.txt
.venv/bin/python -m pip install -q --no-deps "contrastive-lm @ git+https://github.com/Contrastive-LM/CLM@$CLM_COMMIT"

# Pre-download weights (pinned in classifier.py). Xet transfers can stall, so use plain HTTPS.
HF_HUB_DISABLE_XET=1 BENCHMARK_TASK=ag_news PYTHONPATH=../../shared .venv/bin/python -c "
from classifier import encoder_dir, heads_path; print(encoder_dir()); print(heads_path())"

BENCHMARK_TASK=ag_news PYTHONPATH=../../shared .venv/bin/python checks.py
.venv/bin/python -c "import torch; print('Ready. MPS:', torch.backends.mps.is_available())"
