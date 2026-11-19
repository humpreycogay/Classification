#!/usr/bin/env bash
# One-time setup: virtualenv, PyTorch + transformers, the pinned ModernBERT-base backbone (~600 MB, safetensors, into
# models/base/), and the setup checks on both tasks (checks.py). The fine-tuned models are made by the first run per
# task and saved in models/<task>/.
set -euo pipefail
cd "$(dirname "$0")"
. ../../shared/env.sh  # settings from the root .env (FINETUNE_MODERNBERT_DEVICE)
[ -n "${FINETUNE_MODERNBERT_DEVICE:-}" ] && export FINETUNE_MODERNBERT_DEVICE

python3 -m venv .venv
.venv/bin/python -m pip install -q --upgrade pip
.venv/bin/python -m pip install -q -r requirements.txt
BENCHMARK_TASK=ag_news PYTHONPATH=../../shared .venv/bin/python -c "from classifier import base_dir; print('Backbone:', base_dir())"
for task in ag_news banking77; do
  BENCHMARK_TASK=$task PYTHONPATH=../../shared .venv/bin/python checks.py
done
echo "Ready. The first run per task fine-tunes the model (Banking77: minutes; AG News: an hour or more) and saves it in models/<task>/."
