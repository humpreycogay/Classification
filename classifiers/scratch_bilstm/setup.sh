#!/usr/bin/env bash
# One-time setup: virtualenv, PyTorch, and the setup checks on both tasks (checks.py). No model is downloaded:
# the network is trained from scratch on the first run (or by the demo) and saved in models/<task>/.
set -euo pipefail
cd "$(dirname "$0")"
. ../../shared/env.sh  # settings from the root .env (SCRATCH_BILSTM_DEVICE)
[ -n "${SCRATCH_BILSTM_DEVICE:-}" ] && export SCRATCH_BILSTM_DEVICE

python3 -m venv .venv
.venv/bin/python -m pip install -q --upgrade pip
.venv/bin/python -m pip install -q -r requirements.txt
for task in ag_news banking77; do
  BENCHMARK_TASK=$task PYTHONPATH=../../shared .venv/bin/python checks.py
done
echo "Ready. The first run per task trains the model (AG News: minutes) and saves it in models/<task>/."
