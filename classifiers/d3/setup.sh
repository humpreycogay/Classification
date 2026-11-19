#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"; . ../../shared/env.sh
PYTHON=${PYTHON:-python3.12}; "$PYTHON" -m venv .venv
.venv/bin/python -m pip install -q --upgrade pip; .venv/bin/python -m pip install -q -r requirements.txt
HF_HUB_DISABLE_XET=1 .venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
for repo,rev in [('vllm-sr/d3-edge','bee5e872f75b9c5d1405767bf244a6948b95c0af'),('vllm-sr/d3-lite','6c01fba14e74f63b70e1ef442b0b6d1b934339d5'),('vllm-sr/d3-nano','8198c9f763b046605b2f4827b4999971fa2c92c4')]: snapshot_download(repo,revision=rev)
PY
BENCHMARK_TASK=ag_news PYTHONPATH=../../shared .venv/bin/python checks.py