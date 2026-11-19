#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"; . ../../shared/env.sh
PYTHON=${PYTHON:-python3.12}; "$PYTHON" -m venv .venv
.venv/bin/python -m pip install -q --upgrade pip; .venv/bin/python -m pip install -q -r requirements.txt
HF_HUB_DISABLE_XET=1 .venv/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('jaredpalmer/kev-0.8b',revision='bf75a6a8848ea6960ff2ed108d9ed44c2941174f'); snapshot_download('Qwen/Qwen3.5-0.8B-Base',revision='dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68')"
PYTHONPATH=../../shared BENCHMARK_TASK=ag_news .venv/bin/python checks.py overlap
PYTHONPATH=../../shared BENCHMARK_TASK=ag_news .venv/bin/python checks.py probes