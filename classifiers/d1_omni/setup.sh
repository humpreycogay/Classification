#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
. ../../shared/env.sh
PYTHON=${PYTHON:-python3.12}
"$PYTHON" -m venv .venv
.venv/bin/python -m pip install -q --upgrade pip
.venv/bin/python -m pip install -q -r requirements.txt
HF_HUB_DISABLE_XET=1 .venv/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('LiquidAI/d1-omni-600M', revision='02b55d7076f15129e59ab3f94783f32c4b088674')"
BENCHMARK_TASK=ag_news PYTHONPATH=../../shared .venv/bin/python checks.py