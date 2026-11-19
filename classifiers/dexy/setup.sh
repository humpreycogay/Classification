#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"; . ../../shared/env.sh
if [[ "$(uname -s)" == Darwin && "${DEXY_ALLOW_EXPERIMENTAL:-0}" != 1 ]]; then
  echo "Dexy is a 17.9 GB CUDA-oriented BF16 release and is unsafe by default on a 32 GB Mac." >&2
  echo "Set DEXY_ALLOW_EXPERIMENTAL=1 to opt in to the unverified MPS path." >&2; exit 2
fi
PYTHON=${PYTHON:-python3.12}; "$PYTHON" -m venv .venv
.venv/bin/python -m pip install -q --upgrade pip; .venv/bin/python -m pip install -q -r requirements.txt
HF_HUB_DISABLE_XET=1 .venv/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('Patrick-18/Dexy-Decision-9B',revision='afee4baa7ef921221a0e74f695a150edd6887ce2')"
BENCHMARK_TASK=ag_news PYTHONPATH=../../shared .venv/bin/python checks.py