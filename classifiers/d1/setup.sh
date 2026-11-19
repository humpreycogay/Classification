#!/usr/bin/env bash
# One-time setup for LiquidAI/d1-3B (~6.25 GB BF16). The Hub license must be accepted and a token may be required.
set -euo pipefail
cd "$(dirname "$0")"
. ../../shared/env.sh

PYTHON=${PYTHON:-python3.12}
"$PYTHON" -m venv .venv
.venv/bin/python -m pip install -q --upgrade pip
.venv/bin/python -m pip install -q -r requirements.txt

# Loading performs the pinned download and validates the remote-code import; probes fail loudly on a bad runtime.
BENCHMARK_TASK=ag_news PYTHONPATH=../../shared .venv/bin/python checks.py
.venv/bin/python -c "import torch; print('Ready. MPS:', torch.backends.mps.is_available())"