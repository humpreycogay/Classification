#!/usr/bin/env bash
# One-time setup: Python 3.12 virtualenv, SemIf + MLX, and the pinned Qwen3.5-4B checkpoint (~9 GB).
# Python 3.12 because SemIf pins numpy==2.2.6, which has no Python 3.14 wheels.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$(cd ../../shared && pwd)${PYTHONPATH:+:$PYTHONPATH}"  # classifier.py imports the shared library
. ../../shared/env.sh  # settings from the root .env

PYTHON=${PYTHON:-python3.12}
"$PYTHON" -m venv .venv
.venv/bin/python -m pip install -q --upgrade pip
.venv/bin/python -m pip install -q -r requirements.txt

# Pre-download weights so the first run doesn't block. Xet transfers can stall, so use plain HTTPS.
HF_HUB_DISABLE_XET=1 .venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen3.5-4B", revision="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
                  allow_patterns=["*.json", "model*.safetensors", "*.jinja", "*.txt", "*.model"])
PY

.venv/bin/python -c "import mlx.core as mx; print('Ready. Metal:', mx.metal.is_available())"
