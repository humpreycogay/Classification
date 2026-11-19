#!/usr/bin/env bash
# One-time setup: virtualenv, dependencies, and the Verdict model (~600 MB) and its calibrator.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$(cd ../../shared && pwd)${PYTHONPATH:+:$PYTHONPATH}"  # classifier.py imports the shared library
. ../../shared/env.sh  # settings from the root .env
MODEL=${VERDICT_MODEL:-heman10x/rlcd-modernbert-151m}

python3 -m venv .venv
.venv/bin/python -m pip install -q --upgrade pip
.venv/bin/python -m pip install -q -r requirements.txt

# Pre-download weights. Xet transfers can stall, so use plain HTTPS.
HF_HUB_DISABLE_XET=1 .venv/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('$MODEL', allow_patterns=['*.json', 'model.safetensors'])"
.venv/bin/python -c "import torch; print('Ready. MPS:', torch.backends.mps.is_available())"
