#!/usr/bin/env bash
# One-time setup: client virtualenv, the bge-m3 embedding model in the local Ollama install,
# and the shared AG News train split (~29 MB, in data/).
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$(cd ../../shared && pwd)${PYTHONPATH:+:$PYTHONPATH}"  # classifier.py imports the shared library
. ../../shared/env.sh  # settings from the root .env
MODEL=${EMBEDSUP_MODEL:-bge-m3:567m}

command -v ollama >/dev/null || { echo "Ollama not found. Install it from https://ollama.com/download" >&2; exit 1; }
ollama list | awk 'NR>1 {print $1}' | grep -qx "$MODEL" || ollama pull "$MODEL"

python3 -m venv .venv
.venv/bin/python -m pip install -q --upgrade pip
.venv/bin/python -m pip install -q -r requirements.txt
.venv/bin/python -c "from classifier import load_train_sample; load_train_sample(1, 0)"
echo "Ready: $MODEL via Ollama. Run: .venv/bin/python evaluate.py [--method centroid|logreg]"
