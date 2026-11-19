#!/usr/bin/env bash
# One-time setup: client virtualenv and the model(s) in the local Ollama install.
# Usage: ./setup.sh [MODEL ...]   (default: the models set in the root .env: LLMQWEN_MODEL, LLM_MODEL)
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$(cd ../../shared && pwd)${PYTHONPATH:+:$PYTHONPATH}"  # classifier.py imports the shared library
. ../../shared/env.sh  # settings from the root .env
if [ $# -gt 0 ]; then MODELS=("$@")
else MODELS=("${LLMQWEN_MODEL:-qwen3.8:27b-q4_K_M}" ${LLM_MODEL:+"$LLM_MODEL"})
fi

command -v ollama >/dev/null || { echo "Ollama not found. Install it from https://ollama.com/download" >&2; exit 1; }
for MODEL in "${MODELS[@]}"; do
  ollama list | awk 'NR>1 {print $1}' | grep -qx "$MODEL" || ollama pull "$MODEL"
done

python3 -m venv .venv
.venv/bin/python -m pip install -q --upgrade pip
.venv/bin/python -m pip install -q -r requirements.txt
echo "Ready: ${MODELS[*]} via Ollama. Run: .venv/bin/python evaluate.py --model ${MODELS[0]}"
