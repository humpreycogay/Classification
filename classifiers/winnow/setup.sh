#!/usr/bin/env bash
# Client setup. The model runs in the separate Winnow server: see server/setup.sh and server/serve.sh. Then the
# probe check: starts the server if needed, and checks three fixed probes, that probabilities sum to 1, and that
# the longest prompt of each task fits the server's context (exit 1 otherwise).
set -euo pipefail
cd "$(dirname "$0")"
python3 -m venv .venv
.venv/bin/python -m pip install -q --upgrade pip
.venv/bin/python -m pip install -q -r requirements.txt
[ -x server/src/.build/bin/winnow-server ] || server/setup.sh
BENCHMARK_TASK=ag_news PYTHONPATH=../../shared .venv/bin/python checks.py
echo "Client ready."
