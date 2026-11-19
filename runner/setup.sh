#!/usr/bin/env bash
# One-time setup: the runner's own virtualenv (it only reads results; each app keeps its own .venv).
set -euo pipefail
cd "$(dirname "$0")"

python3 -m venv .venv
.venv/bin/python -m pip install -q --upgrade pip
.venv/bin/python -m pip install -q -r requirements.txt
echo "Runner ready. Run: .venv/bin/python run_all.py   (or --report-only to rebuild the report)"
