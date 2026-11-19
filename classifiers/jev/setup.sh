#!/usr/bin/env bash
# One-time setup: client virtualenv. Jev runs on TypeSafe's API; put jev_api_key in the repo-root .env.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$(cd ../../shared && pwd)${PYTHONPATH:+:$PYTHONPATH}"  # classifier.py imports the shared library

python3 -m venv .venv
.venv/bin/python -m pip install -q --upgrade pip
.venv/bin/python -m pip install -q -r requirements.txt

grep -q '^jev_api_key=' ../../.env 2>/dev/null || echo "Warning: jev_api_key not found in ../../.env" >&2
echo "Client ready. Run: .venv/bin/python evaluate.py"
