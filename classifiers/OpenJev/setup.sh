#!/usr/bin/env bash
# Client setup. The model runs in the separate OpenJev server: see server/setup.sh and server/serve.sh.
set -euo pipefail
cd "$(dirname "$0")"
python3 -m venv .venv
.venv/bin/python -m pip install -q --upgrade pip
.venv/bin/python -m pip install -q -r requirements.txt
echo "Client ready. Start the server with server/serve.sh, then run evaluate.py."
