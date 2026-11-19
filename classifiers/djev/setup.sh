#!/usr/bin/env bash
# Local client setup (this machine). The model itself runs on a GPU host: see infra/.
set -euo pipefail
cd "$(dirname "$0")"

python3 -m venv .venv
.venv/bin/python -m pip install -q --upgrade pip
.venv/bin/python -m pip install -q -r requirements.txt
echo "Client ready. Next: deploy the server with infra/deploy_gpu_host.sh, then open infra/tunnel.sh."
