#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
. ../../shared/env.sh
PYTHON=${PYTHON:-python3.12}
"$PYTHON" -m venv .venv
.venv/bin/python -m pip install -q --upgrade pip
.venv/bin/python -m pip install -q -r requirements.txt
HF_HUB_DISABLE_XET=1 .venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
for repo,rev in [('jerichosiahaya/Naluri-279M-encoder','ba8c4c17c1af0790bc00ad9fbf407eb8ea539821'),('jerichosiahaya/Naluri-308M-encoder','f5d1a9feb1ab8ef3a5fd3a287258182110559416')]: snapshot_download(repo,revision=rev)
PY
BENCHMARK_TASK=ag_news PYTHONPATH=../../shared .venv/bin/python checks.py