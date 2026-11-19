#!/usr/bin/env bash
# One-time setup: virtualenv, dependencies, and the shared AG News train split (~29 MB, in data/).
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$(cd ../../shared && pwd)${PYTHONPATH:+:$PYTHONPATH}"  # classifier.py imports the shared library

python3 -m venv .venv
.venv/bin/python -m pip install -q --upgrade pip
.venv/bin/python -m pip install -q -r requirements.txt
.venv/bin/python -c "from classifier import load_train_set; print('Ready. Train articles:', len(load_train_set()))"
