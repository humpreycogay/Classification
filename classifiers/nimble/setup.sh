#!/usr/bin/env bash
# One-time setup: Python 3.12 virtualenv, Nimble at a pinned commit, and Bespoke-Nimble-9B.
# The release is a LoRA adapter on Qwen/Qwen3.5-9B: this downloads the base (~18 GB) and the adapter,
# then merges them once on the CPU into models/<revision>/ (~18 GB more; needs plenty of free RAM).
# The MLX scorer cannot load an adapter or quantized weights, so the merged BF16 folder is what runs.
set -euo pipefail
cd "$(dirname "$0")"
. ../../shared/env.sh  # settings from the root .env

REPO=https://github.com/bespokelabsai/nimble.git
COMMIT=62076b4f2d365b5879dafcf7f6dd072a1fe76df7  # main as of 2026-09-24; bump deliberately
MODEL=bespokelabs/Bespoke-Nimble-9B
REVISION=bd792f44ec8e265be861bfcdf4e05967ffe0e858  # the 2026-09-24 release (8,192 tokens, 255 choices); bump deliberately
PYTHON=${PYTHON:-python3.12}

if [ ! -d src/.git ]; then git clone --quiet "$REPO" src; fi
git -C src fetch --quiet origin
git -C src checkout --quiet "$COMMIT"

"$PYTHON" -m venv .venv
.venv/bin/python -m pip install -q --upgrade pip
.venv/bin/python -m pip install -q -r requirements.txt

# Download and merge (skipped when models/<revision>/READY.json exists). Xet transfers can stall, so use plain HTTPS.
HF_HUB_DISABLE_XET=1 PYTHONPATH=src MODEL="$MODEL" REVISION="$REVISION" .venv/bin/python - <<'PY'
import hashlib, json, os
from pathlib import Path

from huggingface_hub import snapshot_download

model, revision = os.environ["MODEL"], os.environ["REVISION"]
out = Path("models") / revision
if (out / "READY.json").is_file():
    print("Reusing merged model:", out)
    raise SystemExit
snapshot = Path(snapshot_download(model, revision=revision))
contract = json.loads((snapshot / "schema_config.json").read_text())

import torch
from peft import PeftModel
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

from nimble.training.candidate_schema import validate_contract

tokenizer = AutoTokenizer.from_pretrained(snapshot)
validate_contract(contract, tokenizer)  # the checked-out prompt code must match the release's
snapshot_download(contract["model"], revision=contract["revision"])
print(f"Merging {model}@{revision[:7]} into {contract['model']}@{contract['revision'][:7]} on the CPU...", flush=True)
base = Qwen3_5ForConditionalGeneration.from_pretrained(contract["model"], revision=contract["revision"],
                                                       dtype=torch.bfloat16, device_map="cpu")
merged = PeftModel.from_pretrained(base, snapshot).merge_and_unload(safe_merge=True)
merged.save_pretrained(out)
tokenizer.save_pretrained(out)
(out / "schema_config.json").write_text(json.dumps(contract, indent=2))
# The adapter hash lets Nimble pick the release's temperature for the merged folder (calibration.py).
(out / "READY.json").write_text(json.dumps({
    "model": model, "revision": revision, "base_model": contract["model"], "base_revision": contract["revision"],
    "adapter_sha256": hashlib.sha256((snapshot / "adapter_model.safetensors").read_bytes()).hexdigest(),
}, indent=2))
print("Merged:", out)
PY

.venv/bin/python -c "import mlx.core as mx; print('Ready. Metal:', mx.metal.is_available())"
