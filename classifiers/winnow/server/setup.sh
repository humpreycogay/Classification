#!/usr/bin/env bash
# One-time server setup on Apple silicon: winnow-inference (a patched llama.cpp server) at a pinned commit, built
# with Metal, and only the Winnow-12B Q8_0 GGUF (12.7 GB) at a pinned revision: no BF16 (24 GB, too large for 32 GB
# of memory) and no vision projector (text only). Winnow's own setup.py then checks the prerequisites and the
# model's sha256 against its release manifest.
set -euo pipefail
cd "$(dirname "$0")"
. ../../../shared/env.sh  # settings from the root .env

REPO=https://github.com/EldanRing/winnow-inference.git
COMMIT=77d14580c6732ca2f3745750c1dc1fd446d8bcee  # main as of 2026-09-24; bump deliberately
MODEL_REPO=EldanRing/Winnow-12B
MODEL_REVISION=b2b14213dfa252e6d6b543c8b334762e51772d29  # the revision in src/manifests/models.json
MODEL_FILE=gguf/Winnow-12B-Q8_0.gguf

if [ ! -d src/.git ]; then git clone --quiet "$REPO" src; fi
git -C src fetch --quiet origin
git -C src checkout --quiet "$COMMIT"

# The weights go to the Hugging Face cache (plain HTTPS: Xet transfers can stall); models/ links to them in the
# layout setup.py and serve.py expect (models/gguf/<file>).
python3 -m venv .venv
.venv/bin/python -m pip install -q --upgrade pip huggingface_hub
GGUF=$(HF_HUB_DISABLE_XET=1 .venv/bin/python -c "from huggingface_hub import hf_hub_download
print(hf_hub_download('$MODEL_REPO', '$MODEL_FILE', revision='$MODEL_REVISION'))")
mkdir -p models/gguf
ln -sf "$GGUF" "models/$MODEL_FILE"

# Prerequisites, checksum, Metal build (src/.build/bin/winnow-server).
(cd src && python3 scripts/setup.py --profile apple-silicon --text-only --model-dir ../models)
echo "Server ready. Start it with: server/serve.sh"
