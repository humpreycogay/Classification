#!/usr/bin/env bash
# One-time server setup on Apple silicon: OpenJev at a pinned commit with its MLX backend,
# plus the 4-bit DiffusionGemma weights (~16.6 GB download, ~16 GB memory to run).
set -euo pipefail
cd "$(dirname "$0")"
. ../../../shared/env.sh  # settings from the root .env

REPO=https://github.com/razorback16/openjev.git
COMMIT=6250116feeb96b924d9dfe09c61131c9eb5d961e  # main as of 2026-09-24; bump deliberately
MLX_MODEL=${OPENJEV_MLX_MODEL:-mlx-community/diffusiongemma-26B-A4B-it-4bit}
PYTHON=${PYTHON:-python3.12}

if [ ! -d src/.git ]; then git clone --quiet "$REPO" src; fi
git -C src fetch --quiet origin
git -C src checkout --quiet "$COMMIT"

"$PYTHON" -m venv .venv
.venv/bin/python -m pip install -q --upgrade pip
.venv/bin/python -m pip install -q -e 'src[mlx]'

# Pre-download weights so the first start doesn't block. Xet transfers can stall, so use plain HTTPS.
HF_HUB_DISABLE_XET=1 .venv/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('$MLX_MODEL')"
echo "Server ready. Start it with: server/serve.sh"
