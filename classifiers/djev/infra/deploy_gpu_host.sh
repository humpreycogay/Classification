#!/usr/bin/env bash
# Deploy the djev-dev server on a Linux GPU host. Run this ON the GPU host, not locally.
#
# Requirements (from djev-dev docs/runtime.md):
#   - 1x NVIDIA B200 (BF16 weights alone are ~52 GB), driver compatible with CUDA 13
#   - Docker + NVIDIA Container Toolkit
#   - ~16 vCPUs, 128 GiB RAM, 100+ GiB fast disk for weights and caches
#
# The API is published only on the host's loopback (127.0.0.1:8000). Reach it from your
# machine with infra/tunnel.sh. Optional: export DJEV_API_KEY before running to require
# a Bearer key; export HF_TOKEN if your Hugging Face access needs one.
set -euo pipefail

REPO=https://github.com/Davipar/djev-dev.git
COMMIT=3ce907e6835212f27ee82b4cee9039198c4abe35  # main as of 2026-09-19; bump deliberately
SRC_DIR=${SRC_DIR:-$HOME/djev-dev}
CONTAINER=djev-model

command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found: this host has no NVIDIA driver." >&2; exit 1; }
command -v docker >/dev/null || { echo "docker not found." >&2; exit 1; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

if [ ! -d "$SRC_DIR/.git" ]; then git clone "$REPO" "$SRC_DIR"; fi
git -C "$SRC_DIR" fetch --quiet origin
git -C "$SRC_DIR" checkout --quiet "$COMMIT"

echo "Building image (pins vLLM sources and applies djev runtime patches)..."
docker build -f "$SRC_DIR/runtime/Dockerfile" -t djev-local "$SRC_DIR"

docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
ENV_ARGS=()
[ -n "${HF_TOKEN:-}" ] && ENV_ARGS+=(-e HF_TOKEN)
[ -n "${DJEV_API_KEY:-}" ] && ENV_ARGS+=(-e DJEV_API_KEY)
docker run -d --name "$CONTAINER" \
  --gpus device=0 --shm-size=16g \
  -p 127.0.0.1:8000:8000 \
  -v djev-model-cache:/cache \
  "${ENV_ARGS[@]}" \
  djev-local

echo "Waiting for vLLM (first start downloads ~52 GB and compiles kernels; can take a while)..."
until docker exec "$CONTAINER" python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8001/health')" 2>/dev/null; do
  docker inspect -f '{{.State.Running}}' "$CONTAINER" | grep -q true || { docker logs --tail 50 "$CONTAINER"; exit 1; }
  sleep 15
done

echo "Starting djev API..."
docker exec -d "$CONTAINER" python3 -m djev --host 0.0.0.0 --port 8000
AUTH=()
[ -n "${DJEV_API_KEY:-}" ] && AUTH=(-H "Authorization: Bearer $DJEV_API_KEY")
until curl -sf "${AUTH[@]}" http://127.0.0.1:8000/ready >/dev/null; do sleep 5; done
echo "djev is ready on 127.0.0.1:8000. Stop with: docker stop $CONTAINER && docker rm $CONTAINER"
