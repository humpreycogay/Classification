#!/usr/bin/env bash
# Run the OpenJev server on this Mac (MLX backend) at 127.0.0.1:8080. Ctrl+C to stop.
set -euo pipefail
cd "$(dirname "$0")"
. ../../../shared/env.sh  # settings from the root .env
# The server reads its settings (OPENJEV_*: port, API key, ...) from the environment.
for var in $(compgen -v OPENJEV_); do export "$var"; done
export OPENJEV_BACKEND=mlx
export OPENJEV_MLX_MODEL=${OPENJEV_MLX_MODEL:-mlx-community/diffusiongemma-26B-A4B-it-4bit}
export HF_HUB_DISABLE_XET=1
exec .venv/bin/python -m openjev
