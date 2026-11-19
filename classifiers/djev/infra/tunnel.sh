#!/usr/bin/env bash
# Forward the GPU host's loopback-only djev API to this machine's 127.0.0.1:8000.
# Usage: infra/tunnel.sh user@gpu-host [ssh options...]
# Leave it running while you use classifier.py / evaluate.py; Ctrl+C to close.
set -euo pipefail
[ $# -ge 1 ] || { echo "usage: $0 user@gpu-host [ssh options...]" >&2; exit 1; }
LOCAL_PORT=${LOCAL_PORT:-8000}
echo "Tunnel: http://127.0.0.1:$LOCAL_PORT -> $1:127.0.0.1:8000"
exec ssh -N -o ExitOnForwardFailure=yes -L "$LOCAL_PORT:127.0.0.1:8000" "$@"
