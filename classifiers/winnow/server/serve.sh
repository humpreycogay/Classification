#!/usr/bin/env bash
# Run the Winnow server on this Mac (llama.cpp, Metal) at WINNOW_URL (default http://127.0.0.1:8091). Ctrl+C to stop.
# Apple Silicon profile (one decision branch, F16 KV cache), text only, and a small context: WINNOW_CONTEXT tokens
# (default 8192, above the longest prompt of our tasks; setup's probe check measures it) instead of the profile's
# 65,536, so the KV cache takes less memory and the server starts faster.
set -euo pipefail
cd "$(dirname "$0")"
. ../../../shared/env.sh  # settings from the root .env
URL=${WINNOW_URL:-http://127.0.0.1:8091}
PORT=${URL##*:}; PORT=${PORT%%/*}
cd src
exec python3 scripts/serve.py --profile apple-silicon --text-only --model-dir ../models \
  --context "${WINNOW_CONTEXT:-8192}" --host 127.0.0.1 --port "$PORT"
