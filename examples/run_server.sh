#!/usr/bin/env bash
# Start the server with a small model. Adjust --model / --port / --attn as needed.
set -euo pipefail

MODEL="${MODEL:-gpt2}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
LOGPROB_LEVEL="${LOGPROB_LEVEL:-token}"
LOGPROB_DTYPE="${LOGPROB_DTYPE:-float32}"

logprob-engine serve \
    --model "$MODEL" \
    --host "$HOST" \
    --port "$PORT" \
    --dtype bfloat16 \
    --logprob-level "$LOGPROB_LEVEL" \
    --logprob-dtype "$LOGPROB_DTYPE"
