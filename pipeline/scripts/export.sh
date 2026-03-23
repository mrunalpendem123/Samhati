#!/usr/bin/env bash
# Export a trained model to GGUF and optionally push to HuggingFace.
# Usage: ./scripts/export.sh <model_path> <domain> [--push]
set -euo pipefail

MODEL_PATH="${1:?Usage: $0 <model_path> <domain> [--push]}"
DOMAIN="${2:?Usage: $0 <model_path> <domain> [--push]}"
PUSH_FLAG="${3:-}"

ARGS="--model-path $MODEL_PATH --domain $DOMAIN"
if [ "$PUSH_FLAG" = "--push" ]; then
    ARGS="$ARGS --push"
fi

echo "Exporting model for domain: $DOMAIN"
samhati-pipeline export $ARGS
