#!/usr/bin/env bash
# Train a domain specialist SLM.
# Usage: ./scripts/train.sh <domain> [base_model]
set -euo pipefail

DOMAIN="${1:?Usage: $0 <domain> [base_model]}"
BASE_MODEL="${2:-}"

ARGS="--domain $DOMAIN"
if [ -n "$BASE_MODEL" ]; then
    ARGS="$ARGS --base-model $BASE_MODEL"
fi

echo "Starting training for domain: $DOMAIN"
samhati-pipeline train $ARGS
