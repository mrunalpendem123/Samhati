#!/usr/bin/env bash
# ─────────────────────────────────────────────
# Chat with the distributed model.
# Run this AFTER both node servers are up.
#
# Usage:
#   bash scripts/chat.sh <node_id_A> <node_id_B>
# ─────────────────────────────────────────────
set -e
cd "$(dirname "$0")/.."

NODE_A="$1"
NODE_B="$2"

if [ -z "$NODE_A" ] || [ -z "$NODE_B" ]; then
  echo "Usage: bash scripts/chat.sh <node_id_A> <node_id_B>"
  exit 1
fi

echo ""
echo "Connected to distributed Qwen2.5-7B"
echo "  Node A (layers  0-13): $NODE_A"
echo "  Node B (layers 14-27): $NODE_B"
echo "Type your message and press Enter. Ctrl-C to quit."
echo "────────────────────────────────────────"

while true; do
  printf "\nYou: "
  read -r INPUT
  [ -z "$INPUT" ] && continue
  printf "Model: "
  ./target/release/mesh-node dist-run \
    --executor iroh \
    --peers "$NODE_A,$NODE_B" \
    --input "$INPUT" \
    --config ~/qwen25-7b/config.json --params-b 7.0 \
    --max-tokens 100 \
    2>/dev/null | sed 's/^output: //'
done
