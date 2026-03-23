#!/usr/bin/env bash
# ─────────────────────────────────────────────
# Chat with the distributed Qwen2.5-3B.
# Run AFTER both terminals are up.
#
# Usage:  bash scripts/chat.sh <node_id_A> <node_id_B>
# ─────────────────────────────────────────────
set -e
cd "$(dirname "$0")/.."

NODE_A="$1"
NODE_B="$2"

if [ -z "$NODE_A" ] || [ -z "$NODE_B" ]; then
  echo "Usage: bash scripts/chat.sh <node_id_A> <node_id_B>"
  echo ""
  echo "  node_id_A = the id printed by Terminal 1 (run_a.sh)"
  echo "  node_id_B = the id printed by Terminal 2 (run_b.sh)"
  exit 1
fi

echo ""
echo "══════════════════════════════════════════"
echo "  Samhati — Distributed Qwen2.5-3B Chat"
echo "  Node A (layers  0-17): $NODE_A"
echo "  Node B (layers 18-35): $NODE_B"
echo "  Type a message and press Enter. Ctrl-C to quit."
echo "══════════════════════════════════════════"

while true; do
  printf "\nYou: "
  read -r INPUT
  [ -z "$INPUT" ] && continue
  printf "Model: "
  ./target/release/mesh-node dist-run \
    --executor iroh \
    --peers "$NODE_A,$NODE_B" \
    --input "$INPUT" \
    --config ~/qwen25-3b/config.json --params-b 3.0 \
    --max-tokens 200 \
    2>/dev/null | sed 's/^output: //'
done
