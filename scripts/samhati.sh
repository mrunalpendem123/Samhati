#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# samhati.sh — One-command distributed LLM inference
#
# Usage:
#   bash scripts/samhati.sh              # Host: download layers 0-17, start chat
#   bash scripts/samhati.sh <join-code>  # Join: download layers 18-35, serve
#
# The join-code is the node_id printed by the host. Share it once (copy-paste).
# ─────────────────────────────────────────────────────────────────────────────
set -e
cd "$(dirname "$0")/.."

MODEL=Qwen/Qwen2.5-3B
TOTAL_LAYERS=36
HALF=$((TOTAL_LAYERS / 2))   # 18
WEIGHTS=~/qwen25-3b

# Fixed gossip topic for this model network.
TOPIC="73616d6861746976317177656e322e352d33422d676f73736970746f70696330"

ARCH_ARGS="--hidden 2048 --intermediate 11008 --heads 16 --kv-heads 2 \
  --vocab 151936 --rope-theta 1000000 --rms-eps 0.000001"

JOIN_CODE="${1:-}"

# ── 1. Build (always, so code changes take effect) ──────────────────────────
echo "Building Samhati..."
cargo build --release --features burn 2>&1 | grep -E "^(error|warning\[|Compiling mesh-node|Finished)" || true

if [ -z "$JOIN_CODE" ]; then
  # ════════════════════════════════════════════════════════════════════════════
  #  HOST MODE  (Person A — serves layers 0-17, then starts interactive chat)
  # ════════════════════════════════════════════════════════════════════════════
  echo ""
  echo "┌─────────────────────────────────────────┐"
  echo "│  Samhati  ·  Distributed LLM  ·  Host   │"
  echo "│  You will serve layers 0-17              │"
  echo "└─────────────────────────────────────────┘"
  echo ""

  DL_OUT=$(python3 scripts/download_shard.py \
    --repo "$MODEL" \
    --layer-start 0 --layer-end $HALF --total-layers $TOTAL_LAYERS \
    --out "$WEIGHTS")
  echo "$DL_OUT"

  WEIGHT_FILES=$(echo "$DL_OUT" | grep '^SAMHATI_FILES=' | cut -d= -f2-)
  if [ -z "$WEIGHT_FILES" ]; then
    echo "Error: could not determine weight files from download"
    exit 1
  fi

  SERVE_LOG=$(mktemp -t samhati-serve)
  PEER_FILE=$(mktemp -t samhati-peers)
  trap "kill \$SERVE_PID 2>/dev/null; rm -f '$SERVE_LOG' '$PEER_FILE'" EXIT

  eval ./target/release/mesh-node serve \
    --model-path '"$WEIGHT_FILES"' \
    --layer-start 0 --layer-end $HALF --total-layers $TOTAL_LAYERS \
    $ARCH_ARGS \
    --topic "$TOPIC" \
    --peer-file "$PEER_FILE" \
    > "$SERVE_LOG" 2>&1 &
  SERVE_PID=$!

  MY_NODE_ID=""
  for i in $(seq 1 30); do
    MY_NODE_ID=$(grep -m1 "^node_id:" "$SERVE_LOG" 2>/dev/null | awk '{print $2}')
    [ -n "$MY_NODE_ID" ] && break
    sleep 1
  done

  if [ -z "$MY_NODE_ID" ]; then
    echo "Error: server failed to start. Output:"
    cat "$SERVE_LOG"
    exit 1
  fi

  echo ""
  echo "┌──────────────────────────────────────────────────────────────────────┐"
  echo "│  Your join code — share this with your friend:                       │"
  echo "│                                                                      │"
  printf "│  %s  │\n" "$MY_NODE_ID"
  echo "│                                                                      │"
  echo "│  Friend runs:                                                        │"
  printf "│    bash scripts/samhati.sh %s...  │\n" "${MY_NODE_ID:0:20}"
  echo "│    (paste the full code above)                                       │"
  echo "└──────────────────────────────────────────────────────────────────────┘"
  echo ""
  echo "Connecting to the Samhati relay network..."
  echo "Waiting for your friend to join... (0 peers)"

  PEER_NODE_ID=""
  for i in $(seq 1 600); do
    PEER_NODE_ID=$(head -1 "$PEER_FILE" 2>/dev/null | tr -d '[:space:]')
    if [ -n "$PEER_NODE_ID" ]; then
      echo ""
      echo "  Friend connected!  (1 peer in network)"
      break
    fi
    if [ $((i % 10)) -eq 0 ]; then
      printf "."
    fi
    sleep 1
  done

  if [ -z "$PEER_NODE_ID" ]; then
    echo ""
    echo "No friend connected after 10 minutes. Make sure they ran:"
    echo "  bash scripts/samhati.sh $MY_NODE_ID"
    exit 1
  fi

  echo ""
  echo "════════════════════════════════════════════════════"
  echo "  Distributed Qwen2.5-3B is ready!"
  echo "  Layers  0-17 : you"
  echo "  Layers 18-35 : your friend"
  echo "  Type a message and press Enter.  Ctrl-C to quit."
  echo "════════════════════════════════════════════════════"

  while true; do
    printf "\nYou: "
    read -r INPUT
    [ -z "$INPUT" ] && continue
    printf "Model: "
    ./target/release/mesh-node dist-run \
      --executor iroh \
      --peers "$MY_NODE_ID,$PEER_NODE_ID" \
      --input "$INPUT" \
      --config "$WEIGHTS/config.json" \
      --params-b 3.0 \
      --total-layers $TOTAL_LAYERS \
      --max-tokens 200 \
      2>/dev/null | sed 's/^output: //' || echo "(inference error — is your friend still connected?)"
  done

else
  # ════════════════════════════════════════════════════════════════════════════
  #  JOIN MODE  (Person B — serves layers 18-35, bootstraps off the host)
  # ════════════════════════════════════════════════════════════════════════════
  echo ""
  echo "┌─────────────────────────────────────────┐"
  echo "│  Samhati  ·  Distributed LLM  ·  Join   │"
  echo "│  You will serve layers 18-35             │"
  echo "└─────────────────────────────────────────┘"
  echo ""
  echo "Connecting via:  ${JOIN_CODE:0:24}..."
  echo ""

  DL_OUT=$(python3 scripts/download_shard.py \
    --repo "$MODEL" \
    --layer-start $HALF --layer-end $TOTAL_LAYERS --total-layers $TOTAL_LAYERS \
    --out "$WEIGHTS")
  echo "$DL_OUT"

  WEIGHT_FILES=$(echo "$DL_OUT" | grep '^SAMHATI_FILES=' | cut -d= -f2-)
  if [ -z "$WEIGHT_FILES" ]; then
    echo "Error: could not determine weight files from download"
    exit 1
  fi

  echo ""
  echo "════════════════════════════════════════════════════"
  echo "  Starting shard server (layers 18-35)..."
  echo "  Connecting to the Samhati relay network..."
  echo "  Your host will start chatting once you appear."
  echo "  Ctrl-C to stop."
  echo "════════════════════════════════════════════════════"
  echo ""

  ./target/release/mesh-node serve \
    --model-path "$WEIGHT_FILES" \
    --layer-start $HALF --layer-end $TOTAL_LAYERS --total-layers $TOTAL_LAYERS \
    $ARCH_ARGS \
    --topic "$TOPIC" \
    --bootstrap "$JOIN_CODE"
fi
