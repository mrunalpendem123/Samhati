#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# samhati.sh — One-command distributed LLM inference
#
# Usage:
#   bash scripts/samhati.sh              # Host: download layers 0-13, start chat
#   bash scripts/samhati.sh <join-code>  # Join: download layers 14-27, serve
#
# The join-code is the node_id printed by the host. Share it once (copy-paste).
# ─────────────────────────────────────────────────────────────────────────────
set -e
cd "$(dirname "$0")/.."

MODEL=Qwen/Qwen2.5-7B
TOTAL_LAYERS=28
WEIGHTS=~/qwen25-7b

# Fixed gossip topic for this model network.
# Everyone on this topic is running the same model split.
TOPIC="73616d6861746976317177656e322e352d37422d676f7373697020746f706963"

ARCH_ARGS="--hidden 3584 --intermediate 18944 --heads 28 --kv-heads 4 \
  --vocab 152064 --rope-theta 1000000 --rms-eps 0.000001"

JOIN_CODE="${1:-}"

# ── 1. Build binary if not present ──────────────────────────────────────────
if [ ! -f target/release/mesh-node ]; then
  echo "Building Samhati (first time only, takes ~5 min)..."
  cargo build --release --features burn
fi

if [ -z "$JOIN_CODE" ]; then
  # ════════════════════════════════════════════════════════════════════════════
  #  HOST MODE  (Person A — serves layers 0-13, then starts interactive chat)
  # ════════════════════════════════════════════════════════════════════════════
  echo ""
  echo "┌─────────────────────────────────────────┐"
  echo "│  Samhati  ·  Distributed LLM  ·  Host   │"
  echo "│  You will serve layers 0-13              │"
  echo "└─────────────────────────────────────────┘"
  echo ""

  # Download only layers 0-13 (~7.8 GB for Qwen2.5-7B)
  python3 scripts/download_shard.py \
    --repo "$MODEL" \
    --layer-start 0 --layer-end 14 --total-layers $TOTAL_LAYERS \
    --out "$WEIGHTS"

  WEIGHT_FILES="$(ls "$WEIGHTS"/model-*.safetensors 2>/dev/null | tr '\n' ',' | sed 's/,$//')"
  if [ -z "$WEIGHT_FILES" ]; then
    echo "Error: no weight files found in $WEIGHTS"
    exit 1
  fi

  # Temp files for IPC between serve (bg) and this script (fg)
  SERVE_LOG=$(mktemp /tmp/samhati-serve-XXXXXX.log)
  PEER_FILE=$(mktemp /tmp/samhati-peers-XXXXXX.txt)
  trap "kill \$SERVE_PID 2>/dev/null; rm -f '$SERVE_LOG' '$PEER_FILE'" EXIT

  # Start serve in background; direct its stdout to SERVE_LOG so we can grep
  # for the node_id line.  Also pass --peer-file so it writes peer node_ids
  # as they join via gossip.
  eval ./target/release/mesh-node serve \
    --model-path '"$WEIGHT_FILES"' \
    --store-path '"$WEIGHTS/shard-cache"' \
    --layer-start 0 --layer-end 14 --total-layers $TOTAL_LAYERS \
    $ARCH_ARGS \
    --topic "$TOPIC" \
    --peer-file "$PEER_FILE" \
    > "$SERVE_LOG" 2>&1 &
  SERVE_PID=$!

  # Wait for the node_id line to appear (up to 30 s)
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
  printf "│    bash scripts/samhati.sh %s  │\n" "${MY_NODE_ID:0:16}..."
  echo "│    (paste the full code above)                                       │"
  echo "└──────────────────────────────────────────────────────────────────────┘"
  echo ""
  echo "Connecting to the Samhati relay network..."
  echo "Waiting for your friend to join... (0 peers)"

  # Poll peer-file until a peer node_id appears (up to 10 min)
  PEER_NODE_ID=""
  for i in $(seq 1 600); do
    PEER_NODE_ID=$(head -1 "$PEER_FILE" 2>/dev/null | tr -d '[:space:]')
    if [ -n "$PEER_NODE_ID" ]; then
      echo ""
      echo "  Friend connected!  (1 peer in network)"
      break
    fi
    # Progress dots every 10 s
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
  echo "  Distributed Qwen2.5-7B is ready!"
  echo "  Layers  0-13 : you"
  echo "  Layers 14-27 : your friend"
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
      --params-b 7.0 \
      --total-layers $TOTAL_LAYERS \
      --max-tokens 200 \
      2>/dev/null | sed 's/^output: //' || echo "(inference error — is your friend still connected?)"
  done

else
  # ════════════════════════════════════════════════════════════════════════════
  #  JOIN MODE  (Person B — serves layers 14-27, bootstraps off the host)
  # ════════════════════════════════════════════════════════════════════════════
  echo ""
  echo "┌─────────────────────────────────────────┐"
  echo "│  Samhati  ·  Distributed LLM  ·  Join   │"
  echo "│  You will serve layers 14-27             │"
  echo "└─────────────────────────────────────────┘"
  echo ""
  echo "Connecting via:  ${JOIN_CODE:0:24}..."
  echo ""

  # Download only layers 14-27 (~7.4 GB for Qwen2.5-7B)
  python3 scripts/download_shard.py \
    --repo "$MODEL" \
    --layer-start 14 --layer-end 28 --total-layers $TOTAL_LAYERS \
    --out "$WEIGHTS"

  WEIGHT_FILES="$(ls "$WEIGHTS"/model-*.safetensors 2>/dev/null | tr '\n' ',' | sed 's/,$//')"
  if [ -z "$WEIGHT_FILES" ]; then
    echo "Error: no weight files found in $WEIGHTS"
    exit 1
  fi

  echo ""
  echo "════════════════════════════════════════════════════"
  echo "  Starting shard server (layers 14-27)..."
  echo "  Connecting to the Samhati relay network..."
  echo "  Your host will start chatting once you appear."
  echo "  Ctrl-C to stop."
  echo "════════════════════════════════════════════════════"
  echo ""

  ./target/release/mesh-node serve \
    --model-path "$WEIGHT_FILES" \
    --store-path "$WEIGHTS/shard-cache" \
    --layer-start 14 --layer-end 28 --total-layers $TOTAL_LAYERS \
    $ARCH_ARGS \
    --topic "$TOPIC" \
    --bootstrap "$JOIN_CODE"
fi
