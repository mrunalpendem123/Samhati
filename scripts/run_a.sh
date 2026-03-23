#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════
# Terminal 1 — just run: bash scripts/run_a.sh
# Downloads weights, serves layers 0-17, waits for peer,
# then drops you into a chat with the distributed model.
# ═══════════════════════════════════════════════════════════
set -e
cd "$(dirname "$0")/.."

MODEL=Qwen/Qwen2.5-3B
WEIGHTS=~/qwen25-3b
LAYERS_START=0
LAYERS_END=18
TOTAL=36

# Architecture
HIDDEN=2048; INTERMEDIATE=11008; HEADS=16; KV_HEADS=2
VOCAB=151936; ROPE=1000000; EPS=0.000001

ID_FILE_A=/tmp/samhati_node_a
ID_FILE_B=/tmp/samhati_node_b
rm -f "$ID_FILE_A" "$ID_FILE_B"

# ── Build ────────────────────────────────────────────────
if [ ! -f target/release/mesh-node ]; then
  echo "Building Samhati (first time only)..."
  cargo build --release --features burn
fi

# ── Download weights (only this shard's files) ───────────
python3 scripts/download_shard.py \
  --repo "$MODEL" \
  --layer-start $LAYERS_START --layer-end $LAYERS_END --total-layers $TOTAL \
  --out "$WEIGHTS"

# ── Start shard server in background ─────────────────────
MODEL_FILES=$(ls "$WEIGHTS"/model-*.safetensors | tr '\n' ',' | sed 's/,$//')

./target/release/mesh-node serve \
  --model-path "$MODEL_FILES" \
  --store-path "$WEIGHTS/shard-cache" \
  --layer-start $LAYERS_START --layer-end $LAYERS_END --total-layers $TOTAL \
  --hidden $HIDDEN --intermediate $INTERMEDIATE \
  --heads $HEADS --kv-heads $KV_HEADS \
  --vocab $VOCAB --rope-theta $ROPE --rms-eps $EPS \
  2>&1 | tee /tmp/samhati_a.log &

SERVE_PID=$!
trap "kill $SERVE_PID 2>/dev/null" EXIT

# Wait for node_id to appear in log
echo "Waiting for shard to load..."
while ! grep -q "^node_id:" /tmp/samhati_a.log 2>/dev/null; do sleep 0.5; done
NODE_A=$(grep "^node_id:" /tmp/samhati_a.log | head -1 | awk '{print $2}')
echo "$NODE_A" > "$ID_FILE_A"

echo ""
echo "══════════════════════════════════════════"
echo "  Node A ready (layers $LAYERS_START–$((LAYERS_END-1)))"
echo "  ID: $NODE_A"
echo "  Waiting for Terminal 2 (run_b.sh)..."
echo "══════════════════════════════════════════"

# ── Wait for peer B ──────────────────────────────────────
while [ ! -f "$ID_FILE_B" ]; do sleep 0.5; done
NODE_B=$(cat "$ID_FILE_B")
echo ""
echo "  Peer B found: ${NODE_B:0:16}..."
echo ""
echo "══════════════════════════════════════════"
echo "  Samhati — Distributed Qwen2.5-3B"
echo "  Type a message and press Enter."
echo "  Ctrl-C to quit."
echo "══════════════════════════════════════════"

# ── Chat loop ────────────────────────────────────────────
while true; do
  printf "\nYou: "
  read -r INPUT
  [ -z "$INPUT" ] && continue
  printf "Model: "
  ./target/release/mesh-node dist-run \
    --executor iroh \
    --peers "$NODE_A,$NODE_B" \
    --input "$INPUT" \
    --config "$WEIGHTS/config.json" --params-b 3.0 \
    --max-tokens 200 \
    2>&1 | sed 's/^output: //'
done
