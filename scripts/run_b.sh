#!/usr/bin/env bash
# ─────────────────────────────────────────────
# Person B — layers 14-27  (second half + lm_head)
# Run this script, then share your node_id with Person A.
# ─────────────────────────────────────────────
set -e
cd "$(dirname "$0")/.."

WEIGHTS=~/qwen25-7b

# 1. Build if needed
if [ ! -f target/release/mesh-node ]; then
  echo "Building Samhati (takes ~5 min)..."
  cargo build --release --features burn
fi

# 2. Download only Person B's weight files
python3 scripts/download_shard.py \
  --repo Qwen/Qwen2.5-7B \
  --layer-start 14 --layer-end 28 --total-layers 28 \
  --out "$WEIGHTS"

# 3. Serve
echo ""
echo "=============================="
echo " Starting your shard server..."
echo " Copy your node_id and send it to Person A."
echo "=============================="
echo ""

./target/release/mesh-node serve \
  --model-path "$(ls "$WEIGHTS"/model-*.safetensors | tr '\n' ',' | sed 's/,$//')" \
  --store-path "$WEIGHTS/shard-cache" \
  --layer-start 14 --layer-end 28 --total-layers 28 \
  --hidden 3584 --intermediate 18944 \
  --heads 28 --kv-heads 4 \
  --vocab 152064 --rope-theta 1000000 --rms-eps 0.000001
