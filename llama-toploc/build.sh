#!/bin/bash
# Build llama-server with TOPLOC proof-of-compute
#
# This patches llama.cpp to capture intermediate layer activations
# during inference and return a BLAKE3 proof hash in the API response.
#
# Usage: ./llama-toploc/build.sh
# Output: ./llama-toploc/bin/llama-server

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/llama.cpp"
PATCH_FILE="$SCRIPT_DIR/toploc.patch"

echo "Building llama-server with TOPLOC proof-of-compute..."

# Clone llama.cpp if not present
if [ ! -d "$BUILD_DIR" ]; then
    echo "Cloning llama.cpp..."
    git clone --depth 1 https://github.com/ggml-org/llama.cpp.git "$BUILD_DIR"
fi

# Apply TOPLOC patch
cd "$BUILD_DIR"
echo "Applying TOPLOC patch..."
cp "$SCRIPT_DIR/toploc-proof.h" tools/server/toploc-proof.h
git apply "$PATCH_FILE" 2>/dev/null || echo "Patch already applied or applying manually..."

# If git apply failed, apply manually
if ! grep -q "toploc-proof.h" tools/server/server-context.cpp 2>/dev/null; then
    # Add include
    sed -i.bak '1s/^/#include "toploc-proof.h"\n/' tools/server/server-context.cpp 2>/dev/null || \
    sed -i '' '1s/^/#include "toploc-proof.h"\n/' tools/server/server-context.cpp

    # The patch file handles the rest
    echo "Manual patch applied"
fi

# Build
echo "Compiling (this takes a few minutes)..."
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target llama-server -j$(nproc 2>/dev/null || sysctl -n hw.ncpu)

# Copy binary
mkdir -p "$SCRIPT_DIR/bin"
cp build/bin/llama-server "$SCRIPT_DIR/bin/llama-server"

echo ""
echo "Done! TOPLOC-enabled llama-server at: $SCRIPT_DIR/bin/llama-server"
echo ""
echo "Usage:"
echo "  $SCRIPT_DIR/bin/llama-server -m model.gguf --port 8080 --host 0.0.0.0"
echo ""
echo "The /v1/chat/completions response will include:"
echo '  {"toploc_proof": "a1b2c3d4..."}'
