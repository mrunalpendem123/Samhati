#!/bin/bash
# Samhati — one command install and run
# Usage: curl -sSL https://raw.githubusercontent.com/mrunalpendem123/Samhati/main/install.sh | bash

set -e

echo "
  ╔═══════════════════════════════════════╗
  ║   SAMHATI — Free AI for Everyone     ║
  ║   Installing...                       ║
  ╚═══════════════════════════════════════╝
"

# Check Rust
if ! command -v cargo &> /dev/null; then
    echo "Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
fi

# Check llama.cpp
if ! command -v llama-server &> /dev/null; then
    echo "Installing llama.cpp..."
    if command -v brew &> /dev/null; then
        brew install llama.cpp
    else
        echo "Please install llama.cpp: https://github.com/ggerganov/llama.cpp"
        echo "On Ubuntu: sudo apt install cmake build-essential && git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp && cmake -B build && cmake --build build --config Release -t llama-server && sudo cp build/bin/llama-server /usr/local/bin/"
        exit 1
    fi
fi

# Clone or update
SAMHATI_DIR="$HOME/Samhati"
if [ -d "$SAMHATI_DIR" ]; then
    echo "Updating Samhati..."
    cd "$SAMHATI_DIR"
    git pull --quiet
else
    echo "Cloning Samhati..."
    git clone --quiet https://github.com/mrunalpendem123/Samhati.git "$SAMHATI_DIR"
    cd "$SAMHATI_DIR"
fi

# Build
echo "Building (this takes a few minutes first time)..."
cargo build -p samhati-tui --quiet 2>/dev/null || cargo build -p samhati-tui

# Run
echo "
  ╔═══════════════════════════════════════╗
  ║   Starting Samhati...                 ║
  ║   • Models tab → pick a model         ║
  ║   • Chat tab → ask anything           ║
  ╚═══════════════════════════════════════╝
"
cargo run -p samhati-tui
