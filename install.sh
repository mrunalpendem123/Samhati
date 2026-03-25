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

# Check cmake (needed to build llama.cpp)
if ! command -v cmake &> /dev/null; then
    echo "Installing cmake..."
    if command -v brew &> /dev/null; then
        brew install cmake
    elif command -v apt-get &> /dev/null; then
        sudo apt-get install -y cmake build-essential
    else
        echo "Please install cmake: https://cmake.org/download/"
        exit 1
    fi
fi

# Clone or update Samhati
SAMHATI_DIR="$HOME/Samhati"
if [ -d "$SAMHATI_DIR" ]; then
    echo "Updating Samhati..."
    cd "$SAMHATI_DIR"
    git stash --quiet 2>/dev/null || true
    git pull --quiet || git pull
    git stash pop --quiet 2>/dev/null || true
else
    echo "Cloning Samhati..."
    git clone --quiet https://github.com/mrunalpendem123/Samhati.git "$SAMHATI_DIR"
    cd "$SAMHATI_DIR"
fi

# Build TOPLOC-enabled llama-server (PrimeIntellect-strength proofs)
if [ ! -f "$SAMHATI_DIR/llama-toploc/bin/llama-server" ]; then
    echo "Building llama-server with TOPLOC proof-of-compute..."
    bash "$SAMHATI_DIR/llama-toploc/build.sh"
fi

# Build Samhati TUI
echo "Building Samhati TUI..."
cargo build -p samhati-tui --quiet 2>/dev/null || cargo build -p samhati-tui

# Done — tell user to run it manually (can't run TUI from piped bash — stdin is taken)
echo "
  ╔═══════════════════════════════════════╗
  ║   Samhati installed!                  ║
  ║                                       ║
  ║   Run this command:                   ║
  ║   cd ~/Samhati && cargo run -p samhati-tui  ║
  ║                                       ║
  ║   • Models tab → pick a model         ║
  ║   • Chat tab → ask anything           ║
  ╚═══════════════════════════════════════╝
"
