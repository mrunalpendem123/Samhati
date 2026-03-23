#!/usr/bin/env bash
# Collect training examples from swarm round files.
# Usage: ./scripts/collect.sh <round_file_or_directory>
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

if [ $# -lt 1 ]; then
    echo "Usage: $0 <round_file_or_directory>"
    exit 1
fi

INPUT="$1"

if [ -f "$INPUT" ]; then
    samhati-pipeline collect --round-file "$INPUT"
elif [ -d "$INPUT" ]; then
    for f in "$INPUT"/*.json; do
        [ -f "$f" ] || continue
        echo "Processing: $f"
        samhati-pipeline collect --round-file "$f"
    done
else
    echo "Error: $INPUT is not a file or directory"
    exit 1
fi
