#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

echo "=== Cleaning ForgeCLI ==="
if [[ -d "$SCRIPT_DIR/logs" ]]; then
    find "$SCRIPT_DIR/logs" -maxdepth 1 -type f -name '*.log' -delete
fi
if [[ -d "$SCRIPT_DIR/sessions" ]]; then
    find "$SCRIPT_DIR/sessions" -maxdepth 1 -type f -name '*.json' -mtime +30 -delete
fi
find "$SCRIPT_DIR" -type d -name '__pycache__' -prune -exec rm -rf {} +
echo "Clean complete."
