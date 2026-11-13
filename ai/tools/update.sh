#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

if command -v python3 >/dev/null 2>&1; then
    PYTHON=python3
elif command -v python >/dev/null 2>&1; then
    PYTHON=python
else
    echo "[ERROR] Python 3 is required." >&2
    exit 1
fi

echo "=== Updating ForgeCLI dependencies ==="
"$PYTHON" -m pip install --upgrade -r requirements.txt
echo "Update complete."
