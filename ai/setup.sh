#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== ForgeCLI Setup ==="

mkdir -p conf lib tools logs sessions uploads web

if command -v python3 >/dev/null 2>&1; then
    PYTHON=python3
elif command -v python >/dev/null 2>&1; then
    PYTHON=python
else
    echo "[ERROR] Python 3 is required." >&2
    exit 1
fi

echo "[1/3] Installing Python dependencies..."
"$PYTHON" -m pip install -r requirements.txt

for script in tools/*.sh; do
    if [[ -f "$script" ]]; then
        chmod +x "$script"
    fi
done
chmod +x main.py bin/ai.py

if [[ -z "${NVIDIA_API_KEY:-}" ]] && ! grep -q '^NVIDIA_API_KEY=' .env 2>/dev/null; then
    if [[ -t 0 ]]; then
        read -r -s -p "NVIDIA API key (press Enter to configure later): " api_key
        echo
        if [[ -n "$api_key" ]]; then
            umask 077
            printf 'NVIDIA_API_KEY=%s\n' "$api_key" > .env
            chmod 600 .env
            unset api_key
        fi
    else
        echo "[WARN] NVIDIA_API_KEY is not set; configure it before using the CLI."
    fi
fi

echo "[2/3] Setup complete."
echo "[3/3] Run from this directory with: $PYTHON bin/ai.py"
