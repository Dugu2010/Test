#!/bin/bash
set -e

echo "=== AI CLI Agent Setup ==="
echo ""

# Create directory structure
mkdir -p /root/ai/{conf,lib,tools,logs,sessions}

cd /root/ai

# Install dependencies
echo "[1/3] Installing Python dependencies..."
pip install -q -r requirements.txt 2>/dev/null || pip3 install -q -r requirements.txt 2>/dev/null

# Make scripts executable
chmod +x tools/*.sh 2>/dev/null || true
chmod +x main.py

# Check for API key
if [ -z "$NVIDIA_API_KEY" ]; then
    if [ ! -f .env ] || ! grep -q 'NVIDIA_API_KEY' .env 2>/dev/null; then
        echo ""
        echo "[!] NVIDIA_API_KEY not found."
        read -p "    Enter your NVIDIA API key: " api_key
        if [ -n "$api_key" ]; then
            echo "NVIDIA_API_KEY=$api_key" > .env
            echo "    Key saved to /root/ai/.env"
        fi
    fi
fi

echo ""
echo "[2/3] Setup complete."
echo "[3/3] Run with:  cd /root/ai && python main.py"
echo ""
