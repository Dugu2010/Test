#!/bin/bash
set -e
echo "=== Updating ForgeCLI ==="
cd /root/ai
pip install -q --upgrade -r requirements.txt 2>/dev/null
echo "Dependencies updated."
echo "Update complete."
