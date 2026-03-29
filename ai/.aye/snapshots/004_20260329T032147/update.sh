#!/bin/bash
set -e
echo "=== Updating AI Agent ==="
cd /root/ai
pip install -q --upgrade -r requirements.txt 2>/dev/null
echo "Dependencies updated."
echo "Update complete."
