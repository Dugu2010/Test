#!/bin/bash
echo "=== Cleaning ForgeCLI ==="
echo "Removing log files..."
rm -f /root/ai/logs/*.log
echo "Removing old sessions..."
find /root/ai/sessions -name '*.json' -mtime +30 -delete 2>/dev/null
echo "Removing Python cache..."
find /root/ai -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null
echo "Clean complete."
