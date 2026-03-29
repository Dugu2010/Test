#!/bin/bash
set -e
echo "=== Project Initialization ==="
echo "Setting up directories..."
mkdir -p /root/ai/{conf,lib,tools,logs,sessions}
echo "Installing dependencies..."
cd /root/ai && pip install -q -r requirements.txt 2>/dev/null
echo "Initialization complete."
