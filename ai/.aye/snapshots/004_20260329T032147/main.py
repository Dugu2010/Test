#!/usr/bin/env python3
"""AI CLI Agent - Main Entry Point
Production-ready AI terminal agent with tool-calling capabilities.
API: NVIDIA integrate.api.nvidia.com
"""

import sys
import os
import signal

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

def main():
    """Main entry point."""
    # Graceful exit on SIGINT
    signal.signal(signal.SIGINT, lambda s, f: None)

    # Load environment
    env_path = os.path.join(PROJECT_ROOT, '.env')
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, _, value = line.partition('=')
                    os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))

    # Ensure directories exist
    for d in ['logs', 'sessions', 'tools']:
        os.makedirs(os.path.join(PROJECT_ROOT, d), exist_ok=True)

    # Check API key
    api_key = os.environ.get('NVIDIA_API_KEY', '')
    if not api_key:
        print("\033[91m[ERROR]\033[0m NVIDIA_API_KEY not set.")
        print("  Set it via:  export NVIDIA_API_KEY='your-key'")
        print("  Or add to:   /root/ai/.env")
        sys.exit(1)

    from lib.tui import AgentTUI
    tui = AgentTUI()
    tui.run()


if __name__ == '__main__':
    main()
