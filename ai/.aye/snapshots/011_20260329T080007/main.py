#!/usr/bin/env python3
"""ForgeCLI """

import sys
import os
import signal

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)


def _load_env():
    """Load .env file if present."""
    env_path = os.path.join(PROJECT_ROOT, '.env')
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, _, value = line.partition('=')
                    os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _ensure_dirs():
    """Ensure required directories exist."""
    for d in ['logs', 'sessions', 'tools', 'uploads', 'web']:
        os.makedirs(os.path.join(PROJECT_ROOT, d), exist_ok=True)


def _check_api_key() -> bool:
    """Check if API key is available (warning only for web mode)."""
    return bool(os.environ.get('NVIDIA_API_KEY', ''))


def main():
    """Main entry point."""
    signal.signal(signal.SIGINT, lambda s, f: None)
    _load_env()
    _ensure_dirs()

    # Parse args
    args = sys.argv[1:]
    web_mode = '--web' in args or '-w' in args
    port = None
    host = None

    for i, arg in enumerate(args):
        if arg == '--port' and i + 1 < len(args):
            try:
                port = int(args[i + 1])
            except ValueError:
                pass
        if arg == '--host' and i + 1 < len(args):
            host = args[i + 1]

    if web_mode:
        # Web UI mode
        if not _check_api_key():
            print("\033[93m[WARN]\033[0m NVIDIA_API_KEY not set. Set it in Settings or via environment.")
            print("  export NVIDIA_API_KEY='your-key'")
            print("  Or add to .env file")
            print()

        try:
            from lib.web_server import start_server
            start_server(host=host, port=port)
        except ImportError as e:
            print(f"\033[91m[ERROR]\033[0m Web dependencies missing: {e}")
            print("  Install with: pip install fastapi uvicorn python-multipart websockets")
            sys.exit(1)
        except KeyboardInterrupt:
            print("\n\033[90mServer stopped.\033[0m")
    else:
        # CLI mode
        if not _check_api_key():
            print("\033[91m[ERROR]\033[0m NVIDIA_API_KEY not set.")
            print("  Set it via:  export NVIDIA_API_KEY='your-key'")
            print("  Or add to:   .env")
            print("  Or use web mode: python main.py --web")
            sys.exit(1)

        from lib.tui import AgentTUI
        tui = AgentTUI()
        tui.run()


if __name__ == '__main__':
    main()
