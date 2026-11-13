#!/usr/bin/env python3
"""ForgeCLI application entry point."""

import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from lib.utils import logger

SETTINGS_FILE = os.path.join(PROJECT_ROOT, ".forge_settings.json")


def _load_env():
    """Load project-local environment variables without overriding the shell."""
    env_path = os.path.join(PROJECT_ROOT, ".env")
    if not os.path.isfile(env_path):
        return

    try:
        with open(env_path, encoding="utf-8") as env_file:
            for line_number, raw_line in enumerate(env_file, 1):
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("export "):
                    line = line[7:].lstrip()
                key, separator, value = line.partition("=")
                key = key.strip()
                if not separator or not key:
                    logger.warning("Ignoring malformed .env entry at line %d", line_number)
                    continue
                value = value.strip()
                if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
                    value = value[1:-1]
                os.environ.setdefault(key, value)
    except (OSError, UnicodeError):
        logger.exception("Could not read project .env file")


def _load_api_key_from_settings():
    """Load an API key from persisted settings only when no environment key exists."""
    if os.environ.get("NVIDIA_API_KEY"):
        return

    try:
        with open(SETTINGS_FILE, encoding="utf-8") as settings_file:
            settings = json.load(settings_file)
    except FileNotFoundError:
        return
    except (OSError, UnicodeError, json.JSONDecodeError):
        logger.exception("Could not load settings while looking for an API key")
        return

    if not isinstance(settings, dict):
        logger.warning("Ignoring settings file: expected a JSON object")
        return
    key = settings.get("api_key")
    if isinstance(key, str) and key.strip():
        os.environ["NVIDIA_API_KEY"] = key.strip()


def _ensure_dirs():
    """Ensure app data directories exist beside the installed application."""
    for name in ("logs", "sessions", "tools", "uploads", "web"):
        os.makedirs(os.path.join(PROJECT_ROOT, name), exist_ok=True)


def _port(value):
    try:
        port = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("port must be an integer") from exc
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("port must be between 1 and 65535")
    return port


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="ForgeCLI, a tool-enabled terminal AI assistant.")
    parser.add_argument("--version", action="version", version="ForgeCLI 1.0.0")
    parser.add_argument("-w", "--web", action="store_true", help="start the web interface")
    parser.add_argument("--host", help="web server bind address")
    parser.add_argument("--port", type=_port, help="web server port (1-65535)")
    return parser.parse_args(argv)


def main(argv=None):
    """Start the terminal or web application."""
    args = _parse_args(argv)
    _load_env()
    _load_api_key_from_settings()

    try:
        _ensure_dirs()
    except OSError as exc:
        logger.exception("Could not create ForgeCLI data directories")
        print(f"[ERROR] Cannot initialize ForgeCLI directories: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    if args.web:
        if not os.environ.get("NVIDIA_API_KEY"):
            print("[WARN] NVIDIA_API_KEY is not set. Configure it in the web UI or environment.")
        try:
            from lib.web_server import start_server
            start_server(host=args.host, port=args.port)
        except ImportError as exc:
            logger.exception("Web dependencies are unavailable")
            print(
                f"[ERROR] Web dependencies are missing: {exc}\n"
                "Install them with: python -m pip install -r requirements.txt",
                file=sys.stderr,
            )
            raise SystemExit(1) from exc
        except KeyboardInterrupt:
            print("\nServer stopped.")
        except Exception as exc:
            logger.exception("Web server failed to start")
            print(f"[ERROR] Web server failed: {exc}", file=sys.stderr)
            raise SystemExit(1) from exc
        return

    if not os.environ.get("NVIDIA_API_KEY"):
        print(
            "[ERROR] NVIDIA_API_KEY is not set.\n"
            "Set it in the environment, add it to ai/.env, or run with --web.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    try:
        from lib.tui import AgentTUI
        AgentTUI().run()
    except ImportError as exc:
        logger.exception("Terminal UI dependencies are unavailable")
        print(
            f"[ERROR] Required dependencies are missing: {exc}\n"
            "Install them with: python -m pip install -r requirements.txt",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc
    except KeyboardInterrupt:
        print("\nForgeCLI stopped.")
    except Exception as exc:
        logger.exception("ForgeCLI failed")
        print(f"[ERROR] ForgeCLI failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
