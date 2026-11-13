"""Utility functions for ForgeCLI."""

import os
import json
import logging
import datetime
import sys
from conf.conf import LOG_FILE, MAX_TOOL_OUTPUT_CHARS


def setup_logging():
    """Configure file logging, falling back to stderr if the log is unavailable."""
    app_logger = logging.getLogger('forgecli')
    if app_logger.handlers:
        return app_logger

    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    try:
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
        handler = logging.FileHandler(LOG_FILE, encoding='utf-8')
    except OSError as exc:
        handler = logging.StreamHandler(sys.stderr)
        app_logger.addHandler(handler)
        app_logger.setLevel(logging.DEBUG)
        app_logger.propagate = False
        handler.setFormatter(formatter)
        app_logger.warning("File logging unavailable (%s); logging to stderr.", exc)
        return app_logger

    handler.setFormatter(formatter)
    app_logger.addHandler(handler)
    app_logger.setLevel(logging.DEBUG)
    app_logger.propagate = False
    return app_logger


logger = setup_logging()


def truncate(text: str, max_len: int = MAX_TOOL_OUTPUT_CHARS) -> str:
    """Truncate text to max length with indicator."""
    if len(text) <= max_len:
        return text
    half = max_len // 2 - 20
    return text[:half] + f'\n\n... [truncated {len(text) - max_len} chars] ...\n\n' + text[-half:]


def timestamp() -> str:
    """ISO timestamp."""
    return datetime.datetime.now().isoformat()


def safe_json_loads(text: str, default=None):
    """Safely parse JSON."""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return default


def get_terminal_size():
    """Get terminal dimensions safely."""
    try:
        cols, rows = os.get_terminal_size()
        return max(cols, 40), max(rows, 10)
    except OSError:
        return 80, 24
