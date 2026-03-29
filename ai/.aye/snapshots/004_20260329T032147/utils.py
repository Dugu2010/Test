"""Utility functions for AI CLI Agent."""

import os
import json
import logging
import datetime
from conf.conf import LOG_FILE, MAX_TOOL_OUTPUT_CHARS


def setup_logging():
    """Configure logging to file."""
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    logging.basicConfig(
        filename=LOG_FILE,
        level=logging.DEBUG,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    return logging.getLogger('ai_agent')


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
