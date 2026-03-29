"""Parser for PATCH, TOOL_CALL, and TOOL_RESULT blocks in ForgeAI responses.

Extracts structured blocks from AI text output for rich rendering.
Pure parsing logic.
"""

import re
from dataclasses import dataclass
from typing import List, Tuple, Union, Optional


@dataclass
class PatchBlock:
    """Parsed [PATCH]...[/PATCH] block."""
    filepath: str
    line_start: int
    line_end: int
    new_content: str
    raw: str = ""


@dataclass
class ToolCallBlock:
    """Parsed [TOOL_CALL]...[/TOOL_CALL] block."""
    name: str
    arguments: str
    raw: str = ""


@dataclass
class ToolResultBlock:
    """Parsed [TOOL_RESULT]...[/TOOL_RESULT] block."""
    content: str
    raw: str = ""


# Type alias for parsed segments
Segment = Tuple[str, Union[str, PatchBlock, ToolCallBlock, ToolResultBlock]]


# ── Regex patterns ──────────────────────────────────────────────────

_PATCH_RE = re.compile(
    r'\[PATCH\]\s*\n'
    r'filepath:\s*(.+?)\n'
    r'line_start:\s*(\d+)\n'
    r'line_end:\s*(\d+)\n'
    r'new_content:\s*\n'
    r'(.*?)'
    r'\[/PATCH\]',
    re.DOTALL,
)

_TOOL_CALL_RE = re.compile(
    r'\[TOOL_CALL\]\s*\n'
    r'name:\s*(.+?)\n'
    r'arguments:\s*\n'
    r'(.*?)'
    r'\[/TOOL_CALL\]',
    re.DOTALL,
)

_TOOL_RESULT_RE = re.compile(
    r'\[TOOL_RESULT\]\s*\n'
    r'(.*?)'
    r'\[/TOOL_RESULT\]',
    re.DOTALL,
)

_COMBINED_RE = re.compile(
    r'(\[PATCH\].*?\[/PATCH\])'
    r'|(\[TOOL_CALL\].*?\[/TOOL_CALL\])'
    r'|(\[TOOL_RESULT\].*?\[/TOOL_RESULT\])',
    re.DOTALL,
)


# ── Public API ──────────────────────────────────────────────────────

def parse_blocks(text: str) -> List[Segment]:
    """Split *text* into typed segments: plain text, patch, tool_call, tool_result."""
    segments: List[Segment] = []
    last_end = 0

    for match in _COMBINED_RE.finditer(text):
        # preceding plain text
        if match.start() > last_end:
            preceding = text[last_end:match.start()]
            if preceding.strip():
                segments.append(("text", preceding))

        raw = match.group(0)

        if match.group(1):  # PATCH
            m = _PATCH_RE.search(raw)
            if m:
                segments.append(("patch", PatchBlock(
                    filepath=m.group(1).strip(),
                    line_start=int(m.group(2)),
                    line_end=int(m.group(3)),
                    new_content=m.group(4).rstrip(),
                    raw=raw,
                )))
            else:
                segments.append(("text", raw))

        elif match.group(2):  # TOOL_CALL
            m = _TOOL_CALL_RE.search(raw)
            if m:
                segments.append(("tool_call", ToolCallBlock(
                    name=m.group(1).strip(),
                    arguments=m.group(2).strip(),
                    raw=raw,
                )))
            else:
                segments.append(("text", raw))

        elif match.group(3):  # TOOL_RESULT
            m = _TOOL_RESULT_RE.search(raw)
            if m:
                segments.append(("tool_result", ToolResultBlock(
                    content=m.group(1).strip(),
                    raw=raw,
                )))
            else:
                segments.append(("text", raw))

        last_end = match.end()

    # trailing text
    if last_end < len(text):
        remaining = text[last_end:]
        if remaining.strip():
            segments.append(("text", remaining))

    # If nothing was parsed, return original as text
    if not segments:
        segments.append(("text", text))

    return segments


def has_incomplete_block(text: str) -> Optional[str]:
    """Return the block type ('patch', 'tool_call', 'tool_result') if there is
    an unclosed opening tag, else None."""
    for open_tag, close_tag, kind in (
        ("[PATCH]", "[/PATCH]", "patch"),
        ("[TOOL_CALL]", "[/TOOL_CALL]", "tool_call"),
        ("[TOOL_RESULT]", "[/TOOL_RESULT]", "tool_result"),
    ):
        if text.count(open_tag) > text.count(close_tag):
            return kind
    return None


def get_incomplete_block_content(text: str) -> str:
    """Return raw text from the last unclosed opening tag to end-of-string."""
    for tag in ("[PATCH]", "[TOOL_CALL]", "[TOOL_RESULT]"):
        idx = text.rfind(tag)
        if idx != -1:
            close_tag = tag.replace("[", "[/")
            if close_tag not in text[idx:]:
                return text[idx:]
    return ""


def strip_incomplete_block(text: str) -> str:
    """Return *text* with a trailing incomplete block removed."""
    for tag in ("[PATCH]", "[TOOL_CALL]", "[TOOL_RESULT]"):
        idx = text.rfind(tag)
        if idx != -1:
            close_tag = tag.replace("[", "[/")
            if close_tag not in text[idx:]:
                return text[:idx]
    return text
