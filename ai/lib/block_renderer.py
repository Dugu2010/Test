"""Rich renderables for PATCH blocks and tool call/result displays in ForgeCLI.

All functions return Rich renderables (Text, Panel, Group) suitable
for printing with Console or embedding in a Live display.
"""

import os
from pathlib import Path
from typing import List, Dict, Any

from rich.text import Text
from rich.panel import Panel
from rich import box

from lib.block_parser import PatchBlock, ToolCallBlock, ToolResultBlock


# ── Helpers ─────────────────────────────────────────────────────────

def _read_lines(filepath: str, start: int, end: int) -> List[str]:
    """Best-effort read of *filepath* lines [start..end] (1-based, inclusive)."""
    try:
        p = Path(filepath)
        if not p.exists():
            return []
        lines = p.read_text(errors="replace").splitlines()
        return lines[max(0, start - 1):end]
    except Exception:
        return []


# ── PATCH block rendering ───────────────────────────────────────────

def render_patch_block(patch: PatchBlock, width: int = 80) -> Text:
    """Render a [PATCH] block as a diff-style display."""
    old_lines = _read_lines(patch.filepath, patch.line_start, patch.line_end)
    new_lines = patch.new_content.splitlines() if patch.new_content else []
    removed = len(old_lines)
    added = len(new_lines)

    t = Text()
    t.append("\U0001fa79 patch applied  ", style="bold green")
    t.append(f"\u2212{removed}", style="bold red")
    t.append("  ")
    t.append(f"+{added}", style="bold green")
    t.append("\n")
    t.append("\u250c\u2500 ", style="dim cyan")
    t.append(patch.filepath, style="bold cyan")
    t.append(f"  lines {patch.line_start}+", style="dim")
    t.append("\n")

    max_line_width = max(width - 10, 20)
    for line in old_lines:
        t.append("\u2502  ", style="dim cyan")
        t.append("\u2212  ", style="bold red")
        t.append(line[:max_line_width], style="red dim strike")
        t.append("\n")
    for line in new_lines:
        t.append("\u2502  ", style="dim cyan")
        t.append("+  ", style="bold green")
        t.append(line[:max_line_width], style="green")
        t.append("\n")
    t.append("\u2514\u2500", style="dim cyan")
    return t


# ── Tool call display ───────────────────────────────────────────────

def render_tool_call(name: str, args: Dict[str, Any], width: int = 80) -> Panel:
    """Render a tool call as a Rich Panel."""
    compact = width < 60
    t = Text()
    t.append("\U0001f527 ", style="")
    t.append(name, style="bold yellow")
    t.append("\n")
    for key, value in (args or {}).items():
        t.append(f"  {key}: ", style="bold")
        val_str = str(value)
        if key == 'content' and len(val_str) > 120:
            val_str = val_str[:120] + f"\u2026 ({len(val_str)} chars)"
        elif len(val_str) > 200:
            val_str = val_str[:200] + "\u2026"
        t.append(val_str, style="dim")
        t.append("\n")
    return Panel(
        t,
        border_style="yellow",
        box=box.ROUNDED if not compact else box.MINIMAL,
        padding=(0, 1),
        expand=True,
    )


# ── Tool result display ─────────────────────────────────────────────

def render_tool_result(
    name: str,
    args: Dict[str, Any],
    result: Dict[str, Any],
    width: int = 80,
) -> Panel:
    """Render a tool result as a Rich Panel."""
    compact = width < 60
    result = result or {}
    args = args or {}
    success = result.get('success', False)
    output = result.get('output', '')
    icon = '\u2705' if success else '\u274c'
    color = 'green' if success else 'red'

    if name == 'file_write' and success:
        return _render_file_write_result(args, output, width, compact)
    if name == 'file_read' and success:
        return _render_file_read_result(args, output, width, compact)
    if name == 'cmd_exec':
        return _render_cmd_result(args, result, width, compact)

    t = Text()
    t.append(f"{icon} ", style="")
    t.append(name, style=f"bold {color}")
    t.append(" completed\n", style="dim")
    display_output = output[:500] + "\n\u2026 (truncated)" if len(output) > 500 else output
    if display_output:
        t.append(display_output, style="")
    return Panel(
        t,
        border_style=color,
        box=box.ROUNDED if not compact else box.MINIMAL,
        padding=(0, 1),
        expand=True,
    )


def _render_file_write_result(args, output, width, compact):
    path = args.get('path', '?')
    content = args.get('content', '')
    lines = content.splitlines()
    max_preview = 25
    max_line_width = max(width - 10, 20)
    t = Text()
    t.append("\U0001fa79 written  ", style="bold green")
    t.append(f"+{len(lines)} lines", style="bold green")
    t.append("\n")
    t.append("\u250c\u2500 ", style="dim cyan")
    t.append(path, style="bold cyan")
    t.append("\n")
    for line in lines[:max_preview]:
        t.append("\u2502  ", style="dim cyan")
        t.append("+  ", style="bold green")
        t.append(line[:max_line_width], style="green")
        t.append("\n")
    if len(lines) > max_preview:
        t.append("\u2502  ", style="dim cyan")
        t.append(f"   \u2026 ({len(lines) - max_preview} more lines)", style="dim")
        t.append("\n")
    t.append("\u2514\u2500", style="dim cyan")
    t.append("\n")
    t.append(output, style="dim")
    return Panel(t, border_style="green", box=box.ROUNDED if not compact else box.MINIMAL, padding=(0, 1), expand=True)


def _render_file_read_result(args, output, width, compact):
    path = args.get('path', '?')
    lines = output.splitlines()
    max_preview = 30
    max_line_width = max(width - 12, 20)
    t = Text()
    t.append("\U0001f4c4 ", style="")
    t.append(path, style="bold cyan")
    t.append(f"  ({len(lines)} lines)\n", style="dim")
    t.append("\u250c\u2500", style="dim blue")
    t.append("\n")
    for i, line in enumerate(lines[:max_preview]):
        t.append("\u2502 ", style="dim blue")
        t.append(f"{i + 1:4d} ", style="dim")
        t.append(line[:max_line_width], style="")
        t.append("\n")
    if len(lines) > max_preview:
        t.append("\u2502 ", style="dim blue")
        t.append(f"     \u2026 ({len(lines) - max_preview} more lines)", style="dim")
        t.append("\n")
    t.append("\u2514\u2500", style="dim blue")
    return Panel(t, border_style="blue", box=box.ROUNDED if not compact else box.MINIMAL, padding=(0, 1), expand=True)


def _render_cmd_result(args, result, width, compact):
    command = args.get('command', '?')
    success = result.get('success', False)
    exit_code = result.get('exit_code', -1)
    output = result.get('output', '')
    icon = '\u2705' if success else '\u274c'
    color = 'green' if success else 'red'
    t = Text()
    t.append(f"{icon} ", style="")
    t.append("$ ", style="bold")
    t.append(command[:max(width - 10, 20)], style="bold white")
    t.append(f"  (exit {exit_code})\n", style="dim")
    display = output[:600] + "\n\u2026 (truncated)" if len(output) > 600 else output
    if display:
        t.append(display, style="dim white")
    return Panel(t, border_style=color, box=box.ROUNDED if not compact else box.MINIMAL, padding=(0, 1), expand=True)


# ── Streaming indicator (no clock) ───────────────────────────────

def render_streaming_indicator(block_type: str, partial_content: str = "") -> Text:
    """Indicator shown while a block is still being streamed. Uses dots instead of clock."""
    t = Text()

    if block_type == "patch":
        t.append("\U0001fa79 ", style="")
        t.append("receiving patch", style="bold yellow")
        t.append(" \u00b7\u00b7\u00b7", style="yellow")
    elif block_type == "tool_call":
        t.append("\U0001f527 ", style="")
        t.append("receiving tool call", style="bold yellow")
        t.append(" \u00b7\u00b7\u00b7", style="yellow")
    elif block_type == "tool_result":
        t.append("\U0001f4cb ", style="")
        t.append("receiving result", style="bold blue")
        t.append(" \u00b7\u00b7\u00b7", style="blue")
    else:
        t.append("\u00b7\u00b7\u00b7 ", style="dim")
        t.append(f"streaming {block_type}", style="dim italic")

    if partial_content:
        preview_lines = partial_content.splitlines()[:4]
        t.append("\n")
        t.append("\u250c\u2500", style="dim")
        t.append("\n")
        for line in preview_lines:
            t.append("\u2502  ", style="dim")
            t.append(line[:100], style="dim italic")
            t.append("\n")
        if len(partial_content.splitlines()) > 4:
            t.append("\u2502  \u2026", style="dim")
            t.append("\n")
        t.append("\u2514\u2500", style="dim")

    return t


# ── Text-embedded block rendering ───────────────────────────────────

def render_tool_call_block(block: ToolCallBlock, width: int = 80) -> Text:
    max_line_width = max(width - 8, 20)
    t = Text()
    t.append("\U0001f527 ", style="")
    t.append(block.name, style="bold yellow")
    t.append("\n")
    t.append("\u250c\u2500 ", style="dim yellow")
    t.append("arguments", style="dim italic")
    t.append("\n")
    for line in block.arguments.splitlines():
        t.append("\u2502  ", style="dim yellow")
        t.append(line[:max_line_width], style="")
        t.append("\n")
    t.append("\u2514\u2500", style="dim yellow")
    return t


def render_tool_result_block(block: ToolResultBlock, width: int = 80) -> Text:
    max_line_width = max(width - 8, 20)
    t = Text()
    t.append("\U0001f4cb ", style="")
    t.append("tool result", style="bold blue")
    t.append("\n")
    t.append("\u250c\u2500", style="dim blue")
    t.append("\n")
    for line in block.content.splitlines()[:30]:
        t.append("\u2502  ", style="dim blue")
        t.append(line[:max_line_width], style="")
        t.append("\n")
    t.append("\u2514\u2500", style="dim blue")
    return t
