"""Rich renderables for PATCH blocks and tool call/result displays.

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
    """Render a [PATCH] block as a diff-style display.

    Example output:
        🩹 patch applied  −4  +4
        ┌─ src/auth/login.py  lines 42+
        │  −  old line
        │  +  new line
        └─
    """
    old_lines = _read_lines(patch.filepath, patch.line_start, patch.line_end)
    new_lines = patch.new_content.splitlines() if patch.new_content else []
    removed = len(old_lines)
    added = len(new_lines)

    t = Text()

    # header
    t.append("🩹 patch applied  ", style="bold green")
    t.append(f"−{removed}", style="bold red")
    t.append("  ")
    t.append(f"+{added}", style="bold green")
    t.append("\n")

    # box top
    t.append("┌─ ", style="dim cyan")
    t.append(patch.filepath, style="bold cyan")
    t.append(f"  lines {patch.line_start}+", style="dim")
    t.append("\n")

    # removed lines
    for line in old_lines:
        t.append("│  ", style="dim cyan")
        t.append("−  ", style="bold red")
        t.append(line[:width - 10], style="red dim strike")
        t.append("\n")

    # added lines
    for line in new_lines:
        t.append("│  ", style="dim cyan")
        t.append("+  ", style="bold green")
        t.append(line[:width - 10], style="green")
        t.append("\n")

    # box bottom
    t.append("└─", style="dim cyan")

    return t


# ── Tool call display ───────────────────────────────────────────────

def render_tool_call(name: str, args: Dict[str, Any], width: int = 80) -> Panel:
    """Render a tool call as a Rich Panel with name and arguments."""
    compact = width < 60

    t = Text()
    t.append("🔧 ", style="")
    t.append(name, style="bold yellow")
    t.append("\n")

    for key, value in args.items():
        t.append(f"  {key}: ", style="bold")
        val_str = str(value)
        # Truncate long values (e.g. file content)
        if key == 'content' and len(val_str) > 120:
            val_str = val_str[:120] + f"… ({len(val_str)} chars)"
        elif len(val_str) > 200:
            val_str = val_str[:200] + "…"
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
    """Render a tool result as a Rich Panel.

    Special handling for file_write (PATCH-like diff), file_read (code box),
    and cmd_exec (command output).
    """
    compact = width < 60
    success = result.get('success', False)
    output = result.get('output', '')
    icon = '✅' if success else '❌'
    color = 'green' if success else 'red'

    # ── file_write: PATCH-like display ──────────────────────────────
    if name == 'file_write' and success:
        return _render_file_write_result(args, output, width, compact)

    # ── file_read: code display ─────────────────────────────────────
    if name == 'file_read' and success:
        return _render_file_read_result(args, output, width, compact)

    # ── cmd_exec: command output ────────────────────────────────────
    if name == 'cmd_exec':
        return _render_cmd_result(args, result, width, compact)

    # ── generic result ──────────────────────────────────────────────
    t = Text()
    t.append(f"{icon} ", style="")
    t.append(name, style=f"bold {color}")
    t.append(" completed\n", style="dim")

    display_output = output[:500] + "\n… (truncated)" if len(output) > 500 else output
    if display_output:
        t.append(display_output, style="")

    return Panel(
        t,
        border_style=color,
        box=box.ROUNDED if not compact else box.MINIMAL,
        padding=(0, 1),
        expand=True,
    )


def _render_file_write_result(
    args: Dict[str, Any],
    output: str,
    width: int,
    compact: bool,
) -> Panel:
    """Render file_write as a PATCH-style panel with green '+' lines."""
    path = args.get('path', '?')
    content = args.get('content', '')
    lines = content.splitlines()
    max_preview = 25

    t = Text()
    t.append("🩹 written  ", style="bold green")
    t.append(f"+{len(lines)} lines", style="bold green")
    t.append("\n")
    t.append("┌─ ", style="dim cyan")
    t.append(path, style="bold cyan")
    t.append("\n")

    for line in lines[:max_preview]:
        t.append("│  ", style="dim cyan")
        t.append("+  ", style="bold green")
        t.append(line[:width - 10], style="green")
        t.append("\n")

    if len(lines) > max_preview:
        t.append("│  ", style="dim cyan")
        t.append(f"   … ({len(lines) - max_preview} more lines)", style="dim")
        t.append("\n")

    t.append("└─", style="dim cyan")
    t.append("\n")
    t.append(output, style="dim")

    return Panel(
        t,
        border_style="green",
        box=box.ROUNDED if not compact else box.MINIMAL,
        padding=(0, 1),
        expand=True,
    )


def _render_file_read_result(
    args: Dict[str, Any],
    output: str,
    width: int,
    compact: bool,
) -> Panel:
    """Render file_read as a code-style panel."""
    path = args.get('path', '?')
    lines = output.splitlines()
    max_preview = 30

    t = Text()
    t.append("📄 ", style="")
    t.append(path, style="bold cyan")
    t.append(f"  ({len(lines)} lines)\n", style="dim")
    t.append("┌─", style="dim blue")
    t.append("\n")

    for i, line in enumerate(lines[:max_preview]):
        t.append("│ ", style="dim blue")
        t.append(f"{i + 1:4d} ", style="dim")
        t.append(line[:width - 12], style="")
        t.append("\n")

    if len(lines) > max_preview:
        t.append("│ ", style="dim blue")
        t.append(f"     … ({len(lines) - max_preview} more lines)", style="dim")
        t.append("\n")

    t.append("└─", style="dim blue")

    return Panel(
        t,
        border_style="blue",
        box=box.ROUNDED if not compact else box.MINIMAL,
        padding=(0, 1),
        expand=True,
    )


def _render_cmd_result(
    args: Dict[str, Any],
    result: Dict[str, Any],
    width: int,
    compact: bool,
) -> Panel:
    """Render cmd_exec result with exit code."""
    command = args.get('command', '?')
    success = result.get('success', False)
    exit_code = result.get('exit_code', -1)
    output = result.get('output', '')
    icon = '✅' if success else '❌'
    color = 'green' if success else 'red'

    t = Text()
    t.append(f"{icon} ", style="")
    t.append("$ ", style="bold")
    t.append(command[:width - 10], style="bold white")
    t.append(f"  (exit {exit_code})\n", style="dim")

    display = output[:600] + "\n… (truncated)" if len(output) > 600 else output
    if display:
        t.append(display, style="dim white")

    return Panel(
        t,
        border_style=color,
        box=box.ROUNDED if not compact else box.MINIMAL,
        padding=(0, 1),
        expand=True,
    )


# ── Streaming indicator ─────────────────────────────────────────────

def render_streaming_indicator(block_type: str, partial_content: str = "") -> Text:
    """Indicator shown while a block is still being streamed."""
    t = Text()

    if block_type == "patch":
        t.append("🩹 ", style="")
        t.append("receiving patch… ", style="bold yellow")
        t.append("⏳", style="yellow")
    elif block_type == "tool_call":
        t.append("🔧 ", style="")
        t.append("receiving tool call… ", style="bold yellow")
        t.append("⏳", style="yellow")
    elif block_type == "tool_result":
        t.append("📋 ", style="")
        t.append("receiving result… ", style="bold blue")
        t.append("⏳", style="blue")
    else:
        t.append(f"⏳ streaming {block_type}… ", style="dim italic")

    # Small preview of partial content
    if partial_content:
        preview_lines = partial_content.splitlines()[:4]
        t.append("\n")
        t.append("┌─", style="dim")
        t.append("\n")
        for line in preview_lines:
            t.append("│  ", style="dim")
            t.append(line[:100], style="dim italic")
            t.append("\n")
        if len(partial_content.splitlines()) > 4:
            t.append("│  …", style="dim")
            t.append("\n")
        t.append("└─", style="dim")

    return t


# ── Text-embedded block rendering ───────────────────────────────────

def render_tool_call_block(block: ToolCallBlock, width: int = 80) -> Text:
    """Render a [TOOL_CALL] block found in AI text."""
    t = Text()
    t.append("🔧 ", style="")
    t.append(block.name, style="bold yellow")
    t.append("\n")
    t.append("┌─ ", style="dim yellow")
    t.append("arguments", style="dim italic")
    t.append("\n")
    for line in block.arguments.splitlines():
        t.append("│  ", style="dim yellow")
        t.append(line[:width - 8], style="")
        t.append("\n")
    t.append("└─", style="dim yellow")
    return t


def render_tool_result_block(block: ToolResultBlock, width: int = 80) -> Text:
    """Render a [TOOL_RESULT] block found in AI text."""
    t = Text()
    t.append("📋 ", style="")
    t.append("tool result", style="bold blue")
    t.append("\n")
    t.append("┌─", style="dim blue")
    t.append("\n")
    for line in block.content.splitlines()[:30]:
        t.append("│  ", style="dim blue")
        t.append(line[:width - 8], style="")
        t.append("\n")
    t.append("└─", style="dim blue")
    return t
