"""Terminal UI for AI CLI Agent — responsive, rich-based, with streaming."""

import os
import sys
from typing import List
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.markdown import Markdown
from rich.live import Live
from rich.style import Style
from rich import box
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.formatted_text import HTML
from lib.agent import Agent
from lib.utils import logger, get_terminal_size
from lib.tools import TOOL_DEFINITIONS
from lib.block_parser import (
    parse_blocks,
    has_incomplete_block,
    get_incomplete_block_content,
    strip_incomplete_block,
)
from lib.block_renderer import (
    render_patch_block,
    render_tool_call,
    render_tool_result,
    render_tool_call_block,
    render_tool_result_block,
    render_streaming_indicator,
)
from conf.conf import (
    APP_NAME,
    APP_VERSION,
    AVAILABLE_MODELS,
    PROJECT_ROOT,
    LOG_FILE,
    STREAM_ENABLED,
    STREAM_REFRESH_RATE,
)

# ──────────────────────────────────────────────
# Color scheme
# ──────────────────────────────────────────────
STYLE_HEADER = "bold bright_cyan"
STYLE_USER = "bold green"
STYLE_ASSISTANT = "white"
STYLE_TOOL = "bold yellow"
STYLE_ERROR = "bold red"
STYLE_STATUS = "dim cyan"
STYLE_MUTED = "dim white"
STYLE_SUCCESS = "bold green"


class AgentTUI:
    """Rich-based terminal UI for the AI agent."""

    def __init__(self):
        self.console = Console()
        self.agent = Agent(on_status=self._show_status)
        self.verbose = False
        self.streaming = STREAM_ENABLED
        self._status_text = ""
        self._current_status = None

        # Prompt toolkit session with history
        hist_path = os.path.join(PROJECT_ROOT, '.prompt_history')
        self.prompt_session = PromptSession(
            history=FileHistory(hist_path),
            auto_suggest=AutoSuggestFromHistory(),
            enable_history_search=True,
        )

    def run(self):
        """Main run loop."""
        self._show_banner()

        while True:
            try:
                user_input = self.prompt_session.prompt(
                    HTML('<ansigreen><b>❯ </b></ansigreen>'),
                ).strip()

                if not user_input:
                    continue

                # Handle commands
                if user_input.startswith('/'):
                    if self._handle_command(user_input):
                        continue

                # Process through agent
                self.console.print()

                if self.streaming:
                    self._process_streaming(user_input)
                else:
                    self._process_sync(user_input)

            except KeyboardInterrupt:
                self.console.print("\n[dim]Use /exit to quit, or Ctrl+D[/dim]")
                continue
            except EOFError:
                self._exit()
                break
            except Exception as e:
                logger.error(f"TUI error: {e}", exc_info=True)
                self.console.print(f"[{STYLE_ERROR}]Error: {str(e)}[/]")

    # ── Synchronous processing (original path) ──────────────────────

    def _process_sync(self, user_input: str):
        """Process input non-streaming (original behaviour)."""
        with self.console.status("[cyan]Thinking...", spinner="dots") as status:
            self._current_status = status
            response = self.agent.process(user_input)
            self._current_status = None

        self._display_response(response)

    # ── Streaming processing ─────────────────────────────────────────

    def _process_streaming(self, user_input: str):
        """Process input with streaming display using Rich Live."""
        display_parts: List = []       # Completed panels (tool calls, results, text)
        current_text = ''              # Text being streamed right now
        status_text = 'Thinking...'

        def _build_display() -> Group:
            """Rebuild the full Live renderable from accumulated parts."""
            cols, _ = get_terminal_size()
            compact = cols < 60
            parts = list(display_parts)

            # Streaming text panel
            if current_text:
                incomplete = has_incomplete_block(current_text)
                if incomplete:
                    content = self._render_partial_block_content(current_text, cols)
                else:
                    content = Text(current_text + " █", style=STYLE_ASSISTANT)

                parts.append(Panel(
                    content,
                    title="[bold bright_cyan]Assistant[/]",
                    title_align="left",
                    border_style="bright_black",
                    box=box.ROUNDED if not compact else box.MINIMAL,
                    padding=(1, 2) if not compact else (0, 1),
                    expand=True,
                ))

            # Status line
            if status_text:
                parts.append(Text(f"  ⏳ {status_text}", style=STYLE_STATUS))

            return Group(*parts) if parts else Text("⏳ Thinking...", style=STYLE_STATUS)

        try:
            with Live(
                _build_display(),
                console=self.console,
                refresh_per_second=STREAM_REFRESH_RATE,
                vertical_overflow="visible",
            ) as live:
                for event in self.agent.process_streaming(user_input):
                    cols, _ = get_terminal_size()
                    compact = cols < 60
                    etype = event['type']

                    if etype == 'status':
                        status_text = event['text']
                        live.update(_build_display())

                    elif etype == 'content_delta':
                        current_text += event['content']
                        status_text = ''  # Clear status while streaming
                        live.update(_build_display())

                    elif etype == 'content_done':
                        # Finalize text into a proper panel with Markdown + blocks
                        if current_text:
                            final_panel = self._render_final_text_panel(current_text, cols, compact)
                            display_parts.append(final_panel)
                            current_text = ''
                        live.update(_build_display())

                    elif etype == 'tool_exec':
                        status_text = f"🔧 {event['name']}..."
                        tool_panel = render_tool_call(
                            event['name'], event['args'], width=cols
                        )
                        display_parts.append(tool_panel)
                        live.update(_build_display())

                    elif etype == 'tool_result':
                        status_text = ''
                        result_panel = render_tool_result(
                            event['name'],
                            event['args'],
                            event['result'],
                            width=cols,
                        )
                        display_parts.append(result_panel)
                        live.update(_build_display())

                    elif etype == 'error':
                        status_text = ''
                        display_parts.append(Panel(
                            Text(event['text'], style=STYLE_ERROR),
                            border_style="red",
                            box=box.ROUNDED if not compact else box.MINIMAL,
                            expand=True,
                        ))
                        live.update(_build_display())

                    elif etype == 'done':
                        status_text = ''
                        live.update(_build_display())

        except KeyboardInterrupt:
            self.console.print("\n[dim]Response interrupted.[/dim]")

        self.console.print()

    # ── Renderable builders ──────────────────────────────────────────

    def _render_partial_block_content(self, text: str, cols: int):
        """Render text that contains an incomplete block (during streaming)."""
        finished = strip_incomplete_block(text)
        parts = []

        if finished.strip():
            segments = parse_blocks(finished)
            for kind, payload in segments:
                if kind == 'text':
                    parts.append(Text(payload))
                elif kind == 'patch':
                    parts.append(render_patch_block(payload, width=cols))
                elif kind == 'tool_call':
                    parts.append(render_tool_call_block(payload, width=cols))
                elif kind == 'tool_result':
                    parts.append(render_tool_result_block(payload, width=cols))

        # Streaming indicator for the open block
        incomplete_type = has_incomplete_block(text)
        partial = get_incomplete_block_content(text)
        parts.append(render_streaming_indicator(incomplete_type, partial_content=partial))

        return Group(*parts) if parts else Text("⏳ receiving...", style="dim italic")

    def _render_final_text_panel(self, text: str, cols: int, compact: bool) -> Panel:
        """Render finalized text with Markdown + PATCH/TOOL blocks."""
        segments = parse_blocks(text)

        # Fast path: plain text only
        if len(segments) == 1 and segments[0][0] == 'text':
            try:
                content = Markdown(text)
            except Exception:
                content = Text(text)
        else:
            parts = []
            for kind, payload in segments:
                if kind == 'text':
                    try:
                        parts.append(Markdown(payload))
                    except Exception:
                        parts.append(Text(payload))
                elif kind == 'patch':
                    parts.append(render_patch_block(payload, width=cols))
                elif kind == 'tool_call':
                    parts.append(render_tool_call_block(payload, width=cols))
                elif kind == 'tool_result':
                    parts.append(render_tool_result_block(payload, width=cols))
            content = Group(*parts) if len(parts) > 1 else (parts[0] if parts else Text(text))

        return Panel(
            content,
            title="[bold bright_cyan]Assistant[/]",
            title_align="left",
            border_style="bright_black",
            box=box.ROUNDED if not compact else box.MINIMAL,
            padding=(1, 2) if not compact else (0, 1),
            expand=True,
        )

    # ── Status callback ──────────────────────────────────────────────

    def _show_status(self, text: str):
        """Update status display (callback from agent, sync mode only)."""
        self._status_text = text
        if hasattr(self, '_current_status') and self._current_status:
            self._current_status.update(f"[cyan]{text}")
        if self.verbose:
            self.console.print(f"  [{STYLE_STATUS}]{text}[/]")

    # ── Banner ───────────────────────────────────────────────────────

    def _show_banner(self):
        """Display welcome banner - responsive to terminal width."""
        cols, _ = get_terminal_size()

        title = f"🤖 {APP_NAME} v{APP_VERSION}"
        subtitle = f"Model: {self.agent.client.model}"
        stream_indicator = f"Streaming: {'ON' if self.streaming else 'OFF'}"

        if cols >= 60:
            banner_lines = [
                f"[{STYLE_HEADER}]{title}[/]",
                f"[{STYLE_MUTED}]{subtitle}[/]",
                f"[{STYLE_MUTED}]{stream_indicator}[/]",
                "",
                f"[{STYLE_MUTED}]Type a message or /help for commands[/]",
            ]
            self.console.print(Panel(
                "\n".join(banner_lines),
                border_style="cyan",
                box=box.ROUNDED,
                padding=(1, 2),
            ))
        else:
            # Compact banner for narrow terminals
            self.console.print(f"[{STYLE_HEADER}]{title}[/]")
            self.console.print(f"[{STYLE_MUTED}]{subtitle} | {stream_indicator} | /help[/]")

        self.console.print()

    # ── Response display (non-streaming path) ────────────────────────

    def _display_response(self, response: str):
        """Display AI response with Markdown + PATCH block rendering."""
        cols, _ = get_terminal_size()
        compact = cols < 60

        if not response:
            return

        # Parse for structured blocks
        segments = parse_blocks(response)

        # Simple text only — render as Markdown
        if len(segments) == 1 and segments[0][0] == 'text':
            try:
                md = Markdown(response)
                if not compact:
                    self.console.print(Panel(
                        md,
                        title="[bold bright_cyan]Assistant[/]",
                        title_align="left",
                        border_style="bright_black",
                        box=box.ROUNDED,
                        padding=(1, 2),
                        expand=True,
                    ))
                else:
                    self.console.print(f"[{STYLE_HEADER}]Assistant:[/]")
                    self.console.print(md)
            except Exception:
                self.console.print(f"[{STYLE_HEADER}]Assistant:[/]")
                self.console.print(response)

            self.console.print()
            return

        # Has structured blocks — render each segment
        parts = []
        for kind, payload in segments:
            if kind == 'text':
                try:
                    parts.append(Markdown(payload))
                except Exception:
                    parts.append(Text(payload))
            elif kind == 'patch':
                parts.append(render_patch_block(payload, width=cols))
            elif kind == 'tool_call':
                parts.append(render_tool_call_block(payload, width=cols))
            elif kind == 'tool_result':
                parts.append(render_tool_result_block(payload, width=cols))

        content = Group(*parts) if len(parts) > 1 else (parts[0] if parts else Text(response))

        if not compact:
            self.console.print(Panel(
                content,
                title="[bold bright_cyan]Assistant[/]",
                title_align="left",
                border_style="bright_black",
                box=box.ROUNDED,
                padding=(1, 2),
                expand=True,
            ))
        else:
            self.console.print(f"[{STYLE_HEADER}]Assistant:[/]")
            self.console.print(content)

        self.console.print()

    # ── Command handlers ─────────────────────────────────────────────

    def _handle_command(self, cmd: str) -> bool:
        """Handle slash commands. Returns True if command was handled."""
        parts = cmd.strip().split(maxsplit=1)
        command = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ''

        handlers = {
            '/help': self._cmd_help,
            '/h': self._cmd_help,
            '/clear': self._cmd_clear,
            '/cls': self._cmd_clear,
            '/model': self._cmd_model,
            '/m': self._cmd_model,
            '/history': self._cmd_history,
            '/session': self._cmd_session_info,
            '/sessions': self._cmd_sessions_list,
            '/new': self._cmd_new_session,
            '/load': self._cmd_load_session,
            '/save': self._cmd_save_session,
            '/tools': self._cmd_tools,
            '/verbose': self._cmd_verbose,
            '/v': self._cmd_verbose,
            '/stats': self._cmd_stats,
            '/cd': self._cmd_cd,
            '/stream': self._cmd_stream,
            '/s': self._cmd_stream,
            '/exit': self._cmd_exit,
            '/quit': self._cmd_exit,
            '/q': self._cmd_exit,
        }

        handler = handlers.get(command)
        if handler:
            handler(arg)
            return True

        # Unknown command
        self.console.print(f"[{STYLE_ERROR}]Unknown command: {command}[/]")
        self.console.print(f"[{STYLE_MUTED}]Type /help for available commands[/]")
        return True

    def _cmd_help(self, _arg=''):
        """Show help."""
        cols, _ = get_terminal_size()

        table = Table(
            title="Commands",
            box=box.SIMPLE_HEAD if cols >= 60 else box.MINIMAL,
            show_edge=cols >= 60,
            expand=True,
            title_style=STYLE_HEADER,
        )
        table.add_column("Command", style="bold cyan", min_width=12)
        table.add_column("Description", style="white")

        commands = [
            ("/help, /h", "Show this help message"),
            ("/clear, /cls", "Clear screen"),
            ("/model [name]", "Show or change the AI model"),
            ("/tools", "List available tools"),
            ("/history", "Show conversation history"),
            ("/new", "Start new conversation session"),
            ("/save", "Save current session"),
            ("/sessions", "List saved sessions"),
            ("/load <id>", "Load a saved session"),
            ("/session", "Show current session info"),
            ("/stream, /s", "Toggle streaming mode"),
            ("/verbose, /v", "Toggle verbose output"),
            ("/stats", "Show API usage statistics"),
            ("/cd <dir>", "Change working directory"),
            ("/exit, /quit, /q", "Exit the agent"),
        ]

        for cmd_name, desc in commands:
            table.add_row(cmd_name, desc)

        self.console.print(table)
        self.console.print(f"\n[{STYLE_MUTED}]Any other input is sent as a prompt to the AI.[/]")
        self.console.print()

    def _cmd_clear(self, _arg=''):
        self.console.clear()
        self._show_banner()

    def _cmd_model(self, arg=''):
        """Show or change model."""
        if not arg:
            self.console.print(f"[{STYLE_HEADER}]Current model:[/] {self.agent.client.model}\n")
            table = Table(title="Available Models", box=box.SIMPLE, expand=True)
            table.add_column("#", style="dim", width=4)
            table.add_column("Model", style="cyan")
            for i, model in enumerate(AVAILABLE_MODELS, 1):
                marker = " ◀" if model == self.agent.client.model else ""
                table.add_row(str(i), f"{model}{marker}")
            self.console.print(table)
            self.console.print(f"\n[{STYLE_MUTED}]Usage: /model <number or name>[/]")
        else:
            arg = arg.strip()
            # Allow selection by number
            if arg.isdigit():
                idx = int(arg) - 1
                if 0 <= idx < len(AVAILABLE_MODELS):
                    arg = AVAILABLE_MODELS[idx]
                else:
                    self.console.print(f"[{STYLE_ERROR}]Invalid model number[/]")
                    return
            self.agent.client.set_model(arg)
            self.console.print(f"[{STYLE_SUCCESS}]Model changed to: {arg}[/]")
        self.console.print()

    def _cmd_history(self, _arg=''):
        """Show conversation history."""
        history = self.agent.history
        if not history:
            self.console.print(f"[{STYLE_MUTED}]No conversation history yet.[/]")
            return

        for msg in history:
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')

            if role == 'user':
                self.console.print(f"[{STYLE_USER}]You:[/] {content[:200]}")
            elif role == 'assistant':
                if msg.get('tool_calls'):
                    tools = [tc['function']['name'] for tc in msg['tool_calls']]
                    self.console.print(f"[{STYLE_TOOL}]Agent:[/] [dim](called: {', '.join(tools)})[/]")
                if content:
                    self.console.print(f"[{STYLE_HEADER}]Agent:[/] {content[:200]}")
            elif role == 'tool':
                self.console.print(f"  [{STYLE_MUTED}]Tool result: {content[:100]}...[/]")

        self.console.print(f"\n[{STYLE_MUTED}]{len(history)} messages total[/]")
        self.console.print()

    def _cmd_session_info(self, _arg=''):
        self.console.print(f"[{STYLE_HEADER}]Session ID:[/]  {self.agent.session_id}")
        self.console.print(f"[{STYLE_HEADER}]Model:[/]       {self.agent.client.model}")
        self.console.print(f"[{STYLE_HEADER}]Messages:[/]    {len(self.agent.messages)}")
        self.console.print(f"[{STYLE_HEADER}]CWD:[/]         {os.getcwd()}")
        self.console.print(f"[{STYLE_HEADER}]Streaming:[/]   {'ON' if self.streaming else 'OFF'}")
        self.console.print()

    def _cmd_sessions_list(self, _arg=''):
        sessions = self.agent.list_sessions()
        if not sessions:
            self.console.print(f"[{STYLE_MUTED}]No saved sessions.[/]")
            return
        table = Table(title="Saved Sessions", box=box.SIMPLE, expand=True)
        table.add_column("ID", style="cyan")
        table.add_column("Created", style="dim")
        table.add_column("Model", style="white")
        table.add_column("Messages", style="yellow", justify="right")
        for s in sessions:
            table.add_row(s['id'], s['created'][:19], s['model'], str(s['messages']))
        self.console.print(table)
        self.console.print()

    def _cmd_new_session(self, _arg=''):
        self.agent.new_session()
        self.console.print(f"[{STYLE_SUCCESS}]New session started: {self.agent.session_id}[/]")
        self.console.print()

    def _cmd_load_session(self, arg=''):
        if not arg:
            self.console.print(f"[{STYLE_ERROR}]Usage: /load <session_id>[/]")
            return
        if self.agent.load_session(arg.strip()):
            self.console.print(f"[{STYLE_SUCCESS}]Session {arg.strip()} loaded ({len(self.agent.messages)} messages)[/]")
        else:
            self.console.print(f"[{STYLE_ERROR}]Session not found: {arg.strip()}[/]")
        self.console.print()

    def _cmd_save_session(self, _arg=''):
        path = self.agent.save_session()
        if path:
            self.console.print(f"[{STYLE_SUCCESS}]Session saved: {path}[/]")
        else:
            self.console.print(f"[{STYLE_MUTED}]Nothing to save.[/]")
        self.console.print()

    def _cmd_tools(self, _arg=''):
        table = Table(title="Available Tools", box=box.SIMPLE, expand=True)
        table.add_column("Tool", style="bold yellow", min_width=14)
        table.add_column("Description", style="white")
        for tool_def in TOOL_DEFINITIONS:
            fn = tool_def['function']
            table.add_row(fn['name'], fn['description'][:100])
        self.console.print(table)
        self.console.print()

    def _cmd_verbose(self, _arg=''):
        self.verbose = not self.verbose
        self.agent.client.verbose = self.verbose
        state = "ON" if self.verbose else "OFF"
        self.console.print(f"[{STYLE_SUCCESS}]Verbose mode: {state}[/]")
        self.console.print()

    def _cmd_stream(self, _arg=''):
        """Toggle streaming mode."""
        self.streaming = not self.streaming
        state = "ON" if self.streaming else "OFF"
        self.console.print(f"[{STYLE_SUCCESS}]Streaming mode: {state}[/]")
        self.console.print()

    def _cmd_stats(self, _arg=''):
        stats = self.agent.client.stats
        self.console.print(f"[{STYLE_HEADER}]API Statistics:[/]")
        self.console.print(f"  Total requests:  {stats['total_requests']}")
        self.console.print(f"  Total errors:    {stats['total_errors']}")
        self.console.print(f"  Current model:   {stats['model']}")
        self.console.print(f"  Streaming:       {'ON' if self.streaming else 'OFF'}")
        self.console.print(f"  Log file:        {LOG_FILE}")
        self.console.print()

    def _cmd_cd(self, arg=''):
        if not arg:
            self.console.print(f"[{STYLE_HEADER}]CWD:[/] {os.getcwd()}")
            return
        try:
            os.chdir(os.path.expanduser(arg.strip()))
            self.console.print(f"[{STYLE_SUCCESS}]Changed to: {os.getcwd()}[/]")
        except Exception as e:
            self.console.print(f"[{STYLE_ERROR}]Error: {str(e)}[/]")
        self.console.print()

    def _cmd_exit(self, _arg=''):
        self._exit()

    def _exit(self):
        """Clean exit."""
        self.agent.save_session()
        self.console.print(f"\n[{STYLE_MUTED}]Session saved. Goodbye! 👋[/]\n")
        sys.exit(0)
