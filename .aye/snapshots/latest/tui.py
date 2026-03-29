"""Terminal UI for AI CLI Agent - responsive, rich-based."""

import os
import sys
import shutil
from typing import Optional
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.markdown import Markdown
from rich.live import Live
from rich.layout import Layout
from rich.style import Style
from rich.columns import Columns
from rich import box
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.formatted_text import HTML
from lib.agent import Agent
from lib.utils import logger, get_terminal_size
from lib.tools import TOOL_DEFINITIONS
from conf.conf import (
    APP_NAME,
    APP_VERSION,
    AVAILABLE_MODELS,
    PROJECT_ROOT,
    LOG_FILE,
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
        self._status_text = ""

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
                with self.console.status("[cyan]Thinking...", spinner="dots") as status:
                    self._current_status = status
                    response = self.agent.process(user_input)
                    self._current_status = None

                # Display response
                self._display_response(response)

            except KeyboardInterrupt:
                self.console.print("\n[dim]Use /exit to quit, or Ctrl+D[/dim]")
                continue
            except EOFError:
                self._exit()
                break
            except Exception as e:
                logger.error(f"TUI error: {e}", exc_info=True)
                self.console.print(f"[{STYLE_ERROR}]Error: {str(e)}[/]")

    def _show_status(self, text: str):
        """Update status display (callback from agent)."""
        self._status_text = text
        if hasattr(self, '_current_status') and self._current_status:
            self._current_status.update(f"[cyan]{text}")
        if self.verbose:
            self.console.print(f"  [{STYLE_STATUS}]{text}[/]")

    def _show_banner(self):
        """Display welcome banner - responsive to terminal width."""
        cols, _ = get_terminal_size()

        title = f"🤖 {APP_NAME} v{APP_VERSION}"
        subtitle = f"Model: {self.agent.client.model}"

        if cols >= 60:
            banner_lines = [
                f"[{STYLE_HEADER}]{title}[/]",
                f"[{STYLE_MUTED}]{subtitle}[/]",
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
            self.console.print(f"[{STYLE_MUTED}]{subtitle} | /help[/]")

        self.console.print()

    def _display_response(self, response: str):
        """Display AI response with markdown rendering."""
        cols, _ = get_terminal_size()

        if not response:
            return

        try:
            # Try rendering as markdown
            md = Markdown(response)
            if cols >= 60:
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
            # Fallback to plain text
            self.console.print(f"[{STYLE_HEADER}]Assistant:[/]")
            self.console.print(response)

        self.console.print()

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

    def _cmd_stats(self, _arg=''):
        stats = self.agent.client.stats
        self.console.print(f"[{STYLE_HEADER}]API Statistics:[/]")
        self.console.print(f"  Total requests:  {stats['total_requests']}")
        self.console.print(f"  Total errors:    {stats['total_errors']}")
        self.console.print(f"  Current model:   {stats['model']}")
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
