"""Terminal UI for ForgeCLI """

import os
import sys
import json
import math
import tempfile
import time
from typing import List
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.markdown import Markdown
from rich.live import Live
from rich import box
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory, InMemoryHistory
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
    AI_NAME,
    APP_VERSION,
    AVAILABLE_MODELS,
    MODEL_META,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TOP_P,
    PROJECT_ROOT,
    LOG_FILE,
    SETTINGS_FILE,
    API_ENDPOINT,
    STREAM_ENABLED,
    STREAM_REFRESH_RATE,
    WEB_PORT,
    CUSTOM_MODELS,
    configure_custom_models,
    model_is_chat,
    model_supports_tools,
    valid_model_id,
)

STYLE_HEADER = "bold bright_cyan"
STYLE_USER = "bold green"
STYLE_ASSISTANT = "white"
STYLE_TOOL = "bold yellow"
STYLE_ERROR = "bold red"
STYLE_STATUS = "dim cyan"
STYLE_MUTED = "dim white"
STYLE_SUCCESS = "bold green"
STYLE_DIM = "dim"

SPINNER = ["\u00b7  ", "\u00b7\u00b7 ", "\u00b7\u00b7\u00b7", " \u00b7\u00b7", "  \u00b7", "   "]


def _load_persisted_settings() -> dict:
    """Load settings from disk."""
    defaults = {
        'model': DEFAULT_MODEL,
        'auto_model': True,
        'temperature': DEFAULT_TEMPERATURE,
        'max_tokens': DEFAULT_MAX_TOKENS,
        'top_p': DEFAULT_TOP_P,
        'system_prompt': '',
        'theme': 'dark',
        'api_key': '',
    }
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, encoding='utf-8') as f:
                saved = json.load(f)
            if isinstance(saved, dict):
                defaults.update(saved)
            else:
                logger.warning("Ignoring settings file: expected a JSON object")
        except FileNotFoundError:
            pass
        except (OSError, UnicodeError, json.JSONDecodeError):
            logger.exception("Could not load persisted settings")
    return defaults


def _save_persisted_settings(settings: dict):
    """Save settings to disk."""
    temp_path = None
    try:
        settings_dir = os.path.dirname(SETTINGS_FILE)
        fd, temp_path = tempfile.mkstemp(prefix='.forge_settings.', dir=settings_dir)
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(settings, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        if os.name != 'nt':
            os.chmod(temp_path, 0o600)
        os.replace(temp_path, SETTINGS_FILE)
        return True
    except (OSError, TypeError, ValueError):
        logger.exception("Failed to save settings")
        return False
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                logger.exception("Failed to remove temporary settings file")


def _apply_settings_to_agent(agent, settings: dict):
    """Apply persisted settings to an agent."""
    configure_custom_models(settings.get('custom_models', {}))
    # Model
    saved_model = settings.get('model', DEFAULT_MODEL)
    if saved_model and saved_model != 'auto':
        if model_is_chat(saved_model):
            agent.client.set_model(saved_model)
        else:
            logger.warning("Ignoring invalid persisted model setting")
    # Temperature
    temp = settings.get('temperature')
    if temp is not None:
        try:
            temp = float(temp)
            if math.isfinite(temp) and 0 <= temp <= 2:
                agent.client.temperature = temp
            else:
                logger.warning("Ignoring invalid temperature setting: expected 0..2")
        except (TypeError, ValueError):
            logger.warning("Ignoring invalid temperature setting")
    # Max tokens
    mt = settings.get('max_tokens')
    if mt is not None:
        try:
            mt = int(mt)
            if mt > 0:
                agent.client.max_tokens = mt
            else:
                logger.warning("Ignoring invalid max_tokens setting: expected a positive integer")
        except (TypeError, ValueError):
            logger.warning("Ignoring invalid max_tokens setting")
    # Top P
    tp = settings.get('top_p')
    if tp is not None:
        try:
            tp = float(tp)
            if math.isfinite(tp) and 0 <= tp <= 1:
                agent.client.top_p = tp
            else:
                logger.warning("Ignoring invalid top_p setting: expected 0..1")
        except (TypeError, ValueError):
            logger.warning("Ignoring invalid top_p setting")
    # API key
    api_key = settings.get('api_key', '') or os.environ.get('NVIDIA_API_KEY', '')
    if isinstance(api_key, str) and api_key:
        agent.client.api_key = api_key
        agent.client.session.headers['Authorization'] = f'Bearer {api_key}'


class AgentTUI:
    """Rich-based terminal UI for the ForgeAI agent."""

    def __init__(self):
        self.console = Console()
        self.agent = Agent(on_status=self._show_status)
        self.verbose = False
        self.streaming = STREAM_ENABLED
        self._status_text = ""
        self._current_status = None
        self._spin_tick = 0

        hist_path = os.path.join(PROJECT_ROOT, '.prompt_history')
        try:
            with open(hist_path, 'a', encoding='utf-8'):
                pass
            history = FileHistory(hist_path)
        except OSError:
            logger.exception("Could not open prompt history; using in-memory history")
            history = InMemoryHistory()
        self.prompt_session = PromptSession(
            history=history,
            auto_suggest=AutoSuggestFromHistory(),
            enable_history_search=True,
        )

        # Load and apply persisted settings on startup
        self._load_and_apply_settings()

    def _load_and_apply_settings(self):
        """Load persisted settings and apply to agent on startup."""
        settings = _load_persisted_settings()
        _apply_settings_to_agent(self.agent, settings)
        logger.info(
            f"TUI loaded settings: model={self.agent.client.model}, "
            f"temp={self.agent.client.temperature}, "
            f"max_tokens={self.agent.client.max_tokens}, "
            f"api_key={'set' if self.agent.client.api_key else 'not set'}"
        )

    def _available_models(self):
        return list(dict.fromkeys([*AVAILABLE_MODELS, *CUSTOM_MODELS]))

    def run(self):
        """Main run loop."""
        self._show_banner()

        while True:
            try:
                user_input = self.prompt_session.prompt(
                    HTML('<ansigreen><b>\u276f </b></ansigreen>'),
                ).strip()

                if not user_input:
                    continue

                if user_input.startswith('/'):
                    if self._handle_command(user_input):
                        continue

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

    # \u2500\u2500 Synchronous processing \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

    def _process_sync(self, user_input: str):
        try:
            with self.console.status("[dim cyan]thinking", spinner="dots") as status:
                self._current_status = status
                response = self.agent.process(user_input)
        finally:
            self._current_status = None

        self._display_response(response)

    # \u2500\u2500 Streaming processing \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

    def _process_streaming(self, user_input: str):
        display_parts: List = []
        current_text = ''
        status_text = ''
        self._spin_tick = 0
        cols, _ = get_terminal_size()
        compact = cols < 60
        last_refresh = 0.0
        pending_content = False
        refresh_interval = 1 / max(1, STREAM_REFRESH_RATE)

        def _build_display() -> Group:
            self._spin_tick += 1
            parts = list(display_parts)

            if current_text:
                incomplete = has_incomplete_block(current_text)
                if incomplete:
                    content = self._render_partial_block_content(current_text, cols)
                else:
                    content = Text(current_text + " \u2588", style=STYLE_ASSISTANT)

                parts.append(Panel(
                    content,
                    title=f"[bold bright_cyan]{AI_NAME}[/]",
                    title_align="left",
                    border_style="bright_black",
                    box=box.ROUNDED if not compact else box.MINIMAL,
                    padding=(1, 2) if not compact else (0, 1),
                    expand=True,
                ))

            if status_text:
                frame = SPINNER[self._spin_tick % len(SPINNER)]
                parts.append(Text(f"  {frame} {status_text}", style=STYLE_DIM))
            elif not parts:
                frame = SPINNER[self._spin_tick % len(SPINNER)]
                parts.append(Text(f"  {frame} thinking", style=STYLE_DIM))

            return Group(*parts) if parts else Text("  \u00b7\u00b7\u00b7 thinking", style=STYLE_DIM)

        try:
            with Live(
                _build_display(),
                console=self.console,
                auto_refresh=False,
                vertical_overflow="visible",
            ) as live:
                for event in self.agent.process_streaming(user_input):
                    etype = event.get('type', '')

                    if etype == 'status':
                        status_text = event['text']
                        live.update(_build_display())
                        live.refresh()

                    elif etype == 'content_delta':
                        current_text += event['content']
                        status_text = 'Thinking...'
                        pending_content = True
                        now = time.monotonic()
                        if now - last_refresh >= refresh_interval:
                            live.update(_build_display())
                            live.refresh()
                            last_refresh = now
                            pending_content = False

                    elif etype == 'content_done':
                        status_text = ''
                        if not current_text and event.get('content'):
                            current_text = event['content']
                        if pending_content:
                            live.update(_build_display())
                            live.refresh()
                            pending_content = False
                        if current_text:
                            final_panel = self._render_final_text_panel(current_text, cols, compact)
                            display_parts.append(final_panel)
                            current_text = ''
                        live.update(_build_display())
                        live.refresh()

                    elif etype == 'tool_exec':
                        status_text = f"{event['name']}"
                        tool_panel = render_tool_call(
                            event['name'], event['args'], width=cols
                        )
                        display_parts.append(tool_panel)
                        live.update(_build_display())
                        live.refresh()

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
                        live.refresh()

                    elif etype == 'error':
                        status_text = ''
                        display_parts.append(Panel(
                            Text(event['text'], style=STYLE_ERROR),
                            border_style="red",
                            box=box.ROUNDED if not compact else box.MINIMAL,
                            expand=True,
                        ))
                        live.update(_build_display())
                        live.refresh()

                    elif etype == 'warning':
                        status_text = ''
                        display_parts.append(Panel(
                            Text(event.get('message', 'Tool calling was disabled for this request.'),
                                 style="yellow"),
                            border_style="yellow",
                            box=box.ROUNDED if not compact else box.MINIMAL,
                            expand=True,
                        ))
                        live.update(_build_display())
                        live.refresh()

                    elif etype == 'done':
                        status_text = ''
                        live.update(_build_display())
                        live.refresh()

        except KeyboardInterrupt:
            self.console.print("\n[dim]Response interrupted.[/dim]")

        self.console.print()

    # \u2500\u2500 Renderable builders \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

    def _render_partial_block_content(self, text: str, cols: int):
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

        incomplete_type = has_incomplete_block(text)
        partial = get_incomplete_block_content(text)
        parts.append(render_streaming_indicator(incomplete_type, partial_content=partial))

        return Group(*parts) if parts else Text("\u00b7\u00b7\u00b7 receiving", style="dim italic")

    def _render_final_text_panel(self, text: str, cols: int, compact: bool) -> Panel:
        segments = parse_blocks(text)
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
            title=f"[bold bright_cyan]{AI_NAME}[/]",
            title_align="left",
            border_style="bright_black",
            box=box.ROUNDED if not compact else box.MINIMAL,
            padding=(1, 2) if not compact else (0, 1),
            expand=True,
        )

    # \u2500\u2500 Status callback \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

    def _show_status(self, text: str):
        self._status_text = text
        if hasattr(self, '_current_status') and self._current_status:
            self._current_status.update(f"[dim cyan]{text}")
        if self.verbose:
            self.console.print(f"  [{STYLE_STATUS}]{text}[/]")

    # \u2500\u2500 Banner \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

    def _show_banner(self):
        cols, _ = get_terminal_size()
        title = f"\U0001f525 {APP_NAME} v{APP_VERSION}"
        subtitle = f"Model: {self.agent.client.model}"
        tool_indicator = f"Tools: {'ON' if self.agent.client.tools_supported else 'OFF for this model'}"
        stream_indicator = f"Streaming: {'ON' if self.streaming else 'OFF'}"
        key_status = "API Key: set" if self.agent.client.api_key else "API Key: [red]not set[/red]"

        if cols >= 60:
            banner_lines = [
                f"[{STYLE_HEADER}]{title}[/]",
                f"[{STYLE_MUTED}]{subtitle}[/]",
                f"[{STYLE_MUTED}]{tool_indicator}[/]",
                f"[{STYLE_MUTED}]{stream_indicator}[/]",
                f"[{STYLE_MUTED}]{key_status}[/]",
                f"[{STYLE_MUTED}]Workspace: {os.getcwd()}[/]",
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
            self.console.print(f"[{STYLE_HEADER}]{title}[/]")
            self.console.print(
                f"[{STYLE_MUTED}]{subtitle} | {tool_indicator} | "
                f"{stream_indicator} | /help[/]"
            )

        self.console.print()

    # \u2500\u2500 Response display \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

    def _display_response(self, response: str):
        cols, _ = get_terminal_size()
        compact = cols < 60
        if not response:
            return

        segments = parse_blocks(response)

        if len(segments) == 1 and segments[0][0] == 'text':
            try:
                md = Markdown(response)
                if not compact:
                    self.console.print(Panel(
                        md,
                        title=f"[bold bright_cyan]{AI_NAME}[/]",
                        title_align="left",
                        border_style="bright_black",
                        box=box.ROUNDED,
                        padding=(1, 2),
                        expand=True,
                    ))
                else:
                    self.console.print(f"[{STYLE_HEADER}]{AI_NAME}:[/]")
                    self.console.print(md)
            except Exception:
                self.console.print(f"[{STYLE_HEADER}]{AI_NAME}:[/]")
                self.console.print(response)
            self.console.print()
            return

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
                title=f"[bold bright_cyan]{AI_NAME}[/]",
                title_align="left",
                border_style="bright_black",
                box=box.ROUNDED,
                padding=(1, 2),
                expand=True,
            ))
        else:
            self.console.print(f"[{STYLE_HEADER}]{AI_NAME}:[/]")
            self.console.print(content)
        self.console.print()

    # \u2500\u2500 Command handlers \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

    def _handle_command(self, cmd: str) -> bool:
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
            '/api': self._cmd_api,
            '/exit': self._cmd_exit,
            '/quit': self._cmd_exit,
            '/q': self._cmd_exit,
        }

        handler = handlers.get(command)
        if handler:
            handler(arg)
            return True

        self.console.print(f"[{STYLE_ERROR}]Unknown command: {command}[/]")
        self.console.print(f"[{STYLE_MUTED}]Type /help for available commands[/]")
        return True

    def _cmd_help(self, _arg=''):
        cols, _ = get_terminal_size()
        table = Table(
            title=f"{APP_NAME} Commands",
            box=box.SIMPLE_HEAD if cols >= 60 else box.MINIMAL,
            show_edge=cols >= 60,
            expand=True,
            title_style=STYLE_HEADER,
        )
        table.add_column("Command", style="bold cyan", min_width=18)
        table.add_column("Description", style="white")

        commands = [
            ("/help, /h", "Show this help message"),
            ("/clear, /cls", "Clear screen"),
            ("/model [name|#]", "Show or change the model and tool support"),
            ("/model add <org/model> [tools=yes|no]", "Register a custom NVIDIA model"),
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
            ("/cd <dir>", "Change workspace for relative file and shell tools"),
            ("/api <subcmd>", "API management (info/models/settings/health/key)"),
            ("/exit, /quit, /q", "Exit ForgeCLI"),
        ]

        for cmd_name, desc in commands:
            table.add_row(cmd_name, desc)

        self.console.print(table)
        self.console.print(f"\n[{STYLE_MUTED}]Any other input is sent as a prompt to {AI_NAME}.[/]")
        self.console.print()

    def _cmd_clear(self, _arg=''):
        self.console.clear()
        self._show_banner()

    def _cmd_model(self, arg=''):
        if arg.strip().lower().startswith('add '):
            self._cmd_model_add(arg.strip()[4:])
            return

        if not arg:
            self.console.print(f"[{STYLE_HEADER}]Current model:[/] {self.agent.client.model}\n")
            table = Table(title="Available Models", box=box.SIMPLE, expand=True)
            table.add_column("#", style="dim", width=4)
            table.add_column("Model", style="cyan")
            table.add_column("Tags", style="dim")
            table.add_column("Tools", style="green")
            for i, model in enumerate(self._available_models(), 1):
                marker = " \u25c0" if model == self.agent.client.model else ""
                meta = MODEL_META.get(model, {})
                tags = ', '.join(meta.get('tags', [])[:3])
                tool_status = "yes" if model_supports_tools(model) else "no"
                table.add_row(str(i), f"{model}{marker}", tags, tool_status)
            self.console.print(table)
            self.console.print(
                f"\n[{STYLE_MUTED}]Usage: /model <number or name> | "
                "/model add <org/model> [tools=yes|no][/]"
            )
        else:
            arg = arg.strip()
            if arg.isdigit():
                idx = int(arg) - 1
                models = self._available_models()
                if 0 <= idx < len(models):
                    arg = models[idx]
                else:
                    self.console.print(f"[{STYLE_ERROR}]Invalid model number[/]")
                    return
            if not arg or not model_is_chat(arg):
                self.console.print(
                    f"[{STYLE_ERROR}]That model cannot be used for chat. "
                    "Choose a chat model from /model.[/]"
                )
                return
            self.agent.client.set_model(arg)
            self.agent._update_system_prompt()
            # Persist model change
            settings = _load_persisted_settings()
            settings['model'] = arg
            if _save_persisted_settings(settings):
                self.console.print(f"[{STYLE_SUCCESS}]Model changed to: {arg} (saved)[/]")
            else:
                self.console.print(
                    f"[{STYLE_ERROR}]Model changed for this session, but could not be saved.[/]"
                )
            if not self.agent.client.tools_supported:
                self.console.print(
                    f"[{STYLE_ERROR}]This model is not known to support tools. "
                    "Choose a model marked 'yes' to enable them.[/]"
                )
        self.console.print()

    def _cmd_model_add(self, arg: str):
        parts = arg.split()
        if not parts or len(parts) > 2:
            self.console.print(
                f"[{STYLE_ERROR}]Usage: /model add <org/model> [tools=yes|no][/]"
            )
            return

        model = parts[0].strip()
        options = {'tools': 'no'}
        if len(parts) == 2:
            key, separator, value = parts[1].partition('=')
            if not separator or key.lower() != 'tools' or value.lower() not in ('yes', 'no'):
                self.console.print(
                    f"[{STYLE_ERROR}]Use tools=yes or tools=no. "
                    "Only set yes when the model endpoint supports function calling.[/]"
                )
                return
            options['tools'] = value.lower()

        if not valid_model_id(model):
            self.console.print(
                f"[{STYLE_ERROR}]Invalid model ID. Expected an NVIDIA model ID like org/model.[/]"
            )
            return
        if model in AVAILABLE_MODELS:
            self.console.print(
                f"[{STYLE_ERROR}]That model is already in the built-in model list.[/]"
            )
            return
        if model in CUSTOM_MODELS:
            self.console.print(f"[{STYLE_ERROR}]That custom model is already registered.[/]")
            return

        settings = _load_persisted_settings()
        custom_models = settings.get('custom_models', {})
        if not isinstance(custom_models, dict):
            custom_models = {}
        custom_models[model] = {'tools_supported': options['tools'] == 'yes'}
        settings['custom_models'] = custom_models
        if not _save_persisted_settings(settings):
            self.console.print(f"[{STYLE_ERROR}]Could not save custom model.[/]")
            self.console.print()
            return

        configure_custom_models(custom_models)
        self.console.print(f"[{STYLE_SUCCESS}]Added custom model: {model}[/]")
        if options['tools'] == 'yes':
            self.console.print(
                f"[{STYLE_MUTED}]Tool support is enabled in ForgeCLI. "
                "The provider must also support OpenAI-compatible function calling.[/]"
            )
        self.console.print()

    def _cmd_history(self, _arg=''):
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
                    tools = [tc.get('function', {}).get('name', '?') for tc in msg['tool_calls']]
                    self.console.print(f"[{STYLE_TOOL}]{AI_NAME}:[/] [dim](called: {', '.join(tools)})[/]")
                if content:
                    self.console.print(f"[{STYLE_HEADER}]{AI_NAME}:[/] {content[:200]}")
            elif role == 'tool':
                self.console.print(f"  [{STYLE_MUTED}]Tool: {content[:100]}...[/]")
        self.console.print(f"\n[{STYLE_MUTED}]{len(history)} messages total[/]")
        self.console.print()

    def _cmd_session_info(self, _arg=''):
        self.console.print(f"[{STYLE_HEADER}]Session ID:[/]  {self.agent.session_id}")
        self.console.print(f"[{STYLE_HEADER}]Model:[/]       {self.agent.client.model}")
        self.console.print(f"[{STYLE_HEADER}]Messages:[/]    {len(self.agent.messages)}")
        self.console.print(f"[{STYLE_HEADER}]CWD:[/]         {os.getcwd()}")
        self.console.print(f"[{STYLE_HEADER}]Streaming:[/]   {'ON' if self.streaming else 'OFF'}")
        self.console.print(f"[{STYLE_HEADER}]API Key:[/]     {'set' if self.agent.client.api_key else '[red]not set[/red]'}")
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
        table.add_column("Msgs", style="yellow", justify="right")
        for s in sessions:
            table.add_row(s['id'], s['created'][:19], s['model'], str(s['messages']))
        self.console.print(table)
        self.console.print()

    def _cmd_new_session(self, _arg=''):
        self.agent.new_session()
        self.console.print(f"[{STYLE_SUCCESS}]New session: {self.agent.session_id}[/]")
        self.console.print()

    def _cmd_load_session(self, arg=''):
        if not arg:
            self.console.print(f"[{STYLE_ERROR}]Usage: /load <session_id>[/]")
            return
        if self.agent.load_session(arg.strip()):
            self.console.print(f"[{STYLE_SUCCESS}]Session {arg.strip()} loaded ({len(self.agent.messages)} msgs)[/]")
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
        enabled = self.agent.client.tools_supported
        state = "available" if enabled else "disabled for the current model"
        table = Table(title=f"{AI_NAME} Tools ({state})", box=box.SIMPLE, expand=True)
        table.add_column("Tool", style="bold yellow", min_width=14)
        table.add_column("Description", style="white")
        for tool_def in TOOL_DEFINITIONS:
            fn = tool_def['function']
            table.add_row(fn['name'], fn['description'][:100])
        self.console.print(table)
        if not enabled:
            self.console.print(
                f"[{STYLE_ERROR}]Current model {self.agent.client.model} is not "
                "configured for tool calling. Use /model and select a model marked yes.[/]"
            )
        self.console.print()

    def _cmd_verbose(self, _arg=''):
        self.verbose = not self.verbose
        self.agent.client.verbose = self.verbose
        state = "ON" if self.verbose else "OFF"
        self.console.print(f"[{STYLE_SUCCESS}]Verbose mode: {state}[/]")
        self.console.print()

    def _cmd_stream(self, _arg=''):
        self.streaming = not self.streaming
        state = "ON" if self.streaming else "OFF"
        self.console.print(f"[{STYLE_SUCCESS}]Streaming mode: {state}[/]")
        self.console.print()

    def _cmd_stats(self, _arg=''):
        stats = self.agent.client.stats
        self.console.print(f"[{STYLE_HEADER}]{AI_NAME} API Statistics:[/]")
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

    # \u2500\u2500 /api subcommands \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

    def _cmd_api(self, arg=''):
        sub_parts = arg.strip().split(maxsplit=1)
        subcmd = sub_parts[0].lower() if sub_parts else 'help'
        sub_arg = sub_parts[1] if len(sub_parts) > 1 else ''

        api_handlers = {
            'help': self._api_help,
            'info': self._api_info,
            'models': self._api_models,
            'settings': self._api_settings,
            'health': self._api_health,
            'key': self._api_key,
        }

        handler = api_handlers.get(subcmd)
        if handler is None:
            self.console.print(f"[{STYLE_ERROR}]Unknown /api subcommand: {subcmd}[/]")
            self.console.print("[dim]Use /api help to see valid commands.[/]\n")
            return
        handler(sub_arg)

    def _api_help(self, _arg=''):
        self.console.print(f"[{STYLE_HEADER}]/api subcommands:[/]")
        cmds = [
            ("info", "Show API connection info"),
            ("models", "List all models with details"),
            ("settings [key=val]", "View or update persisted settings"),
            ("health", "Check API endpoint connectivity"),
            ("key [api-key]", "View or set NVIDIA API key"),
        ]
        for c, d in cmds:
            self.console.print(f"  [bold cyan]/api {c:<22}[/] {d}")
        self.console.print()

    def _api_info(self, _arg=''):
        self.console.print(f"[{STYLE_HEADER}]API Connection Info:[/]")
        self.console.print(f"  Endpoint:      {API_ENDPOINT}")
        self.console.print(f"  Model:         {self.agent.client.model}")
        self.console.print(f"  API Key:       {'set' if self.agent.client.api_key else '[red]not set[/red]'}")
        self.console.print(f"  Temperature:   {self.agent.client.temperature}")
        self.console.print(f"  Max tokens:    {self.agent.client.max_tokens}")
        self.console.print(f"  Top P:         {self.agent.client.top_p}")
        self.console.print(f"  Settings file: {SETTINGS_FILE}")
        self.console.print(f"  Web UI:        http://localhost:{WEB_PORT}")
        self.console.print()

    def _api_models(self, _arg=''):
        cols, _ = get_terminal_size()
        table = Table(title="NVIDIA Free Models", box=box.SIMPLE, expand=True)
        table.add_column("#", style="dim", width=3)
        table.add_column("Model", style="cyan", min_width=20)
        table.add_column("Tags", style="yellow")
        table.add_column("Tier", style="dim")
        table.add_column("Description", style="white")

        for i, model in enumerate(self._available_models(), 1):
            meta = MODEL_META.get(model, {})
            tags = ', '.join(meta.get('tags', [])[:3])
            tier = meta.get('tier', '')
            desc = meta.get('desc', 'Custom NVIDIA model')[:50]
            marker = " \u25c0" if model == self.agent.client.model else ""
            name = model.split('/')[-1]
            table.add_row(str(i), f"{name}{marker}", tags, tier, desc)

        self.console.print(table)
        self.console.print(f"\n[{STYLE_MUTED}]{len(AVAILABLE_MODELS)} models available[/]")
        self.console.print()

    def _api_settings(self, arg=''):
        settings = _load_persisted_settings()

        if not arg:
            self.console.print(f"[{STYLE_HEADER}]Persisted Settings ({SETTINGS_FILE}):[/]")
            for key, val in settings.items():
                display_val = '***' if key == 'api_key' and val else val
                self.console.print(f"  [bold]{key}:[/] {display_val}")
            self.console.print(f"\n[{STYLE_MUTED}]Update: /api settings key=value[/]")
            self.console.print()
            return

        if '=' not in arg:
            self.console.print(f"[{STYLE_ERROR}]Usage: /api settings key=value[/]")
            return

        key, _, value = arg.partition('=')
        key = key.strip()
        value = value.strip()

        valid_keys = {'model', 'auto_model', 'temperature', 'max_tokens', 'top_p', 'system_prompt', 'theme', 'api_key'}
        if key not in valid_keys:
            self.console.print(f"[{STYLE_ERROR}]Unknown setting: {key}[/]")
            self.console.print(f"[{STYLE_MUTED}]Valid keys: {', '.join(sorted(valid_keys))}[/]")
            return

        # Type coercion
        if key in ('temperature', 'top_p'):
            try:
                value = float(value)
            except ValueError:
                self.console.print(f"[{STYLE_ERROR}]{key} must be a number[/]")
                return
            maximum = 2 if key == 'temperature' else 1
            if not math.isfinite(value) or not 0 <= value <= maximum:
                self.console.print(
                    f"[{STYLE_ERROR}]{key} must be between 0 and {maximum}[/]"
                )
                return
        elif key == 'max_tokens':
            try:
                value = int(value)
            except ValueError:
                self.console.print(f"[{STYLE_ERROR}]{key} must be an integer[/]")
                return
            if value <= 0:
                self.console.print(f"[{STYLE_ERROR}]max_tokens must be positive[/]")
                return
        elif key == 'auto_model':
            normalized = value.lower()
            if normalized not in ('true', '1', 'yes', 'on', 'false', '0', 'no', 'off'):
                self.console.print(
                    f"[{STYLE_ERROR}]auto_model must be true or false[/]"
                )
                return
            value = normalized in ('true', '1', 'yes', 'on')
        elif key == 'model':
            if not model_is_chat(value):
                self.console.print(f"[{STYLE_ERROR}]That model cannot be used for chat.[/]")
                return
            value = value.strip()

        settings[key] = value
        if not _save_persisted_settings(settings):
            self.console.print(f"[{STYLE_ERROR}]Could not save setting {key}.[/]")
            self.console.print()
            return

        # Apply to current agent
        _apply_settings_to_agent(self.agent, settings)

        self.console.print(f"[{STYLE_SUCCESS}]{key} updated and saved[/]")
        if key == 'model' and not self.agent.client.tools_supported:
            self.console.print(
                f"[{STYLE_ERROR}]The selected model is not configured for tool calling.[/]"
            )
        self.console.print()

    def _api_health(self, _arg=''):
        import requests
        self.console.print(f"[{STYLE_MUTED}]Checking API endpoint...[/]")
        try:
            resp = requests.head(API_ENDPOINT.rsplit('/', 1)[0], timeout=10)
            if resp.status_code < 500:
                self.console.print(f"[{STYLE_SUCCESS}]API reachable (HTTP {resp.status_code})[/]")
            else:
                self.console.print(f"[{STYLE_ERROR}]API returned {resp.status_code}[/]")
        except Exception as e:
            logger.exception("API health check failed")
            self.console.print(f"[{STYLE_ERROR}]API unreachable: {e}[/]")

        if self.agent.client.api_key:
            self.console.print(f"[{STYLE_SUCCESS}]API key: set[/]")
        else:
            self.console.print(f"[{STYLE_ERROR}]API key: not set[/]")
        self.console.print()

    def _api_key(self, arg=''):
        if not arg:
            key = self.agent.client.api_key
            if key:
                masked = key[:8] + '\u00b7\u00b7\u00b7' + key[-4:] if len(key) > 12 else '***'
                self.console.print(f"[{STYLE_HEADER}]API Key:[/] {masked}")
            else:
                self.console.print(f"[{STYLE_ERROR}]API key not set[/]")
                self.console.print(f"[{STYLE_MUTED}]Set with: /api key nvapi-...[/]")
        else:
            new_key = arg.strip()
            if not new_key:
                self.console.print(f"[{STYLE_ERROR}]API key cannot be empty[/]")
                self.console.print()
                return
            self.agent.client.api_key = new_key
            self.agent.client.session.headers['Authorization'] = f'Bearer {new_key}'
            os.environ['NVIDIA_API_KEY'] = new_key
            # Persist
            settings = _load_persisted_settings()
            settings['api_key'] = new_key
            if _save_persisted_settings(settings):
                self.console.print(f"[{STYLE_SUCCESS}]API key updated and saved[/]")
            else:
                self.console.print(
                    f"[{STYLE_ERROR}]API key is set for this session, but could not be saved.[/]"
                )
        self.console.print()

    def _cmd_exit(self, _arg=''):
        self._exit()

    def _exit(self):
        self.agent.save_session()
        self.console.print(f"\n[{STYLE_MUTED}]Session saved. Goodbye! \U0001f44b[/]\n")
        sys.exit(0)
