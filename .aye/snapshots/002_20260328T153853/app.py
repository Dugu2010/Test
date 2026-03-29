"""Main Aye Chat TUI application."""

from __future__ import annotations

from textual.app import App, ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.widgets import Header, Footer, Input, Static
from textual.binding import Binding
from textual.css.query import NoMatches
from textual import work, on, events
from rich.text import Text

from .chat_manager import ChatManager
from .message_display import MessageDisplay
from .model_selector import ModelSelector
from .settings import Settings
from .config import load_config


class ChatApp(App):
    """AI-powered terminal chat."""

    CSS_PATH = "styles.tcss"
    TITLE = "aye chat"

    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit", show=True, priority=True),
        Binding("ctrl+n", "new_session", "New session", show=True),
        Binding("ctrl+m", "select_model", "Model", show=True),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.settings = Settings()
        self.chat_manager = ChatManager(settings=self.settings)

    # ── compose ───────────────────────────────────────────────────────

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with VerticalScroll(id="chat-view"):
            yield Static(
                "Welcome to Aye Chat! Type a message to begin.",
                id="welcome-msg",
                classes="welcome",
            )
        with Vertical(id="input-area"):
            yield Input(
                placeholder="Type your message… (Ctrl+Q to quit)",
                id="user-input",
            )
        yield Footer()

    # ── lifecycle / events ────────────────────────────────────────────

    def on_mount(self) -> None:
        self.query_one("#user-input", Input).focus()

    def on_resize(self, event: events.Resize) -> None:
        """Re-render every message bubble so content reflows to the new width."""
        try:
            for widget in self.query(MessageDisplay):
                widget.on_resize()
        except NoMatches:
            pass

    # ── input handling ────────────────────────────────────────────────

    @on(Input.Submitted, "#user-input")
    def on_input_submitted(self, event: Input.Submitted) -> None:
        message = event.value.strip()
        if not message:
            return
        event.input.value = ""
        self.send_message(message)

    # ── send / stream ─────────────────────────────────────────────────

    @work(exclusive=True)
    async def send_message(self, message: str) -> None:
        chat_view = self.query_one("#chat-view", VerticalScroll)

        # remove welcome banner on first message
        try:
            self.query_one("#welcome-msg").remove()
        except NoMatches:
            pass

        # user bubble
        user_display = MessageDisplay(
            message, role="user", classes="message user-message"
        )
        chat_view.mount(user_display)

        # assistant bubble (placeholder)
        assistant_display = MessageDisplay(
            "", role="assistant", classes="message assistant-message"
        )
        chat_view.mount(assistant_display)
        chat_view.scroll_end(animate=False)

        # stream the response
        try:
            full_response = ""
            tick = 0
            async for chunk in self.chat_manager.stream_response(message):
                full_response += chunk
                tick += 1
                assistant_display.update_content(
                    full_response,
                    streaming=True,
                    stream_tick=tick,
                )
                chat_view.scroll_end(animate=False)

            # final render (streaming=False so incomplete-block indicators vanish)
            assistant_display.update_content(full_response, streaming=False)
            self.chat_manager.add_to_history("assistant", full_response)

        except Exception as exc:
            assistant_display.update_content(
                f"⚠️  Error: {exc}",
                streaming=False,
            )

    # ── actions ───────────────────────────────────────────────────────

    def action_new_session(self) -> None:
        self.chat_manager.clear_history()
        chat_view = self.query_one("#chat-view", VerticalScroll)
        chat_view.remove_children()
        chat_view.mount(
            Static(
                "New session started. Type a message to begin.",
                id="welcome-msg",
                classes="welcome",
            )
        )

    def action_select_model(self) -> None:
        def _on_model(model: str | None) -> None:
            if model:
                self.settings.model = model
                self.notify(f"Model set to {model}")

        self.push_screen(ModelSelector(), callback=_on_model)

    async def action_quit(self) -> None:
        await self.chat_manager.close()
        self.exit()
