"""Core ForgeCLI Agent — manages conversation, tool calling loop, sessions.

Supports both synchronous (process) and streaming (process_streaming) modes.
"""

import os
import json
import time
import uuid
from typing import List, Dict, Optional, Callable, Generator
from lib.api import NvidiaAPIClient
from lib.tools import execute_tool, TOOL_DEFINITIONS
from lib.utils import logger, timestamp, safe_json_loads
from conf.conf import (
    MAX_CONTEXT_MESSAGES,
    SESSION_DIR,
    APP_NAME,
    APP_VERSION,
)

SYSTEM_PROMPT = f"""You are {APP_NAME} v{APP_VERSION} (ForgeAI), an expert AI assistant operating in a user's terminal.
You have access to powerful tools to help users accomplish tasks.

Capabilities:
- Execute shell commands (cmd_exec)
- Read, write, append, and delete files (file_read, file_write, file_append, file_delete)
- List directory contents (list_dir)
- Search files by name or content (search_files)
- Execute Python code (python_exec)

Guidelines:
1. Use tools proactively when the user's request requires file operations or commands.
2. Always confirm destructive operations (delete, overwrite) by stating what you'll do before doing it.
3. For multi-step tasks, plan your approach then execute step by step.
4. When a command fails, analyze the error and try to fix it.
5. Keep explanations concise but helpful.
6. Return file contents in full when writing files - never use placeholders or ellipsis.
7. When unsure about the environment, use tools to explore (list_dir, cmd_exec, file_read).
8. Prefer absolute paths when writing files to avoid ambiguity.
"""

MAX_TOOL_ROUNDS = 15  # Safety limit on consecutive tool-call rounds


def _safe_choice(response: dict) -> Optional[dict]:
    """Safely extract the first choice from an API response.

    Returns None if choices is missing or empty, preventing IndexError.
    """
    choices = response.get('choices')
    if not choices:
        return None
    return choices[0]


class Agent:
    """ForgeAI Agent with tool-calling capabilities."""

    def __init__(self, on_status: Optional[Callable] = None):
        self.client = NvidiaAPIClient()
        self.messages: List[Dict] = []
        self.session_id = str(uuid.uuid4())[:8]
        self.on_status = on_status or (lambda *a, **kw: None)
        self._init_session()

    def _init_session(self):
        """Initialize conversation with system prompt."""
        self.messages = [{'role': 'system', 'content': SYSTEM_PROMPT}]
        logger.info(f"New ForgeCLI session: {self.session_id}")

    def new_session(self):
        """Start fresh session."""
        self.save_session()
        self.session_id = str(uuid.uuid4())[:8]
        self._init_session()

    # ── Synchronous processing (original) ────────────────────────────

    def process(self, user_input: str) -> str:
        """Process user input through the agent loop.

        Returns the final assistant text response.
        """
        self.messages.append({'role': 'user', 'content': user_input})
        self._trim_context()

        round_num = 0
        while round_num < MAX_TOOL_ROUNDS:
            round_num += 1
            self.on_status(f"Thinking... (round {round_num})" if round_num > 1 else "Thinking...")

            response = self.client.chat(self.messages, tools=True)

            # Handle errors
            if isinstance(response, dict) and response.get('error'):
                error_msg = response.get('message', 'Unknown API error')
                logger.error(f"API error: {error_msg}")
                return f"\u274c {error_msg}"

            # Safely extract assistant message
            choice = _safe_choice(response)
            if choice is None:
                logger.error(f"Empty choices in response: {json.dumps(response)[:500]}")
                return "\u274c Received empty response from API (no choices)."

            assistant_msg = choice.get('message')
            if assistant_msg is None:
                logger.error(f"No message in choice: {json.dumps(choice)[:500]}")
                return "\u274c Received malformed response from API (no message)."

            # Build message to append
            msg_to_append = {'role': 'assistant'}
            if assistant_msg.get('content'):
                msg_to_append['content'] = assistant_msg['content']
            if assistant_msg.get('tool_calls'):
                msg_to_append['tool_calls'] = assistant_msg['tool_calls']

            self.messages.append(msg_to_append)

            # If no tool calls, we're done
            if not assistant_msg.get('tool_calls'):
                return assistant_msg.get('content', '(no response)')

            # Execute each tool call
            for tool_call in assistant_msg['tool_calls']:
                tool_name = tool_call['function']['name']
                tool_id = tool_call.get('id', f'call_{uuid.uuid4().hex[:8]}')

                # Parse arguments
                raw_args = tool_call['function'].get('arguments', '{}')
                if isinstance(raw_args, str):
                    args = safe_json_loads(raw_args, {})
                else:
                    args = raw_args

                self.on_status(f"\U0001f527 {tool_name}: {self._summarize_args(tool_name, args)}")
                logger.info(f"Tool call: {tool_name}({json.dumps(args)[:200]})")

                # Execute
                result = execute_tool(tool_name, args)

                success = result.get('success', False)
                output = result.get('output', '')

                status_icon = '\u2705' if success else '\u274c'
                self.on_status(f"{status_icon} {tool_name} completed")

                # Append tool result to conversation
                self.messages.append({
                    'role': 'tool',
                    'tool_call_id': tool_id,
                    'content': output,
                })

            # Loop continues so the model can see tool results

        # If we exceeded max rounds
        return "\u26a0\ufe0f Reached maximum tool execution rounds. Please break your task into smaller steps."

    # ── Streaming processing ─────────────────────────────────────────

    def process_streaming(self, user_input: str) -> Generator[Dict, None, None]:
        """Process user input with streaming, yielding events for the TUI.

        Yields dicts with 'type' key:
          - {'type': 'status', 'text': str}        — status message
          - {'type': 'content_delta', 'content': str} — text chunk
          - {'type': 'content_done', 'content': str}  — text segment complete
          - {'type': 'tool_exec', 'name': str, 'args': dict, 'tool_id': str}
          - {'type': 'tool_result', 'name': str, 'args': dict, 'result': dict, 'tool_id': str}
          - {'type': 'error', 'text': str}         — error message
          - {'type': 'done'}                       — all processing complete
        """
        self.messages.append({'role': 'user', 'content': user_input})
        self._trim_context()

        round_num = 0
        while round_num < MAX_TOOL_ROUNDS:
            round_num += 1
            yield {
                'type': 'status',
                'text': f'Thinking... (round {round_num})' if round_num > 1 else 'Thinking...',
            }

            content_text = ''
            tool_calls = []
            got_error = False

            for event in self.client.chat_stream(self.messages, tools=True):
                etype = event.get('type', '')

                if etype == 'content_delta':
                    content_text += event['content']
                    yield event  # pass through to TUI

                elif etype == 'tool_calls':
                    tool_calls = event['tool_calls']

                elif etype == 'error':
                    yield {'type': 'error', 'text': event.get('message', 'Unknown stream error')}
                    got_error = True
                    break

                elif etype == 'stream_end':
                    pass

            if got_error:
                return

            # Build and append assistant message
            msg = {'role': 'assistant'}
            if content_text:
                msg['content'] = content_text
            if tool_calls:
                msg['tool_calls'] = tool_calls
            self.messages.append(msg)

            # Signal that current text is done
            if content_text:
                yield {'type': 'content_done', 'content': content_text}

            # If no tool calls, we're done
            if not tool_calls:
                if not content_text:
                    yield {'type': 'content_done', 'content': '(no response)'}
                yield {'type': 'done'}
                return

            # Execute each tool call
            for tc in tool_calls:
                tool_name = tc['function']['name']
                tool_id = tc.get('id', f'call_{uuid.uuid4().hex[:8]}')

                raw_args = tc['function'].get('arguments', '{}')
                if isinstance(raw_args, str):
                    args = safe_json_loads(raw_args, {})
                else:
                    args = raw_args

                logger.info(f"Tool call: {tool_name}({json.dumps(args)[:200]})")

                yield {
                    'type': 'tool_exec',
                    'name': tool_name,
                    'args': args,
                    'tool_id': tool_id,
                }

                # Execute (blocking)
                result = execute_tool(tool_name, args)

                yield {
                    'type': 'tool_result',
                    'name': tool_name,
                    'args': args,
                    'result': result,
                    'tool_id': tool_id,
                }

                # Append tool result to conversation
                self.messages.append({
                    'role': 'tool',
                    'tool_call_id': tool_id,
                    'content': result.get('output', ''),
                })

            # Loop continues — model sees tool results

        # Exceeded max rounds
        yield {
            'type': 'error',
            'text': '\u26a0\ufe0f Reached maximum tool execution rounds. Break your task into smaller steps.',
        }
        yield {'type': 'done'}

    # ── Helpers ──────────────────────────────────────────────────────

    def _summarize_args(self, name: str, args: dict) -> str:
        """Create a short summary of tool args for status display."""
        if name == 'cmd_exec':
            return args.get('command', '')[:80]
        elif name in ('file_write', 'file_read', 'file_append', 'file_delete'):
            return args.get('path', '')[:80]
        elif name == 'list_dir':
            return args.get('path', '.')[:80]
        elif name == 'search_files':
            parts = []
            if args.get('pattern'):
                parts.append(f"pattern={args['pattern']}")
            if args.get('text'):
                parts.append(f"text='{args['text'][:40]}'")
            return ', '.join(parts) or '.'
        elif name == 'python_exec':
            return args.get('code', '')[:80].replace('\n', ' ')
        return str(args)[:80]

    def _trim_context(self):
        """Trim conversation history to stay within limits."""
        if len(self.messages) > MAX_CONTEXT_MESSAGES + 1:
            system = self.messages[0]
            self.messages = [system] + self.messages[-(MAX_CONTEXT_MESSAGES):]
            logger.info(f"Context trimmed to {len(self.messages)} messages")

    def save_session(self) -> Optional[str]:
        """Save current session to disk."""
        if len(self.messages) <= 1:  # Only system prompt
            return None
        try:
            os.makedirs(SESSION_DIR, exist_ok=True)
            path = os.path.join(SESSION_DIR, f"{self.session_id}.json")
            data = {
                'session_id': self.session_id,
                'model': self.client.model,
                'created': timestamp(),
                'messages': self.messages,
                'stats': self.client.stats,
            }
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Session saved: {path}")
            return path
        except Exception as e:
            logger.error(f"Failed to save session: {e}")
            return None

    def load_session(self, session_id: str) -> bool:
        """Load a previous session."""
        try:
            path = os.path.join(SESSION_DIR, f"{session_id}.json")
            if not os.path.exists(path):
                return False
            with open(path, 'r') as f:
                data = json.load(f)
            self.messages = data.get('messages', [])
            self.session_id = data.get('session_id', session_id)
            if data.get('model'):
                self.client.set_model(data['model'])
            logger.info(f"Session loaded: {session_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to load session: {e}")
            return False

    def list_sessions(self) -> List[Dict]:
        """List saved sessions."""
        sessions = []
        try:
            os.makedirs(SESSION_DIR, exist_ok=True)
            for fname in sorted(os.listdir(SESSION_DIR)):
                if fname.endswith('.json'):
                    path = os.path.join(SESSION_DIR, fname)
                    try:
                        with open(path, 'r') as f:
                            data = json.load(f)
                        msg_count = len([m for m in data.get('messages', []) if m.get('role') == 'user'])
                        sessions.append({
                            'id': data.get('session_id', fname[:-5]),
                            'created': data.get('created', 'unknown'),
                            'model': data.get('model', 'unknown'),
                            'messages': msg_count,
                        })
                    except Exception:
                        pass
        except Exception as e:
            logger.error(f"Error listing sessions: {e}")
        return sessions

    @property
    def history(self) -> List[Dict]:
        """Return conversation history (excluding system prompt)."""
        return [m for m in self.messages if m.get('role') != 'system']
