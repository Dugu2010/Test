"""Core ForgeCLI Agent — tool-calling with model-aware capabilities."""

import os
import json
import time
import uuid
import traceback
from typing import List, Dict, Optional, Callable, Generator
from lib.api import NvidiaAPIClient
from lib.tools import execute_tool, TOOL_DEFINITIONS
from lib.utils import logger, timestamp, safe_json_loads
from conf.conf import (
    MAX_CONTEXT_MESSAGES,
    SESSION_DIR,
    APP_NAME,
    APP_VERSION,
    model_supports_tools,
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

SYSTEM_PROMPT_NO_TOOLS = f"""You are {APP_NAME} v{APP_VERSION} (ForgeAI), an expert AI assistant.
You help users with coding, debugging, analysis, writing, math, and general questions.
Provide clear, concise, and helpful responses.
"""

MAX_TOOL_ROUNDS = 15


def _safe_choice(response: dict) -> Optional[dict]:
    """Safely extract the first choice from an API response."""
    choices = response.get('choices')
    if not choices or not isinstance(choices, list):
        return None
    first = choices[0]
    if not isinstance(first, dict):
        return None
    return first


def _safe_extract_tool_call(tc: dict) -> Optional[dict]:
    """Safely extract tool name, id, args from a tool_call dict."""
    if not tc or not isinstance(tc, dict):
        logger.warning(f"Invalid tool_call (not dict): {tc}")
        return None

    fn = tc.get('function')
    if fn is None or not isinstance(fn, dict):
        logger.warning(f"tool_call no function: {json.dumps(tc, default=str)[:200]}")
        return None

    name = fn.get('name', '')
    if not name or not isinstance(name, str):
        logger.warning(f"tool_call no name: {json.dumps(tc, default=str)[:200]}")
        return None

    tool_id = tc.get('id') or f'call_{uuid.uuid4().hex[:8]}'
    raw_args = fn.get('arguments', '{}')

    return {
        'name': name.strip(),
        'tool_id': tool_id,
        'raw_args': raw_args,
    }


def _parse_tool_args(raw_args) -> dict:
    if isinstance(raw_args, dict):
        return raw_args
    if isinstance(raw_args, str):
        return safe_json_loads(raw_args, {})
    return {}


class Agent:
    """ForgeAI Agent with model-aware tool-calling."""

    def __init__(self, on_status: Optional[Callable] = None):
        self.client = NvidiaAPIClient()
        self.messages: List[Dict] = []
        self.session_id = str(uuid.uuid4())[:8]
        self.on_status = on_status or (lambda *a, **kw: None)
        self._init_session()

    def _init_session(self):
        """Initialize session with appropriate system prompt."""
        prompt = SYSTEM_PROMPT if self.client.tools_supported else SYSTEM_PROMPT_NO_TOOLS
        self.messages = [{'role': 'system', 'content': prompt}]
        logger.info(f"Session init: {self.session_id}, model={self.client.model}, tools={self.client.tools_supported}")

    def _update_system_prompt(self):
        """Update system prompt when model changes (tool support might change)."""
        if self.messages and self.messages[0].get('role') == 'system':
            prompt = SYSTEM_PROMPT if self.client.tools_supported else SYSTEM_PROMPT_NO_TOOLS
            self.messages[0]['content'] = prompt

    def new_session(self):
        self.save_session()
        self.session_id = str(uuid.uuid4())[:8]
        self._init_session()

    # ── Synchronous processing ────────────────────────────────

    def process(self, user_input: str) -> str:
        self.messages.append({'role': 'user', 'content': user_input})
        self._trim_context()
        self._update_system_prompt()

        use_tools = self.client.tools_supported
        logger.info(f"Process: model={self.client.model}, tools={use_tools}")

        round_num = 0
        while round_num < MAX_TOOL_ROUNDS:
            round_num += 1
            self.on_status(f"Thinking... (round {round_num})" if round_num > 1 else "Thinking...")

            response = self.client.chat(self.messages, tools=use_tools)

            if isinstance(response, dict) and response.get('error'):
                error_msg = response.get('message', 'Unknown API error')
                logger.error(f"API error: {error_msg}")
                return f"\u274c {error_msg}"

            choice = _safe_choice(response)
            if choice is None:
                logger.error(f"No choices: {json.dumps(response, default=str)[:500]}")
                return "\u274c Empty response from model."

            assistant_msg = choice.get('message')
            if assistant_msg is None or not isinstance(assistant_msg, dict):
                logger.error(f"No message in choice: {json.dumps(choice, default=str)[:500]}")
                return "\u274c Malformed response from model."

            msg_to_append = {'role': 'assistant'}
            content = assistant_msg.get('content')
            if content:
                msg_to_append['content'] = content

            raw_tcs = assistant_msg.get('tool_calls')
            if raw_tcs and isinstance(raw_tcs, list):
                msg_to_append['tool_calls'] = raw_tcs
            else:
                raw_tcs = None

            self.messages.append(msg_to_append)

            # No tool calls → done
            if not raw_tcs:
                return content or '(no response)'

            # Execute tools
            for tc in raw_tcs:
                extracted = _safe_extract_tool_call(tc)
                if not extracted:
                    continue

                tool_name = extracted['name']
                tool_id = extracted['tool_id']
                args = _parse_tool_args(extracted['raw_args'])

                self.on_status(f"\U0001f527 {tool_name}: {self._summarize_args(tool_name, args)}")
                logger.info(f"Tool: {tool_name}({json.dumps(args, default=str)[:200]})")

                try:
                    result = execute_tool(tool_name, args)
                except Exception as e:
                    logger.error(f"Tool crash: {tool_name}: {e}", exc_info=True)
                    result = {'success': False, 'output': f'Tool error: {e}'}

                self.messages.append({
                    'role': 'tool',
                    'tool_call_id': tool_id,
                    'content': (result or {}).get('output', '') or '(no output)',
                })

        return "\u26a0\ufe0f Max tool rounds reached."

    # ── Streaming processing ──────────────────────────────────

    def process_streaming(self, user_input: str) -> Generator[Dict, None, None]:
        self.messages.append({'role': 'user', 'content': user_input})
        self._trim_context()
        self._update_system_prompt()

        use_tools = self.client.tools_supported
        logger.info(f"Stream: model={self.client.model}, tools={use_tools}")

        round_num = 0
        while round_num < MAX_TOOL_ROUNDS:
            round_num += 1
            yield {
                'type': 'status',
                'text': f'Thinking... (round {round_num})' if round_num > 1 else 'Thinking...',
            }

            content_text = ''
            tool_calls_raw = []
            got_error = False

            try:
                stream = self.client.chat_stream(self.messages, tools=use_tools)
                if stream is None:
                    yield {'type': 'error', 'text': 'Internal error: stream is None'}
                    return

                for event in stream:
                    if event is None:
                        continue
                    etype = event.get('type', '')

                    if etype == 'content_delta':
                        c = event.get('content', '')
                        if c:
                            content_text += c
                            yield event

                    elif etype == 'tool_calls':
                        tc_list = event.get('tool_calls')
                        if tc_list and isinstance(tc_list, list):
                            tool_calls_raw = tc_list

                    elif etype == 'error':
                        error_text = event.get('message') or event.get('text') or 'Stream error'
                        logger.error(f"Stream error: {error_text}")
                        yield {'type': 'error', 'text': error_text}
                        got_error = True
                        break

                    elif etype == 'stream_end':
                        pass

            except Exception as e:
                logger.error(f"process_streaming exception: {e}", exc_info=True)
                yield {'type': 'error', 'text': f'Error: {e}'}
                return

            if got_error:
                return

            # Build assistant message
            msg = {'role': 'assistant'}
            if content_text:
                msg['content'] = content_text
            if tool_calls_raw:
                msg['tool_calls'] = tool_calls_raw
            self.messages.append(msg)

            if content_text:
                yield {'type': 'content_done', 'content': content_text}

            # No tool calls → done
            if not tool_calls_raw:
                if not content_text:
                    yield {'type': 'content_done', 'content': '(no response)'}
                yield {'type': 'done'}
                return

            # Execute tool calls
            for tc in tool_calls_raw:
                extracted = _safe_extract_tool_call(tc)
                if not extracted:
                    continue

                tool_name = extracted['name']
                tool_id = extracted['tool_id']
                args = _parse_tool_args(extracted['raw_args'])

                logger.info(f"Tool: {tool_name}({json.dumps(args, default=str)[:200]})")

                yield {
                    'type': 'tool_exec',
                    'name': tool_name,
                    'args': args,
                    'tool_id': tool_id,
                }

                try:
                    result = execute_tool(tool_name, args)
                except Exception as e:
                    logger.error(f"Tool crash: {tool_name}: {e}", exc_info=True)
                    result = {'success': False, 'output': f'Tool error: {e}'}

                yield {
                    'type': 'tool_result',
                    'name': tool_name,
                    'args': args,
                    'result': result,
                    'tool_id': tool_id,
                }

                self.messages.append({
                    'role': 'tool',
                    'tool_call_id': tool_id,
                    'content': (result or {}).get('output', '') or '(no output)',
                })

        yield {'type': 'error', 'text': 'Max tool rounds reached.'}
        yield {'type': 'done'}

    # ── Helpers ───────────────────────────────────────────────

    def _summarize_args(self, name: str, args: dict) -> str:
        if name == 'cmd_exec':
            return (args.get('command') or '')[:80]
        elif name in ('file_write', 'file_read', 'file_append', 'file_delete'):
            return (args.get('path') or '')[:80]
        elif name == 'list_dir':
            return (args.get('path') or '.')[:80]
        elif name == 'search_files':
            parts = []
            if args.get('pattern'):
                parts.append(f"pattern={args['pattern']}")
            if args.get('text'):
                parts.append(f"text='{args['text'][:40]}'")
            return ', '.join(parts) or '.'
        elif name == 'python_exec':
            return (args.get('code') or '')[:80].replace('\n', ' ')
        return str(args)[:80]

    def _trim_context(self):
        if len(self.messages) > MAX_CONTEXT_MESSAGES + 1:
            system = self.messages[0]
            self.messages = [system] + self.messages[-(MAX_CONTEXT_MESSAGES):]
            logger.info(f"Context trimmed to {len(self.messages)} messages")

    def save_session(self) -> Optional[str]:
        if len(self.messages) <= 1:
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
                json.dump(data, f, indent=2, default=str)
            logger.info(f"Session saved: {path}")
            return path
        except Exception as e:
            logger.error(f"Save session failed: {e}", exc_info=True)
            return None

    def load_session(self, session_id: str) -> bool:
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
            logger.error(f"Load session failed: {e}", exc_info=True)
            return False

    def list_sessions(self) -> List[Dict]:
        sessions = []
        try:
            os.makedirs(SESSION_DIR, exist_ok=True)
            for fname in sorted(os.listdir(SESSION_DIR)):
                if fname.endswith('.json'):
                    try:
                        with open(os.path.join(SESSION_DIR, fname), 'r') as f:
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
            logger.error(f"List sessions error: {e}")
        return sessions

    @property
    def history(self) -> List[Dict]:
        return [m for m in self.messages if m.get('role') != 'system']
