"""NVIDIA API Client for ForgeCLI.

Handles: rate limiting, retry, streaming, tool-aware payload building,
fallback from stream to non-stream, and auto-retry without tools on 400.
"""

import os
import time
import json
import random
import traceback
import requests
from typing import List, Dict, Optional, Generator
from lib.utils import logger
from lib.tools import TOOL_DEFINITIONS
from conf.conf import (
    API_ENDPOINT,
    DEFAULT_MODEL,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    RATE_LIMIT_RPM,
    RATE_LIMIT_BACKOFF_BASE,
    RATE_LIMIT_BACKOFF_MAX,
    RATE_LIMIT_MAX_RETRIES,
    model_supports_tools,
)


# ── Safe extraction helpers ──────────────────────────────────────


def _safe_delta(chunk: dict) -> dict:
    """Safely extract delta from a streaming chunk."""
    choices = chunk.get('choices')
    if not choices or not isinstance(choices, list):
        return {}
    first = choices[0]
    if not isinstance(first, dict):
        return {}
    return first.get('delta') or {}


def _safe_content(delta: dict) -> Optional[str]:
    """Safely extract content string. Returns None if empty/null."""
    c = delta.get('content')
    if c is None or c == '':
        return None
    if not isinstance(c, str):
        try:
            return str(c)
        except Exception:
            return None
    return c


def _safe_tool_calls(delta: dict) -> list:
    """Safely extract tool_calls list from delta.
    
    Handles: missing, null, empty string, list with None entries.
    """
    tc = delta.get('tool_calls')
    if not tc or not isinstance(tc, list):
        return []
    return [item for item in tc if item is not None and isinstance(item, dict)]


def _safe_tc_function(tc: dict) -> dict:
    """Safely extract function dict from a tool_call chunk."""
    fn = tc.get('function')
    if fn is None or not isinstance(fn, dict):
        return {}
    return fn


def _accumulate_tool_call(tool_calls: list, tc: dict):
    """Accumulate a single streaming tool_call chunk."""
    try:
        idx = tc.get('index', 0)
        if not isinstance(idx, int):
            idx = 0

        while len(tool_calls) <= idx:
            tool_calls.append({
                'id': '',
                'type': 'function',
                'function': {'name': '', 'arguments': ''},
            })

        if tc.get('id'):
            tool_calls[idx]['id'] = tc['id']

        fn = _safe_tc_function(tc)
        if fn:
            name = fn.get('name')
            if name and isinstance(name, str):
                tool_calls[idx]['function']['name'] += name
            args = fn.get('arguments')
            if args and isinstance(args, str):
                tool_calls[idx]['function']['arguments'] += args
    except Exception as e:
        logger.error(f"Error accumulating tool_call: {e}, tc={json.dumps(tc, default=str)[:200]}", exc_info=True)


def _safe_message_from_response(response: dict) -> Optional[dict]:
    """Safely extract message dict from a non-streaming API response.
    
    Handles all the weird shapes different models return.
    """
    if not response or not isinstance(response, dict):
        return None
    
    choices = response.get('choices')
    if not choices or not isinstance(choices, list):
        return None
    
    first = choices[0]
    if not isinstance(first, dict):
        return None
    
    msg = first.get('message')
    if msg is None or not isinstance(msg, dict):
        # Some models put content directly in choice
        content = first.get('text') or first.get('content')
        if content:
            return {'role': 'assistant', 'content': str(content)}
        return None
    
    return msg


# ── Rate limiter ─────────────────────────────────────────────────


class RateLimiter:
    def __init__(self, rpm: int = RATE_LIMIT_RPM):
        self.rpm = rpm
        self.window = 60.0
        self.timestamps: List[float] = []

    def wait_if_needed(self):
        now = time.time()
        self.timestamps = [t for t in self.timestamps if now - t < self.window]
        if len(self.timestamps) >= self.rpm:
            sleep_time = self.timestamps[0] + self.window - now + 0.1
            if sleep_time > 0:
                logger.info(f"Rate limiter: sleeping {sleep_time:.1f}s")
                time.sleep(sleep_time)
        self.timestamps.append(time.time())

    def record_rate_limit_hit(self):
        now = time.time()
        for i in range(3):
            self.timestamps.append(now - i * 2)


# ── Main client ──────────────────────────────────────────────────


class NvidiaAPIClient:
    """Client for NVIDIA Chat Completions API."""

    def __init__(self):
        self.api_key = os.environ.get('NVIDIA_API_KEY', '')
        self.model = DEFAULT_MODEL
        self.max_tokens = DEFAULT_MAX_TOKENS
        self.temperature = DEFAULT_TEMPERATURE
        self.top_p = DEFAULT_TOP_P
        self.rate_limiter = RateLimiter()
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
            'Accept': 'application/json',
        })
        self.verbose = False
        self._total_requests = 0
        self._total_errors = 0

    def set_model(self, model: str):
        self.model = model
        logger.info(f"Model changed to: {model}")

    @property
    def tools_supported(self) -> bool:
        """Whether the current model supports function calling."""
        return model_supports_tools(self.model)

    def _build_payload(
        self,
        messages: List[Dict],
        tools: bool,
        stream: bool,
    ) -> Dict:
        """Build API payload. Only includes tools if model supports them."""
        payload = {
            'model': self.model,
            'messages': messages,
            'max_tokens': self.max_tokens,
            'temperature': self.temperature,
            'top_p': self.top_p,
            'stream': stream,
        }
        # CRITICAL: Only send tools to models that support them
        if tools and self.tools_supported:
            payload['tools'] = TOOL_DEFINITIONS
            payload['tool_choice'] = 'auto'
        return payload

    # ── Non-streaming chat ─────────────────────────────────────

    def chat(
        self,
        messages: List[Dict],
        tools: bool = True,
        stream: bool = False,
    ) -> Dict:
        """Send chat completion with retry. Auto-retries without tools on 400."""
        payload = self._build_payload(messages, tools, stream)
        has_tools_in_payload = 'tools' in payload

        retries = 0
        while retries <= RATE_LIMIT_MAX_RETRIES:
            self.rate_limiter.wait_if_needed()
            self._total_requests += 1

            try:
                logger.debug(
                    f"API request #{self._total_requests}: model={self.model}, "
                    f"msgs={len(messages)}, tools={'yes' if has_tools_in_payload else 'no'}, "
                    f"stream={stream}"
                )

                resp = self.session.post(
                    API_ENDPOINT,
                    json=payload,
                    timeout=120,
                    stream=stream,
                )

                # Rate limited
                if resp.status_code == 429:
                    self._total_errors += 1
                    self.rate_limiter.record_rate_limit_hit()
                    backoff = min(
                        RATE_LIMIT_BACKOFF_BASE * (2 ** retries) + random.uniform(0, 1),
                        RATE_LIMIT_BACKOFF_MAX,
                    )
                    logger.warning(f"Rate limited (429). Retry {retries+1}/{RATE_LIMIT_MAX_RETRIES} in {backoff:.1f}s")
                    time.sleep(backoff)
                    retries += 1
                    continue

                # Server error
                if resp.status_code >= 500:
                    self._total_errors += 1
                    backoff = min(
                        RATE_LIMIT_BACKOFF_BASE * (2 ** retries) + random.uniform(0, 1),
                        RATE_LIMIT_BACKOFF_MAX,
                    )
                    logger.warning(f"Server error ({resp.status_code}). Retry {retries+1}/{RATE_LIMIT_MAX_RETRIES}")
                    time.sleep(backoff)
                    retries += 1
                    continue

                # Client error — if we sent tools and got 400, retry WITHOUT tools
                if resp.status_code in (400, 422) and has_tools_in_payload:
                    logger.warning(
                        f"Got {resp.status_code} with tools for {self.model}. "
                        f"Retrying without tools. Body: {resp.text[:300]}"
                    )
                    payload.pop('tools', None)
                    payload.pop('tool_choice', None)
                    has_tools_in_payload = False
                    self._total_errors += 1
                    continue  # immediate retry without tools

                # Other client errors
                if resp.status_code >= 400:
                    self._total_errors += 1
                    error_body = resp.text[:500]
                    logger.error(f"API error {resp.status_code}: {error_body}")
                    return {
                        'error': True,
                        'status_code': resp.status_code,
                        'message': f"API Error {resp.status_code}: {error_body}",
                    }

                # Success
                if stream:
                    return self._collect_stream(resp)

                data = resp.json()
                logger.debug(f"API response OK: {json.dumps(data, default=str)[:500]}")
                return data

            except requests.exceptions.Timeout:
                self._total_errors += 1
                logger.warning(f"Timeout. Retry {retries+1}/{RATE_LIMIT_MAX_RETRIES}")
                retries += 1
                time.sleep(RATE_LIMIT_BACKOFF_BASE * retries)
                continue

            except requests.exceptions.ConnectionError as e:
                self._total_errors += 1
                logger.error(f"Connection error: {e}")
                retries += 1
                time.sleep(RATE_LIMIT_BACKOFF_BASE * (2 ** retries))
                continue

            except Exception as e:
                self._total_errors += 1
                logger.error(f"Unexpected error: {e}", exc_info=True)
                return {'error': True, 'status_code': 0, 'message': f"Unexpected error: {e}"}

        return {
            'error': True,
            'status_code': 429,
            'message': f"Max retries ({RATE_LIMIT_MAX_RETRIES}) exceeded.",
        }

    def _collect_stream(self, resp) -> Dict:
        """Collect a streaming response into a single standard dict."""
        content_parts = []
        tool_calls = []

        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith('data: '):
                continue
            data_str = line[6:]
            if data_str.strip() == '[DONE]':
                break
            try:
                chunk = json.loads(data_str)
            except json.JSONDecodeError:
                continue
            try:
                delta = _safe_delta(chunk)
                if not delta:
                    continue
                c = _safe_content(delta)
                if c:
                    content_parts.append(c)
                for tc in _safe_tool_calls(delta):
                    _accumulate_tool_call(tool_calls, tc)
            except Exception as e:
                logger.error(f"Stream chunk error: {e}", exc_info=True)
                continue

        message = {'role': 'assistant'}
        if content_parts:
            message['content'] = ''.join(content_parts)
        if tool_calls:
            valid = [tc for tc in tool_calls if (tc.get('function') or {}).get('name')]
            if valid:
                message['tool_calls'] = valid

        return {
            'choices': [{'message': message, 'finish_reason': 'tool_calls' if tool_calls else 'stop'}]
        }

    # ── Streaming generator ─────────────────────────────────────

    def chat_stream(
        self,
        messages: List[Dict],
        tools: bool = True,
    ) -> Generator[Dict, None, None]:
        """Stream chat completion, yielding events.

        If streaming fails or returns 0 content, falls back to non-streaming.
        If tools cause 400, auto-retries without tools.
        """
        payload = self._build_payload(messages, tools, stream=True)
        has_tools = 'tools' in payload

        retries = 0
        while retries <= RATE_LIMIT_MAX_RETRIES:
            self.rate_limiter.wait_if_needed()
            self._total_requests += 1

            try:
                logger.debug(
                    f"Stream #{self._total_requests}: model={self.model}, "
                    f"msgs={len(messages)}, tools={'yes' if has_tools else 'no'}"
                )

                resp = self.session.post(
                    API_ENDPOINT,
                    json=payload,
                    timeout=120,
                    stream=True,
                )

                # Rate limited
                if resp.status_code == 429:
                    self._total_errors += 1
                    self.rate_limiter.record_rate_limit_hit()
                    backoff = min(
                        RATE_LIMIT_BACKOFF_BASE * (2 ** retries) + random.uniform(0, 1),
                        RATE_LIMIT_BACKOFF_MAX,
                    )
                    logger.warning(f"Rate limited (429). Retry {retries+1}/{RATE_LIMIT_MAX_RETRIES}")
                    time.sleep(backoff)
                    retries += 1
                    continue

                # Server error
                if resp.status_code >= 500:
                    self._total_errors += 1
                    backoff = min(
                        RATE_LIMIT_BACKOFF_BASE * (2 ** retries) + random.uniform(0, 1),
                        RATE_LIMIT_BACKOFF_MAX,
                    )
                    logger.warning(f"Server error ({resp.status_code}). Retry {retries+1}")
                    time.sleep(backoff)
                    retries += 1
                    continue

                # 400/422 with tools → retry without tools
                if resp.status_code in (400, 422) and has_tools:
                    logger.warning(
                        f"Got {resp.status_code} with tools for {self.model}. "
                        f"Retrying without tools. Body: {resp.text[:300]}"
                    )
                    payload.pop('tools', None)
                    payload.pop('tool_choice', None)
                    has_tools = False
                    self._total_errors += 1
                    continue

                # Other client errors
                if resp.status_code >= 400:
                    self._total_errors += 1
                    error_body = resp.text[:500]
                    logger.error(f"Stream client error {resp.status_code}: {error_body}")
                    yield {'type': 'error', 'message': f"API Error {resp.status_code}: {error_body}"}
                    return

                # ── Process SSE stream ──
                tool_calls = []
                chunk_count = 0
                content_count = 0

                for line in resp.iter_lines(decode_unicode=True):
                    if not line or not line.startswith('data: '):
                        continue
                    data_str = line[6:]
                    if data_str.strip() == '[DONE]':
                        break

                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    chunk_count += 1

                    try:
                        delta = _safe_delta(chunk)
                        if not delta:
                            continue

                        c = _safe_content(delta)
                        if c:
                            content_count += 1
                            yield {'type': 'content_delta', 'content': c}

                        for tc in _safe_tool_calls(delta):
                            _accumulate_tool_call(tool_calls, tc)

                    except Exception as e:
                        logger.error(
                            f"Stream chunk #{chunk_count} error: {e}, "
                            f"model={self.model}, delta={data_str[:200]}",
                            exc_info=True,
                        )
                        continue

                logger.debug(
                    f"Stream done: {chunk_count} chunks, {content_count} content, "
                    f"{len(tool_calls)} tool_calls"
                )

                # If stream returned ZERO chunks, fall back to non-streaming
                if chunk_count == 0 and content_count == 0 and not tool_calls:
                    logger.warning(
                        f"Stream returned 0 chunks for {self.model}. "
                        f"Falling back to non-streaming."
                    )
                    yield from self._fallback_non_stream(messages, has_tools)
                    return

                # Yield tool calls if any
                if tool_calls:
                    valid = [tc for tc in tool_calls if (tc.get('function') or {}).get('name')]
                    if valid:
                        yield {'type': 'tool_calls', 'tool_calls': valid}
                    else:
                        logger.warning(
                            f"All tool_calls had no name: "
                            f"{json.dumps(tool_calls, default=str)[:300]}"
                        )

                yield {'type': 'stream_end'}
                return

            except requests.exceptions.Timeout:
                self._total_errors += 1
                logger.warning(f"Stream timeout. Retry {retries+1}")
                retries += 1
                time.sleep(RATE_LIMIT_BACKOFF_BASE * retries)
                continue

            except requests.exceptions.ConnectionError as e:
                self._total_errors += 1
                logger.error(f"Stream connection error: {e}")
                retries += 1
                time.sleep(RATE_LIMIT_BACKOFF_BASE * (2 ** retries))
                continue

            except Exception as e:
                self._total_errors += 1
                logger.error(
                    f"Stream unexpected error: {e}, model={self.model}",
                    exc_info=True,
                )
                yield {'type': 'error', 'message': f"Stream error: {e}"}
                return

        yield {
            'type': 'error',
            'message': f"Max retries ({RATE_LIMIT_MAX_RETRIES}) exceeded.",
        }

    def _fallback_non_stream(
        self,
        messages: List[Dict],
        has_tools: bool,
    ) -> Generator[Dict, None, None]:
        """Fallback: do a non-streaming request and yield events from it."""
        logger.info(f"Non-stream fallback for {self.model}")

        response = self.chat(messages, tools=has_tools, stream=False)

        if isinstance(response, dict) and response.get('error'):
            yield {
                'type': 'error',
                'message': response.get('message', 'API error in fallback'),
            }
            return

        msg = _safe_message_from_response(response)
        if not msg:
            logger.error(
                f"Fallback got empty message: {json.dumps(response, default=str)[:500]}"
            )
            yield {'type': 'error', 'message': 'Model returned empty response'}
            return

        content = msg.get('content') or ''
        if content:
            # Emit content as a single delta
            yield {'type': 'content_delta', 'content': content}

        tc_list = msg.get('tool_calls')
        if tc_list and isinstance(tc_list, list):
            valid = [tc for tc in tc_list
                     if tc and isinstance(tc, dict)
                     and (tc.get('function') or {}).get('name')]
            if valid:
                yield {'type': 'tool_calls', 'tool_calls': valid}

        yield {'type': 'stream_end'}

    @property
    def stats(self) -> Dict:
        return {
            'total_requests': self._total_requests,
            'total_errors': self._total_errors,
            'model': self.model,
        }
