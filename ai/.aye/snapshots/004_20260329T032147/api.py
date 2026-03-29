"""NVIDIA API Client with rate limiting, retry logic, and streaming support."""

import os
import time
import json
import random
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
)


class RateLimiter:
    """Token-bucket style rate limiter."""

    def __init__(self, rpm: int = RATE_LIMIT_RPM):
        self.rpm = rpm
        self.window = 60.0
        self.timestamps: List[float] = []

    def wait_if_needed(self):
        """Block until a request is allowed."""
        now = time.time()
        # Purge old timestamps
        self.timestamps = [t for t in self.timestamps if now - t < self.window]
        if len(self.timestamps) >= self.rpm:
            sleep_time = self.timestamps[0] + self.window - now + 0.1
            if sleep_time > 0:
                logger.info(f"Rate limiter: sleeping {sleep_time:.1f}s")
                time.sleep(sleep_time)
        self.timestamps.append(time.time())

    def record_rate_limit_hit(self):
        """Record that we hit a 429; be more conservative."""
        now = time.time()
        for i in range(3):
            self.timestamps.append(now - i * 2)


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
        """Switch model."""
        self.model = model
        logger.info(f"Model changed to: {model}")

    def _build_payload(self, messages: List[Dict], tools: bool, stream: bool) -> Dict:
        """Build the request payload."""
        payload = {
            'model': self.model,
            'messages': messages,
            'max_tokens': self.max_tokens,
            'temperature': self.temperature,
            'top_p': self.top_p,
            'stream': stream,
        }
        if tools:
            payload['tools'] = TOOL_DEFINITIONS
            payload['tool_choice'] = 'auto'
        return payload

    # ── Non-streaming chat (original) ────────────────────────────────

    def chat(
        self,
        messages: List[Dict],
        tools: bool = True,
        stream: bool = False,
    ) -> Dict:
        """Send chat completion request with retry logic."""

        payload = self._build_payload(messages, tools, stream)

        retries = 0
        while retries <= RATE_LIMIT_MAX_RETRIES:
            self.rate_limiter.wait_if_needed()
            self._total_requests += 1

            try:
                logger.debug(f"API request #{self._total_requests}: model={self.model}, messages={len(messages)}")

                if self.verbose:
                    logger.debug(f"Payload: {json.dumps(payload, indent=2)[:2000]}")

                resp = self.session.post(
                    API_ENDPOINT,
                    json=payload,
                    timeout=120,
                    stream=stream,
                )

                # Handle rate limit
                if resp.status_code == 429:
                    self._total_errors += 1
                    self.rate_limiter.record_rate_limit_hit()
                    backoff = min(
                        RATE_LIMIT_BACKOFF_BASE * (2 ** retries) + random.uniform(0, 1),
                        RATE_LIMIT_BACKOFF_MAX
                    )
                    logger.warning(f"Rate limited (429). Retry {retries+1}/{RATE_LIMIT_MAX_RETRIES} in {backoff:.1f}s")
                    time.sleep(backoff)
                    retries += 1
                    continue

                # Handle server errors with retry
                if resp.status_code >= 500:
                    self._total_errors += 1
                    backoff = min(
                        RATE_LIMIT_BACKOFF_BASE * (2 ** retries) + random.uniform(0, 1),
                        RATE_LIMIT_BACKOFF_MAX
                    )
                    logger.warning(f"Server error ({resp.status_code}). Retry {retries+1}/{RATE_LIMIT_MAX_RETRIES} in {backoff:.1f}s")
                    time.sleep(backoff)
                    retries += 1
                    continue

                # Client errors - don't retry
                if resp.status_code >= 400:
                    self._total_errors += 1
                    error_body = resp.text[:500]
                    logger.error(f"API client error {resp.status_code}: {error_body}")
                    return {
                        'error': True,
                        'status_code': resp.status_code,
                        'message': f"API Error {resp.status_code}: {error_body}"
                    }

                if stream:
                    return self._handle_stream(resp)

                data = resp.json()
                logger.debug(f"API response: {json.dumps(data, indent=2)[:1000]}")
                return data

            except requests.exceptions.Timeout:
                self._total_errors += 1
                logger.warning(f"Request timeout. Retry {retries+1}/{RATE_LIMIT_MAX_RETRIES}")
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
                logger.error(f"Unexpected API error: {e}", exc_info=True)
                return {'error': True, 'status_code': 0, 'message': f"Unexpected error: {str(e)}"}

        return {
            'error': True,
            'status_code': 429,
            'message': f"Max retries ({RATE_LIMIT_MAX_RETRIES}) exceeded. Try again later."
        }

    def _handle_stream(self, resp) -> Dict:
        """Handle streaming response (collect all chunks, return complete dict)."""
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
                delta = chunk.get('choices', [{}])[0].get('delta', {})
                if 'content' in delta and delta['content']:
                    content_parts.append(delta['content'])
                if 'tool_calls' in delta:
                    for tc in delta['tool_calls']:
                        idx = tc.get('index', 0)
                        while len(tool_calls) <= idx:
                            tool_calls.append({'id': '', 'type': 'function', 'function': {'name': '', 'arguments': ''}})
                        if 'id' in tc:
                            tool_calls[idx]['id'] = tc['id']
                        if 'function' in tc:
                            if 'name' in tc['function']:
                                tool_calls[idx]['function']['name'] += tc['function']['name']
                            if 'arguments' in tc['function']:
                                tool_calls[idx]['function']['arguments'] += tc['function']['arguments']
            except json.JSONDecodeError:
                continue

        # Build response in standard format
        message = {'role': 'assistant'}
        if content_parts:
            message['content'] = ''.join(content_parts)
        if tool_calls:
            message['tool_calls'] = tool_calls

        return {
            'choices': [{'message': message, 'finish_reason': 'tool_calls' if tool_calls else 'stop'}]
        }

    # ── Streaming generator (yields events) ──────────────────────────

    def chat_stream(
        self,
        messages: List[Dict],
        tools: bool = True,
    ) -> Generator[Dict, None, None]:
        """Stream chat completion, yielding events as they arrive.

        Yields dicts with 'type' key:
          - {'type': 'content_delta', 'content': str}  — text chunk
          - {'type': 'tool_calls', 'tool_calls': list}  — completed tool calls
          - {'type': 'stream_end'}                       — stream finished
          - {'type': 'error', 'message': str}            — error occurred
        """
        payload = self._build_payload(messages, tools, stream=True)

        retries = 0
        while retries <= RATE_LIMIT_MAX_RETRIES:
            self.rate_limiter.wait_if_needed()
            self._total_requests += 1

            try:
                logger.debug(
                    f"Stream request #{self._total_requests}: "
                    f"model={self.model}, messages={len(messages)}"
                )

                resp = self.session.post(
                    API_ENDPOINT,
                    json=payload,
                    timeout=120,
                    stream=True,
                )

                # Handle rate limit
                if resp.status_code == 429:
                    self._total_errors += 1
                    self.rate_limiter.record_rate_limit_hit()
                    backoff = min(
                        RATE_LIMIT_BACKOFF_BASE * (2 ** retries) + random.uniform(0, 1),
                        RATE_LIMIT_BACKOFF_MAX,
                    )
                    logger.warning(
                        f"Rate limited (429). Retry {retries+1}/{RATE_LIMIT_MAX_RETRIES} "
                        f"in {backoff:.1f}s"
                    )
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
                    logger.warning(
                        f"Server error ({resp.status_code}). Retry {retries+1}/"
                        f"{RATE_LIMIT_MAX_RETRIES} in {backoff:.1f}s"
                    )
                    time.sleep(backoff)
                    retries += 1
                    continue

                # Client error — don't retry
                if resp.status_code >= 400:
                    self._total_errors += 1
                    error_body = resp.text[:500]
                    logger.error(f"API client error {resp.status_code}: {error_body}")
                    yield {'type': 'error', 'message': f"API Error {resp.status_code}: {error_body}"}
                    return

                # ── Stream the SSE response ──────────────────────────
                tool_calls = []

                for line in resp.iter_lines(decode_unicode=True):
                    if not line or not line.startswith('data: '):
                        continue
                    data_str = line[6:]
                    if data_str.strip() == '[DONE]':
                        break

                    try:
                        chunk = json.loads(data_str)
                        delta = chunk.get('choices', [{}])[0].get('delta', {})

                        # Content token
                        if 'content' in delta and delta['content']:
                            yield {
                                'type': 'content_delta',
                                'content': delta['content'],
                            }

                        # Tool-call accumulation
                        if 'tool_calls' in delta:
                            for tc in delta['tool_calls']:
                                idx = tc.get('index', 0)
                                while len(tool_calls) <= idx:
                                    tool_calls.append({
                                        'id': '',
                                        'type': 'function',
                                        'function': {'name': '', 'arguments': ''},
                                    })
                                if 'id' in tc:
                                    tool_calls[idx]['id'] = tc['id']
                                if 'function' in tc:
                                    if 'name' in tc['function']:
                                        tool_calls[idx]['function']['name'] += tc['function']['name']
                                    if 'arguments' in tc['function']:
                                        tool_calls[idx]['function']['arguments'] += tc['function']['arguments']

                    except json.JSONDecodeError:
                        continue

                # After stream ends
                if tool_calls:
                    yield {'type': 'tool_calls', 'tool_calls': tool_calls}

                yield {'type': 'stream_end'}
                return  # Success — exit retry loop

            except requests.exceptions.Timeout:
                self._total_errors += 1
                logger.warning(f"Stream timeout. Retry {retries+1}/{RATE_LIMIT_MAX_RETRIES}")
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
                logger.error(f"Stream unexpected error: {e}", exc_info=True)
                yield {'type': 'error', 'message': f"Unexpected error: {str(e)}"}
                return

        yield {
            'type': 'error',
            'message': f"Max retries ({RATE_LIMIT_MAX_RETRIES}) exceeded. Try again later.",
        }

    @property
    def stats(self) -> Dict:
        return {
            'total_requests': self._total_requests,
            'total_errors': self._total_errors,
            'model': self.model,
        }
