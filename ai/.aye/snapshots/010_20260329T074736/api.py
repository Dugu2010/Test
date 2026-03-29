"""NVIDIA API Client for ForgeCLI """

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
)


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
    """Safely extract content string from delta. Returns None if missing/null/empty."""
    c = delta.get('content')
    if c is None or c == '':
        return None
    if not isinstance(c, str):
        return str(c)
    return c


def _safe_tool_calls(delta: dict) -> list:
    """Safely extract tool_calls list from delta.

    Guards against:
      - tool_calls missing entirely
      - tool_calls: null
      - tool_calls: "" (some models)
      - tool_calls: [None, ...]
    """
    tc = delta.get('tool_calls')
    if not tc or not isinstance(tc, list):
        return []
    # Filter out any None entries in the list
    return [item for item in tc if item is not None and isinstance(item, dict)]


def _safe_tc_function(tc: dict) -> dict:
    """Safely extract function dict from a tool_call chunk."""
    fn = tc.get('function')
    if fn is None or not isinstance(fn, dict):
        return {}
    return fn


def _accumulate_tool_call(tool_calls: list, tc: dict):
    """Accumulate a single streaming tool_call chunk into tool_calls list.

    Handles all the None/missing field cases.
    """
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
        logger.error(
            f"Error accumulating tool_call chunk: {e}\n"
            f"  tc={json.dumps(tc, default=str)[:300]}\n"
            f"  tool_calls_len={len(tool_calls)}",
            exc_info=True,
        )


class RateLimiter:
    """Token-bucket style rate limiter."""

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

    def _build_payload(self, messages: List[Dict], tools: bool, stream: bool) -> Dict:
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

    # 
    # Non-streaming
    # 

    def chat(
        self,
        messages: List[Dict],
        tools: bool = True,
        stream: bool = False,
    ) -> Dict:
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

                if resp.status_code >= 500:
                    self._total_errors += 1
                    backoff = min(
                        RATE_LIMIT_BACKOFF_BASE * (2 ** retries) + random.uniform(0, 1),
                        RATE_LIMIT_BACKOFF_MAX,
                    )
                    logger.warning(f"Server error ({resp.status_code}). Retry {retries+1}/{RATE_LIMIT_MAX_RETRIES} in {backoff:.1f}s")
                    time.sleep(backoff)
                    retries += 1
                    continue

                if resp.status_code >= 400:
                    self._total_errors += 1
                    error_body = resp.text[:500]
                    logger.error(f"API client error {resp.status_code}: {error_body}")
                    return {
                        'error': True,
                        'status_code': resp.status_code,
                        'message': f"API Error {resp.status_code}: {error_body}",
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
            'message': f"Max retries ({RATE_LIMIT_MAX_RETRIES}) exceeded. Try again later.",
        }

    def _handle_stream(self, resp) -> Dict:
        """Collect streaming response into a standard dict."""
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

                content = _safe_content(delta)
                if content:
                    content_parts.append(content)

                for tc in _safe_tool_calls(delta):
                    _accumulate_tool_call(tool_calls, tc)

            except Exception as e:
                logger.error(
                    f"Error processing stream chunk: {e}\n"
                    f"  chunk={data_str[:300]}",
                    exc_info=True,
                )
                continue

        message = {'role': 'assistant'}
        if content_parts:
            message['content'] = ''.join(content_parts)
        if tool_calls:
            message['tool_calls'] = tool_calls

        return {
            'choices': [{'message': message, 'finish_reason': 'tool_calls' if tool_calls else 'stop'}]
        }

    # 
    # Streaming generator
    # 

    def chat_stream(
        self,
        messages: List[Dict],
        tools: bool = True,
    ) -> Generator[Dict, None, None]:
        """Stream chat completion, yielding events.

        Yields:
          - {'type': 'content_delta', 'content': str}
          - {'type': 'tool_calls', 'tool_calls': list}
          - {'type': 'stream_end'}
          - {'type': 'error', 'message': str}
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

                if resp.status_code >= 500:
                    self._total_errors += 1
                    backoff = min(
                        RATE_LIMIT_BACKOFF_BASE * (2 ** retries) + random.uniform(0, 1),
                        RATE_LIMIT_BACKOFF_MAX,
                    )
                    logger.warning(f"Server error ({resp.status_code}). Retry {retries+1}/{RATE_LIMIT_MAX_RETRIES} in {backoff:.1f}s")
                    time.sleep(backoff)
                    retries += 1
                    continue

                if resp.status_code >= 400:
                    self._total_errors += 1
                    error_body = resp.text[:500]
                    logger.error(f"API client error {resp.status_code}: {error_body}")
                    yield {'type': 'error', 'message': f"API Error {resp.status_code}: {error_body}"}
                    return

                # Stream SSE
                tool_calls = []
                chunk_count = 0

                for line in resp.iter_lines(decode_unicode=True):
                    if not line or not line.startswith('data: '):
                        continue
                    data_str = line[6:]
                    if data_str.strip() == '[DONE]':
                        break

                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        logger.debug(f"Skipping non-JSON chunk: {data_str[:100]}")
                        continue

                    chunk_count += 1

                    try:
                        delta = _safe_delta(chunk)
                        if not delta:
                            continue

                        # Content token
                        content = _safe_content(delta)
                        if content:
                            yield {'type': 'content_delta', 'content': content}

                        # Tool-call accumulation
                        for tc in _safe_tool_calls(delta):
                            _accumulate_tool_call(tool_calls, tc)

                    except Exception as e:
                        logger.error(
                            f"Error processing stream chunk #{chunk_count}: {e}\n"
                            f"  model={self.model}\n"
                            f"  raw_delta={json.dumps(chunk, default=str)[:400]}",
                            exc_info=True,
                        )
                        # Don't break 
                        continue

                # Stream ended
                logger.debug(f"Stream ended: {chunk_count} chunks, {len(tool_calls)} tool_calls")

                if tool_calls:
                    # Validate tool_calls before yielding
                    valid_tcs = []
                    for tc in tool_calls:
                        fn = tc.get('function') or {}
                        name = fn.get('name', '')
                        if name:  # Only yield tool calls with a name
                            valid_tcs.append(tc)
                        else:
                            logger.warning(f"Dropping unnamed tool_call: {json.dumps(tc, default=str)[:200]}")
                    if valid_tcs:
                        yield {'type': 'tool_calls', 'tool_calls': valid_tcs}

                yield {'type': 'stream_end'}
                return

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
                logger.error(
                    f"Stream unexpected error: {e}\n"
                    f"  model={self.model}\n"
                    f"  traceback={traceback.format_exc()}"
                )
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
