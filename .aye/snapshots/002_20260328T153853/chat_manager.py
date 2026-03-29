"""Chat manager — handles history & streaming API calls."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import AsyncIterator, Optional
import httpx
import json

from .config import load_config
from .settings import Settings


@dataclass
class ChatManager:
    settings: Settings
    history: list = field(default_factory=list)
    _client: Optional[httpx.AsyncClient] = field(default=None, repr=False)

    async def get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=120.0)
        return self._client

    def add_to_history(self, role: str, content: str) -> None:
        self.history.append({"role": role, "content": content})

    def clear_history(self) -> None:
        self.history.clear()

    async def stream_response(self, message: str) -> AsyncIterator[str]:
        self.add_to_history("user", message)
        client = await self.get_client()
        config = load_config()

        payload = {
            "model": self.settings.model,
            "messages": self.history,
            "stream": True,
        }

        api_base = config.get("api_base", "https://api.openai.com/v1")
        api_key = config.get("api_key", "")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        async with client.stream(
            "POST",
            f"{api_base}/chat/completions",
            json=payload,
            headers=headers,
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data.strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        delta = chunk["choices"][0].get("delta", {})
                        if "content" in delta:
                            yield delta["content"]
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
