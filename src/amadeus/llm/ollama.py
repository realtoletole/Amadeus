"""Ollama-backed LLM provider (local models)."""

from __future__ import annotations

import json
from typing import AsyncIterator

import httpx

from .base import ChatMessage


class OllamaProvider:
    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        think: bool = False,
        keep_alive: str = "60m",
        num_ctx: int = 8192,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.think = think
        self.keep_alive = keep_alive
        self.num_ctx = num_ctx

    def _payload(self, messages, *, stream: bool, temperature: float,
                 json_mode: bool = False) -> dict:
        rendered = [m.model_dump() for m in messages]
        # Qwen3 soft switch: the API-level "think" flag is ignored by older
        # Ollama versions, in which case the model reasons out loud as plain
        # prose (unfilterable). The trained /no_think marker in the last user
        # turn disables it at the model level, belt and suspenders.
        # only hybrid Qwen3 models understand the soft switch; the 2507
        # variants (instruct/thinking splits) dropped it — appending it there
        # would inject literal junk into the model's context
        hybrid_qwen = "qwen3" in self.model and "2507" not in self.model
        if not self.think and hybrid_qwen and rendered:
            for entry in reversed(rendered):
                if entry["role"] == "user":
                    entry["content"] = entry["content"] + " /no_think"
                    break
        payload: dict = {
            "model": self.model,
            "messages": rendered,
            "stream": stream,
            "think": self.think,
            "keep_alive": self.keep_alive,
            "options": {"temperature": temperature, "num_ctx": self.num_ctx},
        }
        if json_mode:
            payload["format"] = "json"
        return payload

    async def stream_chat(
        self, messages: list[ChatMessage], *, temperature: float = 0.7
    ) -> AsyncIterator[str]:
        payload = self._payload(messages, stream=True, temperature=temperature)
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                "POST", f"{self.base_url}/api/chat", json=payload
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    chunk = json.loads(line)
                    if content := chunk.get("message", {}).get("content"):
                        yield content
                    if chunk.get("done"):
                        return

    async def complete(
        self, messages: list[ChatMessage], *, temperature: float = 0.3, json_mode: bool = False
    ) -> str:
        payload = self._payload(
            messages, stream=False, temperature=temperature, json_mode=json_mode
        )
        async with httpx.AsyncClient(timeout=600.0) as client:
            response = await client.post(f"{self.base_url}/api/chat", json=payload)
            response.raise_for_status()
            return response.json()["message"]["content"]
