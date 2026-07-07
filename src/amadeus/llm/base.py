"""LLM provider abstraction.

Everything above this layer (cognition, prompt building) speaks
:class:`ChatMessage` and consumes an async token stream. Switching from
a local Ollama model to the Anthropic API is a config change.
"""

from __future__ import annotations

from typing import AsyncIterator, Literal, Protocol

from pydantic import BaseModel

Role = Literal["system", "user", "assistant"]


class ChatMessage(BaseModel):
    role: Role
    content: str


class LLMProvider(Protocol):
    async def stream_chat(
        self, messages: list[ChatMessage], *, temperature: float = 0.7
    ) -> AsyncIterator[str]:
        """Yield response tokens as they are generated. Must be cancellable
        (consumer stops iterating => generation stops) so voice barge-in
        can cut a response mid-sentence."""
        ...

    async def complete(
        self, messages: list[ChatMessage], *, temperature: float = 0.3, json_mode: bool = False
    ) -> str:
        """Non-streaming completion. With json_mode=True the provider should
        constrain output to valid JSON (used by memory consolidation)."""
        ...
