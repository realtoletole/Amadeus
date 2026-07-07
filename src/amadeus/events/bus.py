"""Async in-process event bus.

Subsystems (memory, cognition, voice, avatar, UI) communicate through
named topics instead of importing each other. This is what lets Phase 4
(voice) and Phase 5 (avatar) bolt on without touching the core loop.

Topics are plain strings; well-known ones are defined in :class:`Topic`.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)

Handler = Callable[[Any], Awaitable[None]]


class Topic:
    """Well-known event topics."""

    USER_MESSAGE = "user.message"            # user text finalized (typed or transcribed)
    ASSISTANT_TOKEN = "assistant.token"      # streamed LLM token
    ASSISTANT_DONE = "assistant.done"        # full assistant turn completed
    ASSISTANT_INTERRUPTED = "assistant.interrupted"  # user barged in
    MEMORY_STORED = "memory.stored"
    EMOTION_UPDATED = "emotion.updated"
    SESSION_STARTED = "session.started"
    SESSION_ENDED = "session.ended"          # triggers consolidation


class EventBus:
    """Minimal async pub/sub. Handlers for one topic run concurrently;
    a failing handler is logged and never breaks the others."""

    def __init__(self) -> None:
        self._handlers: dict[str, list[Handler]] = defaultdict(list)

    def subscribe(self, topic: str, handler: Handler) -> None:
        self._handlers[topic].append(handler)

    def unsubscribe(self, topic: str, handler: Handler) -> None:
        self._handlers[topic].remove(handler)

    async def publish(self, topic: str, payload: Any = None) -> None:
        handlers = list(self._handlers.get(topic, ()))
        if not handlers:
            return
        results = await asyncio.gather(
            *(h(payload) for h in handlers), return_exceptions=True
        )
        for handler, result in zip(handlers, results):
            if isinstance(result, Exception):
                logger.exception(
                    "handler %s failed on topic %s", handler, topic, exc_info=result
                )
