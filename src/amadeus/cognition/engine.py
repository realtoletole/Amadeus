"""Cognition engine: the per-turn conversation loop.

For each user message:
1. retrieve relevant long-term memories (hybrid scorer),
2. assemble the system prompt (persona + time awareness + memories),
3. stream the LLM response through the think-tag filter,
4. archive both turns and capture a short-term memory of the exchange.

If the consumer stops consuming mid-stream (voice barge-in, UI stop
button), the partial assistant turn is still archived, flagged
``interrupted`` — Amadeus knows she was cut off.
"""

from __future__ import annotations

import logging
from typing import AsyncIterator, Callable

from ..llm.base import ChatMessage, LLMProvider
from ..memory.archive import ConversationArchive
from ..memory.models import Memory, MemoryType
from ..memory.retrieval import MemoryRetriever
from ..memory.store import MemoryStore
from ..memory.state import StateStore
from ..persona.profile import Persona as PersonaModel
from .consolidation import Consolidator
from .emotions import EmotionalState, render_style_hints
from .keywords import extract_keywords
from .prompt import build_system_prompt
from .stream_filter import ThinkTagFilter

logger = logging.getLogger(__name__)

_STATE_KEY = "emotional_state"


class CognitionEngine:
    def __init__(
        self,
        *,
        provider: LLMProvider,
        store: MemoryStore,
        retriever: MemoryRetriever,
        archive: ConversationArchive,
        persona: "PersonaModel | Callable[[], PersonaModel]",
        state_store: StateStore | None = None,
        consolidator: Consolidator | None = None,
        history_turns: int = 20,
        retrieval_top_k: int = 8,
    ) -> None:
        self.provider = provider
        self.store = store
        self.retriever = retriever
        self.archive = archive
        # persona is re-resolved every reply, so in-app edits to persona.md
        # take effect on the next message with no restart
        self._persona_source = persona if callable(persona) else (lambda: persona)
        self.state_store = state_store
        self.consolidator = consolidator
        self.history_turns = history_turns
        self.retrieval_top_k = retrieval_top_k
        # called at every prompt build so the offered expressions always
        # match the avatar model that is installed RIGHT NOW; swapping
        # models never leaves her trying to make faces she no longer has
        self.expression_source: Callable[[], list[str]] | None = None
        self.session_id: str | None = None

    @property
    def persona(self) -> PersonaModel:
        return self._persona_source()

    # -- session lifecycle ----------------------------------------------

    def start_session(self) -> str:
        self.session_id = self.archive.start_session()
        return self.session_id

    def end_session(self) -> None:
        if self.session_id:
            self.archive.end_session(self.session_id)
            self.session_id = None

    async def close_session(self, session_id: str | None = None) -> None:
        """End a session and run memory consolidation (fail-safe: an
        unconsolidated session is retried by consolidate_pending).

        Takes an explicit id so a stale connection's cleanup can never
        close a newer session (e.g. after a browser refresh)."""
        session_id = session_id or self.session_id
        if session_id is None:
            return
        if session_id == self.session_id:
            self.session_id = None
        self.archive.end_session(session_id)
        if self.consolidator:
            try:
                result = await self.consolidator.consolidate(session_id)
            except Exception:  # noqa: BLE001 - never crash shutdown
                logger.exception("consolidation failed for session %s", session_id)
                return
            if result and result.emotional_impact:
                self._apply_emotional_impact(result.emotional_impact)

    async def consolidate_pending(self, limit: int = 2) -> int:
        """Retry consolidation for sessions left unconsolidated (crash,
        Ollama down at quit time). Called at startup.

        Bounded to the most recent ``limit`` sessions: each retry is a
        full LLM job, and Ollama serves requests serially — an unbounded
        backlog (e.g. after a schema migration) would queue in front of
        the user's live conversation and freeze it. Older sessions are
        marked consolidated and skipped; their raw transcripts remain in
        the archive."""
        if not self.consolidator:
            return 0
        pending = self.archive.unconsolidated_sessions()  # oldest first
        skipped, retried = pending[:-limit] if limit else pending, pending[-limit:]
        for session_id in skipped:
            logger.info("skipping backlog consolidation for old session %s", session_id)
            self.archive.mark_consolidated(session_id)
        done = 0
        for session_id in retried:
            try:
                result = await self.consolidator.consolidate(session_id)
            except Exception:  # noqa: BLE001
                logger.exception("pending consolidation failed for %s", session_id)
                continue
            if result and result.emotional_impact:
                self._apply_emotional_impact(result.emotional_impact)
            done += 1
        return done

    # -- emotional state ---------------------------------------------------

    def emotional_state(self) -> EmotionalState:
        """Load persisted state and apply time decay toward baselines."""
        if self.state_store is None:
            return EmotionalState()
        raw = self.state_store.get(_STATE_KEY)
        state = EmotionalState.model_validate_json(raw) if raw else EmotionalState()
        return state.decayed()

    def _apply_emotional_impact(self, deltas: dict[str, float]) -> None:
        if self.state_store is None:
            return
        state = self.emotional_state().apply_deltas(deltas)
        self.state_store.set(_STATE_KEY, state.model_dump_json())

    # -- the turn loop ---------------------------------------------------

    async def respond(self, user_text: str) -> AsyncIterator[str]:
        if self.session_id is None:
            self.start_session()
        session_id = self.session_id
        assert session_id is not None

        memories = self.retriever.retrieve(user_text, top_k=self.retrieval_top_k)
        system = build_system_prompt(
            self.persona,
            memories=memories,
            last_session_ended=self.archive.last_session_ended_at(exclude=session_id),
            style_hints=render_style_hints(self.emotional_state()),
            expressions=self.expression_source() if self.expression_source else [],
        )
        history = self.archive.recent_turns(session_id, limit=self.history_turns)
        messages = [
            ChatMessage(role="system", content=system),
            *(ChatMessage(role=t.role, content=t.content) for t in history),
            ChatMessage(role="user", content=user_text),
        ]

        self.archive.add_turn(session_id, "user", user_text)
        self._capture_short_term(user_text, session_id)

        reply_parts: list[str] = []
        completed = False
        filt = ThinkTagFilter()
        try:
            async for raw in self.provider.stream_chat(messages):
                if visible := filt.feed(raw):
                    reply_parts.append(visible)
                    yield visible
            if tail := filt.flush():
                reply_parts.append(tail)
                yield tail
            completed = True
        finally:
            reply = "".join(reply_parts)
            if reply:
                self.archive.add_turn(
                    session_id, "assistant", reply, interrupted=not completed
                )

    # -- helpers ----------------------------------------------------------

    def _capture_short_term(self, user_text: str, session_id: str) -> None:
        """Raw capture of the user's message; Phase 3 consolidation turns
        these into proper episodic/semantic memories and decays the rest."""
        self.store.add(
            Memory(
                type=MemoryType.SHORT_TERM,
                content=f"The user said: {user_text}",
                importance=0.3,
                keywords=extract_keywords(user_text),
                session_id=session_id,
            )
        )
