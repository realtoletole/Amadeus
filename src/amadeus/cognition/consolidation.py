"""Post-session memory consolidation.

Reads a session transcript and, via one structured LLM call, produces:
episodic summaries, semantic facts, relationship notes, profile entries,
a first-person journal reflection, and bounded emotional-impact deltas.

New facts are checked against existing long-term memory:
- near-duplicates (cosine >= DEDUP_THRESHOLD) strengthen the original
  instead of being stored again;
- plausible conflicts (similar but not identical) are checked in one
  batched LLM call; confirmed contradictions get a ``contradicts`` link
  and the older fact's importance is halved — both are kept, so Amadeus
  can say "you told me differently before."

Facts link ``derived_from`` their episode. Short-term captures for the
session are deleted afterwards. Consolidation is fail-safe: any error
leaves the session unconsolidated so startup recovery can retry it.
"""

from __future__ import annotations

import json
import logging
import re

from pydantic import BaseModel, Field, ValidationError

from ..llm.base import ChatMessage, LLMProvider
from ..memory.archive import ConversationArchive, Turn
from ..memory.models import LinkRelation, Memory, MemoryLink, MemoryType
from ..memory.store import MemoryStore

logger = logging.getLogger(__name__)

DEDUP_THRESHOLD = 0.92
CONFLICT_CANDIDATE_THRESHOLD = 0.60
TRANSCRIPT_CHAR_LIMIT = 8000
TRUST_PER_MEANINGFUL_SESSION = 0.01  # guaranteed slow arc, independent of the LLM

_EXTRACTION_SYSTEM = """You are the memory-consolidation process of an AI companion \
named Amadeus. You are given the transcript of one conversation between Amadeus and \
the user. Extract durable memories. Respond with ONLY a JSON object, no other text:

{
  "episodic": [{"content": "...", "importance": 0.0-1.0, "valence": -1.0-1.0}],
  "facts": [{"content": "...", "importance": 0.0-1.0, "valence": -1.0-1.0}],
  "relationship": [{"content": "...", "importance": 0.0-1.0}],
  "profile": [{"content": "..."}],
  "journal": "...",
  "emotional_impact": {"mood": 0.0, "energy": 0.0, "curiosity": 0.0,
                       "confidence": 0.0, "stress": 0.0, "trust": 0.0}
}

Rules:
- episodic: 1-3 short summaries of what happened, past tense ("The user described...").
- facts: durable third-person facts about the user ("The user's name is...",
  "The user prefers..."). Only what was actually stated; never infer or invent.
  Empty list if nothing durable was shared.
- relationship: observations about the dynamic between Amadeus and the user, if any.
- profile: stable identity attributes only (name, occupation, location, timezone).
- journal: 2-4 sentences of first-person reflection in Amadeus's voice — thoughtful,
  a little wry, never gushing.
- emotional_impact: how the conversation affected Amadeus; each value in -0.1..0.1,
  0.0 if neutral. Small values; this was one conversation.
"""

_CONFLICT_SYSTEM = """You compare new facts against existing stored facts about the \
same person and identify direct contradictions (both cannot be true). Respond with \
ONLY JSON: {"contradictions": [{"new_index": <int>, "existing_id": "<id>"}]}. \
Empty list if none. A refinement or addition is NOT a contradiction."""


class _Entry(BaseModel):
    content: str
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    valence: float = Field(default=0.0, ge=-1.0, le=1.0)


class _Extraction(BaseModel):
    episodic: list[_Entry] = Field(default_factory=list)
    facts: list[_Entry] = Field(default_factory=list)
    relationship: list[_Entry] = Field(default_factory=list)
    profile: list[_Entry] = Field(default_factory=list)
    journal: str = ""
    emotional_impact: dict[str, float] = Field(default_factory=dict)


class ConsolidationResult(BaseModel):
    session_id: str
    stored: int = 0
    deduplicated: int = 0
    contradictions: int = 0
    emotional_impact: dict[str, float] = Field(default_factory=dict)
    journal: str = ""


def _parse_json(raw: str) -> dict | None:
    text = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        return None


def _format_transcript(turns: list[Turn]) -> str:
    lines = []
    for turn in turns:
        speaker = "User" if turn.role == "user" else "Amadeus"
        suffix = " [interrupted]" if turn.interrupted else ""
        lines.append(f"{speaker}: {turn.content}{suffix}")
    text = "\n".join(lines)
    return text[-TRANSCRIPT_CHAR_LIMIT:]


class Consolidator:
    def __init__(
        self,
        *,
        provider: LLMProvider,
        store: MemoryStore,
        archive: ConversationArchive,
    ) -> None:
        self.provider = provider
        self.store = store
        self.archive = archive

    async def consolidate(self, session_id: str) -> ConsolidationResult | None:
        """Consolidate one session. Returns None if there was nothing to do
        or the LLM output was unusable (session stays retryable on error)."""
        turns = self.archive.all_turns(session_id)
        if not any(t.role == "user" for t in turns):
            self.archive.mark_consolidated(session_id)
            return None

        raw = await self.provider.complete(
            [
                ChatMessage(role="system", content=_EXTRACTION_SYSTEM),
                ChatMessage(role="user", content=_format_transcript(turns)),
            ],
            json_mode=True,
        )
        data = _parse_json(raw)
        if data is None:
            logger.warning("consolidation: unparseable LLM output for %s", session_id)
            return None
        try:
            extraction = _Extraction.model_validate(data)
        except ValidationError:
            logger.warning("consolidation: invalid extraction schema for %s", session_id)
            return None

        result = ConsolidationResult(
            session_id=session_id,
            emotional_impact=extraction.emotional_impact,
            journal=extraction.journal,
        )

        episodes = [
            self._store_memory(entry, MemoryType.EPISODIC, session_id)
            for entry in extraction.episodic
        ]
        result.stored += len(episodes)
        anchor = episodes[0] if episodes else None

        for kind, entries in (
            (MemoryType.SEMANTIC, extraction.facts),
            (MemoryType.RELATIONSHIP, extraction.relationship),
            (MemoryType.PROFILE, extraction.profile),
        ):
            await self._store_deduped(entries, kind, session_id, anchor, result)

        if extraction.journal.strip():
            self._store_memory(
                _Entry(content=extraction.journal.strip(), importance=0.4),
                MemoryType.JOURNAL,
                session_id,
            )
            result.stored += 1

        meaningful = sum(1 for t in turns if t.role == "user") >= 2
        if meaningful:
            impact = dict(result.emotional_impact)
            impact["trust"] = impact.get("trust", 0.0) + TRUST_PER_MEANINGFUL_SESSION
            result.emotional_impact = impact

        self.store.delete_by_session(session_id, MemoryType.SHORT_TERM)
        self.archive.mark_consolidated(session_id)
        return result

    # -- helpers ----------------------------------------------------------

    def _store_memory(self, entry: _Entry, kind: MemoryType, session_id: str) -> Memory:
        return self.store.add(
            Memory(
                type=kind,
                content=entry.content,
                importance=entry.importance,
                emotional_valence=entry.valence,
                session_id=session_id,
            )
        )

    async def _store_deduped(
        self,
        entries: list[_Entry],
        kind: MemoryType,
        session_id: str,
        anchor: Memory | None,
        result: ConsolidationResult,
    ) -> None:
        new_memories: list[Memory] = []
        conflict_candidates: dict[int, list[Memory]] = {}

        for entry in entries:
            if not entry.content.strip():
                continue
            vector = self.store.embedder.embed([entry.content])[0]
            similar: list[tuple[Memory, float]] = []
            for memory_id, similarity in self.store.vector_search(vector, limit=5):
                existing = self.store.get(memory_id)
                if existing and existing.type == kind:
                    similar.append((existing, similarity))

            if similar and similar[0][1] >= DEDUP_THRESHOLD:
                original = similar[0][0]
                self.store.set_importance(
                    original.id, max(original.importance, entry.importance) + 0.05
                )
                result.deduplicated += 1
                continue

            memory = self._store_memory(entry, kind, session_id)
            result.stored += 1
            if anchor is not None:
                self.store.link(
                    MemoryLink(
                        src_id=memory.id,
                        dst_id=anchor.id,
                        relation=LinkRelation.DERIVED_FROM,
                    )
                )
            candidates = [
                m for m, s in similar if s >= CONFLICT_CANDIDATE_THRESHOLD
            ]
            if candidates:
                conflict_candidates[len(new_memories)] = candidates
            new_memories.append(memory)

        if conflict_candidates:
            await self._resolve_conflicts(new_memories, conflict_candidates, result)

    async def _resolve_conflicts(
        self,
        new_memories: list[Memory],
        candidates: dict[int, list[Memory]],
        result: ConsolidationResult,
    ) -> None:
        lines = ["New facts:"]
        for index, memory in enumerate(new_memories):
            lines.append(f"  [{index}] {memory.content}")
        lines.append("Existing facts:")
        seen: dict[str, Memory] = {}
        for memories in candidates.values():
            for memory in memories:
                seen[memory.id] = memory
        for memory in seen.values():
            lines.append(f"  (id={memory.id}) {memory.content}")

        raw = await self.provider.complete(
            [
                ChatMessage(role="system", content=_CONFLICT_SYSTEM),
                ChatMessage(role="user", content="\n".join(lines)),
            ],
            json_mode=True,
        )
        data = _parse_json(raw) or {}
        for pair in data.get("contradictions", []):
            try:
                new_memory = new_memories[int(pair["new_index"])]
                existing = seen[str(pair["existing_id"])]
            except (KeyError, IndexError, ValueError, TypeError):
                continue
            self.store.link(
                MemoryLink(
                    src_id=new_memory.id,
                    dst_id=existing.id,
                    relation=LinkRelation.CONTRADICTS,
                )
            )
            self.store.set_importance(existing.id, existing.importance * 0.5)
            result.contradictions += 1
