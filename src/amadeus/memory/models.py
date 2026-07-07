"""Memory data models.

Every memory, regardless of layer, shares one shape (:class:`Memory`) and
is distinguished by :class:`MemoryType`. This keeps storage and retrieval
uniform while letting consolidation treat layers differently.

Layer semantics
---------------
- WORKING: current-turn scratch state; never persisted long-term.
- SHORT_TERM: recent conversational context; decays or is consolidated.
- EPISODIC: "what happened" — summarized events with time and emotion.
- SEMANTIC: "what is true" — extracted facts, preferences, knowledge.
- RELATIONSHIP: facts and dynamics about the user specifically.
- PROFILE: stable user attributes (name, timezone, occupation...).
- JOURNAL: Amadeus's own first-person reflections after sessions.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from enum import StrEnum

from pydantic import BaseModel, Field


class MemoryType(StrEnum):
    WORKING = "working"
    SHORT_TERM = "short_term"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    RELATIONSHIP = "relationship"
    PROFILE = "profile"
    JOURNAL = "journal"


class LinkRelation(StrEnum):
    RELATES_TO = "relates_to"
    CAUSED_BY = "caused_by"
    ABOUT_PERSON = "about_person"
    CONTRADICTS = "contradicts"
    DERIVED_FROM = "derived_from"  # e.g. semantic fact derived from an episode


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Memory(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    type: MemoryType
    content: str
    created_at: datetime = Field(default_factory=_now)
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    emotional_valence: float = Field(default=0.0, ge=-1.0, le=1.0)
    keywords: list[str] = Field(default_factory=list)
    session_id: str | None = None
    access_count: int = 0
    last_accessed: datetime | None = None
    metadata: dict = Field(default_factory=dict)

    def keywords_json(self) -> str:
        return json.dumps(self.keywords)

    def metadata_json(self) -> str:
        return json.dumps(self.metadata)


class MemoryLink(BaseModel):
    """Typed edge in the knowledge graph between two memories."""

    src_id: str
    dst_id: str
    relation: LinkRelation = LinkRelation.RELATES_TO
    weight: float = Field(default=1.0, ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=_now)


class ScoredMemory(BaseModel):
    """Retrieval result with score breakdown for debuggability."""

    memory: Memory
    score: float
    semantic: float = 0.0
    keyword: float = 0.0
    recency: float = 0.0
    importance: float = 0.0
