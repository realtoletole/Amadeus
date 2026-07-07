"""Hybrid memory retrieval.

Candidates come from the union of vector search and FTS5 keyword search,
then each candidate is scored:

    score = w_sem * cosine
          + w_kw  * normalized_bm25
          + w_rec * exp(-age / half_life)
          + w_imp * importance

Weights live in config (:class:`amadeus.config.RetrievalWeights`).
Results carry their per-channel breakdown so retrieval quality can be
inspected and tuned, and retrieved memories are "touched" to record use.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

from ..config import RetrievalWeights
from .models import MemoryType, ScoredMemory
from .store import MemoryStore

_LTM_TYPES = (
    MemoryType.EPISODIC,
    MemoryType.SEMANTIC,
    MemoryType.RELATIONSHIP,
    MemoryType.PROFILE,
    MemoryType.JOURNAL,
)


def _normalize_bm25(ranked: list[tuple[str, float]]) -> dict[str, float]:
    """Map FTS5 bm25 ranks (lower = better, can be negative) to [0, 1]."""
    if not ranked:
        return {}
    ranks = [r for _, r in ranked]
    lo, hi = min(ranks), max(ranks)
    if hi == lo:
        return {mid: 1.0 for mid, _ in ranked}
    return {mid: (hi - rank) / (hi - lo) for mid, rank in ranked}


class MemoryRetriever:
    def __init__(
        self,
        store: MemoryStore,
        weights: RetrievalWeights,
        *,
        recency_half_life_hours: float = 72.0,
    ) -> None:
        self.store = store
        self.weights = weights
        self.half_life = recency_half_life_hours

    def _recency(self, created_at: datetime, now: datetime) -> float:
        age_hours = max((now - created_at).total_seconds() / 3600.0, 0.0)
        return math.exp(-math.log(2) * age_hours / self.half_life)

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 8,
        types: tuple[MemoryType, ...] = _LTM_TYPES,
        touch: bool = True,
    ) -> list[ScoredMemory]:
        query_vec = self.store.embedder.embed([query])[0]
        semantic = dict(self.store.vector_search(query_vec, limit=top_k * 6))
        keyword = _normalize_bm25(self.store.keyword_search(query, limit=top_k * 6))

        now = datetime.now(timezone.utc)
        w = self.weights
        results: list[ScoredMemory] = []
        for memory_id in semantic.keys() | keyword.keys():
            memory = self.store.get(memory_id)
            if memory is None or memory.type not in types:
                continue
            sem = max(semantic.get(memory_id, 0.0), 0.0)
            kw = keyword.get(memory_id, 0.0)
            rec = self._recency(memory.created_at, now)
            imp = memory.importance
            results.append(
                ScoredMemory(
                    memory=memory,
                    score=w.semantic * sem + w.keyword * kw + w.recency * rec + w.importance * imp,
                    semantic=sem,
                    keyword=kw,
                    recency=rec,
                    importance=imp,
                )
            )

        results.sort(key=lambda r: r.score, reverse=True)
        results = results[:top_k]
        if touch and results:
            self.store.touch([r.memory.id for r in results])
        return results
