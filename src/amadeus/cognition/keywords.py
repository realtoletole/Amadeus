"""Lightweight keyword extraction (no LLM) for short-term memory capture.

Phase 3 consolidation does proper LLM-based extraction; this keeps the
FTS index useful for memories created mid-conversation.
"""

from __future__ import annotations

import re
from collections import Counter

_STOPWORDS = frozenset(
    """a about after again all also am an and any are as at be because been before
    being but by can could did do does doing down for from had has have having he
    her here hers him his how i if in into is it its just me more most my no nor
    not now of off on once only or other our out over own same she should so some
    such than that the their them then there these they this those through to too
    under until up very was we were what when where which while who why will with
    would you your yours""".split()
)


def extract_keywords(text: str, limit: int = 6) -> list[str]:
    tokens = re.findall(r"[a-z][a-z0-9'-]+", text.lower())
    counts = Counter(t for t in tokens if t not in _STOPWORDS and len(t) > 2)
    return [word for word, _ in counts.most_common(limit)]
