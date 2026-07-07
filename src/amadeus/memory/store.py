"""SQLite-backed memory store.

One database file holds everything: memories, embeddings (as BLOBs),
an FTS5 full-text index kept in sync by triggers, and the typed link
table that forms the knowledge graph.

Vector search is brute-force cosine over numpy for v1 — at personal
scale (up to ~10^5 memories) this is milliseconds and avoids native
extension headaches. Swap in sqlite-vec later if it ever matters.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .embeddings import EmbeddingProvider
from .models import LinkRelation, Memory, MemoryLink, MemoryType

_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id            TEXT PRIMARY KEY,
    type          TEXT NOT NULL,
    content       TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    importance    REAL NOT NULL DEFAULT 0.5,
    valence       REAL NOT NULL DEFAULT 0.0,
    keywords      TEXT NOT NULL DEFAULT '[]',
    session_id    TEXT,
    access_count  INTEGER NOT NULL DEFAULT 0,
    last_accessed TEXT,
    metadata      TEXT NOT NULL DEFAULT '{}',
    embedding     BLOB
);

CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(type);
CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at);

CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    content, keywords, content='memories', content_rowid='rowid'
);

CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(rowid, content, keywords)
    VALUES (new.rowid, new.content, new.keywords);
END;
CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content, keywords)
    VALUES ('delete', old.rowid, old.content, old.keywords);
END;
CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content, keywords)
    VALUES ('delete', old.rowid, old.content, old.keywords);
    INSERT INTO memories_fts(rowid, content, keywords)
    VALUES (new.rowid, new.content, new.keywords);
END;

CREATE TABLE IF NOT EXISTS memory_links (
    src_id     TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    dst_id     TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    relation   TEXT NOT NULL,
    weight     REAL NOT NULL DEFAULT 1.0,
    created_at TEXT NOT NULL,
    PRIMARY KEY (src_id, dst_id, relation)
);
"""


def _row_to_memory(row: sqlite3.Row) -> Memory:
    return Memory(
        id=row["id"],
        type=MemoryType(row["type"]),
        content=row["content"],
        created_at=datetime.fromisoformat(row["created_at"]),
        importance=row["importance"],
        emotional_valence=row["valence"],
        keywords=json.loads(row["keywords"]),
        session_id=row["session_id"],
        access_count=row["access_count"],
        last_accessed=(
            datetime.fromisoformat(row["last_accessed"]) if row["last_accessed"] else None
        ),
        metadata=json.loads(row["metadata"]),
    )


class MemoryStore:
    def __init__(self, db_path: Path | str, embedder: EmbeddingProvider) -> None:
        # check_same_thread=False: single-user app; access is sequential but
        # may hop threads (test client, UI server worker threads).
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(_SCHEMA)
        self.embedder = embedder

    # -- writes ---------------------------------------------------------

    def add(self, memory: Memory, *, embed: bool = True) -> Memory:
        vector = self.embedder.embed([memory.content])[0] if embed else None
        self._conn.execute(
            """INSERT INTO memories
               (id, type, content, created_at, importance, valence, keywords,
                session_id, access_count, last_accessed, metadata, embedding)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                memory.id,
                memory.type.value,
                memory.content,
                memory.created_at.isoformat(),
                memory.importance,
                memory.emotional_valence,
                memory.keywords_json(),
                memory.session_id,
                memory.access_count,
                memory.last_accessed.isoformat() if memory.last_accessed else None,
                memory.metadata_json(),
                vector.tobytes() if vector is not None else None,
            ),
        )
        self._conn.commit()
        return memory

    def delete(self, memory_id: str) -> None:
        self._conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        self._conn.commit()

    def set_importance(self, memory_id: str, importance: float) -> None:
        importance = min(1.0, max(0.0, importance))
        self._conn.execute(
            "UPDATE memories SET importance = ? WHERE id = ?", (importance, memory_id)
        )
        self._conn.commit()

    def delete_by_session(self, session_id: str, type: MemoryType) -> int:
        cursor = self._conn.execute(
            "DELETE FROM memories WHERE session_id = ? AND type = ?",
            (session_id, type.value),
        )
        self._conn.commit()
        return cursor.rowcount

    def link(self, link: MemoryLink) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO memory_links
               (src_id, dst_id, relation, weight, created_at) VALUES (?,?,?,?,?)""",
            (
                link.src_id,
                link.dst_id,
                link.relation.value,
                link.weight,
                link.created_at.isoformat(),
            ),
        )
        self._conn.commit()

    def touch(self, memory_ids: list[str]) -> None:
        """Record that memories were recalled (strengthens future recency/importance)."""
        now = datetime.now(timezone.utc).isoformat()
        self._conn.executemany(
            "UPDATE memories SET access_count = access_count + 1, last_accessed = ? "
            "WHERE id = ?",
            [(now, mid) for mid in memory_ids],
        )
        self._conn.commit()

    # -- reads ----------------------------------------------------------

    def get(self, memory_id: str) -> Memory | None:
        row = self._conn.execute(
            "SELECT * FROM memories WHERE id = ?", (memory_id,)
        ).fetchone()
        return _row_to_memory(row) if row else None

    def all(self, type: MemoryType | None = None) -> list[Memory]:
        if type:
            rows = self._conn.execute(
                "SELECT * FROM memories WHERE type = ? ORDER BY created_at", (type.value,)
            )
        else:
            rows = self._conn.execute("SELECT * FROM memories ORDER BY created_at")
        return [_row_to_memory(r) for r in rows]

    def neighbors(self, memory_id: str) -> list[tuple[Memory, LinkRelation, float]]:
        """Memories linked from/to the given one (undirected view of the graph)."""
        rows = self._conn.execute(
            """SELECT m.*, l.relation, l.weight FROM memory_links l
               JOIN memories m ON m.id = CASE WHEN l.src_id = ? THEN l.dst_id ELSE l.src_id END
               WHERE l.src_id = ? OR l.dst_id = ?""",
            (memory_id, memory_id, memory_id),
        ).fetchall()
        return [(_row_to_memory(r), LinkRelation(r["relation"]), r["weight"]) for r in rows]

    def keyword_search(self, query: str, limit: int = 50) -> list[tuple[str, float]]:
        """FTS5 match -> [(memory_id, bm25_rank)]. Lower rank = better match."""
        sanitized = " ".join(t for t in query.split() if t.isalnum())
        if not sanitized:
            return []
        rows = self._conn.execute(
            """SELECT m.id AS id, bm25(memories_fts) AS rank
               FROM memories_fts JOIN memories m ON m.rowid = memories_fts.rowid
               WHERE memories_fts MATCH ? ORDER BY rank LIMIT ?""",
            (sanitized, limit),
        ).fetchall()
        return [(r["id"], r["rank"]) for r in rows]

    def vector_search(self, query_vec: np.ndarray, limit: int = 50) -> list[tuple[str, float]]:
        """Brute-force cosine -> [(memory_id, similarity)] descending."""
        rows = self._conn.execute(
            "SELECT id, embedding FROM memories WHERE embedding IS NOT NULL"
        ).fetchall()
        if not rows:
            return []
        ids = [r["id"] for r in rows]
        matrix = np.frombuffer(
            b"".join(r["embedding"] for r in rows), dtype=np.float32
        ).reshape(len(rows), -1)
        sims = matrix @ query_vec.astype(np.float32)
        order = np.argsort(-sims)[:limit]
        return [(ids[i], float(sims[i])) for i in order]

    def close(self) -> None:
        self._conn.close()
