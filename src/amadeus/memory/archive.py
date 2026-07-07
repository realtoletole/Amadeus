"""Conversation archive.

Persists every session and turn verbatim. This is both the source of
in-conversation context (recent turns) and the raw material the Phase 3
consolidation job mines for episodic/semantic memories.
"""

from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id           TEXT PRIMARY KEY,
    started_at   TEXT NOT NULL,
    ended_at     TEXT,
    consolidated INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS turns (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role        TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content     TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    interrupted INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_turns_session ON turns(session_id);
"""


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Turn(BaseModel):
    role: str
    content: str
    created_at: datetime
    interrupted: bool = False


class ConversationArchive:
    def __init__(self, db_path: Path | str) -> None:
        # check_same_thread=False: single-user app; access is sequential but
        # may hop threads (test client, UI server worker threads).
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(_SCHEMA)
        self._migrate()

    def _migrate(self) -> None:
        """Add columns introduced after v0.3 to pre-existing databases."""
        columns = {r[1] for r in self._conn.execute("PRAGMA table_info(sessions)")}
        if "consolidated" not in columns:
            self._conn.execute(
                "ALTER TABLE sessions ADD COLUMN consolidated INTEGER NOT NULL DEFAULT 0"
            )
            self._conn.commit()

    def start_session(self) -> str:
        session_id = uuid.uuid4().hex
        self._conn.execute(
            "INSERT INTO sessions (id, started_at) VALUES (?, ?)",
            (session_id, _now().isoformat()),
        )
        self._conn.commit()
        return session_id

    def end_session(self, session_id: str) -> None:
        self._conn.execute(
            "UPDATE sessions SET ended_at = ? WHERE id = ?",
            (_now().isoformat(), session_id),
        )
        self._conn.commit()

    def add_turn(
        self, session_id: str, role: str, content: str, *, interrupted: bool = False
    ) -> None:
        self._conn.execute(
            "INSERT INTO turns (session_id, role, content, created_at, interrupted) "
            "VALUES (?, ?, ?, ?, ?)",
            (session_id, role, content, _now().isoformat(), int(interrupted)),
        )
        self._conn.commit()

    def recent_turns(self, session_id: str, limit: int = 20) -> list[Turn]:
        rows = self._conn.execute(
            "SELECT role, content, created_at, interrupted FROM turns "
            "WHERE session_id = ? ORDER BY id DESC LIMIT ?",
            (session_id, limit),
        ).fetchall()
        return [
            Turn(
                role=r["role"],
                content=r["content"],
                created_at=datetime.fromisoformat(r["created_at"]),
                interrupted=bool(r["interrupted"]),
            )
            for r in reversed(rows)
        ]

    def all_turns(self, session_id: str) -> list[Turn]:
        rows = self._conn.execute(
            "SELECT role, content, created_at, interrupted FROM turns "
            "WHERE session_id = ? ORDER BY id",
            (session_id,),
        ).fetchall()
        return [
            Turn(
                role=r["role"],
                content=r["content"],
                created_at=datetime.fromisoformat(r["created_at"]),
                interrupted=bool(r["interrupted"]),
            )
            for r in rows
        ]

    def unconsolidated_sessions(self) -> list[str]:
        """Ended sessions whose memories were never consolidated (e.g. the
        app was killed before the job ran). Retried at startup."""
        rows = self._conn.execute(
            "SELECT id FROM sessions WHERE ended_at IS NOT NULL AND consolidated = 0 "
            "ORDER BY ended_at"
        ).fetchall()
        return [r["id"] for r in rows]

    def mark_consolidated(self, session_id: str) -> None:
        self._conn.execute(
            "UPDATE sessions SET consolidated = 1 WHERE id = ?", (session_id,)
        )
        self._conn.commit()

    def last_session_ended_at(self, *, exclude: str | None = None) -> datetime | None:
        """When the most recent *previous* session ended (for time awareness)."""
        row = self._conn.execute(
            "SELECT MAX(ended_at) AS ended FROM sessions "
            "WHERE ended_at IS NOT NULL AND id != COALESCE(?, '')",
            (exclude,),
        ).fetchone()
        return datetime.fromisoformat(row["ended"]) if row and row["ended"] else None

    def close(self) -> None:
        self._conn.close()
