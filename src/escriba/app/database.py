"""SQLite database layer for transcription sessions and segments."""

from __future__ import annotations

import json
import logging
import shutil
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_DB_DIR = Path.home() / "Library" / "Application Support" / "Escriba"
_LEGACY_DB_DIR = Path.home() / "Library" / "Application Support" / "local-transcriber"
_DB_FILENAME = "transcriber.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    started_at TEXT NOT NULL,
    stopped_at TEXT,
    duration_seconds REAL,
    model TEXT,
    language TEXT,
    backend TEXT,
    status TEXT DEFAULT 'active',
    notes_text TEXT,
    parent_session_ids TEXT
);

CREATE TABLE IF NOT EXISTS segments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    start_time REAL,
    end_time REAL,
    text TEXT NOT NULL,
    speaker TEXT
);

CREATE INDEX IF NOT EXISTS idx_segments_session ON segments(session_id);
"""


class Database:
    """Thread-safe SQLite database for transcription history."""

    def __init__(self, db_path: Path | None = None):
        if db_path is None:
            db_dir = _DEFAULT_DB_DIR
            db_dir.mkdir(parents=True, exist_ok=True)
            db_path = db_dir / _DB_FILENAME
            # Migrate from old location if new DB doesn't exist yet
            if not db_path.exists():
                legacy_db = _LEGACY_DB_DIR / _DB_FILENAME
                if legacy_db.exists():
                    shutil.copy2(legacy_db, db_path)
                    logger.info(
                        "Migrated database from %s to %s", legacy_db, db_path
                    )
        self._db_path = db_path
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        logger.info("Database opened: %s", db_path)

    def create_session(
        self, name: str, model: str | None = None, language: str | None = None, backend: str | None = None
    ) -> str:
        session_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "INSERT INTO sessions (id, name, started_at, model, language, backend, status) "
            "VALUES (?, ?, ?, ?, ?, ?, 'active')",
            (session_id, name, now, model, language, backend),
        )
        self._conn.commit()
        return session_id

    def stop_session(self, session_id: str, status: str = "completed"):
        row = self._conn.execute(
            "SELECT started_at FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        duration = None
        if row:
            started = datetime.fromisoformat(row["started_at"])
            duration = (datetime.now(timezone.utc) - started).total_seconds()
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "UPDATE sessions SET stopped_at = ?, duration_seconds = ?, status = ? WHERE id = ?",
            (now, duration, status, session_id),
        )
        self._conn.commit()

    def add_segments(self, session_id: str, segments: list[dict]):
        if not segments:
            return
        self._conn.executemany(
            "INSERT INTO segments (session_id, start_time, end_time, text, speaker) "
            "VALUES (?, ?, ?, ?, ?)",
            [
                (
                    session_id,
                    s.get("start"),
                    s.get("end"),
                    s.get("text", ""),
                    s.get("speaker"),
                )
                for s in segments
            ],
        )
        self._conn.commit()

    def get_session(self, session_id: str) -> dict | None:
        row = self._conn.execute(
            "SELECT *, (SELECT COUNT(*) FROM segments WHERE session_id = s.id) AS segment_count "
            "FROM sessions s WHERE s.id = ?",
            (session_id,),
        ).fetchone()
        return dict(row) if row else None

    def list_sessions(self, limit: int = 50) -> list[dict]:
        rows = self._conn.execute(
            "SELECT s.*, (SELECT COUNT(*) FROM segments WHERE session_id = s.id) AS segment_count "
            "FROM sessions s ORDER BY s.started_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_segments(self, session_id: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM segments WHERE session_id = ? ORDER BY start_time ASC, id ASC",
            (session_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def delete_session(self, session_id: str):
        self._conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        self._conn.commit()

    def merge_sessions(self, session_ids: list[str], name: str) -> str:
        merged_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        # Gather all segments from source sessions, sorted by time
        placeholders = ",".join("?" for _ in session_ids)
        rows = self._conn.execute(
            f"SELECT start_time, end_time, text, speaker FROM segments "
            f"WHERE session_id IN ({placeholders}) ORDER BY start_time ASC, id ASC",
            session_ids,
        ).fetchall()

        self._conn.execute(
            "INSERT INTO sessions (id, name, started_at, status, parent_session_ids) "
            "VALUES (?, ?, ?, 'merged', ?)",
            (merged_id, name, now, json.dumps(session_ids)),
        )

        if rows:
            self._conn.executemany(
                "INSERT INTO segments (session_id, start_time, end_time, text, speaker) "
                "VALUES (?, ?, ?, ?, ?)",
                [(merged_id, r["start_time"], r["end_time"], r["text"], r["speaker"]) for r in rows],
            )

        self._conn.commit()
        return merged_id

    def save_notes(self, session_id: str, notes_text: str):
        self._conn.execute(
            "UPDATE sessions SET notes_text = ? WHERE id = ?",
            (notes_text, session_id),
        )
        self._conn.commit()

    def close(self):
        self._conn.close()
