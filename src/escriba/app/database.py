"""SQLite database layer for transcription sessions and segments."""

from __future__ import annotations

import json
import logging
import shutil
import sqlite3
import threading
import uuid
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_DB_DIR = Path.home() / "Library" / "Application Support" / "Escriba"
_LEGACY_DB_DIR = Path.home() / "Library" / "Application Support" / "local-transcriber"
_DB_FILENAME = "transcriber.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS folders (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    position INTEGER DEFAULT 0
);

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
    parent_session_ids TEXT,
    audio_path TEXT,
    folder_id TEXT REFERENCES folders(id) ON DELETE SET NULL
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

CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER NOT NULL
);
"""

MigrationFn = Callable[[sqlite3.Connection], None]


def _create_session_indexes(conn: sqlite3.Connection) -> None:
    """Create indexes added in schema migration 3."""
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_sessions_folder ON sessions(folder_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_sessions_status ON sessions(status)"
    )


def _dedupe_segments_and_create_unique_index(conn: sqlite3.Connection) -> None:
    """Collapse duplicate segment timings, then enforce uniqueness."""
    deleted = conn.execute(
        """
        DELETE FROM segments
        WHERE id NOT IN (
            SELECT MIN(id)
            FROM segments
            GROUP BY session_id, start_time, end_time
        )
        """
    ).rowcount
    if deleted:
        logger.info("Migration: removed %d duplicate segment row(s)", deleted)
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_segments_session_timing "
        "ON segments(session_id, start_time, end_time)"
    )


_MIGRATIONS: list[tuple[int, MigrationFn]] = [
    (
        1,
        lambda conn: _add_column_if_missing(conn, "sessions", "audio_path", "TEXT"),
    ),
    (
        2,
        lambda conn: _add_column_if_missing(
            conn,
            "sessions",
            "folder_id",
            "TEXT REFERENCES folders(id) ON DELETE SET NULL",
        ),
    ),
    (
        3,
        lambda conn: _create_session_indexes(conn),
    ),
    (
        4,
        lambda conn: _dedupe_segments_and_create_unique_index(conn),
    ),
]


def _add_column_if_missing(
    conn: sqlite3.Connection, table: str, column: str, definition: str
) -> None:
    """Add a column when upgrading legacy databases that predate the column."""
    columns = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    if column not in columns:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
        logger.info("Migration: added %s column to %s", column, table)


def _get_schema_version(conn: sqlite3.Connection) -> int:
    """Return the recorded schema version, or 0 when unset."""
    row = conn.execute("SELECT version FROM schema_version LIMIT 1").fetchone()
    return int(row[0]) if row else 0


def _set_schema_version(conn: sqlite3.Connection, version: int) -> None:
    """Persist the schema version as a single-row table."""
    conn.execute("DELETE FROM schema_version")
    conn.execute("INSERT INTO schema_version (version) VALUES (?)", (version,))


def _run_migrations(conn: sqlite3.Connection) -> None:
    """Apply pending schema migrations in order; safe to call repeatedly."""
    current = _get_schema_version(conn)
    for version, step in _MIGRATIONS:
        if version <= current:
            continue
        logger.info("Applying database migration %d", version)
        with conn:
            step(conn)
            _set_schema_version(conn, version)
        current = version


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
        self._lock = threading.RLock()
        with self._lock:
            self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.executescript(_SCHEMA)
            _run_migrations(self._conn)
            self._close_stale_sessions()
            self._conn.commit()
        logger.info("Database opened: %s", db_path)

    def _close_stale_sessions(self):
        """Mark any leftover 'active' sessions as 'completed' on startup."""
        cursor = self._conn.execute(
            "UPDATE sessions SET status = 'completed' WHERE status = 'active'"
        )
        if cursor.rowcount > 0:
            logger.info("Closed %d stale active session(s)", cursor.rowcount)

    # --- Sessions ---

    def create_session(
        self, name: str, model: str | None = None, language: str | None = None, backend: str | None = None
    ) -> str:
        with self._lock:
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
        with self._lock:
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
        with self._lock:
            self._conn.executemany(
                "INSERT OR IGNORE INTO segments (session_id, start_time, end_time, text, speaker) "
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
        with self._lock:
            row = self._conn.execute(
                "SELECT *, (SELECT COUNT(*) FROM segments WHERE session_id = s.id) AS segment_count "
                "FROM sessions s WHERE s.id = ?",
                (session_id,),
            ).fetchone()
            return dict(row) if row else None

    def list_sessions(self, limit: int = 100) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT s.*, (SELECT COUNT(*) FROM segments WHERE session_id = s.id) AS segment_count "
                "FROM sessions s ORDER BY s.started_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_segment(self, segment_id: int) -> dict | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM segments WHERE id = ?", (segment_id,)
            ).fetchone()
            return dict(row) if row else None

    def get_segments(self, session_id: str) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM segments WHERE session_id = ? ORDER BY start_time ASC, id ASC",
                (session_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    def delete_segments(self, session_id: str):
        with self._lock:
            self._conn.execute("DELETE FROM segments WHERE session_id = ?", (session_id,))
            self._conn.commit()

    def update_audio_path(self, session_id: str, audio_path: str):
        with self._lock:
            self._conn.execute(
                "UPDATE sessions SET audio_path = ? WHERE id = ?",
                (audio_path, session_id),
            )
            self._conn.commit()

    def rename_session(self, session_id: str, name: str):
        with self._lock:
            self._conn.execute(
                "UPDATE sessions SET name = ? WHERE id = ?",
                (name, session_id),
            )
            self._conn.commit()

    def move_session_to_folder(self, session_id: str, folder_id: str | None):
        with self._lock:
            self._conn.execute(
                "UPDATE sessions SET folder_id = ? WHERE id = ?",
                (folder_id, session_id),
            )
            self._conn.commit()

    def move_sessions_to_folder(self, session_ids: list[str], folder_id: str | None):
        with self._lock:
            placeholders = ",".join("?" for _ in session_ids)
            self._conn.execute(
                f"UPDATE sessions SET folder_id = ? WHERE id IN ({placeholders})",
                [folder_id, *session_ids],
            )
            self._conn.commit()

    def delete_session(self, session_id: str):
        with self._lock:
            row = self._conn.execute(
                "SELECT audio_path FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
            if row and row["audio_path"]:
                audio_file = Path(row["audio_path"])
                if audio_file.exists():
                    audio_file.unlink()
                    logger.info("Deleted audio file: %s", audio_file)
            self._conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            self._conn.commit()

    def split_session(
        self, session_id: str, segment_id: int
    ) -> tuple[str, float, str | None]:
        """Split a session into two at the start of `segment_id`.

        Returns ``(new_session_id, split_time_seconds, original_audio_path)``.
        The caller is responsible for physically slicing the WAV file —
        this method only rearranges DB rows, atomically in one transaction.

        Raises ``ValueError`` if the session or segment can't be split
        (missing, not completed, segment not in session, split at index 0).
        """
        with self._lock:
            session = self.get_session(session_id)
            if not session:
                raise ValueError(f"Session not found: {session_id}")
            if session.get("status") == "active":
                raise ValueError("Cannot split an active (recording) session")

            segment_row = self._conn.execute(
                "SELECT id, session_id, start_time FROM segments WHERE id = ?",
                (segment_id,),
            ).fetchone()
            if not segment_row or segment_row["session_id"] != session_id:
                raise ValueError(
                    f"Segment {segment_id} not found in session {session_id}"
                )

            split_time = float(segment_row["start_time"] or 0.0)
            if split_time <= 0:
                raise ValueError("Cannot split at the first segment")

            # Confirm there's at least one segment on each side.
            before = self._conn.execute(
                "SELECT COUNT(*) FROM segments WHERE session_id = ? AND start_time < ?",
                (session_id, split_time),
            ).fetchone()[0]
            after = self._conn.execute(
                "SELECT COUNT(*) FROM segments WHERE session_id = ? AND start_time >= ?",
                (session_id, split_time),
            ).fetchone()[0]
            if before == 0 or after == 0:
                raise ValueError("Split would leave one side empty")

            new_id = str(uuid.uuid4())
            orig_name = session.get("name") or "Session"
            orig_audio_path = session.get("audio_path")
            part1_name = f"{orig_name} (part 1)"
            part2_name = f"{orig_name} (part 2)"

            # Derive timestamps for the second half.
            started_at_iso = session.get("started_at")
            if isinstance(started_at_iso, str):
                try:
                    orig_started = datetime.fromisoformat(started_at_iso)
                except ValueError:
                    orig_started = datetime.now(timezone.utc)
            else:
                orig_started = datetime.now(timezone.utc)
            part2_started_iso = (
                orig_started + timedelta(seconds=split_time)
            ).isoformat()

            orig_duration = session.get("duration_seconds")
            if orig_duration is None:
                orig_duration = max(split_time, split_time + 1.0)
            part2_duration = max(float(orig_duration) - split_time, 0.0)

            with self._conn:
                # Create the second-half session row.
                self._conn.execute(
                    "INSERT INTO sessions ("
                    "id, name, started_at, stopped_at, duration_seconds, "
                    "model, language, backend, status, folder_id"
                    ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'completed', ?)",
                    (
                        new_id,
                        part2_name,
                        part2_started_iso,
                        session.get("stopped_at"),
                        part2_duration,
                        session.get("model"),
                        session.get("language"),
                        session.get("backend"),
                        session.get("folder_id"),
                    ),
                )

                # Move the tail segments onto the new session and rebase their
                # timestamps so both halves start at 0.
                self._conn.execute(
                    "UPDATE segments SET "
                    "session_id = ?, "
                    "start_time = start_time - ?, "
                    "end_time = CASE WHEN end_time IS NULL THEN NULL "
                    "                ELSE end_time - ? END "
                    "WHERE session_id = ? AND start_time >= ?",
                    (new_id, split_time, split_time, session_id, split_time),
                )

                # Rename and shorten the original (which keeps its notes).
                self._conn.execute(
                    "UPDATE sessions SET name = ?, duration_seconds = ? WHERE id = ?",
                    (part1_name, split_time, session_id),
                )

            return new_id, split_time, orig_audio_path

    def merge_sessions(
        self, session_ids: list[str], name: str
    ) -> tuple[str, list[tuple[str | None, float, float]]]:
        """Merge `session_ids` into a new `merged`-status session.

        Returns ``(merged_id, sources)``. ``sources`` is a list, ordered by
        `started_at`, of ``(audio_path, offset_seconds, duration_seconds)``.
        The caller uses it to concatenate the physical WAV files with the
        same offsets used to rebase segment timestamps.
        """
        with self._lock:
            merged_id = str(uuid.uuid4())
            now = datetime.now(timezone.utc).isoformat()

            # Sort source sessions by start time so the merged timeline is
            # chronological and matches the order the user recorded them.
            placeholders = ",".join("?" for _ in session_ids)
            src_rows = self._conn.execute(
                f"SELECT id, started_at, duration_seconds, audio_path "
                f"FROM sessions WHERE id IN ({placeholders}) "
                f"ORDER BY started_at ASC",
                session_ids,
            ).fetchall()

            offsets: list[tuple[str | None, float, float]] = []
            cumulative = 0.0
            for row in src_rows:
                dur = float(row["duration_seconds"] or 0.0)
                offsets.append((row["audio_path"], cumulative, dur))
                cumulative += dur

            merged_started_iso = src_rows[0]["started_at"] if src_rows else now

            with self._conn:
                self._conn.execute(
                    "INSERT INTO sessions ("
                    "id, name, started_at, duration_seconds, "
                    "status, parent_session_ids"
                    ") VALUES (?, ?, ?, ?, 'merged', ?)",
                    (
                        merged_id,
                        name,
                        merged_started_iso,
                        cumulative or None,
                        json.dumps(session_ids),
                    ),
                )

                # Rebase each source session's segments by that session's
                # offset so the merged timeline is continuous from 0.
                for src_id, offset, _dur in [
                    (row["id"], off, dur)
                    for row, (_ap, off, dur) in zip(src_rows, offsets)
                ]:
                    expected = self._conn.execute(
                        "SELECT COUNT(*) FROM segments WHERE session_id = ?",
                        (src_id,),
                    ).fetchone()[0]
                    cursor = self._conn.execute(
                        "INSERT OR IGNORE INTO segments "
                        "(session_id, start_time, end_time, text, speaker) "
                        "SELECT ?, "
                        "       start_time + ?, "
                        "       CASE WHEN end_time IS NULL THEN NULL "
                        "            ELSE end_time + ? END, "
                        "       text, speaker "
                        "FROM segments WHERE session_id = ? "
                        "ORDER BY start_time ASC, id ASC",
                        (merged_id, offset, offset, src_id),
                    )
                    dropped = expected - cursor.rowcount
                    if dropped > 0:
                        logger.warning(
                            "Merge ignored %d duplicate segment(s) from session %s",
                            dropped,
                            src_id,
                        )

            return merged_id, offsets

    def save_notes(self, session_id: str, notes_text: str):
        with self._lock:
            self._conn.execute(
                "UPDATE sessions SET notes_text = ? WHERE id = ?",
                (notes_text, session_id),
            )
            self._conn.commit()

    # --- Folders ---

    def create_folder(self, name: str) -> str:
        with self._lock:
            folder_id = str(uuid.uuid4())
            max_pos = self._conn.execute(
                "SELECT COALESCE(MAX(position), -1) + 1 FROM folders"
            ).fetchone()[0]
            self._conn.execute(
                "INSERT INTO folders (id, name, position) VALUES (?, ?, ?)",
                (folder_id, name, max_pos),
            )
            self._conn.commit()
            return folder_id

    def list_folders(self) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT f.*, (SELECT COUNT(*) FROM sessions WHERE folder_id = f.id) AS session_count "
                "FROM folders f ORDER BY f.position ASC"
            ).fetchall()
            return [dict(r) for r in rows]

    def rename_folder(self, folder_id: str, name: str):
        with self._lock:
            self._conn.execute(
                "UPDATE folders SET name = ? WHERE id = ?",
                (name, folder_id),
            )
            self._conn.commit()

    def delete_folder(self, folder_id: str):
        with self._lock:
            # Move sessions out of folder before deleting
            self._conn.execute(
                "UPDATE sessions SET folder_id = NULL WHERE folder_id = ?",
                (folder_id,),
            )
            self._conn.execute("DELETE FROM folders WHERE id = ?", (folder_id,))
            self._conn.commit()

    def close(self):
        with self._lock:
            self._conn.close()
