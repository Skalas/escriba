"""Tests for database hardening: atomic mutations, concurrency, migrations."""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

import pytest

from escriba.app.database import (
    Database,
    _MIGRATIONS,
    _get_schema_version,
    _run_migrations,
)


LATEST_SCHEMA_VERSION = max(version for version, _ in _MIGRATIONS)


@pytest.fixture
def db(tmp_path: Path) -> Database:
    database = Database(tmp_path / "test.db")
    yield database
    database.close()


def _seed_completed_session(
    database: Database,
    name: str = "Session",
    *,
    duration: float = 120.0,
) -> str:
    session_id = database.create_session(name=name)
    database.stop_session(session_id, status="completed")
    database._conn.execute(
        "UPDATE sessions SET duration_seconds = ? WHERE id = ?",
        (duration, session_id),
    )
    database._conn.commit()
    return session_id


def _add_segments(
    database: Database,
    session_id: str,
    segments: list[tuple[float, str]],
) -> list[int]:
    database.add_segments(
        session_id,
        [
            {"start": start, "end": start + 1.0, "text": text}
            for start, text in segments
        ],
    )
    rows = database.get_segments(session_id)
    return [row["id"] for row in rows]


def _snapshot_session(database: Database, session_id: str) -> dict:
    return {
        "session": database.get_session(session_id),
        "segments": database.get_segments(session_id),
    }


def _assert_segment_integrity(database: Database) -> None:
    """Every segment belongs to an existing session; ids are unique."""
    session_ids = {row["id"] for row in database.list_sessions(limit=1000)}
    rows = database._conn.execute(
        "SELECT id, session_id FROM segments ORDER BY id"
    ).fetchall()
    seen: set[int] = set()
    for row in rows:
        assert row["session_id"] in session_ids
        assert row["id"] not in seen
        seen.add(row["id"])


def test_t4_split_session_rolls_back_on_mid_operation_failure(db: Database) -> None:
    """T4: a failed split leaves no partial session/segment moves."""
    session_id = _seed_completed_session(db)
    seg_ids = _add_segments(
        db,
        session_id,
        [(0.0, "intro"), (10.0, "middle"), (20.0, "outro")],
    )
    split_segment_id = seg_ids[1]
    before = _snapshot_session(db, session_id)
    session_count_before = len(db.list_sessions(limit=1000))

    real_conn = db._conn
    fail_state = {"pending": False, "armed": False}

    class _ConnWrapper:
        """Delegate to the real connection but fail mid-split for rollback tests."""

        def __init__(self, inner: sqlite3.Connection) -> None:
            self._inner = inner

        def execute(self, sql, parameters=()):
            sql_text = sql if isinstance(sql, str) else str(sql)
            if fail_state["armed"] and "UPDATE segments SET" in sql_text:
                raise sqlite3.OperationalError("forced mid-split failure")
            if fail_state["pending"] and "INSERT INTO sessions" in sql_text:
                fail_state["armed"] = True
            return self._inner.execute(sql, parameters)

        def __enter__(self):
            return self._inner.__enter__()

        def __exit__(self, exc_type, exc, tb):
            return self._inner.__exit__(exc_type, exc, tb)

        def __getattr__(self, name: str):
            return getattr(self._inner, name)

    db._conn = _ConnWrapper(real_conn)
    fail_state["pending"] = True
    with pytest.raises(sqlite3.OperationalError, match="forced mid-split"):
        db.split_session(session_id, split_segment_id)
    db._conn = real_conn

    after = _snapshot_session(db, session_id)
    assert len(db.list_sessions(limit=1000)) == session_count_before
    assert after["session"]["name"] == before["session"]["name"]
    assert after["segments"] == before["segments"]
    assert db.get_session(session_id) is not None


def test_t5_concurrent_split_and_merge_do_not_corrupt_db(db: Database) -> None:
    """T5: concurrent split + merge keep segment/session invariants."""
    split_source = _seed_completed_session(db, name="Split me", duration=90.0)
    split_seg_ids = _add_segments(
        db,
        split_source,
        [(0.0, "a"), (15.0, "b"), (30.0, "c"), (45.0, "d")],
    )
    merge_a = _seed_completed_session(db, name="Merge A", duration=30.0)
    merge_b = _seed_completed_session(db, name="Merge B", duration=40.0)
    _add_segments(db, merge_a, [(0.0, "ma"), (5.0, "mb")])
    _add_segments(db, merge_b, [(0.0, "mc"), (8.0, "md")])

    errors: list[Exception] = []
    barrier = threading.Barrier(2)

    def split_worker() -> None:
        try:
            barrier.wait(timeout=5)
            db.split_session(split_source, split_seg_ids[2])
        except Exception as exc:
            errors.append(exc)

    def merge_worker() -> None:
        try:
            barrier.wait(timeout=5)
            db.merge_sessions([merge_a, merge_b], name="Merged")
        except Exception as exc:
            errors.append(exc)

    t_split = threading.Thread(target=split_worker)
    t_merge = threading.Thread(target=merge_worker)
    t_split.start()
    t_merge.start()
    t_split.join(timeout=10)
    t_merge.join(timeout=10)

    assert not errors, errors
    _assert_segment_integrity(db)

    all_segments = db._conn.execute("SELECT COUNT(*) FROM segments").fetchone()[0]
    assert all_segments >= 6


def test_t6_migration_runner_is_idempotent(tmp_path: Path) -> None:
    """T6: init/migrate twice yields stable schema_version and indexes."""
    db_path = tmp_path / "fresh.db"
    db = Database(db_path)
    try:
        assert _get_schema_version(db._conn) == LATEST_SCHEMA_VERSION
        index_names = {
            row[0]
            for row in db._conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='index' AND name LIKE 'idx_sessions_%'"
            ).fetchall()
        }
        assert index_names == {"idx_sessions_folder", "idx_sessions_status"}

        _run_migrations(db._conn)
        assert _get_schema_version(db._conn) == LATEST_SCHEMA_VERSION
    finally:
        db.close()

    db_reopen = Database(db_path)
    try:
        assert _get_schema_version(db_reopen._conn) == LATEST_SCHEMA_VERSION
        _run_migrations(db_reopen._conn)
        assert _get_schema_version(db_reopen._conn) == LATEST_SCHEMA_VERSION
    finally:
        db_reopen.close()
