"""Tests for database hardening: atomic mutations, concurrency, migrations."""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

import pytest

from escriba.app.database import (
    Database,
    _MIGRATIONS,
    _SCHEMA,
    _get_schema_version,
    _run_migrations,
    _set_schema_version,
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


def test_concurrent_split_and_add_segments(db: Database) -> None:
    """Split vs concurrent add_segments: split is all-or-nothing; writer intact."""
    split_source = _seed_completed_session(db, name="Split target", duration=60.0)
    split_seg_ids = _add_segments(
        db,
        split_source,
        [(0.0, "a"), (10.0, "b"), (20.0, "c"), (30.0, "d")],
    )
    split_segment_id = split_seg_ids[2]

    writer_id = db.create_session(name="Live writer")
    writer_target = 50
    writer_errors: list[Exception] = []
    split_errors: list[Exception] = []
    ready = threading.Barrier(2)

    def writer_worker() -> None:
        try:
            ready.wait(timeout=5)
            for count in range(writer_target):
                db.add_segments(
                    writer_id,
                    [
                        {
                            "start": float(count),
                            "end": float(count) + 1.0,
                            "text": f"live-{count}",
                        }
                    ],
                )
        except Exception as exc:
            writer_errors.append(exc)

    def split_worker() -> None:
        try:
            ready.wait(timeout=5)
            db.split_session(split_source, split_segment_id)
        except Exception as exc:
            split_errors.append(exc)

    t_writer = threading.Thread(target=writer_worker)
    t_split = threading.Thread(target=split_worker)
    t_writer.start()
    t_split.start()
    t_split.join(timeout=15)
    t_writer.join(timeout=15)

    assert not split_errors, split_errors
    assert not writer_errors, writer_errors

    original_segments = db.get_segments(split_source)
    assert len(original_segments) == 2
    assert {s["text"] for s in original_segments} == {"a", "b"}

    sessions = db.list_sessions(limit=100)
    part2_candidates = [
        s for s in sessions if s["id"] not in {split_source, writer_id}
    ]
    assert len(part2_candidates) == 1
    part2_id = part2_candidates[0]["id"]
    part2_segments = db.get_segments(part2_id)
    assert len(part2_segments) == 2
    assert {s["text"] for s in part2_segments} == {"c", "d"}

    writer_segments = db.get_segments(writer_id)
    assert len(writer_segments) == writer_target
    assert {s["text"] for s in writer_segments} == {
        f"live-{i}" for i in range(writer_target)
    }

    _assert_segment_integrity(db)


def test_t4_add_segments_ignores_duplicate_timing(db: Database) -> None:
    """T4: duplicate (session_id, start_time, end_time) inserts are skipped."""
    session_id = db.create_session(name="Dedup")
    segment = {"start": 1.0, "end": 2.0, "text": "first"}
    db.add_segments(session_id, [segment])
    db.add_segments(
        session_id,
        [{"start": 1.0, "end": 2.0, "text": "duplicate"}],
    )

    rows = db.get_segments(session_id)
    assert len(rows) == 1
    assert rows[0]["text"] == "first"


def test_b1_merge_sessions_ignores_colliding_segment_timings(db: Database) -> None:
    """B1: merge survives when rebased segments share (start_time, end_time)."""
    session_a = _seed_completed_session(db, name="Merge A", duration=0.0)
    session_b = _seed_completed_session(db, name="Merge B", duration=0.0)
    db._conn.execute(
        "UPDATE sessions SET duration_seconds = NULL WHERE id = ?",
        (session_a,),
    )
    db._conn.commit()

    db.add_segments(
        session_a,
        [{"start": 0.0, "end": 1.0, "text": "from-a"}],
    )
    db.add_segments(
        session_b,
        [{"start": 0.0, "end": 1.0, "text": "from-b"}],
    )

    merged_id, _sources = db.merge_sessions([session_a, session_b], name="Merged")

    segments = db.get_segments(merged_id)
    assert len(segments) == 1
    assert segments[0]["text"] == "from-a"


def test_t4_migration_dedupes_existing_rows_and_creates_unique_index(
    tmp_path: Path,
) -> None:
    """T4: migration 4 collapses legacy duplicates then adds the unique index."""
    db_path = tmp_path / "legacy-duplicates.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(_SCHEMA)
    _set_schema_version(conn, 3)

    session_id = "session-with-dupes"
    now = "2026-01-01T00:00:00+00:00"
    conn.execute(
        "INSERT INTO sessions (id, name, started_at, status) VALUES (?, ?, ?, 'completed')",
        (session_id, "Legacy", now),
    )
    for text in ("keep-me", "drop-me", "drop-me-too"):
        conn.execute(
            "INSERT INTO segments (session_id, start_time, end_time, text) "
            "VALUES (?, ?, ?, ?)",
            (session_id, 5.0, 6.0, text),
        )
    conn.commit()
    conn.close()

    db = Database(db_path)
    try:
        assert _get_schema_version(db._conn) == LATEST_SCHEMA_VERSION
        rows = db.get_segments(session_id)
        assert len(rows) == 1
        assert rows[0]["text"] == "keep-me"

        index_row = db._conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type = 'index' AND name = 'idx_segments_session_timing'"
        ).fetchone()
        assert index_row is not None

        _run_migrations(db._conn)
        assert _get_schema_version(db._conn) == LATEST_SCHEMA_VERSION
        assert len(db.get_segments(session_id)) == 1
    finally:
        db.close()


def test_t4_segment_dedup_migration_is_idempotent(tmp_path: Path) -> None:
    """T4: running migration 4 twice does not error or drop surviving rows."""
    db_path = tmp_path / "legacy-idempotent.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(_SCHEMA)
    _set_schema_version(conn, 3)

    session_id = "session-idempotent"
    now = "2026-01-01T00:00:00+00:00"
    conn.execute(
        "INSERT INTO sessions (id, name, started_at, status) VALUES (?, ?, ?, 'completed')",
        (session_id, "Legacy", now),
    )
    conn.execute(
        "INSERT INTO segments (session_id, start_time, end_time, text) "
        "VALUES (?, ?, ?, ?)",
        (session_id, 0.0, 1.0, "only-one"),
    )
    conn.commit()
    conn.close()

    db = Database(db_path)
    try:
        _run_migrations(db._conn)
        _run_migrations(db._conn)
        assert len(db.get_segments(session_id)) == 1
        assert db.get_segments(session_id)[0]["text"] == "only-one"
    finally:
        db.close()
