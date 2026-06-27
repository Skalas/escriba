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
    _create_speaker_labels_table,
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


def test_merge_sessions_logs_when_segments_are_dropped(db: Database, caplog) -> None:
    """Merge collisions are visible via WARNING, not silently ignored."""
    import logging

    caplog.set_level(logging.WARNING)

    session_a = _seed_completed_session(db, name="Merge A", duration=0.0)
    session_b = _seed_completed_session(db, name="Merge B", duration=0.0)
    db.add_segments(
        session_a,
        [{"start": 0.0, "end": 1.0, "text": "from-a"}],
    )
    db.add_segments(
        session_b,
        [{"start": 0.0, "end": 1.0, "text": "from-b"}],
    )

    db.merge_sessions([session_a, session_b], name="Merged")

    assert any(
        "ignored 1 duplicate segment" in record.getMessage()
        for record in caplog.records
        if record.levelname == "WARNING"
    )


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


def test_search_segments_finds_matches_across_sessions(db: Database) -> None:
    """Cross-session search returns segment matches from multiple sessions."""
    session_a = _seed_completed_session(db, name="Team sync")
    session_b = _seed_completed_session(db, name="Planning call")
    _add_segments(db, session_a, [(0.0, "We discussed the quarterly roadmap")])
    _add_segments(db, session_b, [(5.0, "The quarterly budget needs review")])

    results = db.search_segments("quarterly")
    assert len(results) == 2
    session_ids = {row["session_id"] for row in results}
    assert session_ids == {session_a, session_b}
    texts = {row["snippet"] for row in results}
    assert "We discussed the quarterly roadmap" in texts
    assert "The quarterly budget needs review" in texts


def test_search_segments_is_case_insensitive(db: Database) -> None:
    """Search ignores letter case in segment text."""
    session_id = _seed_completed_session(db, name="Case test")
    _add_segments(db, session_id, [(0.0, "Hello WORLD from Escriba")])

    results = db.search_segments("world")
    assert len(results) == 1
    assert "WORLD" in results[0]["snippet"]


def test_search_segments_respects_limit(db: Database) -> None:
    """Search returns at most the requested number of rows."""
    session_id = _seed_completed_session(db, name="Limit test")
    _add_segments(
        db,
        session_id,
        [(float(i), f"needle segment number {i}") for i in range(10)],
    )

    results = db.search_segments("needle", limit=3)
    assert len(results) == 3


def test_search_segments_escapes_like_wildcards(db: Database) -> None:
    """A literal % in the query must not match every row."""
    session_id = _seed_completed_session(db, name="Wildcard test")
    _add_segments(db, session_id, [(0.0, "100% complete"), (1.0, "no wildcards here")])

    results = db.search_segments("%")
    assert len(results) == 1
    assert results[0]["snippet"] == "100% complete"


def test_search_segments_matches_session_name(db: Database) -> None:
    """Search also matches session names, returning one representative segment."""
    session_id = _seed_completed_session(db, name="Acme Interview Panel")
    seg_ids = _add_segments(db, session_id, [(0.0, "Tell me about yourself")])

    results = db.search_segments("Acme")
    assert len(results) == 1
    assert results[0]["session_id"] == session_id
    assert results[0]["id"] == seg_ids[0]
    assert results[0]["session_name"] == "Acme Interview Panel"


def test_search_segments_name_match_does_not_flood_all_segments(db: Database) -> None:
    """Session-name matches return one row per session, not every segment."""
    session_id = _seed_completed_session(db, name="Acme Weekly Review")
    _add_segments(
        db,
        session_id,
        [(float(i), f"segment {i} with no keyword") for i in range(20)],
    )

    results = db.search_segments("Acme")
    assert len(results) == 1
    assert results[0]["session_id"] == session_id


def test_search_segments_orders_by_session_started_at_desc(db: Database) -> None:
    """Newer sessions appear before older ones in search results."""
    older = _seed_completed_session(db, name="Older session")
    newer = _seed_completed_session(db, name="Newer session")
    db._conn.execute(
        "UPDATE sessions SET started_at = ? WHERE id = ?",
        ("2026-01-01T00:00:00+00:00", older),
    )
    db._conn.execute(
        "UPDATE sessions SET started_at = ? WHERE id = ?",
        ("2026-06-01T00:00:00+00:00", newer),
    )
    db._conn.commit()
    _add_segments(db, older, [(0.0, "shared keyword alpha")])
    _add_segments(db, newer, [(0.0, "shared keyword beta")])

    results = db.search_segments("shared keyword")
    assert [row["session_id"] for row in results] == [newer, older]


def test_set_speaker_label_upsert_and_blank_deletes(db: Database) -> None:
    """Custom speaker names upsert; blank display name removes the mapping."""
    session_id = _seed_completed_session(db, name="Interview")
    _add_segments(
        db,
        session_id,
        [(0.0, "Hello"), (5.0, "Hi there")],
    )
    db._conn.execute(
        "UPDATE segments SET speaker = ? WHERE session_id = ? AND start_time = 0.0",
        ("SPEAKER_00", session_id),
    )
    db._conn.execute(
        "UPDATE segments SET speaker = ? WHERE session_id = ? AND start_time = 5.0",
        ("SPEAKER_01", session_id),
    )
    db._conn.commit()

    db.set_speaker_label(session_id, "SPEAKER_00", "Alice")
    assert db.get_speaker_labels(session_id) == {"SPEAKER_00": "Alice"}

    db.set_speaker_label(session_id, "SPEAKER_00", "Alicia")
    assert db.get_speaker_labels(session_id) == {"SPEAKER_00": "Alicia"}

    db.set_speaker_label(session_id, "SPEAKER_00", "   ")
    assert db.get_speaker_labels(session_id) == {}


def test_get_segments_includes_speaker_display(db: Database) -> None:
    """Segments expose speaker_display from the mapping, or raw when unset."""
    session_id = _seed_completed_session(db, name="Labels")
    _add_segments(db, session_id, [(0.0, "mapped"), (5.0, "raw only")])
    db._conn.execute(
        "UPDATE segments SET speaker = 'SPEAKER_00' WHERE session_id = ? AND start_time = 0.0",
        (session_id,),
    )
    db._conn.execute(
        "UPDATE segments SET speaker = 'SPEAKER_01' WHERE session_id = ? AND start_time = 5.0",
        (session_id,),
    )
    db._conn.commit()
    db.set_speaker_label(session_id, "SPEAKER_00", "Bob")

    segments = db.get_segments(session_id)
    by_start = {seg["start_time"]: seg for seg in segments}
    assert by_start[0.0]["speaker"] == "SPEAKER_00"
    assert by_start[0.0]["speaker_display"] == "Bob"
    assert by_start[5.0]["speaker"] == "SPEAKER_01"
    assert by_start[5.0]["speaker_display"] == "SPEAKER_01"


def test_list_speakers_returns_distinct_keys_with_labels(db: Database) -> None:
    """list_speakers enumerates unique raw keys with resolved display names."""
    session_id = _seed_completed_session(db, name="Panel")
    _add_segments(
        db,
        session_id,
        [(0.0, "a"), (5.0, "b"), (10.0, "c"), (15.0, "d")],
    )
    db._conn.execute(
        "UPDATE segments SET speaker = 'SPEAKER_00' WHERE session_id = ? AND start_time IN (0.0, 5.0)",
        (session_id,),
    )
    db._conn.execute(
        "UPDATE segments SET speaker = 'SPEAKER_01' WHERE session_id = ? AND start_time IN (10.0, 15.0)",
        (session_id,),
    )
    db._conn.commit()
    db.set_speaker_label(session_id, "SPEAKER_01", "Carol")

    speakers = db.list_speakers(session_id)
    assert speakers == [
        {"speaker": "SPEAKER_00", "display_name": "SPEAKER_00"},
        {"speaker": "SPEAKER_01", "display_name": "Carol"},
    ]


def test_speaker_labels_migration_is_idempotent_with_existing_segments(
    tmp_path: Path,
) -> None:
    """Migration 5 creates speaker_labels on a DB that already has segments."""
    db_path = tmp_path / "legacy-speakers.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(_SCHEMA)
    _set_schema_version(conn, 4)

    session_id = "session-with-speakers"
    now = "2026-01-01T00:00:00+00:00"
    conn.execute(
        "INSERT INTO sessions (id, name, started_at, status) VALUES (?, ?, ?, 'completed')",
        (session_id, "Legacy", now),
    )
    conn.execute(
        "INSERT INTO segments (session_id, start_time, end_time, text, speaker) "
        "VALUES (?, ?, ?, ?, ?)",
        (session_id, 0.0, 1.0, "hello", "SPEAKER_00"),
    )
    conn.commit()
    conn.close()

    db = Database(db_path)
    try:
        assert _get_schema_version(db._conn) == LATEST_SCHEMA_VERSION
        table_row = db._conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type = 'table' AND name = 'speaker_labels'"
        ).fetchone()
        assert table_row is not None

        db.set_speaker_label(session_id, "SPEAKER_00", "Legacy Name")
        assert db.get_segments(session_id)[0]["speaker_display"] == "Legacy Name"

        _create_speaker_labels_table(db._conn)
        _run_migrations(db._conn)
        assert _get_schema_version(db._conn) == LATEST_SCHEMA_VERSION
        assert db.get_speaker_labels(session_id) == {"SPEAKER_00": "Legacy Name"}
    finally:
        db.close()


def _create_orphan_wav(db_path: Path, session_id: str) -> Path:
    """Create a canonical WAV file for a session under the db's audio dir."""
    audio_dir = db_path.parent / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    wav_path = audio_dir / f"{session_id}.wav"
    wav_path.write_bytes(b"RIFF" + b"\x00" * 12)
    return wav_path


def test_get_session_relinks_orphaned_audio(tmp_path: Path) -> None:
    """get_session backfills audio_path when the canonical WAV exists."""
    db_path = tmp_path / "transcriber.db"
    db = Database(db_path)
    session_id = _seed_completed_session(db)
    db._conn.execute(
        "UPDATE sessions SET audio_path = NULL WHERE id = ?",
        (session_id,),
    )
    db._conn.commit()
    wav_path = _create_orphan_wav(db_path, session_id)

    session = db.get_session(session_id)

    assert session is not None
    assert session["audio_path"] == str(wav_path.resolve())
    row = db._conn.execute(
        "SELECT audio_path FROM sessions WHERE id = ?",
        (session_id,),
    ).fetchone()
    assert row["audio_path"] == str(wav_path.resolve())
    db.close()


def test_relink_orphaned_audio_batch(tmp_path: Path) -> None:
    """relink_orphaned_audio links every session whose WAV file is present."""
    db_path = tmp_path / "transcriber.db"
    db = Database(db_path)
    linked_id = _seed_completed_session(db, name="Has WAV")
    missing_id = _seed_completed_session(db, name="No WAV")
    for sid in (linked_id, missing_id):
        db._conn.execute(
            "UPDATE sessions SET audio_path = '' WHERE id = ?",
            (sid,),
        )
    db._conn.commit()
    wav_path = _create_orphan_wav(db_path, linked_id)

    count = db.relink_orphaned_audio()

    assert count == 1
    linked = db.get_session(linked_id)
    missing = db.get_session(missing_id)
    assert linked is not None
    assert missing is not None
    assert linked["audio_path"] == str(wav_path.resolve())
    assert missing["audio_path"] == ""
    db.close()


def test_relink_orphaned_audio_skips_when_path_already_set(tmp_path: Path) -> None:
    """Sessions with audio_path already set are not overwritten."""
    db_path = tmp_path / "transcriber.db"
    db = Database(db_path)
    session_id = _seed_completed_session(db)
    existing_path = str(tmp_path / "custom" / "recording.wav")
    db.update_audio_path(session_id, existing_path)
    _create_orphan_wav(db_path, session_id)

    count = db.relink_orphaned_audio()

    assert count == 0
    session = db.get_session(session_id)
    assert session is not None
    assert session["audio_path"] == existing_path
    db.close()


def test_database_init_relinks_orphaned_audio(tmp_path: Path) -> None:
    """Opening the database relinks orphaned audio once at startup."""
    db_path = tmp_path / "transcriber.db"
    db = Database(db_path)
    session_id = _seed_completed_session(db)
    db._conn.execute(
        "UPDATE sessions SET audio_path = NULL WHERE id = ?",
        (session_id,),
    )
    db._conn.commit()
    db.close()

    wav_path = _create_orphan_wav(db_path, session_id)
    db_reopen = Database(db_path)
    try:
        session = db_reopen.get_session(session_id)
        assert session is not None
        assert session["audio_path"] == str(wav_path.resolve())
    finally:
        db_reopen.close()
