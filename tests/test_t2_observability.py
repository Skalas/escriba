"""T2: Observability — correlation IDs, structured logs, provider latency."""
from __future__ import annotations

import logging
import threading
import time
import urllib.request
from pathlib import Path
from unittest.mock import patch

import pytest

from escriba.app.database import Database
from escriba.app.observability import LatencyStore, get_correlation_id, set_correlation_id
from escriba.app.server import AppState, start_server
from escriba.config import AppConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def minimal_config(tmp_path: Path) -> AppConfig:
    cfg_path = tmp_path / "escriba.toml"
    cfg_path.write_text(
        """
[audio]
audio_source = "mic"
sample_rate = 16000
channels = 1

[streaming]
backend = "mlx-whisper"
model_size = "tiny"
chunk_duration = 0.5

[auto_name]
enabled = false
""".strip(),
        encoding="utf-8",
    )
    return AppConfig.load(cfg_path)


@pytest.fixture
def app_state(minimal_config: AppConfig, tmp_path: Path) -> AppState:
    db = Database(tmp_path / "t2-test.db")
    return AppState(config=minimal_config, db=db)




# ---------------------------------------------------------------------------
# LatencyStore unit tests
# ---------------------------------------------------------------------------


def test_latency_store_records_and_retrieves_percentiles() -> None:
    store = LatencyStore()
    for i in range(100):
        store.record("op.test", float(i))

    p50 = store.p50("op.test")
    p99 = store.p99("op.test")
    assert p50 is not None
    assert p99 is not None
    assert p50 < p99


def test_latency_store_returns_none_for_unknown_key() -> None:
    store = LatencyStore()
    assert store.p50("does.not.exist") is None
    assert store.p99("does.not.exist") is None


def test_latency_store_caps_samples() -> None:
    store = LatencyStore()
    for i in range(1500):
        store.record("k", float(i))
    with store._lock:
        assert len(store._data["k"]) == store._MAX_SAMPLES


def test_latency_store_snapshot_has_all_keys() -> None:
    store = LatencyStore()
    store.record("llm.gemini", 100.0)
    store.record("llm.claude", 200.0)
    store.record("llm.local", 50.0)

    snap = store.snapshot()
    assert "llm.gemini" in snap
    assert "llm.claude" in snap
    assert "llm.local" in snap
    for v in snap.values():
        assert "p50_ms" in v and "p99_ms" in v


def test_latency_store_distinguishes_providers() -> None:
    """Different provider keys are tracked independently."""
    store = LatencyStore()
    store.record("llm.gemini", 300.0)
    store.record("llm.local", 50.0)

    assert store.p50("llm.gemini") != store.p50("llm.local")


# ---------------------------------------------------------------------------
# Correlation ID thread-local
# ---------------------------------------------------------------------------


def test_correlation_id_is_thread_local() -> None:
    """Each thread has an independent correlation ID."""
    ids: list[str | None] = []

    def set_and_capture(cid: str) -> None:
        set_correlation_id(cid)
        time.sleep(0.05)
        ids.append(get_correlation_id())

    t1 = threading.Thread(target=set_and_capture, args=("aaa",))
    t2 = threading.Thread(target=set_and_capture, args=("bbb",))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert set(ids) == {"aaa", "bbb"}


# ---------------------------------------------------------------------------
# Server integration: correlation ID in response header
# ---------------------------------------------------------------------------


def test_response_includes_correlation_id_header(
    tmp_path: Path, minimal_config: AppConfig
) -> None:
    """Every JSON response carries X-Correlation-ID."""
    db = Database(tmp_path / "obs.db")
    state = AppState(config=minimal_config, db=db)
    server = start_server(state, port=0)
    port = server.server_address[1]
    try:
        resp = urllib.request.urlopen(
            f"http://127.0.0.1:{port}/api/status", timeout=5
        )
        cid = resp.headers.get("X-Correlation-ID")
    finally:
        server.shutdown()
        db.close()

    assert cid is not None, "X-Correlation-ID header missing"
    assert len(cid) > 0


def test_each_request_gets_unique_correlation_id(
    tmp_path: Path, minimal_config: AppConfig
) -> None:
    """Two separate requests get different correlation IDs."""
    db = Database(tmp_path / "obs2.db")
    state = AppState(config=minimal_config, db=db)
    server = start_server(state, port=0)
    port = server.server_address[1]
    try:
        r1 = urllib.request.urlopen(f"http://127.0.0.1:{port}/api/status", timeout=5)
        r2 = urllib.request.urlopen(f"http://127.0.0.1:{port}/api/status", timeout=5)
        cid1 = r1.headers.get("X-Correlation-ID")
        cid2 = r2.headers.get("X-Correlation-ID")
    finally:
        server.shutdown()
        db.close()

    assert cid1 != cid2, "Correlation IDs must be unique per request"


# ---------------------------------------------------------------------------
# Server integration: handler latency recorded in latency_store
# ---------------------------------------------------------------------------


def test_handler_latency_recorded_after_request(
    tmp_path: Path, minimal_config: AppConfig
) -> None:
    """After a GET /api/status, handler.GET has a recorded latency."""
    from escriba.app.observability import latency_store

    db = Database(tmp_path / "obs3.db")
    state = AppState(config=minimal_config, db=db)
    server = start_server(state, port=0)
    port = server.server_address[1]
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{port}/api/status", timeout=5)
    finally:
        server.shutdown()
        db.close()

    assert latency_store.p50("handler.GET") is not None


# ---------------------------------------------------------------------------
# LLM latency: provider keys are distinct in latency_store
# ---------------------------------------------------------------------------


def test_llm_latency_keys_distinguish_provider() -> None:
    """Gemini and Claude calls record to separate latency_store keys."""
    from escriba.app.observability import latency_store
    from escriba.summarize import llm_summary

    # Reset any prior state
    with latency_store._lock:
        latency_store._data.pop("llm.gemini", None)
        latency_store._data.pop("llm.claude", None)

    # Simulate a Gemini call raising (key path still records latency)
    with patch("escriba.summarize.llm_summary._retry_cloud_call",
               side_effect=ValueError("no key")):
        with patch.dict("os.environ", {"GEMINI_API_KEY": "fake"}):
            try:
                llm_summary._gemini_generate_text("gemini-2.5-flash", "hi")
            except Exception:
                pass

    assert latency_store.p50("llm.gemini") is not None

    # Simulate a Claude call raising
    with patch("escriba.summarize.llm_summary._retry_cloud_call",
               side_effect=ValueError("no key")):
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "fake"}):
            try:
                llm_summary._claude_generate_text("claude-sonnet-4-6-20250514", "hi")
            except Exception:
                pass

    assert latency_store.p50("llm.claude") is not None

    # Both keys must be distinct (different values)
    snap = latency_store.snapshot()
    assert "llm.gemini" in snap
    assert "llm.claude" in snap


# ---------------------------------------------------------------------------
# Structured log fields
# ---------------------------------------------------------------------------


def test_request_logs_contain_structured_fields(
    tmp_path: Path, minimal_config: AppConfig, caplog: pytest.LogCaptureFixture
) -> None:
    """Completed requests are logged with corr_id and duration_ms extra fields."""
    db = Database(tmp_path / "obs4.db")
    state = AppState(config=minimal_config, db=db)
    server = start_server(state, port=0)
    port = server.server_address[1]

    with caplog.at_level(logging.DEBUG, logger="escriba.app.server"):
        urllib.request.urlopen(f"http://127.0.0.1:{port}/api/status", timeout=5)

    server.shutdown()
    db.close()

    done_records = [
        r for r in caplog.records if "request.done" in r.getMessage()
    ]
    assert done_records, "No request.done log record found"
    rec = done_records[0]
    # The extra dict fields should be attached to the record
    assert hasattr(rec, "corr_id"), "corr_id missing from log record"
    assert hasattr(rec, "duration_ms"), "duration_ms missing from log record"
    assert rec.corr_id != "-"
