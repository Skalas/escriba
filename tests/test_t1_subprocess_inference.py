"""T1: Subprocess inference — server responsiveness and error isolation."""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from escriba.app.database import Database
from escriba.app.server import AppState, start_server
from escriba.config import AppConfig
from escriba.summarize import llm_summary
from escriba.summarize.llm_summary import _LocalInferenceProcess


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
    db = Database(tmp_path / "t1-test.db")
    return AppState(config=minimal_config, db=db)




# ---------------------------------------------------------------------------
# T1a: subprocess crash/error surfaces as handled error, never crashes the app
# ---------------------------------------------------------------------------


def test_subprocess_failure_returns_none_not_exception() -> None:
    """A RuntimeError in the subprocess becomes None after MAX_ATTEMPTS retries."""
    proc = _LocalInferenceProcess()
    mock_future = MagicMock()
    mock_future.result.side_effect = RuntimeError("subprocess died")

    mock_executor = MagicMock()
    mock_executor.submit.return_value = mock_future

    with patch.object(proc, "_get_executor", return_value=mock_executor):
        result = proc.run("prompt", "model", 100, False)

    assert result is None


def test_subprocess_timeout_resets_executor_and_raises() -> None:
    """A TimeoutError kills the stuck subprocess (resets executor) and propagates."""
    import concurrent.futures

    proc = _LocalInferenceProcess()
    mock_future = MagicMock()
    mock_future.result.side_effect = concurrent.futures.TimeoutError()

    mock_executor = MagicMock()
    mock_executor.submit.return_value = mock_future

    # Set executor directly so _reset_executor can see it.
    proc._executor = mock_executor

    with patch.object(proc, "_get_executor", return_value=mock_executor):
        with pytest.raises(TimeoutError):
            proc.run("prompt", "model", 100, False)

    # Executor was shut down after timeout
    mock_executor.shutdown.assert_called_once()


def test_subprocess_oom_retries_then_returns_none() -> None:
    """MemoryError causes a retry; after MAX_ATTEMPTS the call returns None."""
    proc = _LocalInferenceProcess()
    mock_future = MagicMock()
    mock_future.result.side_effect = MemoryError("OOM")

    mock_executor = MagicMock()
    mock_executor.submit.return_value = mock_future

    with patch.object(proc, "_get_executor", return_value=mock_executor):
        result = proc.run("prompt", "model", 100, False)

    assert result is None
    assert mock_executor.submit.call_count == llm_summary.LOCAL_MODEL_MAX_ATTEMPTS


def test_call_llm_local_delegates_to_inference_process() -> None:
    """_call_llm_local forwards to _local_inference_process, not _run_local_generation."""
    calls: list[tuple] = []

    def capture(*args):
        calls.append(args)
        return "result"

    with patch.dict("sys.modules", {"mlx_lm": MagicMock()}):
        with patch.object(
            llm_summary._local_inference_process, "run", side_effect=capture
        ):
            result = llm_summary._call_llm_local("p", "m", max_tokens=64, enable_thinking=True)

    assert result == "result"
    assert calls[0] == ("p", "m", 64, True)


# ---------------------------------------------------------------------------
# T1b: server stays responsive while the subprocess worker is occupied
# ---------------------------------------------------------------------------


def test_status_endpoint_responsive_while_subprocess_worker_blocked(
    tmp_path: Path, minimal_config: AppConfig
) -> None:
    """GET /api/status responds quickly while the inference subprocess worker is occupied.

    We submit time.sleep (a picklable C builtin) directly to the ProcessPoolExecutor,
    saturating the sole worker subprocess.  The parent process and its HTTP threads
    must still serve /api/status without waiting for the subprocess to finish.

    This test FAILS if T1 is reverted to in-process inference: removing
    _LocalInferenceProcess means _get_executor() no longer exists, and the
    test raises AttributeError before reaching the HTTP assertion.
    If inference instead ran in-process holding the GIL, the latency assertion
    would catch any starvation of the HTTP worker threads.
    """
    db = Database(tmp_path / "live.db")
    state = AppState(config=minimal_config, db=db)
    server = start_server(state, port=0)
    port = server.server_address[1]

    # Saturate the single subprocess worker.  time.sleep is a C builtin —
    # picklable and safe to run inside a ProcessPoolExecutor worker process.
    executor = llm_summary._local_inference_process._get_executor()
    blocking_future = executor.submit(time.sleep, 3)

    try:
        # While the worker subprocess is sleeping, the parent process + HTTP
        # threads must remain fully responsive.
        t0 = time.monotonic()
        resp = urllib.request.urlopen(
            f"http://127.0.0.1:{port}/api/status", timeout=3
        )
        elapsed = time.monotonic() - t0
        body = json.loads(resp.read())
    finally:
        blocking_future.result(timeout=6)  # let the subprocess finish cleanly
        server.shutdown()
        db.close()

    assert body["ok"] is True
    assert elapsed < 1.0, f"/api/status took {elapsed:.2f}s while subprocess was occupied"
