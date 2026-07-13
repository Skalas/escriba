"""Unit tests for ModelDownloadService (strand C / T8–T10)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from escriba.app.model_download import (
    DownloadAlreadyInProgress,
    DownloadStartFailed,
    ModelDownloadError,
    ModelDownloadService,
    NoDownloadInProgress,
    run_model_download,
)


def test_try_begin_claims_download_slot() -> None:
    service = ModelDownloadService()

    service.try_begin("mlx-community/test-model")

    downloading, result, total = service.get_status()
    assert downloading == "mlx-community/test-model"
    assert result is None
    assert total == 0


def test_try_begin_rejects_concurrent_download() -> None:
    service = ModelDownloadService()
    service.try_begin("model-a")

    with pytest.raises(DownloadAlreadyInProgress):
        service.try_begin("model-b")


def test_cancel_marks_download_cancelled() -> None:
    service = ModelDownloadService()
    service.try_begin("mlx-community/test-model")
    proc = MagicMock()
    service.set_process(proc)

    service.cancel()

    proc.terminate.assert_called_once()
    assert service.was_cancelled() is True


def test_cancel_raises_when_idle() -> None:
    service = ModelDownloadService()

    with pytest.raises(NoDownloadInProgress):
        service.cancel()


def test_finish_clears_active_download() -> None:
    service = ModelDownloadService()
    service.try_begin("mlx-community/test-model")

    service.finish({"ok": True, "model": "mlx-community/test-model"})

    downloading, result, total = service.get_status()
    assert downloading is None
    assert result == {"ok": True, "model": "mlx-community/test-model"}
    assert total == 0


def test_download_blocking_delegates_to_snapshot() -> None:
    service = ModelDownloadService()

    with patch("escriba.summarize.llm_summary.download_model_snapshot") as snapshot:
        service.download_blocking("mlx-community/test-model")

    snapshot.assert_called_once_with("mlx-community/test-model")


def test_download_blocking_raises_safe_error_without_leaking_details() -> None:
    service = ModelDownloadService()

    with patch(
        "escriba.summarize.llm_summary.download_model_snapshot",
        side_effect=RuntimeError("/secret/path token=abc"),
    ):
        with pytest.raises(ModelDownloadError, match="check logs") as exc_info:
            service.download_blocking("mlx-community/test-model")

    assert "/secret/path" not in str(exc_info.value)
    assert "token=abc" not in str(exc_info.value)


def test_start_background_download_claims_slot_and_starts_process() -> None:
    service = ModelDownloadService()
    proc = MagicMock()
    proc.join = MagicMock(side_effect=lambda: None)
    monitor_targets: list[object] = []

    class _ImmediateThread:
        def __init__(self, target=None, daemon=None) -> None:
            self._target = target

        def start(self) -> None:
            monitor_targets.append(self._target)

    with (
        patch(
            "escriba.summarize.llm_summary.hf_repo_total_bytes",
            return_value=1024,
        ),
        patch("multiprocessing.get_context") as get_context,
        patch("escriba.app.model_download.threading.Thread", _ImmediateThread),
    ):
        context = MagicMock()
        context.Process.return_value = proc
        get_context.return_value = context

        started = service.start_background_download("mlx-community/test-model")

    assert started.model_id == "mlx-community/test-model"
    assert started.total_bytes == 1024
    proc.start.assert_called_once()
    downloading, _result, total = service.get_status()
    assert downloading == "mlx-community/test-model"
    assert total == 1024
    assert monitor_targets


def test_start_background_download_releases_claim_when_proc_start_fails() -> None:
    service = ModelDownloadService()
    proc = MagicMock()
    proc.start.side_effect = OSError("spawn failed")

    with (
        patch(
            "escriba.summarize.llm_summary.hf_repo_total_bytes",
            return_value=1024,
        ),
        patch("multiprocessing.get_context") as get_context,
    ):
        context = MagicMock()
        context.Process.return_value = proc
        get_context.return_value = context

        with pytest.raises(DownloadStartFailed):
            service.start_background_download("mlx-community/test-model")

    downloading, result, total = service.get_status()
    assert downloading is None
    assert result == {"ok": False, "error": "Model download failed; check logs"}
    assert total == 0

    # A follow-up download must not be blocked by a stuck claim.
    service.try_begin("mlx-community/other-model")
    downloading, _result, _total = service.get_status()
    assert downloading == "mlx-community/other-model"


def test_run_model_download_exits_nonzero_on_failure() -> None:
    with (
        patch("escriba.summarize.llm_summary.download_model_snapshot", side_effect=OSError("disk full")),
        patch("sys.exit") as exit_mock,
    ):
        run_model_download("mlx-community/test-model")

    exit_mock.assert_called_once_with(1)
