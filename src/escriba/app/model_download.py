"""Application-layer orchestration for local LLM model downloads."""

from __future__ import annotations

import logging
import multiprocessing as mp
import sys
import threading
from dataclasses import dataclass
from typing import Any, Callable

logger = logging.getLogger(__name__)

USER_DOWNLOAD_FAILED = "Model download failed; check logs"


class DownloadAlreadyInProgress(Exception):
    """Raised when a second download is requested while one is active."""


class NoDownloadInProgress(Exception):
    """Raised when cancel is called with no active download process."""


class DownloadStartFailed(Exception):
    """Raised when the download subprocess cannot be started after claiming the slot."""


class ModelDownloadError(Exception):
    """Blocking CLI download failed. Carries a safe user-facing message only."""

    def __init__(self, model_id: str, message: str = USER_DOWNLOAD_FAILED) -> None:
        self.model_id = model_id
        self.message = message
        super().__init__(message)


@dataclass(frozen=True)
class StartedDownload:
    """A background model download that was successfully spawned."""

    model_id: str
    total_bytes: int
    message: str


def run_model_download(model_id: str) -> None:
    """Subprocess entry point for a cancellable model download.

    Runs at module level so it is importable by ``multiprocessing`` (spawn).
    A clean return is exit code 0; an exception exits non-zero; a cancel is a
    SIGTERM from the parent — the monitor thread interprets the exit code.
    """
    import traceback

    from dotenv import load_dotenv

    from escriba.summarize.llm_summary import download_model_snapshot

    load_dotenv()
    try:
        download_model_snapshot(model_id)
    except Exception:
        traceback.print_exc()
        sys.exit(1)


class ModelDownloadService:
    """Owns claim/cancel/progress/completion for local model downloads."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._downloading_model: str | None = None
        self._download_result: dict[str, Any] | None = None
        self._download_total_bytes: int = 0
        self._download_proc: Any = None
        self._download_cancelled: bool = False

    def get_status(self) -> tuple[str | None, dict[str, Any] | None, int]:
        """Return current download state, finished result, and expected size."""
        with self._lock:
            return (
                self._downloading_model,
                self._download_result,
                self._download_total_bytes,
            )

    def try_begin(self, model_id: str, total_bytes: int = 0) -> None:
        """Claim a model download; only one may run at a time.

        Raises:
            DownloadAlreadyInProgress: When another download is active.
        """
        with self._lock:
            if self._downloading_model:
                raise DownloadAlreadyInProgress()
            self._downloading_model = model_id
            self._download_result = None
            self._download_total_bytes = total_bytes
            self._download_cancelled = False
            self._download_proc = None

    def set_total_bytes(self, total_bytes: int) -> None:
        """Record expected size after the claim (kept off the busy fast-path)."""
        with self._lock:
            if self._downloading_model:
                self._download_total_bytes = total_bytes

    def set_process(self, proc: Any) -> None:
        """Record the running download subprocess so it can be cancelled."""
        with self._lock:
            self._download_proc = proc

    def cancel(self) -> None:
        """Terminate the in-flight download subprocess.

        Raises:
            NoDownloadInProgress: When there is no cancellable download.
        """
        with self._lock:
            proc = self._download_proc
            if not self._downloading_model or proc is None:
                raise NoDownloadInProgress()
            self._download_cancelled = True
        try:
            proc.terminate()
        except Exception:
            logger.warning("Failed to terminate download process", exc_info=True)

    def was_cancelled(self) -> bool:
        """Return whether the active download was cancelled."""
        with self._lock:
            return self._download_cancelled

    def finish(self, result: dict[str, Any]) -> None:
        """Record the outcome of a background model download."""
        with self._lock:
            self._download_result = result
            self._downloading_model = None
            self._download_total_bytes = 0
            self._download_proc = None

    def download_blocking(self, model_id: str) -> None:
        """Download a model synchronously (CLI path).

        Args:
            model_id: HuggingFace repo id to cache locally.

        Raises:
            ModelDownloadError: When the download fails.
        """
        from escriba.summarize.llm_summary import download_model_snapshot

        try:
            download_model_snapshot(model_id)
        except Exception:
            logger.error("Model download failed for %s", model_id, exc_info=True)
            raise ModelDownloadError(model_id) from None

    def start_background_download(
        self,
        model_id: str,
        *,
        commit: bool = False,
        persist_model: Callable[[str], None] | None = None,
    ) -> StartedDownload:
        """Start a cancellable background snapshot download.

        Args:
            model_id: HuggingFace repo id to download.
            commit: When true, persist the model as the active summary model on success.
            persist_model: Callback invoked after a successful download when ``commit``
                is true.

        Returns:
            Metadata for the started download.

        Raises:
            DownloadAlreadyInProgress: When another download is active.
            DownloadStartFailed: When the subprocess cannot be started.
        """
        from escriba.summarize.llm_summary import (
            hf_repo_total_bytes,
            invalidate_models_cache,
            is_model_cached,
        )

        try:
            self.try_begin(model_id)
        except DownloadAlreadyInProgress:
            raise

        total_bytes = hf_repo_total_bytes(model_id)
        self.set_total_bytes(total_bytes)

        proc = mp.get_context("spawn").Process(
            target=run_model_download, args=(model_id,), daemon=True
        )
        logger.info("Downloading model: %s", model_id)
        try:
            proc.start()
            self.set_process(proc)
        except Exception:
            logger.error(
                "Failed to start model download process for %s",
                model_id,
                exc_info=True,
            )
            self.finish({"ok": False, "error": USER_DOWNLOAD_FAILED})
            raise DownloadStartFailed(model_id) from None

        service = self

        def _monitor() -> None:
            proc.join()
            if proc.exitcode == 0 and is_model_cached(model_id):
                logger.info("Model download complete: %s", model_id)
                if commit and persist_model is not None:
                    persist_model(model_id)
                service.finish({"ok": True, "model": model_id})
            elif service.was_cancelled():
                logger.info("Model download cancelled: %s", model_id)
                service.finish(
                    {"ok": False, "cancelled": True, "error": "Download cancelled"}
                )
            else:
                logger.error(
                    "Model download failed (exit %s): %s", proc.exitcode, model_id
                )
                service.finish({"ok": False, "error": USER_DOWNLOAD_FAILED})
            invalidate_models_cache()

        threading.Thread(target=_monitor, daemon=True).start()
        return StartedDownload(
            model_id=model_id,
            total_bytes=total_bytes,
            message=f"Downloading {model_id}...",
        )
