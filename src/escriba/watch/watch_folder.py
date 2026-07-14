from __future__ import annotations

import logging
import os
import queue
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Iterable, Optional

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from escriba.transcribe.whisper import TranscriptionError, transcribe_file


SUPPORTED_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".m4a",
    ".flac",
    ".aac",
    ".ogg",
    ".mp4",
}

LOGGER = logging.getLogger("escriba.watch")

DEFAULT_PROCESSED_MAX = 10_000


def _get_watch_processed_max() -> int:
    """Parse WATCH_PROCESSED_MAX with validation and a safe default."""
    raw = os.getenv("WATCH_PROCESSED_MAX", str(DEFAULT_PROCESSED_MAX)).strip()
    try:
        value = int(raw)
    except ValueError:
        LOGGER.warning(
            "Invalid WATCH_PROCESSED_MAX=%r, using default %s",
            raw,
            DEFAULT_PROCESSED_MAX,
        )
        return DEFAULT_PROCESSED_MAX
    if value < 1:
        LOGGER.warning(
            "WATCH_PROCESSED_MAX must be >= 1, got %s; using default %s",
            value,
            DEFAULT_PROCESSED_MAX,
        )
        return DEFAULT_PROCESSED_MAX
    return value


class _BoundedProcessedSet:
    """Track successfully processed paths; FIFO-evict only completed entries."""

    def __init__(self, max_size: int) -> None:
        if max_size < 1:
            raise ValueError("max_size must be >= 1")
        self._max_size = max_size
        self._paths: OrderedDict[Path, None] = OrderedDict()

    def __contains__(self, path: Path) -> bool:
        return path in self._paths

    def add(self, path: Path) -> bool:
        """Mark path processed. Returns False if already present."""
        if path in self._paths:
            return False
        while len(self._paths) >= self._max_size:
            self._paths.popitem(last=False)
        self._paths[path] = None
        return True

    def discard(self, path: Path) -> None:
        self._paths.pop(path, None)


def _resolve_under_input_dir(path: Path, input_dir: Path) -> Path | None:
    """Resolve ``path`` and ensure it stays under ``input_dir`` (symlink-safe)."""
    try:
        resolved = path.resolve()
        root = input_dir.resolve()
    except OSError:
        LOGGER.warning("Could not resolve watch path: %s", path)
        return None
    if root not in resolved.parents and resolved != root:
        LOGGER.warning("Rejecting path outside watch directory: %s", path)
        return None
    return resolved


def watch_folder(
    input_dir: Path,
    output_dir: Path,
    combined_transcript: Optional[Path] = None,
    extensions: Optional[Iterable[str]] = None,
    stop_event: Optional[threading.Event] = None,
    skip_stability_check: bool = False,
) -> tuple[queue.Queue[Path], threading.Thread]:
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    stop_event = stop_event or threading.Event()
    watch_root = input_dir.resolve()

    LOGGER.info("Watching folder: %s", input_dir)
    LOGGER.info("Output dir: %s", output_dir)
    if combined_transcript is not None:
        LOGGER.info("Combined transcript: %s", combined_transcript)

    normalized_ext = {ext.lower() for ext in (extensions or SUPPORTED_EXTENSIONS)}
    work_queue: queue.Queue[Path] = queue.Queue()
    completed = _BoundedProcessedSet(_get_watch_processed_max())
    active: set[Path] = set()
    lock = threading.Lock()

    def enqueue_path(raw_path: Path) -> None:
        resolved = _resolve_under_input_dir(raw_path, watch_root)
        if resolved is None:
            return
        if resolved.suffix.lower() not in normalized_ext:
            return
        with lock:
            if resolved in completed or resolved in active:
                return
            active.add(resolved)
        LOGGER.info("Queued file: %s", resolved)
        work_queue.put(resolved)

    def worker() -> None:
        LOGGER.info("Worker started")
        while True:
            # Si stop_event está set, intenta procesar lo que queda en la cola
            if stop_event.is_set():
                try:
                    path = work_queue.get(timeout=0.1)
                except queue.Empty:
                    # Cola vacía, podemos salir
                    break
            else:
                try:
                    path = work_queue.get(timeout=0.5)
                except queue.Empty:
                    continue

            try:
                LOGGER.info("Processing file: %s", path)
                if not skip_stability_check:
                    _wait_for_stable_file(path)
                transcribed = _retry_transcribe(path, output_dir, combined_transcript)
                with lock:
                    active.discard(path)
                    if transcribed:
                        completed.add(path)
            finally:
                work_queue.task_done()
        LOGGER.info("Worker stopped")

    worker_thread = threading.Thread(target=worker, daemon=True)
    worker_thread.start()

    class Handler(FileSystemEventHandler):
        def on_created(self, event) -> None:  # type: ignore[override]
            if event.is_directory:
                return
            enqueue_path(Path(event.src_path))

        def on_moved(self, event) -> None:  # type: ignore[override]
            if event.is_directory:
                return
            enqueue_path(Path(event.dest_path))

    observer = Observer()
    observer.schedule(Handler(), str(input_dir), recursive=False)
    observer.start()
    LOGGER.info("Observer started")

    try:
        while not stop_event.is_set():
            time.sleep(0.5)
    finally:
        observer.stop()
        observer.join()
        LOGGER.info("Observer stopped")

    return work_queue, worker_thread


def wait_for_queue_empty(
    work_queue: queue.Queue[Path],
    worker_thread: threading.Thread,
    timeout: float = 300.0,
) -> None:
    """
    Espera a que la cola de trabajo esté vacía y el worker termine.
    
    Args:
        work_queue: Cola de trabajo a esperar
        worker_thread: Thread del worker a esperar
        timeout: Tiempo máximo de espera en segundos
    """
    LOGGER.info("Waiting for queue to empty...")
    start_time = time.time()

    while True:
        if work_queue.empty():
            # Espera un poco más para asegurar que no hay nada en proceso
            time.sleep(1.0)
            if work_queue.empty():
                LOGGER.info("Queue is empty, waiting for worker to finish...")
                break

        if time.time() - start_time > timeout:
            LOGGER.warning(f"Timeout waiting for queue to empty after {timeout}s")
            break

        time.sleep(0.5)

    # Espera a que el worker termine
    worker_thread.join(timeout=30.0)
    if worker_thread.is_alive():
        LOGGER.warning("Worker thread did not finish in time")
    else:
        LOGGER.info("All transcriptions completed")


def _wait_for_stable_file(
    path: Path,
    attempts: int = 40,
    delay: float = 0.5,
    min_size: int | None = None,
    stable_seconds: float | None = None,
) -> None:
    min_size = min_size or int(os.getenv("MIN_STABLE_SIZE", "1024"))
    stable_seconds = stable_seconds or float(os.getenv("STABLE_SECONDS", "2.0"))
    last_size = -1
    last_change = time.monotonic()
    LOGGER.info("Waiting for stable file: %s", path)
    for _ in range(attempts):
        try:
            current_size = path.stat().st_size
        except FileNotFoundError:
            time.sleep(delay)
            continue
        now = time.monotonic()
        if current_size != last_size:
            last_change = now
            last_size = current_size
        if current_size >= min_size and (now - last_change) >= stable_seconds:
            LOGGER.info(
                "File stabilized: %s (%d bytes, %.1fs stable)",
                path,
                current_size,
                now - last_change,
            )
            return
        time.sleep(delay)
    LOGGER.warning("File did not stabilize in time: %s", path)


def _retry_transcribe(
    path: Path,
    output_dir: Path,
    combined_transcript: Optional[Path],
    attempts: int = 3,
    delay: float = 1.0,
) -> bool:
    for attempt in range(attempts):
        try:
            if not path.exists() or path.stat().st_size == 0:
                time.sleep(delay)
                continue
            result = transcribe_file(path, output_dir, combined_transcript)
            if result is None:
                LOGGER.warning("Transcript not created for %s", path.name)
                if attempt >= attempts - 1:
                    return False
                time.sleep(delay)
                continue
            LOGGER.info("Transcribed: %s", path.name)
            return True
        except TranscriptionError as exc:
            if attempt >= attempts - 1:
                LOGGER.error(
                    "Skipping %s due to transcription error: %s",
                    path.name,
                    exc,
                )
                return False
            time.sleep(delay)
        except Exception:
            if attempt >= attempts - 1:
                LOGGER.error("Skipping %s due to error", path.name, exc_info=True)
                return False
            time.sleep(delay)
    return False
