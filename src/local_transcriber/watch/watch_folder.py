from __future__ import annotations

import logging
import os
import queue
import threading
import time
from pathlib import Path
from typing import Iterable, Optional

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from local_transcriber.transcribe.whisper import transcribe_file


SUPPORTED_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".m4a",
    ".flac",
    ".aac",
    ".ogg",
    ".mp4",
}

LOGGER = logging.getLogger("local_transcriber.watch")


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

    LOGGER.info("Watching folder: %s", input_dir)
    LOGGER.info("Output dir: %s", output_dir)
    if combined_transcript is not None:
        LOGGER.info("Combined transcript: %s", combined_transcript)

    normalized_ext = {ext.lower() for ext in (extensions or SUPPORTED_EXTENSIONS)}
    work_queue: queue.Queue[Path] = queue.Queue()
    processed: set[Path] = set()
    lock = threading.Lock()

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
                _retry_transcribe(path, output_dir, combined_transcript)
            finally:
                work_queue.task_done()
        LOGGER.info("Worker stopped")

    worker_thread = threading.Thread(target=worker, daemon=True)
    worker_thread.start()

    class Handler(FileSystemEventHandler):
        def on_created(self, event) -> None:  # type: ignore[override]
            if event.is_directory:
                return
            path = Path(event.src_path)
            if path.suffix.lower() not in normalized_ext:
                return
            with lock:
                if path in processed:
                    return
                processed.add(path)
            LOGGER.info("Queued file: %s", path)
            work_queue.put(path)

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
    """Espera a que la cola de trabajo esté vacía y el worker termine."""
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
) -> None:
    for attempt in range(attempts):
        try:
            if not path.exists() or path.stat().st_size == 0:
                time.sleep(delay)
                continue
            result = transcribe_file(path, output_dir, combined_transcript)
            if result is None:
                LOGGER.warning("Transcript not created for %s", path)
                if attempt >= attempts - 1:
                    return
                time.sleep(delay)
                continue
            LOGGER.info("Transcribed: %s", path)
            return
        except Exception:
            if attempt >= attempts - 1:
                LOGGER.exception("Skipping %s due to error", path)
                return
            time.sleep(delay)
