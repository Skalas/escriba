from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path

from local_transcriber.watch.watch_folder import watch_folder, wait_for_queue_empty

logger = logging.getLogger(__name__)


def run_live_capture(output_dir: Path, combined_transcript: Path | None = None) -> None:
    system_device = _get_str_env("SYSTEM_DEVICE", "0")
    mic_device = _get_str_env("MIC_DEVICE", "1")
    sample_rate = _get_int_env("SAMPLE_RATE", 16000, min_value=8000)
    channels = _get_int_env("CHANNELS", 1, min_value=1)
    segment_seconds = _get_int_env("SEGMENT_SECONDS", 30, min_value=1)

    # ffmpeg escribe en temp_dir, el mover los completa a watched_dir
    temp_segment_dir = Path(tempfile.mkdtemp(prefix="local-transcriber-temp-"))
    watched_segment_dir = Path(tempfile.mkdtemp(prefix="local-transcriber-watched-"))

    ffmpeg_cmd = _build_ffmpeg_command(
        system_device=system_device,
        mic_device=mic_device,
        sample_rate=sample_rate,
        channels=channels,
        segment_seconds=segment_seconds,
        output_dir=temp_segment_dir,
    )

    stop_event = threading.Event()
    process = subprocess.Popen(ffmpeg_cmd)

    # Thread que mueve segmentos completos de temp a watched
    mover_thread = threading.Thread(
        target=_move_completed_segments,
        args=(temp_segment_dir, watched_segment_dir, stop_event, segment_seconds),
        daemon=False,  # No daemon para poder esperarlo
    )
    mover_thread.start()

    work_queue = None
    worker_thread = None

    try:
        work_queue, worker_thread = watch_folder(
            input_dir=watched_segment_dir,
            output_dir=output_dir,
            combined_transcript=combined_transcript,
            extensions=[".wav"],
            stop_event=stop_event,
            skip_stability_check=True,  # Los archivos ya vienen completos del mover
        )
    except KeyboardInterrupt:
        logger.info("Interrupted, stopping gracefully...")
    finally:
        logger.info("Stopping ffmpeg...")
        stop_event.set()

        # Detener ffmpeg
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning("ffmpeg did not stop, killing...")
                process.kill()

        # Esperar a que el mover termine (moverá el último segmento)
        logger.info("Waiting for segment mover to finish...")
        mover_thread.join(timeout=10.0)
        if mover_thread.is_alive():
            logger.warning("Segment mover did not finish in time")
        else:
            logger.info("Segment mover finished")

        # Esperar a que todas las transcripciones terminen
        if work_queue is not None and worker_thread is not None:
            wait_for_queue_empty(work_queue, worker_thread)

        _cleanup_temp_dir(temp_segment_dir, "temp segments")
        _cleanup_temp_dir(watched_segment_dir, "watched segments")
        logger.info("Shutdown complete")


def _build_ffmpeg_command(
    system_device: str,
    mic_device: str,
    sample_rate: int,
    channels: int,
    segment_seconds: int,
    output_dir: Path,
) -> list[str]:
    system_input = _format_device(system_device)
    mic_input = _format_device(mic_device)
    segment_pattern = str(output_dir / "segment_%Y%m%d_%H%M%S.wav")
    return [
        "ffmpeg",
        "-f",
        "avfoundation",
        "-i",
        system_input,
        "-f",
        "avfoundation",
        "-i",
        mic_input,
        "-filter_complex",
        "amix=inputs=2:duration=longest",
        "-ar",
        str(sample_rate),
        "-ac",
        str(channels),
        "-f",
        "segment",
        "-segment_time",
        str(segment_seconds),
        "-reset_timestamps",
        "1",
        "-strftime",
        "1",
        segment_pattern,
    ]


def _move_completed_segments(
    temp_dir: Path, watched_dir: Path, stop_event: threading.Event, segment_seconds: int
) -> None:
    """Mueve segmentos completos de temp_dir a watched_dir.

    Detecta que un segmento está completo cuando:
    - Aparece el siguiente segmento (ffmpeg abrió el nuevo archivo)
    - O después de segment_seconds + margen de seguridad
    """
    watched_dir.mkdir(parents=True, exist_ok=True)
    processed: set[Path] = set()
    last_seen: dict[str, tuple[int, float]] = {}

    while not stop_event.is_set():
        try:
            current_files = sorted(temp_dir.glob("segment_*.wav"))

            for file_path in current_files:
                if file_path in processed:
                    continue

                file_key = file_path.name
                current_size = file_path.stat().st_size
                current_time = time.time()

                # Si es el último archivo en orden, puede estar escribiéndose aún
                is_last = file_path == current_files[-1] if current_files else False

                # Si no es el último, o si pasó suficiente tiempo desde que vimos este archivo
                if not is_last:
                    # No es el último, entonces está completo
                    _move_file(file_path, watched_dir)
                    processed.add(file_path)
                    last_seen.pop(file_key, None)
                else:
                    # Es el último, verifica si está estable
                    if file_key in last_seen:
                        # Ya lo vimos antes, verifica si el tamaño cambió
                        last_size, last_time = last_seen[file_key]
                        if current_size == last_size and current_size > 1024:
                            # Tamaño estable y > 1KB, espera un poco más para asegurar
                            if current_time - last_time > 1.0:  # 1 segundo sin cambios
                                _move_file(file_path, watched_dir)
                                processed.add(file_path)
                                last_seen.pop(file_key, None)
                        else:
                            # Tamaño cambió, actualiza
                            last_seen[file_key] = (current_size, current_time)
                    else:
                        # Primera vez que lo vemos
                        last_seen[file_key] = (current_size, current_time)

            time.sleep(0.5)
        except Exception as e:
            logger.error(f"Error moving segments: {e}", exc_info=True)
            time.sleep(1.0)

    # Al detenerse, mueve el último segmento pendiente si existe
    logger.info("Stopping segment mover, moving final segment if any...")
    time.sleep(2.0)  # Espera a que ffmpeg termine de escribir
    final_files = sorted(temp_dir.glob("segment_*.wav"))
    logger.info(
        f"Found {len(final_files)} files in temp dir, {len(processed)} already processed"
    )

    for file_path in final_files:
        if file_path not in processed:
            logger.info(f"Attempting to move final segment: {file_path.name}")
            # Espera a que el archivo se estabilice (ffmpeg puede estar cerrando)
            moved = False
            for attempt in range(10):
                try:
                    size1 = file_path.stat().st_size
                    time.sleep(0.5)
                    size2 = file_path.stat().st_size
                    if size1 == size2 and size1 > 1024:
                        _move_file(file_path, watched_dir)
                        logger.info(
                            f"Moved final segment: {file_path.name} ({size1} bytes)"
                        )
                        moved = True
                        break
                    elif size1 != size2:
                        logger.debug(
                            f"File still changing: {size1} -> {size2}, waiting..."
                        )
                except FileNotFoundError:
                    logger.warning(f"File disappeared: {file_path.name}")
                    break
                except Exception as e:
                    logger.error(
                        f"Error checking final segment {file_path}: {e}", exc_info=True
                    )
                    break

            if not moved:
                logger.warning(f"Could not stabilize final segment: {file_path.name}")


def _move_file(source: Path, dest_dir: Path) -> None:
    """Mueve un archivo de source a dest_dir."""
    try:
        dest = dest_dir / source.name
        shutil.move(str(source), str(dest))
        logger.info(f"Moved completed segment: {source.name} -> {dest}")
    except Exception as e:
        logger.error(f"Failed to move {source}: {e}", exc_info=True)


def _format_device(device: str) -> str:
    device = device.strip()
    if device.startswith(":") or device.startswith("["):
        return device
    if device.isdigit():
        return f":{device}"
    return device


def _get_str_env(name: str, default: str) -> str:
    value = os.getenv(name, default).strip()
    if not value:
        raise ValueError(f"{name} must not be empty")
    return value


def _get_int_env(name: str, default: int, min_value: int | None = None) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if min_value is not None and value < min_value:
        raise ValueError(f"{name} must be >= {min_value}, got {value}")
    return value


def _cleanup_temp_dir(path: Path, label: str) -> None:
    if not path.exists():
        return
    try:
        shutil.rmtree(path)
        logger.info("Removed %s directory: %s", label, path)
    except Exception:
        logger.exception("Failed to remove %s directory: %s", label, path)
