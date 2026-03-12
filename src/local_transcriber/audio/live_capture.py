from __future__ import annotations

import logging
import os
import shutil
import struct
import subprocess
import tempfile
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, TYPE_CHECKING

import numpy as np

from local_transcriber.audio.device_detection import auto_detect_devices
from local_transcriber.summarize import generate_summary
from local_transcriber.transcribe.metrics import CaptureMetrics
from local_transcriber.transcribe.streaming import StreamingTranscriber
from local_transcriber.utils.env import (
    get_bool_env,
    get_float_env,
    get_int_env,
    get_str_env,
)
from local_transcriber.watch.watch_folder import watch_folder, wait_for_queue_empty

if TYPE_CHECKING:
    from local_transcriber.config import AppConfig

# Intentar importar StreamingTranscriberMPS (requiere openai-whisper y torch)
try:
    from local_transcriber.transcribe.streaming_mps import StreamingTranscriberMPS

    MPS_AVAILABLE = True
except ImportError:
    MPS_AVAILABLE = False
    StreamingTranscriberMPS = None

# Intentar importar StreamingTranscriberMLX (requiere mlx-whisper)
try:
    from local_transcriber.transcribe.streaming_mlx import (
        StreamingTranscriberMLX,
        MLX_AVAILABLE,
    )

    MLX_WHISPER_AVAILABLE = MLX_AVAILABLE
except ImportError:
    MLX_WHISPER_AVAILABLE = False
    StreamingTranscriberMLX = None

# Intentar importar ScreenCaptureKit (CLI Swift)
try:
    from local_transcriber.audio.screen_capture import (
        ScreenCaptureAudioCapture,
        check_screen_recording_permission,
        request_screen_recording_permission,
        SWIFT_CLI_AVAILABLE,
    )

    SCREENCAPTUREKIT_AVAILABLE = SWIFT_CLI_AVAILABLE
except ImportError:
    SCREENCAPTUREKIT_AVAILABLE = False
    SWIFT_CLI_AVAILABLE = False

logger = logging.getLogger(__name__)


def run_live_capture(
    output_dir: Path,
    combined_transcript: Path | None = None,
    *,
    config: AppConfig | None = None,
) -> None:
    """
    Captura audio en vivo usando ffmpeg y transcribe segmentos (modo legacy).

    Este modo captura audio en segmentos de 30 segundos y los transcribe
    usando Whisper CLI. Tiene mayor latencia que el modo streaming.

    Args:
        output_dir: Directorio donde guardar transcripciones
        combined_transcript: Archivo opcional para transcripción combinada
        config: AppConfig opcional (TOML/env resuelto)
    """
    if config is None:
        from local_transcriber.config import AppConfig

        config = AppConfig.load(None)

    # Detección automática de dispositivos (como Notion AI)
    auto_detect = config.audio.auto_detect_devices

    if auto_detect:
        logger.info("Auto-detecting audio devices...")
        detected_system, detected_mic = auto_detect_devices()
        system_device = detected_system or config.audio.system_device
        mic_device = detected_mic or config.audio.mic_device
        logger.info("Using devices - System: %s, Mic: %s", system_device, mic_device)
    else:
        system_device = config.audio.system_device
        mic_device = config.audio.mic_device

    sample_rate = config.audio.sample_rate
    channels = config.audio.channels
    segment_seconds = get_int_env("SEGMENT_SECONDS", 30, min_value=1)

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
    """
    Construye el comando ffmpeg para captura en modo legacy.

    Args:
        system_device: Índice o nombre del dispositivo de audio del sistema
        mic_device: Índice o nombre del dispositivo de micrófono
        sample_rate: Sample rate en Hz
        channels: Número de canales
        segment_seconds: Duración de cada segmento en segundos
        output_dir: Directorio donde guardar segmentos

    Returns:
        Lista de argumentos para subprocess.Popen
    """
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
    """
    Mueve segmentos completos de temp_dir a watched_dir.

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
            logger.error("Error moving segments: %s", e, exc_info=True)
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
            logger.info("Attempting to move final segment: %s", file_path.name)
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
                    logger.warning("File disappeared: %s", file_path.name)
                    break
                except Exception as e:
                    logger.error(
                        f"Error checking final segment {file_path}: {e}", exc_info=True
                    )
                    break

            if not moved:
                logger.warning("Could not stabilize final segment: %s", file_path.name)


def _move_file(source: Path, dest_dir: Path) -> None:
    """Mueve un archivo de source a dest_dir."""
    try:
        dest = dest_dir / source.name
        shutil.move(str(source), str(dest))
        logger.info("Moved completed segment: %s -> %s", source.name, dest)
    except Exception as e:
        logger.error("Failed to move %s: %s", source, e, exc_info=True)


def _format_device(device: str) -> str:
    """
    Formatea un dispositivo de audio para uso con ffmpeg.

    Args:
        device: Índice o nombre del dispositivo

    Returns:
        String formateado para ffmpeg (ej: ":0" o nombre completo)
    """
    device = device.strip()
    if device.startswith(":") or device.startswith("["):
        return device
    if device.isdigit():
        return f":{device}"
    return device


def mix_audio(
    system: np.ndarray, mic: np.ndarray, mic_boost: float = 1.2
) -> np.ndarray:
    """
    Mezcla audio del sistema y micrófono con normalización dinámica.

    Aplica un boost al micrófono (usualmente más bajo que el sistema),
    mezcla ambos canales y normaliza para evitar clipping.

    Args:
        system: Array numpy con audio del sistema (int16)
        mic: Array numpy con audio del micrófono (int16)
        mic_boost: Factor de boost para el micrófono (default: 1.2)

    Returns:
        Array numpy con audio mezclado (int16)
    """
    # Manejar arrays vacíos
    if len(system) == 0 or len(mic) == 0:
        # Si alguno está vacío, retornar array vacío del mismo tipo
        return np.array([], dtype=np.int16)

    # Boost del micrófono (usualmente más bajo que sistema)
    mic_boosted = mic.astype(np.float32) * mic_boost

    # Mezclar en float32 para evitar overflow
    mixed = system.astype(np.float32) + mic_boosted

    # Normalizar para evitar clipping
    peak = np.abs(mixed).max()
    if peak > 32767.0:
        # Escalar para que el pico máximo sea 32767
        mixed = mixed * (32767.0 / peak)

    # Convertir de vuelta a int16
    return mixed.astype(np.int16)


def run_streaming_capture(
    output_dir: Path,
    combined_transcript: Path | None = None,
    *,
    config: AppConfig | None = None,
    streaming_overrides: dict[str, object] | None = None,
) -> None:
    """
    Captura y transcribe audio en tiempo real usando streaming.

    Similar a Notion AI Meeting Notes, procesa chunks pequeños (1-3s)
    y muestra transcripciones mientras ocurre la llamada.
    """
    if config is None:
        from local_transcriber.config import AppConfig

        config = AppConfig.load(None)
    overrides = streaming_overrides or {}

    # Detección automática de dispositivos (como Notion AI)
    auto_detect = config.audio.auto_detect_devices

    if auto_detect:
        logger.info("Auto-detecting audio devices...")
        detected_system, detected_mic = auto_detect_devices()
        system_device = detected_system or config.audio.system_device
        mic_device = detected_mic or config.audio.mic_device
        logger.info("Using devices - System: %s, Mic: %s", system_device, mic_device)

        # ScreenCaptureKit captura el audio del sistema directamente
        # No necesitamos un dispositivo virtual
        logger.info(
            "Using ScreenCaptureKit for system audio capture (no virtual device needed)"
        )
    else:
        system_device = config.audio.system_device
        mic_device = config.audio.mic_device

    sample_rate = config.audio.sample_rate
    channels = config.audio.channels

    chunk_duration = float(
        overrides.get("chunk_duration", config.streaming.chunk_duration)
    )
    model_size = str(overrides.get("model_size", config.streaming.model_size))
    language = str(overrides.get("language", config.streaming.language))
    device = str(overrides.get("device", config.streaming.device))
    backend = str(overrides.get("backend", config.streaming.backend))
    vad_enabled = bool(overrides.get("vad_enabled", config.streaming.vad_enabled))
    realtime_output = bool(
        overrides.get("realtime_output", config.streaming.realtime_output)
    )
    export_formats = overrides.get("export_formats", config.streaming.export_formats)
    if not isinstance(export_formats, list):
        export_formats = config.streaming.export_formats

    show_metrics = bool(overrides.get("show_metrics", config.streaming.show_metrics))
    summarize_enabled = bool(overrides.get("summarize", config.streaming.summarize))
    summary_model = str(overrides.get("summary_model", config.streaming.summary_model))
    speaker_mode = str(overrides.get("speaker_mode", config.streaming.speaker.mode))

    # Configurar archivo de salida
    output_file = combined_transcript or output_dir / "live_transcription.txt"

    # Crear instancia de métricas si está habilitado
    metrics = CaptureMetrics() if show_metrics else None

    transcriber: Any | None = None

    # Inicializar transcriber según el backend
    if backend == "mlx-whisper":
        if not MLX_WHISPER_AVAILABLE:
            logger.warning(
                "mlx-whisper backend requested but not available. "
                "Falling back to faster-whisper. "
                "Install: pip install mlx-whisper"
            )
            backend = "faster-whisper"
        else:
            logger.info("Using mlx-whisper backend (Apple Silicon GPU optimized)")
            transcriber = StreamingTranscriberMLX(
                model_size=model_size,
                language=language,
                output_file=output_file,
                vad_enabled=vad_enabled,
                realtime_output=realtime_output,
                metrics=metrics,
                hallucination_config=config.hallucination,
            )
    elif backend == "openai-whisper" or backend == "mps":
        if not MPS_AVAILABLE:
            logger.warning(
                "openai-whisper backend requested but not available. "
                "Falling back to faster-whisper. "
                "Install: pip install openai-whisper torch"
            )
            backend = "faster-whisper"
        else:
            logger.info("Using openai-whisper backend")
            logger.warning(
                "openai-whisper has known stability issues with MPS (GPU). "
                "It will use CPU by default. For better performance, consider using faster-whisper."
            )
            transcriber = StreamingTranscriberMPS(
                model_size=model_size,
                language=language,
                output_file=output_file,
                vad_enabled=vad_enabled,
                realtime_output=realtime_output,
            )
    else:
        # faster-whisper (default)
        logger.info("Using faster-whisper backend (CPU optimized)")
        transcriber = StreamingTranscriber(
            model_size=model_size,
            language=language,
            output_file=output_file,
            device=device,
            vad_enabled=vad_enabled,
            realtime_output=realtime_output,
            metrics=metrics,
            speaker_mode=speaker_mode,
            speaker_threshold=config.streaming.speaker.threshold,
            vad_config=config.vad,
            hallucination_config=config.hallucination,
        )

    # Verificar si debemos usar Swift CLI para audio del sistema
    use_screen_capture = SCREENCAPTUREKIT_AVAILABLE and config.audio.audio_source in ("system", "both")

    # Por defecto usamos ScreenCaptureKit (mismo que Notion): captura sistema fiable.
    # Core Audio Taps en algunos macOS devuelve silencio; usar USE_CORE_AUDIO_TAPS=1 para probarlo.
    use_screen_capture_kit = not (
        overrides.get("use_core_audio_taps") is True
        or get_bool_env("USE_CORE_AUDIO_TAPS", False)
    )

    if use_screen_capture:
        # Verificar permisos de Screen Recording
        if not check_screen_recording_permission():
            logger.warning("Screen Recording permission not granted")
            request_screen_recording_permission()
            logger.info("Will attempt to use ScreenCaptureKit anyway...")

    # Validar que tenemos un dispositivo de micrófono
    if not mic_device:
        logger.error(
            "No microphone device detected or configured.\n"
            "Run 'ffmpeg -f avfoundation -list_devices true -i \"\"' to see available devices.\n"
            "Then set AUDIO_MIC_DEVICE=<index> in your environment or config."
        )
        raise RuntimeError("No microphone device available")

    logger.info("Using microphone device: %s", mic_device)

    # Construir comando ffmpeg para streaming (solo micrófono)
    ffmpeg_cmd = _build_streaming_ffmpeg_command(
        system_device=system_device,
        mic_device=mic_device,
        sample_rate=sample_rate,
        channels=channels,
        chunk_duration=chunk_duration,
    )
    logger.info("ffmpeg command: %s", ' '.join(ffmpeg_cmd))

    stop_event = threading.Event()
    process = None
    screen_capture = None
    # transcriber already initialized above

    try:
        logger.info("Starting streaming capture...")
        logger.info("Chunk duration: %ss", chunk_duration)
        logger.info("Model: %s, Language: %s", model_size, language)
        logger.info("Output file: %s", output_file)

        if use_screen_capture:
            logger.info(
                "Using Swift CLI (%s) for system audio + ffmpeg for microphone",
                "ScreenCaptureKit" if use_screen_capture_kit else "Core Audio Taps",
            )
        else:
            logger.info("Using ffmpeg for microphone only")
            logger.info("ffmpeg command: %s", ' '.join(ffmpeg_cmd))

        # Buffer para combinar audio del sistema + micrófono
        combined_audio_buffer = bytearray()
        system_audio_buffer = bytearray()
        mic_audio_buffer = bytearray()

        # Iniciar ScreenCaptureKit si está disponible
        if use_screen_capture:
            try:

                def system_audio_callback(pcm_data: bytes):
                    """Callback para audio del sistema desde ScreenCaptureKit"""
                    system_audio_buffer.extend(pcm_data)
                    # Log solo ocasionalmente para no saturar los logs
                    if (
                        len(system_audio_buffer) % 64000 == 0
                    ):  # ~4 segundos de audio a 16kHz
                        logger.debug(
                            f"System audio buffer: {len(system_audio_buffer)} bytes"
                        )

                screen_capture = ScreenCaptureAudioCapture(
                    sample_rate=sample_rate,
                    channels=channels,
                    audio_callback=system_audio_callback,
                    use_screen_capture=use_screen_capture_kit,
                )

                if screen_capture.start():
                    logger.info(
                        "✓ Swift CLI (ScreenCaptureKit) started for system audio"
                    )
                else:
                    logger.warning(
                        "⚠️  Failed to start ScreenCaptureKit.\n"
                        "   Continuing with microphone-only capture.\n"
                        "   System audio will not be captured."
                    )
                    screen_capture = None
            except Exception as e:
                logger.error("Error starting ScreenCaptureKit: %s", e, exc_info=True)
                screen_capture = None

        # Thread para monitorear y reconectar Swift CLI si falla
        max_retries = 3
        retry_count = 0

        def monitor_swift_cli():
            """Monitorea el proceso Swift CLI y lo reinicia si falla."""
            nonlocal screen_capture, retry_count, system_audio_buffer

            if not use_screen_capture or not screen_capture:
                return

            while not stop_event.is_set():
                # Verificar cada 5 segundos si el proceso está vivo
                time.sleep(5.0)

                if stop_event.is_set():
                    break

                # Verificar si el proceso Swift terminó inesperadamente
                if screen_capture and screen_capture.process:
                    if screen_capture.process.poll() is not None:
                        # Proceso terminó
                        if retry_count < max_retries:
                            retry_count += 1
                            backoff_time = min(
                                2.0**retry_count, 10.0
                            )  # Exponential backoff, max 10s

                            logger.warning(
                                f"Swift CLI process ended unexpectedly. "
                                f"Attempting to restart ({retry_count}/{max_retries}) "
                                f"after {backoff_time:.1f}s..."
                            )

                            time.sleep(backoff_time)

                            # Limpiar buffer de audio del sistema
                            system_audio_buffer.clear()

                            # Intentar reiniciar
                            if screen_capture.restart():
                                logger.info("✓ Swift CLI restarted successfully")
                                retry_count = 0  # Reset retry count on success
                            else:
                                logger.error(
                                    f"Failed to restart Swift CLI ({retry_count}/{max_retries})"
                                )
                        else:
                            logger.error(
                                f"Max retries ({max_retries}) reached for Swift CLI. "
                                "Continuing with microphone-only capture."
                            )
                            screen_capture = None
                            break
                elif screen_capture and not screen_capture.is_capturing:
                    # Si no está capturando pero debería, intentar reiniciar
                    if retry_count < max_retries:
                        retry_count += 1
                        logger.warning(
                            f"Swift CLI not capturing. Attempting restart ({retry_count}/{max_retries})..."
                        )
                        if screen_capture.restart():
                            logger.info("✓ Swift CLI restarted successfully")
                            retry_count = 0
                        else:
                            logger.error(
                                f"Failed to restart Swift CLI ({retry_count}/{max_retries})"
                            )
                    else:
                        logger.error("Max retries reached for Swift CLI")
                        screen_capture = None
                        break

        # Iniciar thread de monitoreo si ScreenCaptureKit está activo
        monitor_thread = None
        if use_screen_capture and screen_capture:
            monitor_thread = threading.Thread(target=monitor_swift_cli, daemon=True)
            monitor_thread.start()
            logger.debug("Started Swift CLI monitoring thread")

        # Iniciar ffmpeg con stdout como pipe (micrófono)
        process = subprocess.Popen(
            ffmpeg_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,  # Sin buffering
        )

        # Thread para leer stderr (logs de ffmpeg)
        stderr_tail = deque(maxlen=50)

        def read_stderr():
            if process.stderr:
                for line in iter(process.stderr.readline, b""):
                    if line:
                        line_str = line.decode("utf-8", errors="ignore").strip()
                        stderr_tail.append(line_str)
                        # Mostrar errores y warnings en nivel INFO
                        if (
                            "error" in line_str.lower()
                            or "warning" in line_str.lower()
                            or "failed" in line_str.lower()
                        ):
                            logger.warning("ffmpeg: %s", line_str)
                        else:
                            logger.debug("ffmpeg: %s", line_str)

        stderr_thread = threading.Thread(target=read_stderr, daemon=True)
        stderr_thread.start()

        # Leer stream de audio desde stdout
        # ffmpeg con -f wav - escribe un WAV continuo con header al inicio
        logger.info("Capturing and transcribing audio... Press Ctrl+C to stop")

        # Leer header WAV completo (44 bytes típicamente) del micrófono
        # ffmpeg siempre envía header WAV, incluso cuando usamos ScreenCaptureKit
        wav_header = b""
        while len(wav_header) < 44 and not stop_event.is_set():
            poll_result = process.poll()
            if poll_result is not None:
                logger.error(
                    f"ffmpeg process died while reading WAV header. Exit code: {poll_result}"
                )
                # Give stderr_thread time to read, then dump what we have
                time.sleep(0.5)
                if process.stderr:
                    try:
                        remaining = process.stderr.read()
                        if remaining:
                            logger.error(
                                f"ffmpeg stderr (remaining):\n{remaining.decode('utf-8', errors='ignore')}"
                            )
                    except Exception:
                        pass
                logger.error(
                    f"ffmpeg command was: {' '.join(ffmpeg_cmd)}\n"
                    f"This usually means the microphone device could not be opened.\n"
                    f"Run 'ffmpeg -f avfoundation -list_devices true -i \"\"' to see available devices."
                )
                return
            chunk = process.stdout.read(44 - len(wav_header))
            if not chunk:
                if stop_event.is_set():
                    return
                time.sleep(0.1)
                continue
            wav_header += chunk

        if len(wav_header) < 44:
            logger.error("Could not read complete WAV header")
            return

        # Parsear header para obtener información
        # bytes 22-23: número de canales
        # bytes 24-27: sample rate
        # bytes 34-35: bits per sample
        n_channels = struct.unpack("<H", wav_header[22:24])[0]
        file_sample_rate = struct.unpack("<I", wav_header[24:28])[0]
        bits_per_sample = struct.unpack("<H", wav_header[34:36])[0]
        bytes_per_sample = bits_per_sample // 8

        logger.info(
            f"WAV format: {n_channels} channels, {file_sample_rate} Hz, {bits_per_sample} bits"
        )

        # Verificar que el sample rate coincida con el configurado
        if file_sample_rate != sample_rate:
            logger.warning(
                f"Sample rate mismatch: WAV={file_sample_rate}Hz, config={sample_rate}Hz"
            )

        # Calcular tamaño de chunk de audio (en bytes de datos PCM)
        chunk_audio_bytes = int(
            file_sample_rate * n_channels * bytes_per_sample * chunk_duration
        )

        # Persist full session audio for optional diarization (post-run).
        session_wav_path = output_dir / f"session_{int(time.time())}.wav"
        session_wav_writer = None
        try:
            import wave

            session_wav_path.parent.mkdir(parents=True, exist_ok=True)
            session_wav_writer = wave.open(str(session_wav_path), "wb")
            session_wav_writer.setnchannels(n_channels)
            session_wav_writer.setsampwidth(bytes_per_sample)
            session_wav_writer.setframerate(file_sample_rate)
            logger.info("Recording session audio to: %s", session_wav_path)
        except Exception:
            logger.warning("Failed to open session WAV for diarization (continuing)")
            session_wav_writer = None

        # Buffer para acumular datos PCM (solo micrófono cuando no hay ScreenCaptureKit)
        pcm_buffer = bytearray()
        warned_no_system_audio = [False]
        logged_system_audio_ok = [False]
        system_amplitude_log_count = [0]

        # Thread para leer y procesar chunks
        def process_audio_stream(transcriber_local: Any) -> None:
            nonlocal \
                pcm_buffer, \
                system_audio_buffer, \
                mic_audio_buffer, \
                combined_audio_buffer

            logger.debug("process_audio_stream started")
            iteration = 0

            while not stop_event.is_set():
                iteration += 1
                poll_result = process.poll()
                if poll_result is not None:
                    # ffmpeg process died - capture stderr for diagnosis
                    logger.error(
                        f"ffmpeg process died unexpectedly after {iteration} iterations. "
                        f"Exit code: {poll_result}"
                    )
                    logger.error("ffmpeg command was: %s", ' '.join(ffmpeg_cmd))
                    if stderr_tail:
                        logger.error(
                            "ffmpeg stderr (recent):\n%s", "\n".join(stderr_tail)
                        )
                    # Read any remaining stderr for the error message
                    if process.stderr:
                        try:
                            stderr_data = process.stderr.read()
                            if stderr_data:
                                stderr_text = stderr_data.decode(
                                    "utf-8", errors="ignore"
                                )
                                logger.error("ffmpeg stderr output:\n%s", stderr_text)
                        except Exception as e:
                            logger.error("Could not read ffmpeg stderr: %s", e)
                    break

                # Leer datos PCM del micrófono (sin header WAV, solo datos PCM)
                chunk = process.stdout.read(8192)
                if chunk:
                    # El header WAV ya fue leído, estos son datos PCM puros
                    mic_audio_buffer.extend(chunk)

                # Combinar audio del sistema + micrófono si ScreenCaptureKit está activo
                if screen_capture and screen_capture.is_capturing:
                    # Combinar buffers cuando tengamos datos de ambos
                    # Necesitamos sincronizar los dos streams
                    # Por simplicidad, combinamos cuando ambos tienen datos suficientes

                    # Calcular tamaño necesario para un chunk combinado
                    chunk_size = chunk_audio_bytes

                    # Log solo ocasionalmente para no saturar los logs
                    # (comentado para reducir logs, descomentar si necesitas debuggear)
                    # logger.debug(
                    #     f"Buffers - System: {len(system_audio_buffer)}/{chunk_size} bytes, "
                    #     f"Mic: {len(mic_audio_buffer)}/{chunk_size} bytes"
                    # )

                    if (
                        len(system_audio_buffer) >= chunk_size
                        and len(mic_audio_buffer) >= chunk_size
                    ):
                        if not logged_system_audio_ok[0]:
                            logger.info(
                                "System audio (Core Audio Taps) + mic: mixing and recording"
                            )
                            logged_system_audio_ok[0] = True
                        # Extraer chunks del mismo tamaño
                        system_chunk = bytes(system_audio_buffer[:chunk_size])
                        mic_chunk = bytes(mic_audio_buffer[:chunk_size])

                        # Remover del buffer
                        system_audio_buffer = system_audio_buffer[chunk_size:]
                        mic_audio_buffer = mic_audio_buffer[chunk_size:]

                        # Combinar audio (mixear)
                        # Convertir a numpy arrays para mezclar
                        system_array = np.frombuffer(system_chunk, dtype=np.int16)
                        mic_array = np.frombuffer(mic_chunk, dtype=np.int16)

                        # Diagnóstico: si el sistema es todo ceros, el WAV suena solo a mic
                        if system_amplitude_log_count[0] < 5:
                            sys_max = int(np.abs(system_array).max())
                            logger.info(
                                "System chunk amplitude (max abs): %d (0 = silence from tap)",
                                sys_max,
                            )
                            if sys_max == 0 and system_amplitude_log_count[0] == 0:
                                logger.warning(
                                    "System audio is silence. Check permissions: "
                                    "System Settings > Privacy & Security > Screen & System Audio Recording — "
                                    "add your terminal app (Terminal, iTerm, Cursor) and enable it. Restart the terminal after granting."
                                )
                            system_amplitude_log_count[0] += 1

                        # Mezclar con normalización dinámica
                        mic_boost = get_float_env(
                            "AUDIO_MIC_BOOST", 1.2, min_value=0.1, max_value=5.0
                        )
                        combined_array = mix_audio(system_array, mic_array, mic_boost)
                        combined_chunk = combined_array.tobytes()

                        # Crear WAV chunk completo con header
                        wav_chunk = _create_wav_chunk(
                            wav_header,
                            combined_chunk,
                            n_channels,
                            file_sample_rate,
                            bits_per_sample,
                        )

                        # Procesar chunk
                        try:
                            logger.debug(
                                f"🎵 Processing COMBINED audio: System={len(system_chunk)} bytes, "
                                f"Mic={len(mic_chunk)} bytes, Combined={len(combined_chunk)} bytes"
                            )
                            if session_wav_writer:
                                session_wav_writer.writeframes(combined_chunk)
                            result = transcriber_local.process_wav_chunk(wav_chunk)
                            if result:
                                logger.info("✅ Transcription (combined): %s", result)
                            else:
                                logger.debug(
                                    "No transcription result (silence or VAD filtered)"
                                )
                        except Exception as e:
                            logger.error(
                                f"Error processing audio chunk: {e}", exc_info=True
                            )
                    else:
                        # Si no tenemos datos suficientes, esperar un poco
                        # Pero también procesar solo micrófono si el sistema no tiene datos
                        if (
                            len(mic_audio_buffer) >= chunk_size
                            and len(system_audio_buffer) == 0
                        ):
                            # Solo micrófono disponible: Core Audio Taps no está enviando audio del sistema
                            if not warned_no_system_audio[0]:
                                logger.warning(
                                    "No system audio from Swift CLI (Core Audio Taps). "
                                    "Recording mic only. Check Audio Capture permission and that audio is playing."
                                )
                                warned_no_system_audio[0] = True
                            logger.debug(
                                "⚠️  No system audio available, processing microphone only"
                            )
                            mic_chunk = bytes(mic_audio_buffer[:chunk_size])
                            mic_audio_buffer = mic_audio_buffer[chunk_size:]

                            # Crear WAV chunk solo con micrófono
                            wav_chunk = _create_wav_chunk(
                                wav_header,
                                mic_chunk,
                                n_channels,
                                file_sample_rate,
                                bits_per_sample,
                            )

                            try:
                                logger.debug(
                                    f"🎤 Processing microphone-only chunk: {len(mic_chunk)} bytes"
                                )
                                if session_wav_writer:
                                    session_wav_writer.writeframes(mic_chunk)
                                result = transcriber_local.process_wav_chunk(wav_chunk)
                                if result:
                                    logger.info(
                                        f"✅ Transcription result (mic only): {result}"
                                    )
                            except Exception as e:
                                logger.error(
                                    f"Error processing mic-only chunk: {e}",
                                    exc_info=True,
                                )
                        elif (
                            len(system_audio_buffer) >= chunk_size
                            and len(mic_audio_buffer) == 0
                        ):
                            # Solo sistema disponible, procesar solo eso
                            logger.debug(
                                "🔊 Processing system audio only (no microphone)"
                            )
                            system_chunk = bytes(system_audio_buffer[:chunk_size])
                            system_audio_buffer = system_audio_buffer[chunk_size:]

                            # Crear WAV chunk solo con sistema
                            wav_chunk = _create_wav_chunk(
                                wav_header,
                                system_chunk,
                                n_channels,
                                file_sample_rate,
                                bits_per_sample,
                            )

                            try:
                                logger.debug(
                                    f"🔊 Processing system-only chunk: {len(system_chunk)} bytes"
                                )
                                if session_wav_writer:
                                    session_wav_writer.writeframes(system_chunk)
                                result = transcriber_local.process_wav_chunk(wav_chunk)
                                if result:
                                    logger.info(
                                        f"✅ Transcription result (system only): {result}"
                                    )
                            except Exception as e:
                                logger.error(
                                    f"Error processing system-only chunk: {e}",
                                    exc_info=True,
                                )
                        else:
                            time.sleep(0.01)
                else:
                    # Solo micrófono (modo fallback)
                    if not chunk:
                        if stop_event.is_set():
                            break
                        time.sleep(0.05)
                        continue

                    pcm_buffer.extend(chunk)

                    # Procesar chunks completos
                    while len(pcm_buffer) >= chunk_audio_bytes:
                        # Extraer chunk de audio PCM
                        pcm_chunk = bytes(pcm_buffer[:chunk_audio_bytes])
                        pcm_buffer = pcm_buffer[chunk_audio_bytes:]

                        # Crear WAV chunk completo con header
                        wav_chunk = _create_wav_chunk(
                            wav_header,
                            pcm_chunk,
                            n_channels,
                            file_sample_rate,
                            bits_per_sample,
                        )

                        # Procesar chunk
                        try:
                            if session_wav_writer:
                                session_wav_writer.writeframes(pcm_chunk)
                            result = transcriber_local.process_wav_chunk(wav_chunk)
                            if result:
                                logger.debug("Transcription result: %s...", result[:50])
                            else:
                                logger.debug(
                                    "No transcription result (silence or VAD filtered)"
                                )
                        except Exception as e:
                            logger.error(
                                f"Error processing audio chunk: {e}", exc_info=True
                            )

            # Log why the thread is ending
            if stop_event.is_set():
                logger.info("process_audio_stream ending: stop_event was set")
            else:
                logger.warning(
                    f"process_audio_stream ending after {iteration} iterations "
                    "(ffmpeg died or no more data)"
                )

        # Iniciar thread de procesamiento
        if transcriber is not None:
            process_thread = threading.Thread(
                target=process_audio_stream, args=(transcriber,), daemon=True
            )
            process_thread.start()
            logger.info("Audio processing thread started")
        else:
            process_thread = None
            logger.warning("Transcriber is None, audio processing thread NOT started!")

        interrupted = False

        # Esperar hasta que se detenga (solo si el hilo se inició correctamente)
        try:
            while (
                process_thread is not None
                and process_thread.is_alive()
                and not stop_event.is_set()
            ):
                time.sleep(0.5)

            # Log why we exited the loop
            if process_thread is None:
                logger.warning("Main loop exited: process_thread was None")
            elif not process_thread.is_alive():
                logger.warning("Main loop exited: process_thread died")
            elif stop_event.is_set():
                logger.info("Main loop exited: stop_event was set")
        except KeyboardInterrupt:
            logger.info("Interrupted, stopping gracefully...")
            interrupted = True
    except Exception as e:
        logger.error("Error in streaming capture: %s", e, exc_info=True)
    finally:
        logger.info("Stopping capture...")
        stop_event.set()

        # Detener ScreenCaptureKit
        if screen_capture:
            try:
                screen_capture.stop()
                logger.info("ScreenCaptureKit stopped")
            except Exception as e:
                logger.error("Error stopping ScreenCaptureKit: %s", e)

        # Close session WAV writer.
        try:
            if "session_wav_writer" in locals() and session_wav_writer:
                session_wav_writer.close()
        except Exception:
            logger.debug("Failed to close session WAV writer", exc_info=True)

        # Detener ffmpeg
        if process:
            try:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        logger.warning("ffmpeg did not stop, killing...")
                        process.kill()
                        try:
                            process.wait(timeout=2)
                        except subprocess.TimeoutExpired:
                            logger.warning("ffmpeg still running after kill()")
            finally:
                # Close pipes so daemon threads can exit
                try:
                    if process.stdout:
                        process.stdout.close()
                except Exception:
                    logger.debug("Failed to close ffmpeg stdout", exc_info=True)
                try:
                    if process.stderr:
                        process.stderr.close()
                except Exception:
                    logger.debug("Failed to close ffmpeg stderr", exc_info=True)

        # Asegurar que el thread de stderr termine (no bloqueante)
        try:
            if "stderr_thread" in locals() and stderr_thread is not None:
                stderr_thread.join(timeout=2.0)
        except Exception:
            logger.debug("Failed to join ffmpeg stderr thread", exc_info=True)

        # Asegurar que el thread de procesamiento termine antes de exportar/resumir
        try:
            if "process_thread" in locals() and process_thread is not None:
                process_thread.join(timeout=5.0)
        except Exception:
            logger.debug("Failed to join processing thread", exc_info=True)

        # Exportar en formatos adicionales si se especificó
        # Optional: apply real diarization post-run.
        #
        run_pyannote = (
            speaker_mode == "pyannote"
            and "session_wav_path" in locals()
            and session_wav_path.exists()
            and transcriber is not None
        )
        if speaker_mode == "pyannote" and transcriber is not None and not run_pyannote:
            logger.warning(
                "Skipping pyannote diarization: no session WAV file "
                "(session_wav_path missing or file not created). "
                "Speaker labels will not be applied."
            )
        if run_pyannote:
            try:
                from local_transcriber.speaker.diarization import (
                    assign_speakers_to_segments,
                    diarize_wav,
                )

                # Check if file has enough audio before attempting diarization
                try:
                    import wave

                    with wave.open(str(session_wav_path), "rb") as wav_check:
                        sample_rate = wav_check.getframerate()
                        n_frames = wav_check.getnframes()
                        duration = n_frames / sample_rate if sample_rate > 0 else 0.0

                    if duration < 2.0:
                        logger.warning(
                            f"Session audio too short ({duration:.2f}s) for pyannote diarization. "
                            f"Need at least 2-3 seconds. Skipping diarization."
                        )
                    else:
                        logger.info(
                            f"Running pyannote diarization on session audio ({duration:.2f}s)..."
                        )
                        turns = diarize_wav(session_wav_path)
                        # Update segments in-place for exports
                        if hasattr(transcriber, "lock") and hasattr(
                            transcriber, "segments"
                        ):
                            lock = getattr(transcriber, "lock", None)
                            if lock:
                                with lock:
                                    transcriber.segments = assign_speakers_to_segments(  # type: ignore[attr-defined]
                                        transcriber.segments, turns
                                    )
                            else:
                                transcriber.segments = assign_speakers_to_segments(  # type: ignore[attr-defined]
                                    transcriber.segments, turns
                                )
                        logger.info("Applied speaker labels from pyannote")
                except Exception as check_error:
                    # If duration check fails, try diarization anyway
                    logger.debug(
                        f"Could not check audio duration: {check_error}. Attempting diarization anyway..."
                    )
                    logger.info("Running pyannote diarization on session audio...")
                    turns = diarize_wav(session_wav_path)
                    # Update segments in-place for exports
                    if hasattr(transcriber, "lock") and hasattr(
                        transcriber, "segments"
                    ):
                        lock = getattr(transcriber, "lock", None)
                        if lock:
                            with lock:
                                transcriber.segments = assign_speakers_to_segments(  # type: ignore[attr-defined]
                                    transcriber.segments, turns
                                )
                        else:
                            transcriber.segments = assign_speakers_to_segments(  # type: ignore[attr-defined]
                                transcriber.segments, turns
                            )
                    logger.info("Applied speaker labels from pyannote")
            except RuntimeError as e:
                # RuntimeError from diarization.py includes helpful instructions
                error_msg = str(e)
                logger.error("pyannote diarization failed: %s", error_msg)
                # Only show full traceback if it's not a known issue (access denied, file too small, etc.)
                if "Access denied" not in error_msg and "too small" not in error_msg:
                    logger.debug("Full error details:", exc_info=True)
                logger.warning(
                    "Speaker diarization skipped. Transcription will continue without speaker labels."
                )
            except Exception as e:
                logger.error("pyannote diarization failed: %s", e, exc_info=True)
                logger.warning(
                    "Speaker diarization skipped. Transcription will continue without speaker labels."
                )
        if export_formats and transcriber is not None:
            logger.info("Exporting transcript in formats: %s", ', '.join(export_formats))
            try:
                transcriber.export_transcript(export_formats, output_dir)
            except Exception as e:
                logger.error("Error exporting transcript: %s", e, exc_info=True)

        logger.info("Post-run: export step completed")

        # Generar resumen si está habilitado
        summary_output = None
        if summarize_enabled and transcriber:
            try:
                logger.info("Generating summary...")
                full_transcript = transcriber.get_full_transcript()
                if full_transcript.strip():
                    summary_output = (
                        output_file.parent / f"{output_file.stem}_summary.json"
                    )
                    summary_data = generate_summary(
                        full_transcript, model=summary_model, output_path=summary_output
                    )
                    if summary_data:
                        logger.info("Summary generated and saved to: %s", summary_output)
                    else:
                        logger.warning("Failed to generate summary")
                else:
                    logger.warning("No transcript content to summarize")
            except Exception as e:
                logger.error("Error generating summary: %s", e, exc_info=True)

        logger.info("Post-run: summary step completed")

        # Enviar notificación si está habilitado
        notify_enabled = get_bool_env("STREAMING_NOTIFY", False)
        notify_platform = get_str_env("STREAMING_NOTIFY_PLATFORM", "telegram")
        if notify_enabled and summary_output and summary_output.exists():
            try:
                if notify_platform == "telegram":
                    from local_transcriber.notify.telegram import send_summary

                    send_summary(summary_output)
            except Exception as e:
                logger.error("Error sending notification: %s", e, exc_info=True)

        logger.info("Post-run: notify step completed")

        # Mostrar métricas si están habilitadas (lock-timeout para evitar hangs)
        if metrics:
            try:
                metrics.print_summary(timeout_seconds=0.5)
            except Exception:
                logger.debug("Failed to print metrics", exc_info=True)

        logger.info("Post-run: metrics step completed")

        # Debug: list non-daemon threads that may keep process alive
        try:
            import threading as _threading

            alive = _threading.enumerate()
            non_daemon = [
                t
                for t in alive
                if not t.daemon and t is not _threading.current_thread()
            ]
            if non_daemon:
                logger.warning(
                    "Non-daemon threads still alive at shutdown: %s",
                    ", ".join(f"{t.name}({t.__class__.__name__})" for t in non_daemon),
                )
            else:
                logger.info("No non-daemon threads alive at shutdown")
        except Exception:
            logger.debug("Failed to enumerate threads", exc_info=True)

        logger.info("Shutdown complete")
        logger.info("Full transcript saved to: %s", output_file)


def _create_wav_chunk(
    original_header: bytes,
    pcm_data: bytes,
    n_channels: int,
    sample_rate: int,
    bits_per_sample: int,
) -> bytes:
    """
    Crea un chunk WAV completo con header y datos PCM.

    Args:
        original_header: Header WAV original (no se usa, pero se mantiene para compatibilidad)
        pcm_data: Datos PCM en formato int16
        n_channels: Número de canales
        sample_rate: Sample rate en Hz
        bits_per_sample: Bits por muestra (típicamente 16)

    Returns:
        Bytes del chunk WAV completo (header + datos)
    """
    # Calcular tamaños
    data_size = len(pcm_data)
    # Tamaño del archivo = 4 (RIFF) + 4 (file_size) + 4 (WAVE) +
    #                     4 (fmt ) + 4 (fmt_size) + 18 (fmt_data) +
    #                     4 (data) + 4 (data_size) + data_size
    # Total header = 44 bytes
    file_size = 36 + data_size  # 36 = tamaño del header sin RIFF y file_size

    # Crear header WAV completo desde cero
    header = bytearray()

    # RIFF chunk descriptor
    header.extend(b"RIFF")  # ChunkID
    header.extend(struct.pack("<I", file_size))  # ChunkSize
    header.extend(b"WAVE")  # Format

    # fmt sub-chunk
    header.extend(b"fmt ")  # Subchunk1ID
    header.extend(struct.pack("<I", 16))  # Subchunk1Size (16 para PCM)
    header.extend(struct.pack("<H", 1))  # AudioFormat (1 = PCM)
    header.extend(struct.pack("<H", n_channels))  # NumChannels
    header.extend(struct.pack("<I", sample_rate))  # SampleRate
    header.extend(
        struct.pack("<I", sample_rate * n_channels * (bits_per_sample // 8))
    )  # ByteRate
    header.extend(struct.pack("<H", n_channels * (bits_per_sample // 8)))  # BlockAlign
    header.extend(struct.pack("<H", bits_per_sample))  # BitsPerSample

    # data sub-chunk
    header.extend(b"data")  # Subchunk2ID
    header.extend(struct.pack("<I", data_size))  # Subchunk2Size

    # Combinar header + datos
    return bytes(header) + pcm_data


def _build_streaming_ffmpeg_command(
    system_device: str,
    mic_device: str,
    sample_rate: int,
    channels: int,
    chunk_duration: float,
) -> list[str]:
    """
    Construye comando ffmpeg para captura en streaming.

    Nota: chunk_duration no se usa en el comando actual, pero se mantiene
    para compatibilidad futura.

    Output a stdout en formato WAV para procesamiento en tiempo real.

    Nota: El audio del sistema se captura usando ScreenCaptureKit (API nativa de macOS).
    No se requiere BlackHole ni dispositivos virtuales.
    """
    # ScreenCaptureKit captura el audio del sistema directamente
    # Solo necesitamos el micrófono aquí
    mic_input = _format_device(mic_device)

    # Por ahora, solo capturamos el micrófono
    # El audio del sistema se capturará con ScreenCaptureKit en otro módulo
    # Por ahora solo capturamos el micrófono
    # El audio del sistema se capturará con ScreenCaptureKit
    return [
        "ffmpeg",
        "-f",
        "avfoundation",
        "-i",
        mic_input,  # Micrófono
        "-ar",
        str(sample_rate),
        "-ac",
        str(channels),
        "-f",
        "wav",  # Formato WAV
        "-",  # Output a stdout
    ]


def _cleanup_temp_dir(path: Path, label: str) -> None:
    """
    Elimina un directorio temporal.

    Args:
        path: Ruta del directorio a eliminar
        label: Etiqueta descriptiva para logs
    """
    if not path.exists():
        return
    try:
        shutil.rmtree(path)
        logger.info("Removed %s directory: %s", label, path)
    except Exception:
        logger.exception("Failed to remove %s directory: %s", label, path)
