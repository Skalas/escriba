from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path

from local_transcriber.audio.device_detection import auto_detect_devices
from local_transcriber.transcribe.streaming import StreamingTranscriber
from local_transcriber.watch.watch_folder import watch_folder, wait_for_queue_empty

logger = logging.getLogger(__name__)


def run_live_capture(output_dir: Path, combined_transcript: Path | None = None) -> None:
    # Detección automática de dispositivos (como Notion AI)
    auto_detect = _get_bool_env("AUTO_DETECT_DEVICES", True)

    if auto_detect:
        logger.info("Auto-detecting audio devices...")
        detected_system, detected_mic = auto_detect_devices()
        system_device = detected_system or _get_str_env("SYSTEM_DEVICE", "0")
        mic_device = detected_mic or _get_str_env("MIC_DEVICE", "1")
        logger.info(f"Using devices - System: {system_device}, Mic: {mic_device}")
    else:
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


def run_streaming_capture(
    output_dir: Path,
    combined_transcript: Path | None = None,
) -> None:
    """
    Captura y transcribe audio en tiempo real usando streaming.

    Similar a Notion AI Meeting Notes, procesa chunks pequeños (1-3s)
    y muestra transcripciones mientras ocurre la llamada.
    """
    # Detección automática de dispositivos (como Notion AI)
    auto_detect = _get_bool_env("AUTO_DETECT_DEVICES", True)

    if auto_detect:
        logger.info("Auto-detecting audio devices...")
        detected_system, detected_mic = auto_detect_devices()
        system_device = detected_system or _get_str_env("SYSTEM_DEVICE", "0")
        mic_device = detected_mic or _get_str_env("MIC_DEVICE", "1")
        logger.info(f"Using devices - System: {system_device}, Mic: {mic_device}")

        # Información sobre captura de audio del sistema
        if not detected_system:
            logger.warning(
                "⚠️  No se detectó dispositivo para capturar audio del sistema.\n"
                "  - Solo se capturará el micrófono\n"
                "  - Para capturar audio del sistema, instala BlackHole: brew install blackhole-2ch"
            )
        else:
            # Verificar si el dispositivo de salida actual es compatible
            try:
                from local_transcriber.audio.system_audio_capture import (
                    get_current_output_device,
                )

                current_output = get_current_output_device()
                if current_output:
                    current_name = current_output.get("name", "").lower()
                    if "blackhole" not in current_name:
                        logger.info(
                            f"💡 Dispositivo de salida actual: {current_output.get('name')}\n"
                            f"  - El audio del sistema puede no capturarse si no está configurado BlackHole\n"
                            f"  - Considera crear un Multi-Output Device que combine {current_output.get('name')} + BlackHole"
                        )
            except Exception as e:
                logger.debug(f"Could not check current output device: {e}")
    else:
        system_device = _get_str_env("SYSTEM_DEVICE", "0")
        mic_device = _get_str_env("MIC_DEVICE", "1")
    sample_rate = _get_int_env("SAMPLE_RATE", 16000, min_value=8000)
    channels = _get_int_env("CHANNELS", 1, min_value=1)
    chunk_duration = _get_float_env("STREAMING_CHUNK_DURATION", 2.0, min_value=0.5)
    model_size = _get_str_env("STREAMING_MODEL_SIZE", "base")
    language = _get_str_env("STREAMING_LANGUAGE", "es")
    vad_enabled = _get_bool_env("STREAMING_VAD_ENABLED", True)
    realtime_output = _get_bool_env("STREAMING_REALTIME_OUTPUT", True)

    # Configurar archivo de salida
    output_file = combined_transcript or output_dir / "live_transcription.txt"

    # Inicializar transcriber
    transcriber = StreamingTranscriber(
        model_size=model_size,
        language=language,
        output_file=output_file,
        vad_enabled=vad_enabled,
        realtime_output=realtime_output,
    )

    # Construir comando ffmpeg para streaming
    ffmpeg_cmd = _build_streaming_ffmpeg_command(
        system_device=system_device,
        mic_device=mic_device,
        sample_rate=sample_rate,
        channels=channels,
        chunk_duration=chunk_duration,
    )

    stop_event = threading.Event()
    process = None

    try:
        logger.info("Starting streaming capture...")
        logger.info(f"Chunk duration: {chunk_duration}s")
        logger.info(f"Model: {model_size}, Language: {language}")
        logger.info(f"Output file: {output_file}")
        logger.info(f"ffmpeg command: {' '.join(ffmpeg_cmd)}")

        # Iniciar ffmpeg con stdout como pipe
        process = subprocess.Popen(
            ffmpeg_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,  # Sin buffering
        )

        # Thread para leer stderr (logs de ffmpeg)
        def read_stderr():
            if process.stderr:
                for line in iter(process.stderr.readline, b""):
                    if line:
                        line_str = line.decode("utf-8", errors="ignore").strip()
                        # Mostrar errores y warnings en nivel INFO
                        if (
                            "error" in line_str.lower()
                            or "warning" in line_str.lower()
                            or "failed" in line_str.lower()
                        ):
                            logger.warning(f"ffmpeg: {line_str}")
                        else:
                            logger.debug(f"ffmpeg: {line_str}")

        stderr_thread = threading.Thread(target=read_stderr, daemon=True)
        stderr_thread.start()

        # Leer stream de audio desde stdout
        # ffmpeg con -f wav - escribe un WAV continuo con header al inicio
        import struct

        logger.info("Capturing and transcribing audio... Press Ctrl+C to stop")

        # Leer header WAV completo (44 bytes típicamente)
        wav_header = b""
        while len(wav_header) < 44 and not stop_event.is_set():
            if process.poll() is not None:
                logger.warning("ffmpeg process ended unexpectedly")
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

        # Calcular tamaño de chunk de audio (en bytes de datos PCM)
        chunk_audio_bytes = int(
            file_sample_rate * n_channels * bytes_per_sample * chunk_duration
        )

        # Buffer para acumular datos PCM
        pcm_buffer = bytearray()

        # Thread para leer y procesar chunks
        def process_audio_stream():
            nonlocal pcm_buffer

            while not stop_event.is_set():
                if process.poll() is not None:
                    break

                # Leer datos PCM (sin header)
                chunk = process.stdout.read(8192)
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
                        result = transcriber.process_wav_chunk(wav_chunk)
                        if result:
                            logger.debug(f"Transcription result: {result[:50]}...")
                        else:
                            logger.debug(
                                "No transcription result (silence or VAD filtered)"
                            )
                    except Exception as e:
                        logger.error(
                            f"Error processing audio chunk: {e}", exc_info=True
                        )

        # Iniciar thread de procesamiento
        process_thread = threading.Thread(target=process_audio_stream, daemon=True)
        process_thread.start()

        # Esperar hasta que se detenga
        try:
            while process_thread.is_alive() and not stop_event.is_set():
                time.sleep(0.5)
        except KeyboardInterrupt:
            logger.info("Interrupted, stopping gracefully...")
    except Exception as e:
        logger.error(f"Error in streaming capture: {e}", exc_info=True)
    finally:
        logger.info("Stopping ffmpeg...")
        stop_event.set()

        # Detener ffmpeg
        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning("ffmpeg did not stop, killing...")
                process.kill()

        logger.info("Shutdown complete")
        logger.info(f"Full transcript saved to: {output_file}")


def _create_wav_chunk(
    original_header: bytes,
    pcm_data: bytes,
    n_channels: int,
    sample_rate: int,
    bits_per_sample: int,
) -> bytes:
    """Crea un chunk WAV completo con header y datos PCM."""
    import struct

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

    Output a stdout en formato WAV para procesamiento en tiempo real.

    Nota: Para capturar audio del sistema en macOS, necesitas:
    1. Instalar BlackHole (https://github.com/ExistentialAudio/BlackHole)
    2. Configurar BlackHole como dispositivo de salida en Preferencias del Sistema > Sonido
    3. O usar Multi-Output Device que combine tu salida normal + BlackHole
    """
    system_input = _format_device(system_device)
    mic_input = _format_device(mic_device)

    # Usar dos inputs separados y combinarlos con amix
    # system_input captura el audio del sistema (BlackHole)
    # mic_input captura el micrófono (AirPods, etc.)
    return [
        "ffmpeg",
        "-f",
        "avfoundation",
        "-i",
        system_input,  # Audio del sistema (BlackHole)
        "-f",
        "avfoundation",
        "-i",
        mic_input,  # Micrófono
        "-filter_complex",
        "amix=inputs=2:duration=longest:dropout_transition=0",
        "-ar",
        str(sample_rate),
        "-ac",
        str(channels),
        "-f",
        "wav",  # Formato WAV
        "-",  # Output a stdout
    ]


def _get_float_env(name: str, default: float, min_value: float | None = None) -> float:
    """Obtiene variable de entorno como float."""
    raw = os.getenv(name, str(default)).strip()
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a float, got {raw!r}") from exc
    if min_value is not None and value < min_value:
        raise ValueError(f"{name} must be >= {min_value}, got {value}")
    return value


def _get_bool_env(name: str, default: bool) -> bool:
    """Obtiene variable de entorno como bool."""
    raw = os.getenv(name, str(default)).strip().lower()
    return raw in ("true", "1", "yes", "on")


def _cleanup_temp_dir(path: Path, label: str) -> None:
    if not path.exists():
        return
    try:
        shutil.rmtree(path)
        logger.info("Removed %s directory: %s", label, path)
    except Exception:
        logger.exception("Failed to remove %s directory: %s", label, path)
