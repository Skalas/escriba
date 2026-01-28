from __future__ import annotations

import logging
import shutil
import struct
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

from local_transcriber.audio.device_detection import auto_detect_devices
from local_transcriber.transcribe.streaming import StreamingTranscriber
from local_transcriber.utils.env import (
    get_bool_env,
    get_float_env,
    get_int_env,
    get_str_env,
)
from local_transcriber.watch.watch_folder import watch_folder, wait_for_queue_empty

# Intentar importar StreamingTranscriberMPS (requiere openai-whisper y torch)
try:
    from local_transcriber.transcribe.streaming_mps import StreamingTranscriberMPS

    MPS_AVAILABLE = True
except ImportError:
    MPS_AVAILABLE = False
    StreamingTranscriberMPS = None

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


def run_live_capture(output_dir: Path, combined_transcript: Path | None = None) -> None:
    """
    Captura audio en vivo usando ffmpeg y transcribe segmentos (modo legacy).

    Este modo captura audio en segmentos de 30 segundos y los transcribe
    usando Whisper CLI. Tiene mayor latencia que el modo streaming.

    Args:
        output_dir: Directorio donde guardar transcripciones
        combined_transcript: Archivo opcional para transcripción combinada
    """
    # Detección automática de dispositivos (como Notion AI)
    auto_detect = get_bool_env("AUTO_DETECT_DEVICES", True)

    if auto_detect:
        logger.info("Auto-detecting audio devices...")
        detected_system, detected_mic = auto_detect_devices()
        system_device = detected_system or get_str_env("SYSTEM_DEVICE", "0")
        mic_device = detected_mic or get_str_env("MIC_DEVICE", "1")
        logger.info(f"Using devices - System: {system_device}, Mic: {mic_device}")
    else:
        system_device = get_str_env("SYSTEM_DEVICE", "0")
        mic_device = get_str_env("MIC_DEVICE", "1")
    sample_rate = get_int_env("SAMPLE_RATE", 16000, min_value=8000)
    channels = get_int_env("CHANNELS", 1, min_value=1)
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
) -> None:
    """
    Captura y transcribe audio en tiempo real usando streaming.

    Similar a Notion AI Meeting Notes, procesa chunks pequeños (1-3s)
    y muestra transcripciones mientras ocurre la llamada.
    """
    # Detección automática de dispositivos (como Notion AI)
    auto_detect = get_bool_env("AUTO_DETECT_DEVICES", True)

    if auto_detect:
        logger.info("Auto-detecting audio devices...")
        detected_system, detected_mic = auto_detect_devices()
        system_device = detected_system or get_str_env("SYSTEM_DEVICE", "0")
        mic_device = detected_mic or get_str_env("MIC_DEVICE", "1")
        logger.info(f"Using devices - System: {system_device}, Mic: {mic_device}")

        # ScreenCaptureKit captura el audio del sistema directamente
        # No necesitamos un dispositivo virtual
        logger.info(
            "Using ScreenCaptureKit for system audio capture (no virtual device needed)"
        )
    else:
        system_device = get_str_env("SYSTEM_DEVICE", "0")
        mic_device = get_str_env("MIC_DEVICE", "1")
    sample_rate = get_int_env("SAMPLE_RATE", 16000, min_value=8000)
    channels = get_int_env("CHANNELS", 1, min_value=1)
    chunk_duration = get_float_env("STREAMING_CHUNK_DURATION", 30.0, min_value=0.5)
    model_size = get_str_env("STREAMING_MODEL_SIZE", "base")
    language = get_str_env("STREAMING_LANGUAGE", "es")
    device = get_str_env("STREAMING_DEVICE", "auto")
    backend = get_str_env(
        "STREAMING_BACKEND", "faster-whisper"
    )  # faster-whisper o openai-whisper
    vad_enabled = get_bool_env(
        "STREAMING_VAD_ENABLED", False
    )  # Deshabilitado por defecto para capturar todo el audio
    realtime_output = get_bool_env("STREAMING_REALTIME_OUTPUT", True)

    # Configurar archivo de salida
    output_file = combined_transcript or output_dir / "live_transcription.txt"

    # Inicializar transcriber según el backend
    if backend == "openai-whisper" or backend == "mps":
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
        )

    # Verificar si debemos usar ScreenCaptureKit para audio del sistema
    use_screen_capture = SCREENCAPTUREKIT_AVAILABLE and not get_bool_env(
        "MIC_ONLY", False
    )

    if use_screen_capture:
        # Verificar permisos de Screen Recording
        if not check_screen_recording_permission():
            logger.warning("Screen Recording permission not granted")
            request_screen_recording_permission()
            logger.info("Will attempt to use ScreenCaptureKit anyway...")

    # Construir comando ffmpeg para streaming (solo micrófono)
    ffmpeg_cmd = _build_streaming_ffmpeg_command(
        system_device=system_device,
        mic_device=mic_device,
        sample_rate=sample_rate,
        channels=channels,
        chunk_duration=chunk_duration,
    )

    stop_event = threading.Event()
    process = None
    screen_capture = None

    try:
        logger.info("Starting streaming capture...")
        logger.info(f"Chunk duration: {chunk_duration}s")
        logger.info(f"Model: {model_size}, Language: {language}")
        logger.info(f"Output file: {output_file}")

        if use_screen_capture:
            logger.info(
                "Using Swift CLI (ScreenCaptureKit) for system audio + ffmpeg for microphone"
            )
        else:
            logger.info("Using ffmpeg for microphone only")
            logger.info(f"ffmpeg command: {' '.join(ffmpeg_cmd)}")

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
                logger.error(f"Error starting ScreenCaptureKit: {e}", exc_info=True)
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
        logger.info("Capturing and transcribing audio... Press Ctrl+C to stop")

        # Leer header WAV completo (44 bytes típicamente) del micrófono
        # ffmpeg siempre envía header WAV, incluso cuando usamos ScreenCaptureKit
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

        # Verificar que el sample rate coincida con el configurado
        if file_sample_rate != sample_rate:
            logger.warning(
                f"Sample rate mismatch: WAV={file_sample_rate}Hz, config={sample_rate}Hz"
            )

        # Calcular tamaño de chunk de audio (en bytes de datos PCM)
        chunk_audio_bytes = int(
            file_sample_rate * n_channels * bytes_per_sample * chunk_duration
        )

        # Buffer para acumular datos PCM (solo micrófono cuando no hay ScreenCaptureKit)
        pcm_buffer = bytearray()

        # Thread para leer y procesar chunks
        def process_audio_stream():
            nonlocal \
                pcm_buffer, \
                system_audio_buffer, \
                mic_audio_buffer, \
                combined_audio_buffer

            while not stop_event.is_set():
                if process.poll() is not None:
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
                            logger.info(
                                f"🎵 Processing COMBINED audio: System={len(system_chunk)} bytes, "
                                f"Mic={len(mic_chunk)} bytes, Combined={len(combined_chunk)} bytes"
                            )
                            result = transcriber.process_wav_chunk(wav_chunk)
                            if result:
                                logger.info(f"✅ Transcription (combined): {result}")
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
                            # Solo micrófono disponible, procesar solo eso
                            logger.warning(
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
                                logger.info(
                                    f"🎤 Processing microphone-only chunk: {len(mic_chunk)} bytes"
                                )
                                result = transcriber.process_wav_chunk(wav_chunk)
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
                            logger.info(
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
                                logger.info(
                                    f"🔊 Processing system-only chunk: {len(system_chunk)} bytes"
                                )
                                result = transcriber.process_wav_chunk(wav_chunk)
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
        logger.info("Stopping capture...")
        stop_event.set()

        # Detener ScreenCaptureKit
        if screen_capture:
            try:
                screen_capture.stop()
                logger.info("ScreenCaptureKit stopped")
            except Exception as e:
                logger.error(f"Error stopping ScreenCaptureKit: {e}")

        # Detener ffmpeg
        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning("ffmpeg did not stop, killing...")
                process.kill()

        # Exportar en formatos adicionales si se especificó
        export_formats_str = get_str_env(
            "STREAMING_EXPORT_FORMATS", "", allow_empty=True
        )
        if export_formats_str:
            export_formats = [
                f.strip() for f in export_formats_str.split(",") if f.strip()
            ]
            if export_formats:
                logger.info(
                    f"Exporting transcript in formats: {', '.join(export_formats)}"
                )
                try:
                    transcriber.export_transcript(export_formats, output_dir)
                except Exception as e:
                    logger.error(f"Error exporting transcript: {e}", exc_info=True)

        logger.info("Shutdown complete")
        logger.info(f"Full transcript saved to: {output_file}")


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
