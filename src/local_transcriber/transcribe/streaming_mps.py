from __future__ import annotations

import io
import logging
import os
import ssl
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import whisper

from local_transcriber.transcribe.config import VADConfig

logger = logging.getLogger(__name__)


class StreamingTranscriberMPS:
    """
    Transcripción en tiempo real usando openai-whisper.

    Esta implementación usa openai-whisper. Por defecto usa CPU debido a problemas
    conocidos con MPS que producen NaN. Para forzar MPS, establece WHISPER_FORCE_MPS=true.

    Nota: MPS tiene problemas de estabilidad con openai-whisper. Se recomienda usar
    faster-whisper (backend por defecto) para mejor rendimiento y estabilidad.
    """

    def __init__(
        self,
        model_size: str = "base",
        language: str = "es",
        output_file: Optional[Path] = None,
        vad_enabled: bool = True,
        realtime_output: bool = True,
        vad_config: Optional[VADConfig] = None,
    ):
        """
        Inicializa el transcriber en streaming con MPS.

        Args:
            model_size: Tamaño del modelo (tiny, base, small, medium, large)
            language: Idioma para transcripción (código ISO 639-1)
            output_file: Archivo donde escribir transcripciones en tiempo real
            vad_enabled: Habilitar Voice Activity Detection
            realtime_output: Mostrar transcripciones en consola en tiempo real
            vad_config: Configuración VAD. Si es None, se carga desde variables de entorno.
        """
        self.model_size = model_size
        self.language = language
        self.output_file = output_file
        self.vad_enabled = vad_enabled
        self.realtime_output = realtime_output

        # Configurar VAD
        self.vad_config = vad_config or VADConfig.from_env()

        # Detectar device
        # Nota: openai-whisper tiene problemas conocidos con MPS que producen NaN
        # Usamos CPU por defecto para mayor estabilidad, pero permitimos forzar MPS
        use_mps = os.getenv("WHISPER_FORCE_MPS", "false").lower() in (
            "true",
            "1",
            "yes",
        )

        if use_mps and torch.backends.mps.is_available():
            self.device = "mps"
            self.use_fp16 = False  # fp16 puede causar NaN en MPS
            logger.warning(
                "Using MPS (Apple Silicon GPU) - WARNING: MPS has known issues with openai-whisper "
                "that may cause NaN errors. If you encounter errors, use CPU instead."
            )
        else:
            self.device = "cpu"
            self.use_fp16 = True  # fp16 es seguro en CPU
            if use_mps:
                logger.warning("MPS requested but not available, using CPU")
            else:
                logger.info(
                    "Using CPU for transcription (MPS disabled due to stability issues)"
                )

        # Buffer para mantener contexto entre chunks
        self.transcription_buffer: list[str] = []
        # Segmentos con timestamps para exportación estructurada
        self.segments: list[dict[str, Any]] = []
        self.start_time = time.time()

        # Lock para thread-safety
        self.lock = threading.Lock()

        # Cargar modelo
        logger.info(f"Loading Whisper model: {model_size} (device={self.device})")
        try:
            # Intentar cargar modelo con manejo de SSL
            download_root = os.getenv("WHISPER_CACHE_DIR")
            if download_root:
                download_root = Path(download_root)
                logger.info(f"Using custom cache directory: {download_root}")

            self.model = whisper.load_model(
                model_size, device=self.device, download_root=download_root
            )
            logger.info("Model loaded successfully")
        except Exception as e:
            # Manejar errores de descarga (SSL, red, etc.)
            error_msg = str(e)
            if "SSL" in error_msg or "certificate" in error_msg.lower():
                logger.error(
                    "SSL certificate verification failed while downloading model.\n"
                    "This is often caused by corporate proxies or network security settings.\n\n"
                    "Solutions:\n"
                    "  1. Install Python certificates:\n"
                    "     /Applications/Python\\ 3.13/Install\\ Certificates.command\n"
                    "  2. Set WHISPER_CACHE_DIR to use a pre-downloaded model:\n"
                    "     export WHISPER_CACHE_DIR=~/.cache/whisper\n"
                    "  3. Download model manually from HuggingFace and place in cache\n"
                    "  4. Use faster-whisper backend instead (doesn't require model download):\n"
                    "     local-transcriber live-stream --backend faster-whisper"
                )
            elif "URLError" in error_msg or "urlopen" in error_msg:
                logger.error(
                    "Failed to download model from internet.\n"
                    "This could be due to network issues or firewall restrictions.\n\n"
                    "Solutions:\n"
                    "  1. Check your internet connection\n"
                    "  2. Download model manually from HuggingFace\n"
                    "  3. Set WHISPER_CACHE_DIR environment variable to point to downloaded model\n"
                    "  4. Use faster-whisper backend instead (downloads models automatically):\n"
                    "     local-transcriber live-stream --backend faster-whisper"
                )
            else:
                logger.error(f"Failed to load Whisper model: {e}")
            raise

    def process_wav_chunk(self, wav_data: bytes) -> Optional[str]:
        """
        Procesa un chunk de audio en formato WAV.

        Args:
            wav_data: Audio en formato WAV completo (con header)

        Returns:
            Texto transcrito o None si no hay voz detectada
        """
        try:
            import wave

            wav_io = io.BytesIO(wav_data)

            # Verificar que sea un WAV válido
            if len(wav_data) < 44:
                logger.warning(f"WAV chunk too small: {len(wav_data)} bytes")
                return None

            # Verificar header básico
            if wav_data[:4] != b"RIFF" or wav_data[8:12] != b"WAVE":
                logger.warning("Invalid WAV header")
                return None

            try:
                with wave.open(wav_io, "rb") as wav_file:
                    sample_rate = wav_file.getframerate()
                    n_channels = wav_file.getnchannels()
                    sample_width = wav_file.getsampwidth()
                    frames = wav_file.readframes(wav_file.getnframes())

                    if len(frames) == 0:
                        return None

                    # Convertir a numpy array
                    if sample_width == 2:  # 16-bit
                        audio_array = np.frombuffer(frames, dtype=np.int16)
                    elif sample_width == 4:  # 32-bit
                        audio_array = np.frombuffer(frames, dtype=np.int32)
                    else:
                        logger.warning(f"Unsupported sample width: {sample_width}")
                        return None

                    # Convertir a mono si es estéreo
                    if n_channels == 2:
                        audio_array = (
                            audio_array.reshape(-1, 2).mean(axis=1).astype(np.int16)
                        )

                    # Convertir a float32 normalizado
                    audio_float = audio_array.astype(np.float32) / 32768.0

                    # Transcribir con openai-whisper
                    # Nota: Si MPS produce NaN, haremos fallback a CPU
                    try:
                        result = self.model.transcribe(
                            audio_float,
                            language=self.language if self.language != "auto" else None,
                            fp16=self.use_fp16,
                            verbose=False,
                        )
                    except (ValueError, RuntimeError) as e:
                        error_msg = str(e)
                        # Detectar errores de NaN en MPS
                        if "nan" in error_msg.lower() and self.device == "mps":
                            logger.warning(
                                "NaN error detected with MPS. Falling back to CPU for this chunk. "
                                "Consider using CPU or faster-whisper backend for better stability."
                            )
                            # Fallback a CPU para este chunk
                            if self.device == "mps":
                                # Cargar modelo en CPU si no está ya cargado
                                if not hasattr(self, "_cpu_model"):
                                    logger.info("Loading model on CPU for fallback...")
                                    self._cpu_model = whisper.load_model(
                                        self.model_size, device="cpu"
                                    )
                                result = self._cpu_model.transcribe(
                                    audio_float,
                                    language=self.language
                                    if self.language != "auto"
                                    else None,
                                    fp16=False,
                                    verbose=False,
                                )
                            else:
                                raise
                        else:
                            raise

                    # Procesar segmentos
                    texts = []
                    for segment in result.get("segments", []):
                        text = segment.get("text", "").strip()
                        if text:
                            texts.append(text)
                            start_time = segment.get("start", 0.0)
                            end_time = segment.get("end", 0.0)
                            self._handle_transcription(text, start_time, end_time)

                    return " ".join(texts) if texts else None

            except wave.Error as e:
                logger.debug(f"wave.open failed: {e}")
                return None

        except Exception as e:
            logger.error(f"Error processing WAV chunk: {e}", exc_info=True)
            return None

    def _handle_transcription(
        self, text: str, start_time: float, end_time: float
    ) -> None:
        """
        Maneja una transcripción parcial.

        Args:
            text: Texto transcrito
            start_time: Tiempo de inicio relativo (segundos)
            end_time: Tiempo de fin relativo (segundos)
        """
        with self.lock:
            # Agregar a buffer
            self.transcription_buffer.append(text)

            # Agregar a segmentos con timestamps
            self.segments.append(
                {
                    "start": start_time,
                    "end": end_time,
                    "text": text,
                }
            )

            # Calcular timestamp absoluto
            absolute_start = self.start_time + start_time
            timestamp = time.strftime("%H:%M:%S", time.localtime(absolute_start))

            # Mostrar en tiempo real
            logger.info(f"Transcription: [{timestamp}] {text}")
            if self.realtime_output:
                print(f"[{timestamp}] {text}", flush=True)

            # Escribir a archivo si está configurado
            if self.output_file:
                self._append_to_file(
                    text, absolute_start, absolute_start + (end_time - start_time)
                )

    def _append_to_file(self, text: str, start_time: float, end_time: float) -> None:
        """Escribe transcripción a archivo en tiempo real."""
        try:
            self.output_file.parent.mkdir(parents=True, exist_ok=True)

            # Modo append para escribir mientras transcribe
            timestamp = time.strftime("%H:%M:%S", time.localtime(start_time))
            with self.output_file.open("a", encoding="utf-8") as f:
                f.write(f"[{timestamp}] {text}\n")
                f.flush()  # Asegurar que se escribe inmediatamente
            logger.debug(f"Written to file: {self.output_file}")
        except Exception as e:
            logger.error(f"Error writing to file: {e}", exc_info=True)

    def get_full_transcript(self) -> str:
        """Obtiene la transcripción completa hasta el momento."""
        with self.lock:
            return " ".join(self.transcription_buffer)

    def reset(self) -> None:
        """Reinicia el buffer de transcripciones."""
        with self.lock:
            self.transcription_buffer.clear()
            self.segments.clear()
            self.start_time = time.time()

    def export_transcript(self, formats: list[str], output_dir: Path) -> None:
        """
        Exporta la transcripción completa en los formatos especificados.

        Args:
            formats: Lista de formatos a exportar ('txt', 'json', 'srt', 'markdown')
            output_dir: Directorio donde guardar los archivos exportados
        """
        # Import here to avoid circular dependency
        from local_transcriber.transcribe.formats import (
            export_to_json,
            export_to_txt,
            export_to_srt,
            export_to_markdown,
        )

        with self.lock:
            segments = self.segments.copy()

        if not segments:
            logger.warning("No segments to export")
            return

        # Metadata para JSON
        metadata = {
            "model": self.model_size,
            "language": self.language,
            "device": self.device,
            "backend": "openai-whisper",
        }

        # Generar nombre base del archivo
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"transcript_{timestamp_str}"

        # Exportar en cada formato
        for fmt in formats:
            if fmt == "json":
                output_path = output_dir / f"{base_name}.json"
                export_to_json(segments, output_path, metadata)
            elif fmt == "txt":
                output_path = output_dir / f"{base_name}.txt"
                export_to_txt(segments, output_path)
            elif fmt == "srt":
                output_path = output_dir / f"{base_name}.srt"
                export_to_srt(segments, output_path)
            elif fmt == "markdown":
                output_path = output_dir / f"{base_name}.md"
                export_to_markdown(segments, output_path)
            else:
                logger.warning(f"Unknown format: {fmt}, skipping")
