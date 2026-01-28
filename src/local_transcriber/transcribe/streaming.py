from __future__ import annotations

import io
import logging
import os
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
from faster_whisper import WhisperModel

from local_transcriber.transcribe.config import VADConfig
from local_transcriber.transcribe.metrics import CaptureMetrics
from local_transcriber.utils.env import get_bool_env, get_float_env

logger = logging.getLogger(__name__)


class StreamingTranscriber:
    """
    Transcripción en tiempo real usando faster-whisper.

    Procesa chunks de audio en tiempo real y muestra transcripciones
    mientras ocurre la llamada, similar a Notion AI Meeting Notes.
    """

    def __init__(
        self,
        model_size: str = "base",
        language: str = "es",
        output_file: Optional[Path] = None,
        device: str = "auto",
        compute_type: Optional[str] = None,
        vad_enabled: bool = True,
        realtime_output: bool = True,
        vad_config: Optional[VADConfig] = None,
        metrics: Optional[CaptureMetrics] = None,
    ):
        """
        Inicializa el transcriber en streaming.

        Args:
            model_size: Tamaño del modelo (tiny, base, small, medium, large)
            language: Idioma para transcripción (código ISO 639-1)
            output_file: Archivo donde escribir transcripciones en tiempo real
            device: Dispositivo a usar (auto, cpu, cuda). "auto" detecta automáticamente.
                Nota: MPS no está soportado por faster-whisper (solo cpu y cuda).
            compute_type: Tipo de computación (int8, int8_float16, float16, float32).
                Si es None, se determina automáticamente según el device.
            vad_enabled: Habilitar Voice Activity Detection
            realtime_output: Mostrar transcripciones en consola en tiempo real
            vad_config: Configuración VAD. Si es None, se carga desde variables de entorno.
            metrics: Instancia de CaptureMetrics para tracking de métricas (opcional)
        """
        self.model_size = model_size
        self.language = language
        self.output_file = output_file
        self.vad_enabled = vad_enabled
        self.realtime_output = realtime_output
        self.metrics = metrics

        # Configurar VAD
        self.vad_config = vad_config or VADConfig.from_env()

        # Speaker detection (opcional)
        self.speaker_detection_enabled = get_bool_env("STREAMING_SPEAKER_DETECTION", False)
        self.speaker_detector = None
        if self.speaker_detection_enabled:
            try:
                from local_transcriber.speaker.detection import SpeakerDetector
                self.speaker_detector = SpeakerDetector(threshold=get_float_env("SPEAKER_DETECTION_THRESHOLD", 0.3))
            except ImportError:
                logger.warning("Speaker detection requested but module not available")
                self.speaker_detection_enabled = False

        # Detectar device y compute_type si es necesario
        if device == "auto":
            device, compute_type = get_device_config()
        elif compute_type is None:
            # Si device no es auto pero compute_type no está especificado,
            # usar valores por defecto según el device
            if device == "cuda":
                compute_type = "float16"  # CUDA puede usar float16
            elif device == "mps":
                # MPS no está soportado por faster-whisper (ctranslate2)
                logger.warning(
                    "MPS is not supported by faster-whisper (ctranslate2). "
                    "Falling back to CPU."
                )
                device = "cpu"
                compute_type = "int8"
            else:
                compute_type = "int8"

        # Validar que el device sea soportado por faster-whisper
        if device == "mps":
            logger.error(
                "MPS (Apple Silicon GPU) is not supported by faster-whisper. "
                "faster-whisper only supports 'cpu' and 'cuda'. "
                "Falling back to CPU."
            )
            device = "cpu"
            compute_type = "int8"

        self.device = device
        self.compute_type = compute_type

        # Buffer para mantener contexto entre chunks
        self.transcription_buffer: list[str] = []
        # Segmentos con timestamps para exportación estructurada
        # Nota: En transcripciones muy largas (>10k segmentos), considerar flush periódico
        self.segments: list[dict[str, Any]] = []
        self.start_time = time.time()
        # Tiempo acumulado de audio procesado (para calcular timestamps correctos)
        self.accumulated_audio_time = 0.0

        # Lock para thread-safety
        self.lock = threading.Lock()

        # Cargar modelo
        logger.info(
            f"Loading Whisper model: {model_size} (device={device}, compute_type={compute_type})"
        )
        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
        )
        logger.info("Model loaded successfully")

    def process_chunk(
        self, audio_data: bytes, sample_rate: int = 16000
    ) -> Optional[str]:
        """
        Procesa un chunk de audio y retorna la transcripción.

        Args:
            audio_data: Audio en formato WAV (bytes)
            sample_rate: Sample rate del audio (default: 16000)

        Returns:
            Texto transcrito o None si no hay voz detectada
        """
        try:
            # Convertir bytes a numpy array
            # WAV header tiene 44 bytes, luego viene el audio PCM
            if len(audio_data) < 44:
                return None

            # Leer audio PCM (skip WAV header)
            audio_array = np.frombuffer(audio_data[44:], dtype=np.int16)

            # Convertir a float32 normalizado
            audio_float = audio_array.astype(np.float32) / 32768.0

            # Transcribir con faster-whisper
            segments, info = self.model.transcribe(
                audio_float,
                language=self.language if self.language != "auto" else None,
                beam_size=5,
                vad_filter=self.vad_enabled,
                vad_parameters=dict(
                    min_silence_duration_ms=self.vad_config.min_silence_duration_ms,
                    threshold=self.vad_config.threshold,
                )
                if self.vad_enabled
                else None,
            )

            # Detectar speaker si está habilitado
            speaker_tag = None
            if self.speaker_detection_enabled and self.speaker_detector:
                speaker_tag = self.speaker_detector.detect_change(audio_data)

            # Procesar segmentos
            texts = []
            for segment in segments:
                text = segment.text.strip()
                if text:
                    texts.append(text)
                    # Sumar tiempo acumulado a los tiempos relativos del segmento
                    self._handle_transcription(
                        text,
                        self.accumulated_audio_time + segment.start,
                        self.accumulated_audio_time + segment.end,
                        speaker=speaker_tag,
                    )

            # Actualizar tiempo acumulado con la duración del chunk procesado
            # Calcular duración del chunk desde el audio
            if len(audio_float) > 0:
                chunk_duration = len(audio_float) / sample_rate
                self.accumulated_audio_time += chunk_duration

            return " ".join(texts) if texts else None

        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}", exc_info=True)
            return None

    def process_wav_chunk(self, wav_data: bytes) -> Optional[str]:
        """
        Procesa un chunk de audio en formato WAV.

        Args:
            wav_data: Audio en formato WAV completo (con header)

        Returns:
            Texto transcrito o None si no hay voz detectada
        """
        # Registrar inicio de procesamiento para métricas
        start_timestamp = None
        if self.metrics:
            start_timestamp = self.metrics.record_chunk_start()
            self.metrics.record_audio_level(wav_data)

        try:
            # Intentar leer WAV usando wave module
            import wave
            import struct

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

                    # Transcribir
                    segments, info = self.model.transcribe(
                        audio_float,
                        language=self.language if self.language != "auto" else None,
                        beam_size=5,
                        vad_filter=self.vad_enabled,
                        vad_parameters=dict(
                            min_silence_duration_ms=self.vad_config.min_silence_duration_ms,
                            threshold=self.vad_config.threshold,
                        )
                        if self.vad_enabled
                        else None,
                    )

                    # Detectar speaker si está habilitado
                    speaker_tag = None
                    if self.speaker_detection_enabled and self.speaker_detector:
                        speaker_tag = self.speaker_detector.detect_change(wav_data)

                    # Procesar segmentos
                    texts = []
                    segment_count = 0
                    for segment in segments:
                        segment_count += 1
                        text = segment.text.strip()
                        if text:
                            texts.append(text)
                            logger.info(f"Found transcription segment: {text[:50]}")
                            # Sumar tiempo acumulado a los tiempos relativos del segmento
                            self._handle_transcription(
                                text,
                                self.accumulated_audio_time + segment.start,
                                self.accumulated_audio_time + segment.end,
                                speaker=speaker_tag,
                            )

                    # Actualizar tiempo acumulado con la duración del chunk procesado
                    if len(audio_float) > 0:
                        chunk_duration = len(audio_float) / sample_rate
                        self.accumulated_audio_time += chunk_duration
                        if self.metrics:
                            self.metrics.record_audio_duration(chunk_duration)

                    if segment_count == 0:
                        logger.debug(
                            "No segments found in audio chunk (may be silence or VAD filtered)"
                        )
                    elif not texts:
                        logger.debug(
                            f"Found {segment_count} segments but all were empty"
                        )
                    else:
                        logger.debug(
                            f"Processed {segment_count} segments, {len(texts)} with text"
                        )

                    result = " ".join(texts) if texts else None
                    # Registrar métricas
                    if self.metrics and start_timestamp:
                        self.metrics.record_chunk_end(start_timestamp, had_transcription=(result is not None))
                    return result

            except wave.Error as e:
                # Si wave.open falla, intentar parsear manualmente
                logger.debug(f"wave.open failed, parsing manually: {e}")
                result = self._process_wav_manual(wav_data)
                if self.metrics and start_timestamp:
                    self.metrics.record_chunk_end(start_timestamp, had_transcription=(result is not None))
                return result

        except Exception as e:
            logger.error(f"Error processing WAV chunk: {e}", exc_info=True)
            if self.metrics:
                self.metrics.record_error()
                if start_timestamp:
                    self.metrics.record_chunk_end(start_timestamp, had_transcription=False)
            return None

    def _process_wav_manual(self, wav_data: bytes) -> Optional[str]:
        """Procesa WAV parseando manualmente el header."""
        try:
            import struct

            if len(wav_data) < 44:
                return None

            # Parsear header
            # bytes 22-23: número de canales
            # bytes 24-27: sample rate
            # bytes 34-35: bits per sample
            # bytes 40-43: data chunk size
            n_channels = struct.unpack("<H", wav_data[22:24])[0]
            sample_rate = struct.unpack("<I", wav_data[24:28])[0]
            bits_per_sample = struct.unpack("<H", wav_data[34:36])[0]
            data_size = struct.unpack("<I", wav_data[40:44])[0]

            # Extraer datos PCM (después del header de 44 bytes)
            if len(wav_data) < 44 + data_size:
                return None

            pcm_data = wav_data[44 : 44 + data_size]
            bytes_per_sample = bits_per_sample // 8

            # Convertir a numpy array
            if bytes_per_sample == 2:  # 16-bit
                audio_array = np.frombuffer(pcm_data, dtype=np.int16)
            elif bytes_per_sample == 4:  # 32-bit
                audio_array = np.frombuffer(pcm_data, dtype=np.int32)
            else:
                logger.warning(f"Unsupported bytes per sample: {bytes_per_sample}")
                return None

            # Convertir a mono si es estéreo
            if n_channels == 2:
                audio_array = audio_array.reshape(-1, 2).mean(axis=1).astype(np.int16)

            # Convertir a float32 normalizado
            audio_float = audio_array.astype(np.float32) / 32768.0

            # Transcribir
            logger.debug(
                f"Transcribing audio chunk (manual): {len(audio_float)} samples, VAD={self.vad_enabled}"
            )
            segments, info = self.model.transcribe(
                audio_float,
                language=self.language if self.language != "auto" else None,
                beam_size=5,
                vad_filter=self.vad_enabled,
                vad_parameters=dict(
                    min_silence_duration_ms=self.vad_config.min_silence_duration_ms,
                    threshold=self.vad_config.threshold,
                )
                if self.vad_enabled
                else None,
            )

            # Procesar segmentos
            texts = []
            segment_count = 0
            for segment in segments:
                segment_count += 1
                text = segment.text.strip()
                if text:
                    texts.append(text)
                    logger.info(f"Found transcription segment (manual): {text[:50]}")
                    self._handle_transcription(
                        text,
                        segment.start,
                        segment.end,
                    )

            if segment_count == 0:
                logger.debug(
                    "No segments found in audio chunk (manual) (may be silence or VAD filtered)"
                )
            elif not texts:
                logger.debug(
                    f"Found {segment_count} segments (manual) but all were empty"
                )
            else:
                logger.debug(
                    f"Processed {segment_count} segments (manual), {len(texts)} with text"
                )

            return " ".join(texts) if texts else None

        except Exception as e:
            logger.error(f"Error in manual WAV parsing: {e}", exc_info=True)
            return None

    def _handle_transcription(
        self, text: str, start_time: float, end_time: float, speaker: str | None = None
    ) -> None:
        """
        Maneja una transcripción parcial.

        Args:
            text: Texto transcrito
            start_time: Tiempo de inicio relativo (segundos)
            end_time: Tiempo de fin relativo (segundos)
            speaker: Etiqueta del speaker (opcional)
        """
        with self.lock:
            # Agregar a buffer
            if speaker:
                self.transcription_buffer.append(f"[{speaker}] {text}")
            else:
                self.transcription_buffer.append(text)

            # Agregar a segmentos con timestamps
            segment = {
                "start": start_time,
                "end": end_time,
                "text": text,
            }
            if speaker:
                segment["speaker"] = speaker
            self.segments.append(segment)

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
            "compute_type": self.compute_type,
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


def get_device_config() -> tuple[str, str]:
    """
    Detecta y retorna la configuración óptima de device y compute_type.

    Nota: faster-whisper (que usa ctranslate2) solo soporta 'cpu' y 'cuda'.
    MPS (Apple Silicon GPU) no está soportado actualmente por ctranslate2.

    Para usar GPU en Apple Silicon, usa 'openai-whisper' con el backend 'mps'.

    Returns:
        Tupla (device, compute_type)
    """
    # faster-whisper solo soporta 'cpu' y 'cuda', no 'mps'
    # Aunque MPS esté disponible en PyTorch, ctranslate2 no lo soporta
    logger.info("Using CPU (faster-whisper only supports cpu and cuda, not mps)")
    logger.info("To use GPU on Apple Silicon, use --backend openai-whisper")
    return "cpu", "int8"
