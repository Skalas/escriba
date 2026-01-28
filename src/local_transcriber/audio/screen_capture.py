"""
Captura de audio del sistema usando ScreenCaptureKit (CLI Swift).

Similar a Notion AI, usa ScreenCaptureKit para capturar el audio del sistema
sin necesidad de dispositivos virtuales como BlackHole.

Requiere:
- macOS 13.0+ (ScreenCaptureKit)
- Permisos de Screen Recording
- CLI Swift compilado: swift-audio-capture/.build/release/audio-capture
"""

from __future__ import annotations

import logging
import subprocess
import threading
import shutil
from pathlib import Path
from typing import Optional, Callable

logger = logging.getLogger(__name__)


def _find_swift_cli() -> Optional[Path]:
    """Busca el ejecutable del CLI Swift."""
    # Buscar en el directorio del proyecto
    project_root = Path(__file__).parent.parent.parent.parent
    swift_capture_dir = project_root / "swift-audio-capture"

    # Intentar release primero
    release_path = swift_capture_dir / ".build" / "release" / "audio-capture"
    if release_path.exists():
        return release_path

    # Intentar debug
    debug_path = swift_capture_dir / ".build" / "debug" / "audio-capture"
    if debug_path.exists():
        return debug_path

    # Intentar en PATH
    which_path = shutil.which("audio-capture")
    if which_path:
        return Path(which_path)

    return None


SWIFT_CLI_AVAILABLE = _find_swift_cli() is not None

if not SWIFT_CLI_AVAILABLE:
    logger.warning(
        "Swift audio-capture CLI not found.\n"
        "Build it with: cd swift-audio-capture && swift build -c release"
    )


class ScreenCaptureAudioCapture:
    """
    Captura audio del sistema usando el CLI Swift de ScreenCaptureKit.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        audio_callback: Optional[Callable[[bytes], None]] = None,
    ):
        if not SWIFT_CLI_AVAILABLE:
            raise ImportError(
                "Swift audio-capture CLI not available. "
                "Build it with: cd swift-audio-capture && swift build -c release"
            )

        self.sample_rate = sample_rate
        self.channels = channels
        self.audio_callback = audio_callback
        self.process: Optional[subprocess.Popen] = None
        self.read_thread: Optional[threading.Thread] = None
        self.is_capturing = False
        self.stop_event = threading.Event()
        self.swift_cli_path = _find_swift_cli()

    def start(self) -> bool:
        """Inicia la captura de audio del sistema."""
        if self.is_capturing:
            logger.warning("Capture already started")
            return False

        if not self.swift_cli_path:
            logger.error("Swift CLI not found")
            return False

        try:
            # Iniciar proceso Swift
            self.process = subprocess.Popen(
                [
                    str(self.swift_cli_path),
                    "--sample-rate",
                    str(self.sample_rate),
                    "--channels",
                    str(self.channels),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,  # Sin buffering
            )

            # Iniciar thread para leer PCM desde stdout
            self.read_thread = threading.Thread(
                target=self._read_audio_stream, daemon=True
            )
            self.read_thread.start()

            self.is_capturing = True
            logger.info("✓ Started system audio capture with Swift CLI")

            return True

        except Exception as e:
            logger.error(f"Error starting screen capture: {e}", exc_info=True)
            return False

    def _read_audio_stream(self):
        """Lee datos PCM desde stdout del proceso Swift."""
        if not self.process or not self.process.stdout:
            logger.error("Swift process or stdout not available")
            return

        logger.info("🎧 Audio reading thread started")

        # Track consecutive empty reads to detect stalled process
        consecutive_empty_reads = 0
        max_empty_reads = 100  # ~1 second of empty reads before warning

        try:
            # Leer chunks de PCM (int16, little-endian)
            # Leemos en bloques de ~1 segundo de audio
            chunk_size = (
                self.sample_rate * self.channels * 2
            )  # 2 bytes por sample (int16)

            while not self.stop_event.is_set() and self.is_capturing:
                if self.process.poll() is not None:
                    # Proceso terminó - obtener código de salida
                    exit_code = self.process.returncode
                    logger.warning(
                        f"Swift CLI process ended unexpectedly (exit code: {exit_code})"
                    )
                    # Leer stderr para diagnóstico
                    if self.process.stderr:
                        try:
                            stderr_output = self.process.stderr.read()
                            if stderr_output:
                                logger.warning(
                                    f"Swift CLI stderr: {stderr_output.decode('utf-8', errors='ignore')}"
                                )
                        except Exception:
                            pass
                    break

                chunk = self.process.stdout.read(chunk_size)
                if not chunk:
                    consecutive_empty_reads += 1
                    if consecutive_empty_reads >= max_empty_reads:
                        logger.debug(
                            f"No audio data for {consecutive_empty_reads} reads, Swift CLI may be starting up"
                        )
                        consecutive_empty_reads = 0  # Reset to avoid log spam
                    if self.stop_event.is_set():
                        break
                    # Esperar un poco si no hay datos
                    threading.Event().wait(0.01)
                    continue

                consecutive_empty_reads = 0  # Reset on successful read

                # El CLI Swift ya entrega PCM int16, solo pasarlo al callback
                if self.audio_callback:
                    self.audio_callback(chunk)

        except Exception as e:
            logger.error(f"Error reading audio stream: {e}", exc_info=True)
        finally:
            logger.info("Audio reading thread stopped")
            # Marcar que ya no está capturando - use lock for thread safety
            self.is_capturing = False

    def restart(self) -> bool:
        """
        Reinicia la captura de audio del sistema.

        Útil para recuperarse de fallos del proceso Swift.

        Returns:
            True si se reinició exitosamente, False en caso contrario
        """
        logger.info("Attempting to restart Swift CLI...")

        # Detener captura actual si está activa
        if self.is_capturing:
            self.stop()
            # Esperar un poco antes de reiniciar
            threading.Event().wait(1.0)

        # Reiniciar
        return self.start()

    def stop(self):
        """Detiene la captura de audio."""
        if not self.is_capturing:
            return

        self.is_capturing = False
        self.stop_event.set()

        # Detener proceso Swift
        if self.process:
            try:
                self.process.terminate()
                try:
                    self.process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    logger.warning("Swift CLI did not stop, killing...")
                    self.process.kill()
                    self.process.wait()
            except Exception as e:
                logger.debug(f"Error stopping Swift CLI: {e}")

        # Esperar thread de lectura
        if self.read_thread:
            self.read_thread.join(timeout=2.0)

        logger.info("Stopped system audio capture")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


def check_screen_recording_permission() -> bool:
    """Verifica si hay permisos de Screen Recording usando el CLI Swift."""
    if not SWIFT_CLI_AVAILABLE:
        return False

    swift_cli_path = _find_swift_cli()
    if not swift_cli_path:
        return False

    try:
        # Intentar ejecutar el CLI - si no hay permisos, fallará
        result = subprocess.run(
            [str(swift_cli_path), "--list"],
            capture_output=True,
            timeout=5.0,
        )
        # Si el comando se ejecuta sin error, probablemente hay permisos
        # (aunque podría fallar por otras razones)
        return result.returncode == 0
    except Exception:
        return False


def request_screen_recording_permission():
    """Muestra mensaje para solicitar permisos."""
    logger.info(
        "Screen Recording permission required.\n"
        "Please grant permission in:\n"
        "  System Settings > Privacy & Security > Screen Recording\n"
        "  Add your terminal app (Terminal, iTerm, etc.)\n\n"
        "You can test permissions by running:\n"
        f"  {_find_swift_cli()} --list"
    )
