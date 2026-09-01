"""
Captura de audio del sistema usando Core Audio Taps (CLI Swift).

Usa la API Core Audio Taps (macOS 14.2+) para capturar el audio del sistema
sin necesidad del permiso de Screen Recording; solo requiere Audio Capture.

Requiere:
- macOS 14.2+ (Core Audio Taps)
- Permisos de Audio Capture (Screen & System Audio Recording)
- CLI Swift compilado: swift-audio-capture/.build/release/audio-capture
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
import shutil
from pathlib import Path
from typing import Optional, Callable

logger = logging.getLogger(__name__)


def _find_swift_cli() -> Optional[Path]:
    """Busca el ejecutable del CLI Swift."""
    # Check .app bundle first
    if getattr(sys, "frozen", False):
        bundle_path = Path(sys.executable).parent.parent / "Resources" / "audio-capture"
        if bundle_path.exists():
            return bundle_path

    # When launched from .app, launcher sets ESCRIBA_PROJECT_ROOT to install dir
    env_root = os.environ.get("ESCRIBA_PROJECT_ROOT", "").strip()
    if env_root:
        project_root = Path(env_root)
        if (project_root / "swift-audio-capture").exists():
            swift_capture_dir = project_root / "swift-audio-capture"
            release_path = swift_capture_dir / ".build" / "release" / "audio-capture"
            if release_path.exists():
                return release_path
            debug_path = swift_capture_dir / ".build" / "debug" / "audio-capture"
            if debug_path.exists():
                return debug_path

    # Buscar en el directorio del proyecto (from __file__)
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
    Captura audio del sistema usando el CLI Swift (Core Audio Taps).
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        audio_callback: Optional[Callable[[bytes], None]] = None,
        use_screen_capture: bool = False,
        on_tap_dead: Optional[Callable[[], None]] = None,
    ):
        if not SWIFT_CLI_AVAILABLE:
            raise ImportError(
                "Swift audio-capture CLI not available. "
                "Build it with: cd swift-audio-capture && swift build -c release"
            )

        self.sample_rate = sample_rate
        self.channels = channels
        self.audio_callback = audio_callback
        self.use_screen_capture = use_screen_capture
        # M6: on_tap_dead is IMMUTABLE after construction. stop() sets
        # _callbacks_disabled to gate delivery without destroying the callback,
        # so restart() (stop→start) does not permanently silence dead-tap propagation.
        self.on_tap_dead = on_tap_dead
        self._callbacks_disabled = False
        self.process: Optional[subprocess.Popen] = None
        self.read_thread: Optional[threading.Thread] = None
        self._stderr_thread: Optional[threading.Thread] = None
        self.is_capturing = False
        self._lock = threading.Lock()
        self.stop_event = threading.Event()
        self.swift_cli_path = _find_swift_cli()

    def start(self) -> bool:
        """Inicia la captura de audio del sistema."""
        if not self.swift_cli_path:
            logger.error("Swift CLI not found")
            return False

        # Hold the lock across the entire check → spawn → flag → thread-start
        # sequence so a concurrent start()/restart() cannot both pass the guard
        # and double-spawn the Swift child (the second Popen would leak the
        # first, unterminated). _read_audio_stream blocks on _is_capturing()
        # until we release the lock, so there is no deadlock.
        with self._lock:
            if self.is_capturing:
                logger.warning("Capture already started")
                return False

            # M6: re-enable callbacks for this new capture session.
            self._callbacks_disabled = False
            self.stop_event.clear()

            try:
                # --use-screen-capture = ScreenCaptureKit, captura sistema fiable
                cmd = [
                    str(self.swift_cli_path),
                    "--sample-rate",
                    str(self.sample_rate),
                    "--channels",
                    str(self.channels),
                ]
                if self.use_screen_capture:
                    cmd.append("--use-screen-capture")
                self.process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    bufsize=0,  # Sin buffering
                )

                self.is_capturing = True

                self.read_thread = threading.Thread(
                    target=self._read_audio_stream, daemon=True
                )
                self.read_thread.start()

                self._stderr_thread = threading.Thread(
                    target=self._drain_stderr, daemon=True
                )
                self._stderr_thread.start()

                logger.info("✓ Started system audio capture with Swift CLI")
                return True

            except Exception as e:
                logger.error("Error starting screen capture: %s", e, exc_info=True)
                self.is_capturing = False
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

            while not self.stop_event.is_set() and self._is_capturing():
                if self.process.poll() is not None:
                    exit_code = self.process.returncode
                    logger.warning(
                        "Swift CLI process ended unexpectedly (exit code: %s)", exit_code
                    )
                    break

                chunk = self.process.stdout.read(chunk_size)
                if not chunk:
                    consecutive_empty_reads += 1
                    if consecutive_empty_reads >= max_empty_reads:
                        logger.debug(
                            "No audio data for %s reads, Swift CLI may be starting up", consecutive_empty_reads
                        )
                        consecutive_empty_reads = 0  # Reset to avoid log spam
                    if self.stop_event.is_set():
                        break
                    # Esperar un poco si no hay datos
                    self.stop_event.wait(0.01)
                    continue

                consecutive_empty_reads = 0  # Reset on successful read

                # El CLI Swift ya entrega PCM int16, solo pasarlo al callback
                if self.audio_callback:
                    self.audio_callback(chunk)

        except Exception as e:
            logger.error("Error reading audio stream: %s", e, exc_info=True)
        finally:
            logger.info("Audio reading thread stopped")
            with self._lock:
                self.is_capturing = False

    def _drain_stderr(self):
        """Drain Swift CLI stderr continuously into the Python logger.

        Running as a daemon thread so every [tap] log line (including rebuild
        events) reaches app.log in real time, not only on process failure.
        Does not block capture and cannot deadlock: it reads from a pipe whose
        write-end is held only by the Swift child; when the child exits the
        pipe EOF arrives and this thread exits naturally.
        """
        proc = self.process
        if not proc or not proc.stderr:
            return
        try:
            for raw in proc.stderr:
                line = raw.decode("utf-8", errors="replace").rstrip().replace("\r", "")
                if not line:
                    continue
                logger.info("[swift] %s", line)
                # M6: read into a local so the check and call use the same value;
                # also honour _callbacks_disabled set by stop() so a late line
                # arriving after stop cannot fire on a dead session.
                cb = self.on_tap_dead
                if line.startswith("[tap] dead") and cb and not self._callbacks_disabled:
                    cb()
        except Exception as e:
            logger.debug("stderr drain ended: %s", e)

    def restart(self) -> bool:
        """
        Reinicia la captura de audio del sistema.

        Útil para recuperarse de fallos del proceso Swift.

        Returns:
            True si se reinició exitosamente, False en caso contrario
        """
        logger.info("Attempting to restart Swift CLI...")

        # Always run cleanup before restarting, regardless of the is_capturing
        # flag: the Swift process can die on its own, leaving is_capturing=False
        # while the child is still alive / the read thread has not unwound.
        # stop() is idempotent, so an already-stopped capture is a no-op.
        self.stop()
        self.stop_event.wait(1.0)

        return self.start()

    def stop(self):
        """Detiene la captura de audio (idempotente)."""
        # M6+M8: snapshot both thread handles and set _callbacks_disabled under
        # the lock so concurrent stop() callers cannot race on the check-then-join,
        # and so _drain_stderr sees callbacks_disabled atomically with is_capturing.
        with self._lock:
            self.is_capturing = False
            self._callbacks_disabled = True
            read_thread = self.read_thread
            stderr_thread = self._stderr_thread
            self.read_thread = None
            self._stderr_thread = None

        self.stop_event.set()

        # Detener proceso Swift — siempre, aunque is_capturing ya fuera False:
        # un proceso que murió por su cuenta deja el flag en False pero el hijo
        # puede seguir vivo (o el pipe sin cerrar).
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
                logger.debug("Error stopping Swift CLI: %s", e)
            self.process = None

        # M8: join using local snapshots — no TOCTOU race on self.read_thread /
        # self._stderr_thread, which concurrent callers may have already nulled.
        if read_thread and read_thread != threading.current_thread():
            read_thread.join(timeout=2.0)

        if stderr_thread and stderr_thread != threading.current_thread():
            stderr_thread.join(timeout=1.0)

        logger.info("Stopped system audio capture")

    def _is_capturing(self) -> bool:
        """Devuelve estado de captura con lock."""
        with self._lock:
            return self.is_capturing

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


def check_screen_recording_permission() -> bool:
    """Verifica si hay permisos de Audio Capture usando el CLI Swift."""
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
    """Muestra mensaje para solicitar permisos de Audio Capture."""
    logger.info(
        "Audio Capture permission required.\n"
        "Please grant permission in:\n"
        "  System Settings > Privacy & Security > Screen & System Audio Recording\n"
        "  Add your terminal app (Terminal, iTerm, etc.)\n\n"
        "You can test permissions by running:\n"
        f"  {_find_swift_cli()} --list"
    )
