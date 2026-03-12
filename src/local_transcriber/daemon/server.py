"""Daemon server for local-transcriber."""

from __future__ import annotations

import json
import logging
import os
import signal
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional

from local_transcriber.audio.live_capture import run_streaming_capture

logger = logging.getLogger(__name__)

# Socket path
DAEMON_SOCKET_PATH = Path.home() / ".local-transcriber" / "daemon.sock"
DAEMON_PID_FILE = Path.home() / ".local-transcriber" / "daemon.pid"


class DaemonServer:
    """Server daemon que mantiene el modelo Whisper cargado en memoria."""

    def __init__(self, model_size: str = "base"):
        """
        Inicializa el servidor daemon.

        Args:
            model_size: Tamaño del modelo a cargar
        """
        self.model_size = model_size
        self.socket_path = DAEMON_SOCKET_PATH
        self.server_socket: Optional[socket.socket] = None
        self.running = False
        self.current_recording: Optional[threading.Thread] = None
        self.stop_recording_event = threading.Event()

        # Crear directorio para socket si no existe
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)

    def start(self) -> bool:
        """Inicia el servidor daemon."""
        if self.running:
            logger.warning("Daemon already running")
            return False

        # Crear socket Unix
        try:
            # Eliminar socket anterior si existe
            if self.socket_path.exists():
                self.socket_path.unlink()

            self.server_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self.server_socket.bind(str(self.socket_path))
            self.server_socket.listen(5)
            logger.info("Daemon socket listening on: %s", self.socket_path)

            # Guardar PID
            with DAEMON_PID_FILE.open("w") as f:
                f.write(str(os.getpid()))

            self.running = True

            # Registrar handlers de señales
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)

            # Iniciar loop de aceptación de conexiones
            self._accept_connections()

            return True
        except Exception as e:
            logger.error("Failed to start daemon: %s", e, exc_info=True)
            return False

    def _signal_handler(self, signum, frame):
        """Maneja señales para shutdown graceful."""
        logger.info("Received signal %s, shutting down...", signum)
        self.stop()

    def _accept_connections(self):
        """Acepta conexiones y procesa comandos."""
        while self.running:
            try:
                if not self.server_socket:
                    break

                # Timeout para poder verificar self.running periódicamente
                self.server_socket.settimeout(1.0)
                try:
                    client_socket, _ = self.server_socket.accept()
                except socket.timeout:
                    continue

                # Procesar comando en thread separado
                client_thread = threading.Thread(
                    target=self._handle_client, args=(client_socket,), daemon=True
                )
                client_thread.start()

            except Exception as e:
                if self.running:
                    logger.error("Error accepting connection: %s", e, exc_info=True)
                break

    def _handle_client(self, client_socket: socket.socket):
        """Maneja un cliente conectado."""
        try:
            # Leer comando
            data = client_socket.recv(4096)
            if not data:
                return

            command_data = json.loads(data.decode("utf-8"))
            command = command_data.get("command")
            args = command_data.get("args", {})

            # Procesar comando
            response = self._process_command(command, args)

            # Enviar respuesta
            response_json = json.dumps(response).encode("utf-8")
            client_socket.sendall(response_json)

        except Exception as e:
            logger.error("Error handling client: %s", e, exc_info=True)
            error_response = {"success": False, "error": str(e)}
            try:
                client_socket.sendall(json.dumps(error_response).encode("utf-8"))
            except Exception:
                pass
        finally:
            client_socket.close()

    def _process_command(self, command: str, args: dict[str, Any]) -> dict[str, Any]:
        """
        Procesa un comando.

        Args:
            command: Nombre del comando
            args: Argumentos del comando

        Returns:
            Respuesta del comando
        """
        if command == "status":
            return self._cmd_status()
        elif command == "start-recording":
            return self._cmd_start_recording(args)
        elif command == "stop-recording":
            return self._cmd_stop_recording()
        elif command == "stop":
            return self._cmd_stop()
        else:
            return {"success": False, "error": f"Unknown command: {command}"}

    def _cmd_status(self) -> dict[str, Any]:
        """Comando: status"""
        return {
            "success": True,
            "status": {
                "running": self.running,
                "model_loaded": False,
                "model_size": self.model_size,
                "recording": self.current_recording is not None
                and self.current_recording.is_alive(),
            },
        }

    def _cmd_start_recording(self, args: dict[str, Any]) -> dict[str, Any]:
        """Comando: start-recording"""
        if self.current_recording and self.current_recording.is_alive():
            return {"success": False, "error": "Recording already in progress"}

        output_dir = Path(args.get("output_dir", "transcripts"))
        combined = args.get("combined")

        self.stop_recording_event.clear()

        def recording_thread():
            try:
                combined_path = Path(combined) if combined else None
                run_streaming_capture(output_dir, combined_path)
            except Exception as e:
                logger.error("Error in recording thread: %s", e, exc_info=True)

        self.current_recording = threading.Thread(target=recording_thread, daemon=True)
        self.current_recording.start()

        return {"success": True, "message": "Recording started"}

    def _cmd_stop_recording(self) -> dict[str, Any]:
        """Comando: stop-recording"""
        if not self.current_recording or not self.current_recording.is_alive():
            return {"success": False, "error": "No recording in progress"}

        self.stop_recording_event.set()
        # Esperar a que termine (con timeout)
        self.current_recording.join(timeout=5.0)

        return {"success": True, "message": "Recording stopped"}

    def _cmd_stop(self) -> dict[str, Any]:
        """Comando: stop (detener daemon)"""
        self.stop()
        return {"success": True, "message": "Daemon stopped"}

    def stop(self):
        """Detiene el servidor daemon."""
        logger.info("Stopping daemon...")
        self.running = False

        # Detener grabación si está activa
        if self.current_recording and self.current_recording.is_alive():
            self.stop_recording_event.set()
            self.current_recording.join(timeout=5.0)

        # Cerrar socket
        if self.server_socket:
            try:
                self.server_socket.close()
            except Exception:
                pass

        # Eliminar socket file
        if self.socket_path.exists():
            try:
                self.socket_path.unlink()
            except Exception:
                pass

        # Eliminar PID file
        if DAEMON_PID_FILE.exists():
            try:
                DAEMON_PID_FILE.unlink()
            except Exception:
                pass

        logger.info("Daemon stopped")


def run_daemon(model_size: str = "base") -> None:
    """
    Ejecuta el servidor daemon.

    Args:
        model_size: Tamaño del modelo a cargar
    """
    server = DaemonServer(model_size=model_size)
    try:
        if not server.start():
            sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Interrupted, shutting down...")
        server.stop()
        sys.exit(0)
