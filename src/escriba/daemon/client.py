"""Client for communicating with Escriba daemon."""

from __future__ import annotations

import json
import logging
import socket
from pathlib import Path
from typing import Any, Optional

from escriba.daemon.server import DAEMON_SOCKET_PATH

logger = logging.getLogger(__name__)


class DaemonClient:
    """Cliente para comunicarse con el daemon."""

    def __init__(self, socket_path: Path | None = None):
        """
        Inicializa el cliente.

        Args:
            socket_path: Ruta al socket Unix (default: DAEMON_SOCKET_PATH)
        """
        self.socket_path = socket_path or DAEMON_SOCKET_PATH

    def _send_command(self, command: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        Envía un comando al daemon.

        Args:
            command: Nombre del comando
            args: Argumentos del comando

        Returns:
            Respuesta del daemon

        Raises:
            ConnectionError: Si no se puede conectar al daemon
        """
        if not self.socket_path.exists():
            raise ConnectionError(f"Daemon socket not found: {self.socket_path}")

        try:
            client_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            client_socket.connect(str(self.socket_path))

            # Enviar comando
            command_data = {"command": command, "args": args or {}}
            client_socket.sendall(json.dumps(command_data).encode("utf-8"))

            # Recibir respuesta
            response_data = b""
            while True:
                chunk = client_socket.recv(4096)
                if not chunk:
                    break
                response_data += chunk

            client_socket.close()

            # Parsear respuesta
            response = json.loads(response_data.decode("utf-8"))
            return response

        except FileNotFoundError:
            raise ConnectionError(f"Daemon socket not found: {self.socket_path}")
        except ConnectionRefusedError:
            raise ConnectionError("Daemon is not running")
        except Exception as e:
            raise ConnectionError(f"Error communicating with daemon: {e}")

    def status(self) -> dict[str, Any]:
        """Obtiene el estado del daemon."""
        return self._send_command("status")

    def start_recording(
        self, output_dir: str = "transcripts", combined: str | None = None
    ) -> dict[str, Any]:
        """
        Inicia una grabación.

        Args:
            output_dir: Directorio donde guardar transcripciones
            combined: Archivo opcional para transcripción combinada

        Returns:
            Respuesta del daemon
        """
        return self._send_command(
            "start-recording", {"output_dir": output_dir, "combined": combined}
        )

    def stop_recording(self) -> dict[str, Any]:
        """Detiene la grabación actual."""
        return self._send_command("stop-recording")

    def stop_daemon(self) -> dict[str, Any]:
        """Detiene el daemon."""
        return self._send_command("stop")


def is_daemon_running() -> bool:
    """Verifica si el daemon está corriendo."""
    return DAEMON_SOCKET_PATH.exists()
