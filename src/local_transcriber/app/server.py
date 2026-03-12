"""Lightweight HTTP server for the transcriber web UI."""

from __future__ import annotations

import json
import logging
import threading
from functools import partial
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from local_transcriber.app.session import TranscriptionSession
    from local_transcriber.config import AppConfig

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"
PORT = 19876


class _Handler(BaseHTTPRequestHandler):
    """HTTP request handler with API endpoints for the transcriber app."""

    app_state: dict = {}  # Shared state: session, config, etc.

    def log_message(self, format, *args):
        # Suppress default request logging
        pass

    def do_GET(self):
        if self.path == "/":
            self._serve_file("index.html", "text/html")
        elif self.path == "/api/status":
            self._json_response(self._get_status())
        elif self.path == "/api/transcript":
            self._json_response(self._get_transcript())
        elif self.path == "/api/sessions":
            self._json_response(self._get_sessions())
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == "/api/recording/start":
            self._json_response(self._start_recording())
        elif self.path == "/api/recording/stop":
            self._json_response(self._stop_recording())
        elif self.path == "/api/notes":
            body = self._read_body()
            self._json_response(self._generate_notes(body))
        else:
            self.send_error(404)

    # --- Helpers ---

    def _serve_file(self, filename: str, content_type: str):
        filepath = STATIC_DIR / filename
        if not filepath.exists():
            self.send_error(404, f"File not found: {filename}")
            return
        content = filepath.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _json_response(self, data: dict):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return {}

    # --- API handlers ---

    def _get_status(self) -> dict:
        session: TranscriptionSession | None = self.app_state.get("session")
        if session:
            return {"ok": True, **session.get_status()}
        return {"ok": True, "is_active": False, "session_id": None}

    def _get_transcript(self) -> dict:
        session: TranscriptionSession | None = self.app_state.get("session")
        if not session:
            return {"ok": True, "text": "", "segments": []}
        return {
            "ok": True,
            "text": session.get_transcript(),
            "segments": session.get_segments(),
        }

    def _get_sessions(self) -> dict:
        output_dir = Path("transcripts")
        sessions = []
        if output_dir.exists():
            for f in sorted(output_dir.glob("transcript_*.txt"), reverse=True)[:20]:
                sessions.append(
                    {"name": f.stem, "path": str(f), "size": f.stat().st_size}
                )
        return {"ok": True, "sessions": sessions}

    def _start_recording(self) -> dict:
        session: TranscriptionSession | None = self.app_state.get("session")
        if session and session.is_active:
            return {"ok": False, "error": "Already recording"}

        from local_transcriber.app.session import TranscriptionSession
        from local_transcriber.config import AppConfig

        config: AppConfig = self.app_state.get("config", AppConfig.load())
        session = TranscriptionSession(config)
        session.start()
        self.app_state["session"] = session

        if session.error:
            return {"ok": False, "error": session.error}
        return {"ok": True}

    def _stop_recording(self) -> dict:
        session: TranscriptionSession | None = self.app_state.get("session")
        if not session or not session.is_active:
            return {"ok": False, "error": "Not recording"}

        session.stop()
        return {"ok": True}

    def _generate_notes(self, body: dict) -> dict:
        session: TranscriptionSession | None = self.app_state.get("session")
        if not session:
            return {"ok": False, "error": "No session"}

        prompt = body.get("prompt", "").strip() or None
        model = body.get("model") or None

        try:
            notes = session.generate_notes(prompt=prompt, model=model)
            if notes:
                return {"ok": True, "notes": notes}
            return {
                "ok": False,
                "error": "No transcript to summarize, or API key not set",
            }
        except Exception as e:
            logger.error("Error generating notes: %s", e, exc_info=True)
            return {"ok": False, "error": str(e)}


def start_server(app_state: dict, port: int = PORT) -> HTTPServer:
    """Start the HTTP server in a background thread. Returns the server instance."""
    _Handler.app_state = app_state
    server = HTTPServer(("127.0.0.1", port), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    logger.info("Web UI server started on http://127.0.0.1:%s", port)
    return server
