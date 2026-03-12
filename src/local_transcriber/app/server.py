"""Lightweight HTTP server for the transcriber web UI."""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
from functools import partial
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlparse, parse_qs

if TYPE_CHECKING:
    from local_transcriber.app.database import Database
    from local_transcriber.app.session import TranscriptionSession
    from local_transcriber.config import AppConfig

logger = logging.getLogger(__name__)

if getattr(sys, "frozen", False):
    STATIC_DIR = Path(sys.executable).parent.parent / "Resources" / "static"
else:
    STATIC_DIR = Path(__file__).parent / "static"

PORT = 19876


class _Handler(BaseHTTPRequestHandler):
    """HTTP request handler with API endpoints for the transcriber app."""

    app_state: dict = {}

    def log_message(self, format, *args):
        pass

    # --- Routing ---

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        params = parse_qs(parsed.query)

        if path == "/":
            self._serve_file("index.html", "text/html")
        elif path == "/api/status":
            self._json_response(self._get_status())
        elif path == "/api/transcript":
            session_id = params.get("session_id", [None])[0]
            self._json_response(self._get_transcript(session_id))
        elif path == "/api/config":
            self._json_response(self._get_config())
        elif path == "/api/sessions":
            self._json_response(self._get_sessions())
        elif path.startswith("/api/sessions/"):
            session_id = path.split("/api/sessions/")[1]
            if session_id:
                self._json_response(self._get_session_detail(session_id))
            else:
                self._json_response({"ok": False, "error": "Not found"}, status=404)
        else:
            self._json_response({"ok": False, "error": "Not found"}, status=404)

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"

        if path == "/api/config/reload":
            self._json_response(self._reload_config())
        elif path == "/api/recording/start":
            self._json_response(self._start_recording())
        elif path == "/api/recording/stop":
            self._json_response(self._stop_recording())
        elif path == "/api/notes":
            body = self._read_body()
            self._json_response(self._generate_notes(body))
        elif path == "/api/sessions/merge":
            body = self._read_body()
            self._json_response(self._merge_sessions(body))
        elif path.startswith("/api/sessions/") and path.endswith("/generate-notes"):
            session_id = path.split("/api/sessions/")[1].rsplit("/generate-notes", 1)[0]
            body = self._read_body()
            self._json_response(self._generate_session_notes(session_id, body))
        elif path.startswith("/api/sessions/") and path.endswith("/notes"):
            session_id = path.split("/api/sessions/")[1].rsplit("/notes", 1)[0]
            body = self._read_body()
            self._json_response(self._save_notes(session_id, body))
        else:
            self._json_response({"ok": False, "error": "Not found"}, status=404)

    def do_PUT(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"

        if path == "/api/config":
            body = self._read_body()
            self._json_response(self._put_config(body))
        else:
            self._json_response({"ok": False, "error": "Not found"}, status=404)

    def do_DELETE(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"

        if path.startswith("/api/sessions/"):
            session_id = path.split("/api/sessions/")[1]
            if session_id:
                self._json_response(self._delete_session(session_id))
            else:
                self._json_response({"ok": False, "error": "Not found"}, status=404)
        else:
            self._json_response({"ok": False, "error": "Not found"}, status=404)

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

    def _json_response(self, data: dict, status: int = 200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
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

    def _get_db(self) -> Database | None:
        return self.app_state.get("db")

    # --- API handlers ---

    def _get_status(self) -> dict:
        session: TranscriptionSession | None = self.app_state.get("session")
        if session:
            return {"ok": True, **session.get_status()}
        return {"ok": True, "is_active": False, "session_id": None}

    def _get_transcript(self, session_id: str | None = None) -> dict:
        # If a session_id is given, load from DB
        if session_id:
            db = self._get_db()
            if not db:
                return {"ok": False, "error": "Database not available"}
            segments = db.get_segments(session_id)
            text = " ".join(s["text"] for s in segments)
            session_info = db.get_session(session_id)
            return {
                "ok": True,
                "text": text,
                "segments": segments,
                "notes": session_info.get("notes_text") if session_info else None,
            }

        # Otherwise return the live session transcript
        session: TranscriptionSession | None = self.app_state.get("session")
        if not session:
            return {"ok": True, "text": "", "segments": []}
        return {
            "ok": True,
            "text": session.get_transcript(),
            "segments": session.get_segments(),
        }

    def _get_sessions(self) -> dict:
        db = self._get_db()
        if not db:
            return {"ok": True, "sessions": []}
        sessions = db.list_sessions()
        return {"ok": True, "sessions": sessions}

    def _get_session_detail(self, session_id: str) -> dict:
        db = self._get_db()
        if not db:
            return {"ok": False, "error": "Database not available"}
        session = db.get_session(session_id)
        if not session:
            return {"ok": False, "error": "Session not found"}
        segments = db.get_segments(session_id)
        return {"ok": True, "session": session, "segments": segments}

    def _start_recording(self) -> dict:
        session: TranscriptionSession | None = self.app_state.get("session")
        if session and session.is_active:
            return {"ok": False, "error": "Already recording"}

        from local_transcriber.app.session import TranscriptionSession
        from local_transcriber.config import AppConfig

        config: AppConfig = self.app_state.get("config", AppConfig.load())
        db = self._get_db()
        session = TranscriptionSession(config, database=db)
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

    def _merge_sessions(self, body: dict) -> dict:
        db = self._get_db()
        if not db:
            return {"ok": False, "error": "Database not available"}
        session_ids = body.get("session_ids", [])
        name = body.get("name", "Merged Session")
        if len(session_ids) < 2:
            return {"ok": False, "error": "Need at least 2 sessions to merge"}
        merged_id = db.merge_sessions(session_ids, name)
        return {"ok": True, "session_id": merged_id}

    def _delete_session(self, session_id: str) -> dict:
        db = self._get_db()
        if not db:
            return {"ok": False, "error": "Database not available"}
        db.delete_session(session_id)
        return {"ok": True}

    def _generate_session_notes(self, session_id: str, body: dict) -> dict:
        db = self._get_db()
        if not db:
            return {"ok": False, "error": "Database not available"}
        segments = db.get_segments(session_id)
        if not segments:
            return {"ok": False, "error": "No segments in this session"}
        transcript = " ".join(s["text"] for s in segments)
        prompt = body.get("prompt", "").strip() or "Summarize the key points, decisions, and action items."
        model = body.get("model", "gemini")
        try:
            from local_transcriber.app.session import _generate_custom_notes
            notes = _generate_custom_notes(transcript, prompt, model=model)
            if notes:
                return {"ok": True, "notes": notes}
            return {"ok": False, "error": "Failed to generate notes"}
        except Exception as e:
            logger.error("Error generating session notes: %s", e, exc_info=True)
            return {"ok": False, "error": str(e)}

    def _get_config(self) -> dict:
        from local_transcriber.config import AppConfig, config_to_dict

        config: AppConfig = self.app_state.get("config", AppConfig.load())
        cfg_dict = config_to_dict(config)
        env_keys = {
            "GEMINI_API_KEY": bool(os.environ.get("GEMINI_API_KEY", "").strip()),
            "ANTHROPIC_API_KEY": bool(os.environ.get("ANTHROPIC_API_KEY", "").strip()),
            "HUGGINGFACE_TOKEN": bool(os.environ.get("HUGGINGFACE_TOKEN", "").strip()),
        }
        return {"ok": True, "config": cfg_dict, "env_keys": env_keys}

    def _put_config(self, body: dict) -> dict:
        session: TranscriptionSession | None = self.app_state.get("session")
        if session and session.is_active:
            return {"ok": False, "error": "Stop recording before changing settings"}

        from local_transcriber.config import (
            AppConfig,
            config_to_dict,
            resolve_config_path,
            save_config_to_toml,
        )

        # Separate env keys from TOML config
        env_updates = {}
        env_key_names = ["GEMINI_API_KEY", "ANTHROPIC_API_KEY", "HUGGINGFACE_TOKEN"]
        for key in env_key_names:
            if key in body and body[key].strip():
                env_updates[key] = body[key].strip()

        # Update .env file if needed
        if env_updates:
            self._update_env_file(env_updates)
            # Also set in current process
            for k, v in env_updates.items():
                os.environ[k] = v

        # Write TOML config — use the path the current config was loaded from
        toml_data = {k: v for k, v in body.items() if k not in env_key_names}
        if toml_data:
            current_config = self.app_state.get("config")
            config_path = (
                current_config.config_path
                if isinstance(current_config, AppConfig) and current_config.config_path
                else resolve_config_path() or Path("local-transcriber.toml")
            )
            save_config_to_toml(toml_data, config_path)
            logger.info("Settings saved to %s", config_path)

        # Trigger reload
        reload_fn = self.app_state.get("reload_config")
        if reload_fn:
            new_config = reload_fn()
        else:
            from dotenv import load_dotenv
            load_dotenv(override=True)
            new_config = AppConfig.load()
            self.app_state["config"] = new_config

        return {"ok": True, "config": config_to_dict(new_config)}

    @staticmethod
    def _update_env_file(updates: dict):
        env_path = Path(".env")
        lines = []
        if env_path.exists():
            lines = env_path.read_text().splitlines()

        updated_keys = set()
        new_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped and not stripped.startswith("#") and "=" in stripped:
                key = stripped.split("=", 1)[0].strip()
                if key in updates:
                    new_lines.append(f"{key}={updates[key]}")
                    updated_keys.add(key)
                    continue
            new_lines.append(line)

        for key, value in updates.items():
            if key not in updated_keys:
                new_lines.append(f"{key}={value}")

        env_path.write_text("\n".join(new_lines) + "\n")

    def _reload_config(self) -> dict:
        from local_transcriber.config import config_to_dict

        reload_fn = self.app_state.get("reload_config")
        if reload_fn:
            new_config = reload_fn()
            return {"ok": True, "config": config_to_dict(new_config)}

        from local_transcriber.config import AppConfig
        from dotenv import load_dotenv
        load_dotenv(override=True)
        new_config = AppConfig.load()
        self.app_state["config"] = new_config
        return {"ok": True, "config": config_to_dict(new_config)}

    def _save_notes(self, session_id: str, body: dict) -> dict:
        db = self._get_db()
        if not db:
            return {"ok": False, "error": "Database not available"}
        notes = body.get("notes_text", "")
        db.save_notes(session_id, notes)
        return {"ok": True}


def start_server(app_state: dict, port: int = PORT) -> HTTPServer:
    """Start the HTTP server in a background thread. Returns the server instance."""
    _Handler.app_state = app_state
    server = HTTPServer(("127.0.0.1", port), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    logger.info("Web UI server started on http://127.0.0.1:%s", port)
    return server
