"""Lightweight HTTP server for the transcriber web UI."""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
import uuid
from collections.abc import Callable, Sequence
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from socketserver import ThreadingMixIn
from typing import TYPE_CHECKING, Any, BinaryIO
from urllib.parse import parse_qs, urlparse

from escriba.app.observability import (
    get_correlation_id,
    latency_store,
    new_correlation_id,
    set_correlation_id,
)

if TYPE_CHECKING:
    from escriba.app.database import Database
    from escriba.app.session import TranscriptionSession
    from escriba.config import AppConfig

logger = logging.getLogger(__name__)

MAX_BODY_BYTES = 1_048_576
REQUEST_TIMEOUT_SECONDS = 30
MAX_SPEAKER_DISPLAY_NAME = 200
_CLIENT_DISCONNECT_ERRORS = (BrokenPipeError, ConnectionResetError)


def _git_info() -> dict[str, Any] | None:
    """Best-effort git state of the working tree: short commit + dirty flag."""
    import subprocess

    project_dir = Path(__file__).resolve().parents[3]
    if not (project_dir / ".git").exists():
        return None

    def _run(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=project_dir,
                capture_output=True,
                text=True,
                timeout=2,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return result.stdout.strip() if result.returncode == 0 else None

    commit = _run("rev-parse", "--short", "HEAD")
    if commit is None:
        return None
    status = _run("status", "--porcelain")
    return {"commit": commit, "dirty": bool(status)}


def _segment_speaker_label(segment: dict[str, Any]) -> str | None:
    """Resolved speaker label for display/export (custom name or raw)."""
    display = segment.get("speaker_display")
    if display:
        return str(display)
    raw = segment.get("speaker")
    return str(raw) if raw else None


def _segments_to_transcript(segments: list[dict[str, Any]]) -> str:
    """Build transcript text using display speaker names when available."""
    parts: list[str] = []
    for seg in segments:
        text = seg.get("text") or ""
        speaker = _segment_speaker_label(seg)
        if speaker:
            parts.append(f"[{speaker}] {text}")
        else:
            parts.append(text)
    return " ".join(parts)


def format_export_timestamp(seconds: float | int | None) -> str:
    """Format segment start time as HH:MM:SS for export."""
    total = int(seconds or 0)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def format_export_duration(seconds: float | int | None) -> str:
    """Format session duration as HH:MM:SS for export metadata."""
    return format_export_timestamp(seconds)


def safe_export_filename(name: str, ext: str) -> str:
    """Build a filesystem-safe export filename."""
    safe_name = "".join(
        c if c.isalnum() or c in " -_" else "_"
        for c in name
    ).strip() or "transcript"
    return f"{safe_name}.{ext}"


def format_path_for_display(path: Path) -> str:
    """Return a user-friendly path (~-prefixed when under home)."""
    home = Path.home()
    try:
        return "~/" + str(path.relative_to(home))
    except ValueError:
        return str(path)


def unique_export_path(directory: Path, filename: str) -> Path:
    """Pick a non-colliding path under directory for filename."""
    target = directory / filename
    if not target.exists():
        return target
    stem = Path(filename).stem
    ext = Path(filename).suffix
    counter = 2
    while True:
        candidate = directory / f"{stem} ({counter}){ext}"
        if not candidate.exists():
            return candidate
        counter += 1


def save_session_export_to_downloads(
    content: str,
    filename: str,
    downloads_dir: Path | None = None,
) -> Path:
    """Write export content to ~/Downloads with a de-duplicated filename."""
    directory = downloads_dir if downloads_dir is not None else Path.home() / "Downloads"
    directory.mkdir(parents=True, exist_ok=True)
    path = unique_export_path(directory, filename)
    path.write_text(content, encoding="utf-8")
    return path


def build_session_export_markdown(session: dict[str, Any], segments: list[dict[str, Any]]) -> str:
    """Build a Markdown export bundle for a session."""
    lines: list[str] = [f"# {session.get('name', 'Session')}", ""]

    metadata: list[str] = []
    if session.get("started_at"):
        metadata.append(f"**Date:** {session['started_at']}")
    duration = session.get("duration_seconds")
    if duration is not None:
        metadata.append(f"**Duration:** {format_export_duration(duration)}")
    if metadata:
        lines.extend(metadata)
        lines.append("")

    notes_text = (session.get("notes_text") or "").strip()
    if notes_text:
        lines.extend(["## Notes", "", notes_text, ""])

    lines.extend(["## Transcript", ""])
    for seg in segments:
        seg_id = seg.get("id")
        anchor = ""
        if seg_id is not None:
            anchor = f'<a id="seg-{int(seg_id)}"></a>'
        timestamp = format_export_timestamp(seg.get("start_time"))
        text = seg.get("text") or ""
        speaker = _segment_speaker_label(seg)
        if speaker:
            lines.append(f"{anchor}[{timestamp}] **{speaker}**: {text}")
        else:
            lines.append(f"{anchor}[{timestamp}] {text}")

    return "\n".join(lines)


def build_session_export_txt(session: dict[str, Any], segments: list[dict[str, Any]]) -> str:
    """Build a plain-text export bundle for a session."""
    lines: list[str] = [session.get("name", "Session"), ""]

    if session.get("started_at"):
        lines.append(f"Date: {session['started_at']}")
    duration = session.get("duration_seconds")
    if duration is not None:
        lines.append(f"Duration: {format_export_duration(duration)}")
    if session.get("started_at") or duration is not None:
        lines.append("")

    notes_text = (session.get("notes_text") or "").strip()
    if notes_text:
        lines.extend(["Notes", notes_text, ""])

    lines.append("Transcript")
    for seg in segments:
        timestamp = format_export_timestamp(seg.get("start_time"))
        text = seg.get("text") or ""
        speaker = _segment_speaker_label(seg)
        if speaker:
            lines.append(f"{timestamp}  [{speaker}] {text}")
        else:
            lines.append(f"{timestamp}  {text}")

    return "\n".join(lines)


class ApiError(Exception):
    """Structured API error with an HTTP status code."""

    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.status = status


class AppState:
    """Thread-safe shared state for the HTTP server and menu bar app."""

    def __init__(
        self,
        config: AppConfig,
        db: Database,
        reload_config: Callable[[], AppConfig] | None = None,
    ) -> None:
        self._lock = threading.RLock()
        self._download_lock = threading.Lock()
        self._stop_in_progress = False
        self._start_in_progress = False
        self.config = config
        self.db = db
        self.session: TranscriptionSession | None = None
        self.reload_config = reload_config
        self._downloading_model: str | None = None
        self._download_result: dict[str, Any] | None = None

    def get_active_session(self) -> TranscriptionSession | None:
        """Return the current session reference under the state lock."""
        with self._lock:
            return self.session

    def try_start_recording(
        self, detected_app: str | None = None
    ) -> tuple[dict[str, Any], int]:
        """Start recording if idle; at most one active session."""
        from escriba.app.session import TranscriptionSession

        with self._lock:
            if self._stop_in_progress:
                return {"ok": False, "error": "Recording stop in progress"}, 409
            session = self.session
            if session and session.is_active:
                return {"ok": False, "error": "Already recording"}, 409
            if self._start_in_progress:
                return {"ok": False, "error": "Already recording"}, 409

            new_session = TranscriptionSession(self.config, database=self.db)
            if detected_app:
                new_session.detected_app = detected_app
            self.session = new_session
            self._start_in_progress = True

        try:
            new_session.start()
        finally:
            with self._lock:
                self._start_in_progress = False
                if new_session.error:
                    self.session = None
                    response: tuple[dict[str, Any], int] = (
                        {"ok": False, "error": new_session.error},
                        503,
                    )
                else:
                    response = ({"ok": True}, 200)

        return response

    def begin_stop_recording(
        self,
    ) -> tuple[dict[str, Any], int, TranscriptionSession | None]:
        """Claim an in-flight stop so concurrent starts cannot interleave."""
        with self._lock:
            if self._stop_in_progress:
                return {"ok": False, "error": "Stop already in progress"}, 409, None
            session = self.session
            if not session or not session.is_active:
                return {"ok": False, "error": "Not recording"}, 409, None
            self._stop_in_progress = True
            return {"ok": True}, 200, session

    def finish_stop_recording(self) -> None:
        """Release the stop-in-progress guard."""
        with self._lock:
            self._stop_in_progress = False

    def get_download_status(self) -> tuple[str | None, dict[str, Any] | None]:
        """Return current download state and consume any finished result."""
        with self._download_lock:
            downloading = self._downloading_model
            result = self._download_result
            self._download_result = None
            return downloading, result

    def try_begin_model_download(self, model_id: str) -> tuple[dict[str, Any], int]:
        """Claim a model download; only one may run at a time."""
        with self._download_lock:
            if self._downloading_model:
                return {
                    "ok": False,
                    "error": "A download is already in progress",
                }, 409
            self._downloading_model = model_id
            self._download_result = None
        return {"ok": True}, 200

    def finish_model_download(self, result: dict[str, Any]) -> None:
        """Record the outcome of a background model download."""
        with self._download_lock:
            self._download_result = result
            self._downloading_model = None

if getattr(sys, "frozen", False):
    STATIC_DIR = Path(sys.executable).parent.parent / "Resources" / "static"
else:
    STATIC_DIR = Path(__file__).parent / "static"

PORT = 19876


def _concat_wav(
    sources: Sequence[tuple[str | Path | None, float, float]], out_path: Path
) -> None:
    """Concatenate WAV files into `out_path`, filling silence where needed.

    `sources` is a list of ``(audio_path, offset_seconds, duration_seconds)``
    tuples, ordered chronologically. Sessions with no audio_path (or a
    missing file) are represented by silence of `duration_seconds`.
    Raises ``ValueError`` if source formats disagree.
    """
    import wave

    fmt: tuple[int, int, int] | None = None
    for p, _offset, _dur in sources:
        if p and Path(p).exists():
            with wave.open(str(p), "rb") as src:
                fmt = (
                    src.getnchannels(),
                    src.getsampwidth(),
                    src.getframerate(),
                )
            break
    if fmt is None:
        return
    nchannels, sampwidth, framerate = fmt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out_path), "wb") as dst:
        dst.setnchannels(nchannels)
        dst.setsampwidth(sampwidth)
        dst.setframerate(framerate)

        for audio_path, _offset, dur in sources:
            src_path = Path(audio_path) if audio_path else None
            if src_path and src_path.exists():
                with wave.open(str(src_path), "rb") as src:
                    if (
                        src.getnchannels(),
                        src.getsampwidth(),
                        src.getframerate(),
                    ) != fmt:
                        raise ValueError(
                            f"Audio format mismatch in {src_path}: "
                            f"cannot merge across different sample rates "
                            f"or channel counts"
                        )
                    dst.writeframes(src.readframes(src.getnframes()))
            elif dur > 0:
                # Fill with silence to keep the audio timeline aligned
                # with the rebased segment timestamps.
                n_frames = int(dur * framerate)
                dst.writeframes(b"\x00" * (n_frames * nchannels * sampwidth))


def _slice_wav(
    source: Path, split_at_seconds: float, out_first: Path, out_second: Path
) -> None:
    """Write two WAV files from `source`, cut at `split_at_seconds`.

    Uses the stdlib `wave` module. Preserves the original format
    (channels, sample width, framerate). Raises if the cut is outside
    the file or the file can't be read.
    """
    import wave

    with wave.open(str(source), "rb") as src:
        nchannels = src.getnchannels()
        sampwidth = src.getsampwidth()
        framerate = src.getframerate()
        total_frames = src.getnframes()

        split_frame = int(split_at_seconds * framerate)
        if split_frame <= 0 or split_frame >= total_frames:
            raise ValueError(
                f"Split at {split_at_seconds}s is outside the audio "
                f"(file has {total_frames / framerate:.2f}s)"
            )

        first_frames = src.readframes(split_frame)
        second_frames = src.readframes(total_frames - split_frame)

    out_first.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out_first), "wb") as dst:
        dst.setnchannels(nchannels)
        dst.setsampwidth(sampwidth)
        dst.setframerate(framerate)
        dst.writeframes(first_frames)

    with wave.open(str(out_second), "wb") as dst:
        dst.setnchannels(nchannels)
        dst.setsampwidth(sampwidth)
        dst.setframerate(framerate)
        dst.writeframes(second_frames)


class _ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    """Serve each request on its own thread so long jobs don't block polling."""

    daemon_threads = True
    allow_reuse_address = True


class _Handler(BaseHTTPRequestHandler):
    """HTTP request handler with API endpoints for the transcriber app."""

    app_state: AppState

    def log_message(self, format, *args):
        pass

    def handle(self) -> None:
        """Apply a per-request socket timeout before handling."""
        try:
            self.connection.settimeout(REQUEST_TIMEOUT_SECONDS)
            super().handle()
        except TimeoutError:
            logger.warning("Request timed out from %s", self.client_address)
            self.close_connection = True

    def _respond(self, result: dict[str, Any] | tuple[dict[str, Any], int]) -> None:
        """Send a JSON response, mapping ``ApiError`` to structured 4xx/5xx."""
        if isinstance(result, tuple):
            data, status = result
        else:
            data, status = result, 200
        self._json_response(data, status=status)

    def _respond_error(self, exc: ApiError) -> None:
        self._json_response({"ok": False, "error": exc.message}, status=exc.status)

    def _respond_unexpected(self, exc: Exception) -> None:
        logger.error("Unhandled server error: %s", exc, exc_info=True)
        self._json_response(
            {"ok": False, "error": "Internal server error"},
            status=500,
        )

    def _log_client_disconnect(self, context: str) -> None:
        client = getattr(self, "client_address", None)
        logger.debug(
            "Client disconnected during %s from %s",
            context,
            client,
        )

    def _handle_request_exception(self, exc: Exception) -> None:
        if isinstance(exc, ApiError):
            self._respond_error(exc)
        elif isinstance(exc, _CLIENT_DISCONNECT_ERRORS):
            self._log_client_disconnect("request handling")
        else:
            self._respond_unexpected(exc)

    def _stream_file_to_client(self, f: BinaryIO, length: int | None = None) -> None:
        """Stream an open file to the client; stop quietly on disconnect."""
        remaining = length
        while True:
            chunk_size = 65536 if remaining is None else min(65536, remaining)
            if remaining is not None and remaining <= 0:
                break
            chunk = f.read(chunk_size)
            if not chunk:
                break
            try:
                self.wfile.write(chunk)
            except _CLIENT_DISCONNECT_ERRORS:
                self._log_client_disconnect("audio stream")
                return
            if remaining is not None:
                remaining -= len(chunk)

    # --- Routing ---

    def do_GET(self) -> None:
        cid, t0 = self._begin_request("GET")
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        params = parse_qs(parsed.query)

        try:
            if path == "/":
                self._serve_file("index.html", "text/html")
            elif path == "/api/status":
                self._respond(self._get_status())
            elif path == "/api/version":
                self._respond(self._get_version())
            elif path == "/api/transcript":
                session_id = params.get("session_id", [None])[0]
                self._respond(self._get_transcript(session_id))
            elif path == "/api/config":
                self._respond(self._get_config())
            elif path == "/api/sessions":
                self._respond(self._get_sessions())
            elif path == "/api/folders":
                self._respond(self._get_folders())
            elif path == "/api/models":
                self._respond(self._list_models())
            elif path == "/api/search":
                q = params.get("q", [""])[0]
                self._respond(self._search_segments(q))
            elif path == "/api/download-model/status":
                downloading, result = self.app_state.get_download_status()
                self._respond(
                    {"ok": True, "downloading": downloading, "result": result}
                )
            elif path.startswith("/api/sessions/") and path.endswith("/audio"):
                session_id = path.split("/api/sessions/")[1].rsplit("/audio", 1)[0]
                self._serve_audio(session_id)
            elif path.startswith("/api/sessions/") and path.endswith("/export"):
                session_id = path.split("/api/sessions/")[1].rsplit("/export", 1)[0]
                export_format = params.get("format", ["md"])[0]
                self._respond(self._export_session(session_id, export_format))
            elif path.startswith("/api/sessions/"):
                session_id = path.split("/api/sessions/")[1]
                if session_id:
                    self._respond(self._get_session_detail(session_id))
                else:
                    self._respond(({"ok": False, "error": "Not found"}, 404))
            else:
                self._respond(({"ok": False, "error": "Not found"}, 404))
        except ApiError as exc:
            self._respond_error(exc)
        except _CLIENT_DISCONNECT_ERRORS:
            self._log_client_disconnect("GET")
        except Exception as exc:
            self._handle_request_exception(exc)
        finally:
            self._end_request("GET", cid, t0)

    def do_POST(self) -> None:
        cid, t0 = self._begin_request("POST")
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"

        try:
            if path == "/api/config/reload":
                self._respond(self._reload_config())
            elif path == "/api/recording/start":
                self._respond(self._start_recording())
            elif path == "/api/recording/stop":
                self._respond(self._stop_recording())
            elif path == "/api/recording/user-notes":
                self._respond(self._save_recording_user_notes(self._parse_json_body()))
            elif path == "/api/notes":
                self._respond(self._generate_notes(self._parse_json_body()))
            elif path == "/api/prompts/enhance":
                self._respond(self._enhance_prompt(self._parse_json_body()))
            elif path == "/api/sessions/merge":
                self._respond(self._merge_sessions(self._parse_json_body()))
            elif path == "/api/sessions/move":
                self._respond(self._move_sessions(self._parse_json_body()))
            elif path == "/api/folders":
                self._respond(self._create_folder(self._parse_json_body()))
            elif path.startswith("/api/sessions/") and path.endswith("/export"):
                session_id = path.split("/api/sessions/")[1].rsplit("/export", 1)[0]
                body = self._parse_json_body()
                if body.get("save"):
                    export_format = str(body.get("format", "md"))
                    self._respond(self._save_session_export(session_id, export_format))
                else:
                    self._respond(
                        ({"ok": False, "error": "Use GET for JSON export or POST with save:true"}, 400)
                    )
            elif path.startswith("/api/sessions/") and path.endswith("/retranscribe"):
                session_id = path.split("/api/sessions/")[1].rsplit("/retranscribe", 1)[0]
                self._respond(self._retranscribe_session(session_id))
            elif path.startswith("/api/sessions/") and path.endswith("/split"):
                session_id = path.split("/api/sessions/")[1].rsplit("/split", 1)[0]
                self._respond(self._split_session(session_id, self._parse_json_body()))
            elif path.startswith("/api/sessions/") and path.endswith("/generate-notes"):
                session_id = path.split("/api/sessions/")[1].rsplit("/generate-notes", 1)[0]
                self._respond(
                    self._generate_session_notes(session_id, self._parse_json_body())
                )
            elif path.startswith("/api/sessions/") and path.endswith("/notes"):
                session_id = path.split("/api/sessions/")[1].rsplit("/notes", 1)[0]
                self._respond(self._save_notes(session_id, self._parse_json_body()))
            elif path == "/api/download-model":
                self._respond(self._download_model(self._parse_json_body()))
            else:
                self._respond(({"ok": False, "error": "Not found"}, 404))
        except ApiError as exc:
            self._respond_error(exc)
        except _CLIENT_DISCONNECT_ERRORS:
            self._log_client_disconnect("POST")
        except Exception as exc:
            self._handle_request_exception(exc)
        finally:
            self._end_request("POST", cid, t0)

    def do_PUT(self) -> None:
        cid, t0 = self._begin_request("PUT")
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"

        try:
            if path == "/api/config":
                self._respond(self._put_config(self._parse_json_body()))
            elif path.startswith("/api/sessions/") and path.endswith("/rename"):
                session_id = path.split("/api/sessions/")[1].rsplit("/rename", 1)[0]
                self._respond(self._rename_session(session_id, self._parse_json_body()))
            elif path.startswith("/api/sessions/") and path.endswith("/speakers"):
                session_id = path.split("/api/sessions/")[1].rsplit("/speakers", 1)[0]
                self._respond(self._set_speaker_label(session_id, self._parse_json_body()))
            elif path.startswith("/api/folders/") and path.endswith("/rename"):
                folder_id = path.split("/api/folders/")[1].rsplit("/rename", 1)[0]
                self._respond(self._rename_folder(folder_id, self._parse_json_body()))
            else:
                self._respond(({"ok": False, "error": "Not found"}, 404))
        except ApiError as exc:
            self._respond_error(exc)
        except _CLIENT_DISCONNECT_ERRORS:
            self._log_client_disconnect("PUT")
        except Exception as exc:
            self._handle_request_exception(exc)
        finally:
            self._end_request("PUT", cid, t0)

    def do_DELETE(self) -> None:
        cid, t0 = self._begin_request("DELETE")
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"

        try:
            if path.startswith("/api/folders/"):
                folder_id = path.split("/api/folders/")[1]
                if folder_id:
                    self._respond(self._delete_folder(folder_id))
                else:
                    self._respond(({"ok": False, "error": "Not found"}, 404))
            elif path.startswith("/api/sessions/"):
                session_id = path.split("/api/sessions/")[1]
                if session_id:
                    self._respond(self._delete_session(session_id))
                else:
                    self._respond(({"ok": False, "error": "Not found"}, 404))
            else:
                self._respond(({"ok": False, "error": "Not found"}, 404))
        except ApiError as exc:
            self._respond_error(exc)
        except _CLIENT_DISCONNECT_ERRORS:
            self._log_client_disconnect("DELETE")
        except Exception as exc:
            self._handle_request_exception(exc)
        finally:
            self._end_request("DELETE", cid, t0)

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

    def _resolve_session_export(
        self, session_id: str, fmt: str
    ) -> tuple[dict[str, Any], int] | tuple[str, str, str, str]:
        """Build export payload or return an error response tuple."""
        db = self._require_db()
        session = db.get_session(session_id)
        if not session:
            return {"ok": False, "error": "Session not found"}, 404

        segments = db.get_segments(session_id)
        export_format = (fmt or "md").strip().lower()
        if export_format == "md":
            content = build_session_export_markdown(session, segments)
            filename = safe_export_filename(session.get("name", "Session"), "md")
            content_type = "text/markdown"
        elif export_format == "txt":
            content = build_session_export_txt(session, segments)
            filename = safe_export_filename(session.get("name", "Session"), "txt")
            content_type = "text/plain"
        else:
            return {"ok": False, "error": f"Unsupported format: {fmt}"}, 400

        return content, filename, export_format, content_type

    def _export_session(self, session_id: str, fmt: str = "md") -> tuple[dict, int]:
        """Build an in-browser export bundle for a session (JSON for Copy)."""
        resolved = self._resolve_session_export(session_id, fmt)
        if isinstance(resolved[0], dict):
            return resolved  # type: ignore[return-value]
        content, filename, export_format, _content_type = resolved
        return {
            "ok": True,
            "filename": filename,
            "content": content,
            "format": export_format,
        }, 200

    def _save_session_export(self, session_id: str, fmt: str = "md") -> tuple[dict, int]:
        """Write a session export to ~/Downloads and return its path."""
        resolved = self._resolve_session_export(session_id, fmt)
        if isinstance(resolved[0], dict):
            return resolved  # type: ignore[return-value]
        content, filename, export_format, _content_type = resolved
        try:
            saved_path = save_session_export_to_downloads(content, filename)
        except OSError as exc:
            logger.error("Failed to save export for session %s: %s", session_id, exc, exc_info=True)
            return {"ok": False, "error": f"Failed to save export: {exc}"}, 500
        return {
            "ok": True,
            "path": str(saved_path),
            "display_path": format_path_for_display(saved_path),
            "format": export_format,
        }, 200

    def _serve_audio(self, session_id: str) -> None:
        """Serve the WAV audio file for a session with HTTP Range support."""
        db = self._require_db()
        session = db.get_session(session_id)
        if not session or not session.get("audio_path"):
            raise ApiError("No audio for this session", 404)
        audio_path = Path(session["audio_path"])
        if not audio_path.exists():
            raise ApiError("Audio file not found", 404)

        file_size = audio_path.stat().st_size
        if file_size == 0:
            raise ApiError("Audio file is empty", 416)

        range_header = self.headers.get("Range")

        if range_header:
            if not range_header.startswith("bytes="):
                raise ApiError("Invalid Range header", 416)
            range_spec = range_header[6:]
            if "," in range_spec:
                raise ApiError("Multiple ranges not supported", 416)
            parts = range_spec.split("-", 1)
            if len(parts) != 2:
                raise ApiError("Invalid Range header", 416)
            try:
                if parts[0] == "":
                    if not parts[1]:
                        raise ValueError("empty suffix")
                    suffix_len = int(parts[1])
                    if suffix_len <= 0:
                        raise ValueError("invalid suffix")
                    start = max(file_size - suffix_len, 0)
                    end = file_size - 1
                else:
                    start = int(parts[0])
                    end = int(parts[1]) if parts[1] else file_size - 1
                start = max(0, min(start, file_size - 1))
                end = max(start, min(end, file_size - 1))
            except ValueError as exc:
                raise ApiError("Invalid Range header", 416) from exc
            length = end - start + 1

            self.send_response(206)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(length))
            self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()

            with open(audio_path, "rb") as f:
                f.seek(start)
                self._stream_file_to_client(f, length)
        else:
            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(file_size))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()

            with open(audio_path, "rb") as f:
                self._stream_file_to_client(f)

    def _json_response(self, data: dict, status: int = 200) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            cid = get_correlation_id()
            if cid:
                self.send_header("X-Correlation-ID", cid)
            self.end_headers()
            self.wfile.write(body)
        except _CLIENT_DISCONNECT_ERRORS:
            self._log_client_disconnect("JSON response")

    # --- Observability helpers ---

    def _begin_request(self, method: str) -> tuple[str, float]:
        """Generate a correlation ID, set it in thread-local, start the clock."""
        cid = new_correlation_id()
        set_correlation_id(cid)
        safe_path = "".join(c for c in self.path if c >= " ")
        logger.debug(
            "request.start method=%s path=%s corr_id=%s",
            method,
            safe_path,
            cid,
            extra={"corr_id": cid, "op": f"request.{method}"},
        )
        return cid, time.monotonic()

    def _end_request(self, method: str, cid: str, t0: float) -> None:
        dur_ms = (time.monotonic() - t0) * 1000
        latency_store.record(f"handler.{method}", dur_ms)
        safe_path = "".join(c for c in self.path if c >= " ")
        logger.debug(
            "request.done method=%s path=%s corr_id=%s duration_ms=%.1f",
            method,
            safe_path,
            cid,
            dur_ms,
            extra={"corr_id": cid, "op": f"request.{method}", "duration_ms": dur_ms},
        )

    def _parse_json_body(self) -> dict[str, Any]:
        """Read and parse a JSON object body with size limits."""
        raw = self._read_body_bytes()
        if not raw:
            return {}
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ApiError("Invalid JSON body", 400) from exc
        if not isinstance(data, dict):
            raise ApiError("JSON body must be an object", 400)
        return data

    def _read_body_bytes(self) -> bytes:
        """Read the request body, enforcing MAX_BODY_BYTES on actual bytes read.

        When Content-Length is present and valid it is used directly (after cap
        check).  When absent — chunked transfer-encoding, HTTP/1.0 bodies, or
        any other case — we stream-and-count so oversized payloads are still
        rejected with 413 regardless of what the headers claim.
        """
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            # Stream-and-count: read one byte over the cap so we can detect it.
            data = self.rfile.read(MAX_BODY_BYTES + 1)
            if len(data) > MAX_BODY_BYTES:
                raise ApiError(
                    f"Request body exceeds {MAX_BODY_BYTES} bytes", 413
                )
            return data
        try:
            length = int(raw_length)
        except ValueError as exc:
            raise ApiError("Invalid Content-Length header", 400) from exc
        if length < 0:
            raise ApiError("Invalid Content-Length header", 400)
        if length > MAX_BODY_BYTES:
            raise ApiError(
                f"Request body exceeds {MAX_BODY_BYTES} bytes", 413
            )
        if length == 0:
            return b""
        return self.rfile.read(length)

    def _require_db(self) -> Database:
        db = self.app_state.db
        if not db:
            raise ApiError("Database not available", 503)
        return db

    def _get_db(self) -> Database | None:
        return self.app_state.db

    # --- API handlers ---

    def _get_status(self) -> dict:
        with self.app_state._lock:
            session = self.app_state.session
            if session:
                return {"ok": True, **session.get_status()}
        return {"ok": True, "is_active": False, "session_id": None}

    def _get_version(self) -> dict:
        import platform

        from escriba import __version__

        with self.app_state._lock:
            config = self.app_state.config

        backend = model = None
        if config is not None:
            streaming = getattr(config, "streaming", None)
            backend = getattr(streaming, "backend", None)
            model = getattr(streaming, "model_size", None)

        return {
            "ok": True,
            "version": __version__,
            "git": _git_info(),
            "python_version": platform.python_version(),
            "platform": f"{platform.system()} {platform.release()}",
            "machine": platform.machine(),
            "project_dir": str(Path(__file__).resolve().parents[3]),
            "backend": backend,
            "model": model,
            "repo_url": "https://github.com/Skalas/escriba",
        }

    def _get_transcript(self, session_id: str | None = None) -> dict:
        if session_id:
            db = self._require_db()
            segments = db.get_segments(session_id)
            text = _segments_to_transcript(segments)
            session_info = db.get_session(session_id)
            return {
                "ok": True,
                "text": text,
                "segments": segments,
                "notes": session_info.get("notes_text") if session_info else None,
            }

        with self.app_state._lock:
            session = self.app_state.session
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
            return {"ok": True, "sessions": [], "folders": []}
        sessions = db.list_sessions()
        folders = db.list_folders()
        return {"ok": True, "sessions": sessions, "folders": folders}

    def _search_segments(self, query: str) -> tuple[dict, int]:
        q = query.strip()
        if not q:
            return {"ok": False, "error": "Query required"}, 400
        if len(q) > 200:
            return {"ok": False, "error": "Query too long (max 200 characters)"}, 400
        db = self._require_db()
        results = db.search_segments(q)
        return {"ok": True, "query": q, "results": results}, 200

    def _get_folders(self) -> dict:
        db = self._get_db()
        if not db:
            return {"ok": True, "folders": []}
        return {"ok": True, "folders": db.list_folders()}

    def _get_session_detail(self, session_id: str) -> tuple[dict, int]:
        db = self._require_db()
        session = db.get_session(session_id)
        if not session:
            return {"ok": False, "error": "Session not found"}, 404
        segments = db.get_segments(session_id)
        speakers = db.list_speakers(session_id)
        return {
            "ok": True,
            "session": session,
            "segments": segments,
            "speakers": speakers,
        }, 200

    def _set_speaker_label(self, session_id: str, body: dict) -> tuple[dict, int]:
        db = self._require_db()
        if not db.get_session(session_id):
            return {"ok": False, "error": "Session not found"}, 404
        speaker_key = (body.get("speaker") or "").strip()
        if not speaker_key:
            return {"ok": False, "error": "Speaker key required"}, 400
        display_name = (body.get("name") or "").strip()
        if display_name and len(display_name) > MAX_SPEAKER_DISPLAY_NAME:
            return {
                "ok": False,
                "error": f"Name too long (max {MAX_SPEAKER_DISPLAY_NAME} characters)",
            }, 400
        if display_name:
            known_speakers = {s["speaker"] for s in db.list_speakers(session_id)}
            if speaker_key not in known_speakers:
                return {"ok": False, "error": "Unknown speaker for session"}, 400
        db.set_speaker_label(session_id, speaker_key, display_name)
        return {"ok": True}, 200

    def _start_recording(self) -> tuple[dict, int]:
        return self.app_state.try_start_recording()

    def _save_recording_user_notes(self, body: dict) -> tuple[dict, int]:
        with self.app_state._lock:
            session = self.app_state.session
            if not session or not session.is_active:
                return {"ok": False, "error": "Not recording"}, 409
            db_session_id = session.db_session_id
        if not db_session_id:
            return {"ok": False, "error": "Session not yet created"}, 409
        db = self._require_db()
        user_notes = body.get("user_notes", "")
        if user_notes is not None and not isinstance(user_notes, str):
            raise ApiError("user_notes must be a string", 400)
        db.save_user_notes(db_session_id, user_notes or "")
        return {"ok": True}, 200

    def _stop_recording(self) -> tuple[dict, int]:
        data, status, session = self.app_state.begin_stop_recording()
        if status != 200 or session is None:
            return data, status
        db_session_id = session.db_session_id
        try:
            session.stop()
        finally:
            self.app_state.finish_stop_recording()
        return {"ok": True, "session_id": db_session_id}, 200

    def _generate_notes(self, body: dict) -> tuple[dict, int]:
        with self.app_state._lock:
            session = self.app_state.session
            if not session:
                return {"ok": False, "error": "No session"}, 400

        prompt = (body.get("prompt") or "").strip() or None
        model = body.get("model") or None

        try:
            notes = session.generate_notes(prompt=prompt, model=model)
            if notes:
                return {"ok": True, "notes": notes}, 200
            return {
                "ok": False,
                "error": "No transcript to summarize, or API key not set",
            }, 400
        except Exception as e:
            logger.error("Error generating notes: %s", e, exc_info=True)
            return {"ok": False, "error": "Notes generation failed; check logs"}, 503

    def _enhance_prompt(self, body: dict) -> tuple[dict, int]:
        text = (body.get("text") or "").strip()
        if not text:
            return {"ok": False, "error": "Nothing to enhance"}, 400
        with self.app_state._lock:
            config = self.app_state.config
        default_model = config.streaming.summary_model if config else "auto"
        model = body.get("model") or default_model
        preserve = bool(body.get("preserve_placeholders"))
        try:
            from escriba.summarize.llm_summary import enhance_prompt

            improved = enhance_prompt(
                text, model=model, preserve_placeholders=preserve
            )
            if improved:
                return {"ok": True, "prompt": improved}, 200
            return {
                "ok": False,
                "error": "Could not enhance prompt (check AI model / API key)",
            }, 503
        except Exception as e:
            logger.error("Error enhancing prompt: %s", e, exc_info=True)
            return {"ok": False, "error": "Prompt enhancement failed; check logs"}, 503

    def _merge_sessions(self, body: dict) -> tuple[dict, int]:
        db = self._require_db()
        session_ids = body.get("session_ids")
        if not isinstance(session_ids, list):
            raise ApiError("session_ids must be a list", 400)
        name = body.get("name", "Merged Session")
        if not isinstance(name, str):
            raise ApiError("name must be a string", 400)
        if len(session_ids) < 2:
            return {"ok": False, "error": "Need at least 2 sessions to merge"}, 400

        try:
            merged_id, sources = db.merge_sessions(session_ids, name)
        except Exception as e:
            logger.error("Merge DB step failed: %s", e, exc_info=True)
            return {"ok": False, "error": "Merge failed"}, 503

        # Concatenate WAVs (skip silently on failure — the merged session
        # is still valid as a transcript-only session, same as before).
        audio_dir = (
            Path.home()
            / "Library"
            / "Application Support"
            / "Escriba"
            / "audio"
        )
        has_any_audio = any(p for p, _off, _d in sources if p)
        if has_any_audio:
            merged_audio = audio_dir / f"{merged_id}.wav"
            try:
                _concat_wav(sources, merged_audio)
                db.update_audio_path(merged_id, str(merged_audio))
            except ValueError as e:
                logger.warning(
                    "Merge audio skipped: %s", e,
                )
            except Exception:
                logger.error("Merge audio concat failed", exc_info=True)

        # Regenerate the merged session's title from its transcript —
        # matches split symmetry; user's manually-typed name is replaced
        # if the LLM returns a better one.
        self._regenerate_title(db, merged_id)

        return {"ok": True, "session_id": merged_id}, 200

    def _split_session(self, session_id: str, body: dict) -> tuple[dict, int]:
        """Split a session into two at an existing segment boundary.

        Ordering matters for safety:
          1. Validate request + look up split_time from the chosen segment.
          2. Slice the original WAV to two *temp* files.
          3. Run the DB transaction (moves segments, creates new row).
          4. Atomic-rename the temp files into place.
        If step 2 fails → nothing changed. If step 3 fails → delete temps.
        If step 4 fails after step 3 succeeded we still end up with a valid
        DB state and the original WAV unchanged (only the second half is
        missing); we log and recover the audio_path pointer.
        """
        db = self._require_db()

        segment_id = body.get("segment_id")
        if segment_id is None:
            return {"ok": False, "error": "segment_id is required"}, 400
        try:
            segment_id = int(segment_id)
        except (TypeError, ValueError):
            return {"ok": False, "error": "segment_id must be an integer"}, 400

        session = db.get_session(session_id)
        if not session:
            return {"ok": False, "error": "Session not found"}, 404
        if session.get("status") == "active":
            return {
                "ok": False,
                "error": "Cannot split an active (recording) session",
            }, 409

        orig_audio_path = session.get("audio_path")

        # Pre-compute split_time so we can slice the WAV *before* touching
        # the DB. The DB method re-validates this before it writes.
        segment_row = db.get_segment(segment_id)
        if not segment_row or segment_row.get("session_id") != session_id:
            return {"ok": False, "error": "Segment not in this session"}, 400
        split_time = float(segment_row.get("start_time") or 0.0)
        if split_time <= 0:
            return {"ok": False, "error": "Cannot split at the first segment"}, 400

        audio_dir = (
            Path.home()
            / "Library"
            / "Application Support"
            / "Escriba"
            / "audio"
        )
        new_id_preview = str(uuid.uuid4())  # used only for temp filenames
        part1_tmp = None
        part2_tmp = None

        if orig_audio_path:
            orig_path = Path(orig_audio_path)
            if not orig_path.exists():
                orig_audio_path = None  # treat as "no audio"
            else:
                try:
                    part1_tmp = audio_dir / f".split-{new_id_preview}.part1.wav"
                    part2_tmp = audio_dir / f".split-{new_id_preview}.part2.wav"
                    _slice_wav(orig_path, split_time, part1_tmp, part2_tmp)
                except Exception as e:
                    logger.error("WAV slice failed: %s", e, exc_info=True)
                    for p in (part1_tmp, part2_tmp):
                        if p and p.exists():
                            p.unlink(missing_ok=True)
                    return {"ok": False, "error": "Audio split failed; check logs"}, 503

        try:
            new_id, split_time, _ = db.split_session(session_id, segment_id)
        except ValueError as e:
            for p in (part1_tmp, part2_tmp):
                if p and p.exists():
                    p.unlink(missing_ok=True)
            return {"ok": False, "error": str(e)}, 400
        except Exception as e:
            logger.error("Split DB step failed: %s", e, exc_info=True)
            for p in (part1_tmp, part2_tmp):
                if p and p.exists():
                    p.unlink(missing_ok=True)
            return {"ok": False, "error": "Split failed"}, 503

        # Move temp audio files into their final paths.
        if orig_audio_path and part1_tmp and part2_tmp:
            new_audio_path = audio_dir / f"{new_id}.wav"
            try:
                part2_tmp.replace(new_audio_path)
                db.update_audio_path(new_id, str(new_audio_path))
            except Exception:
                logger.error(
                    "Could not finalize second-half WAV", exc_info=True,
                )
            try:
                part1_tmp.replace(Path(orig_audio_path))
            except Exception:
                logger.error(
                    "Could not finalize first-half WAV", exc_info=True,
                )

        # Regenerate titles for both halves — splits usually separate two
        # unrelated meetings, so "(part 1)"/"(part 2)" is rarely right.
        # Synchronous on purpose: the mlx-lm cache lock protects us from
        # gemma-gemma races, and keeping this inline avoids the
        # whisper-race that a new recording could trigger.
        self._regenerate_title(db, session_id)
        self._regenerate_title(db, new_id)

        return {
            "ok": True,
            "first_session_id": session_id,
            "second_session_id": new_id,
            "split_at_seconds": split_time,
        }, 200

    def _regenerate_title(self, db, session_id: str) -> None:
        """Run `generate_session_title` against a session's transcript.

        Silent on failure — the caller doesn't block on the title and the
        session always has a valid fallback name ("(part 1)"/"(part 2)").
        """
        config: AppConfig | None
        with self.app_state._lock:
            config = self.app_state.config
        if not config or not getattr(config.auto_name, "enabled", True):
            return

        segments = db.get_segments(session_id)
        if not segments:
            return

        words = " ".join((s.get("text") or "") for s in segments[:40]).split()
        max_words = getattr(config.auto_name, "max_snippet_words", 500)
        snippet = " ".join(words[:max_words])
        if not snippet.strip():
            return

        try:
            from escriba.summarize.llm_summary import generate_session_title

            title = generate_session_title(
                snippet,
                app_name=None,
                model=config.streaming.summary_model,
            )
            if title:
                db.rename_session(session_id, title)
                logger.info("Post-split title for %s: %s", session_id, title)
        except Exception:
            logger.debug("Post-split title generation failed", exc_info=True)

    def _move_sessions(self, body: dict) -> tuple[dict, int]:
        db = self._require_db()
        session_ids = body.get("session_ids")
        if not isinstance(session_ids, list):
            raise ApiError("session_ids must be a list", 400)
        folder_id = body.get("folder_id")  # None means "unfiled"
        if not session_ids:
            return {"ok": False, "error": "No sessions specified"}, 400
        db.move_sessions_to_folder(session_ids, folder_id)
        return {"ok": True}, 200

    def _rename_session(self, session_id: str, body: dict) -> tuple[dict, int]:
        db = self._require_db()
        name = (body.get("name") or "").strip()
        if not name:
            return {"ok": False, "error": "Name cannot be empty"}, 400
        if not db.get_session(session_id):
            return {"ok": False, "error": "Session not found"}, 404
        db.rename_session(session_id, name)
        return {"ok": True}, 200

    def _delete_session(self, session_id: str) -> tuple[dict, int]:
        db = self._require_db()
        if not db.get_session(session_id):
            return {"ok": False, "error": "Session not found"}, 404
        db.delete_session(session_id)
        return {"ok": True}, 200

    def _retranscribe_session(self, session_id: str) -> tuple[dict, int]:
        """Re-transcribe a session from its saved WAV audio file."""
        db = self._require_db()
        session = db.get_session(session_id)
        if not session:
            return {"ok": False, "error": "Session not found"}, 404
        if not session.get("audio_path"):
            return {"ok": False, "error": "No audio file for this session"}, 400

        audio_path = Path(session["audio_path"])
        if not audio_path.exists():
            return {"ok": False, "error": "Audio file not found on disk"}, 404

        with self.app_state._lock:
            active_session = self.app_state.session
            if active_session and active_session.is_active:
                return {
                    "ok": False,
                    "error": "Stop recording before re-transcribing",
                }, 409

        try:
            from escriba.app.session import retranscribe_from_wav

            with self.app_state._lock:
                config = self.app_state.config
            segments = retranscribe_from_wav(audio_path, config)

            db.delete_segments(session_id)
            if segments:
                db.add_segments(session_id, segments)

            return {"ok": True, "segment_count": len(segments)}, 200
        except Exception as e:
            logger.error("Re-transcribe failed: %s", e, exc_info=True)
            return {"ok": False, "error": "Re-transcription failed; check logs"}, 503

    def _generate_session_notes(self, session_id: str, body: dict) -> tuple[dict, int]:
        db = self._require_db()
        session = db.get_session(session_id)
        if not session:
            return {"ok": False, "error": "Session not found"}, 404
        segments = db.get_segments(session_id)
        if not segments:
            return {"ok": False, "error": "No segments in this session"}, 400
        transcript = _segments_to_transcript(segments)
        prompt = (body.get("prompt") or "").strip() or "Summarize the key points, decisions, and action items. Respond in the same language as the transcript."
        with self.app_state._lock:
            config = self.app_state.config
        default_model = config.streaming.summary_model if config else "auto"
        model = body.get("model", default_model)
        system_prompt = config.prompts.effective_system_prompt if config else None
        user_notes = session.get("user_notes") or ""
        try:
            from escriba.app.session import _generate_custom_notes

            notes = _generate_custom_notes(
                transcript, prompt, model=model, system_prompt=system_prompt,
                user_notes=user_notes,
            )
            if notes:
                return {"ok": True, "notes": notes}, 200
            return {"ok": False, "error": "Failed to generate notes"}, 503
        except Exception as e:
            logger.error("Error generating session notes: %s", e, exc_info=True)
            return {"ok": False, "error": "Notes generation failed; check logs"}, 503

    def _list_models(self) -> tuple[dict, int]:
        try:
            from escriba.summarize.llm_summary import list_available_models

            result = list_available_models()
            return {"ok": True, **result}, 200
        except Exception as e:
            logger.error("Error listing models: %s", e, exc_info=True)
            return {"ok": False, "error": "Could not list models; check logs"}, 503

    def _download_model(self, body: dict) -> tuple[dict, int]:
        """Download a local LLM model in the background."""
        from escriba.summarize.llm_summary import recommend_model

        model_id = (body.get("model") or "").strip()
        if not model_id or model_id == "auto":
            model_id = recommend_model()
        if not model_id:
            return {
                "ok": False,
                "error": "No local model available for this hardware",
            }, 503

        claim, status = self.app_state.try_begin_model_download(model_id)
        if status != 200:
            return claim, status

        state = self.app_state

        def _do_download() -> None:
            try:
                from mlx_lm import load

                logger.info("Downloading model: %s", model_id)
                load(model_id)
                logger.info("Model download complete: %s", model_id)
                state.finish_model_download({"ok": True, "model": model_id})
            except Exception as e:
                logger.error("Model download failed: %s", e, exc_info=True)
                state.finish_model_download({"ok": False, "error": str(e)})

        threading.Thread(target=_do_download, daemon=True).start()
        return {
            "ok": True,
            "message": f"Downloading {model_id}...",
            "model": model_id,
        }, 200

    def _get_config(self) -> dict:
        from escriba.config import AppConfig, config_to_dict

        with self.app_state._lock:
            config = self.app_state.config
        if config is None:
            config = AppConfig.load()
        cfg_dict = config_to_dict(config)
        env_keys = {
            "GEMINI_API_KEY": bool(os.environ.get("GEMINI_API_KEY", "").strip()),
            "ANTHROPIC_API_KEY": bool(os.environ.get("ANTHROPIC_API_KEY", "").strip()),
            "HUGGINGFACE_TOKEN": bool(os.environ.get("HUGGINGFACE_TOKEN", "").strip()),
        }
        return {"ok": True, "config": cfg_dict, "env_keys": env_keys}

    def _put_config(self, body: dict) -> tuple[dict, int]:
        with self.app_state._lock:
            session = self.app_state.session
            if session and session.is_active:
                return {"ok": False, "error": "Stop recording before changing settings"}, 409

        import shutil
        import tempfile

        from escriba.config import (
            AppConfig,
            ConfigValidationError,
            config_to_dict,
            resolve_config_path,
            update_config_toml,
        )

        env_updates = {}
        env_key_names = ["GEMINI_API_KEY", "ANTHROPIC_API_KEY", "HUGGINGFACE_TOKEN"]
        for key in env_key_names:
            if key in body:
                value = body[key]
                if not isinstance(value, str):
                    raise ApiError(f"{key} must be a string", 400)
                if value.strip():
                    env_updates[key] = value.strip()

        if env_updates:
            self._update_env_file(env_updates)
            for k, v in env_updates.items():
                os.environ[k] = v
            # API keys changed — flush the models cache so next /api/models reflects them.
            from escriba.summarize.llm_summary import invalidate_models_cache
            invalidate_models_cache()

        toml_data = {k: v for k, v in body.items() if k not in env_key_names}
        # Drop read-only fields surfaced by config_to_dict for the UI.
        if isinstance(toml_data.get("prompts"), dict):
            toml_data["prompts"] = {
                k: v
                for k, v in toml_data["prompts"].items()
                if k in ("system_prompt", "templates")
            }
        if toml_data:
            with self.app_state._lock:
                current_config = self.app_state.config
            config_path = (
                current_config.config_path
                if isinstance(current_config, AppConfig) and current_config.config_path
                else resolve_config_path() or Path("escriba.toml")
            )
            # Validate the merged config IN MEMORY before touching the real file.
            # Strategy: copy the real TOML to a temp file, apply the update there,
            # then run AppConfig.load() (which calls validate()). Only on success
            # do we write to the real path.
            tmp_path = Path(
                tempfile.mktemp(suffix=".toml", dir=config_path.parent)
            )
            try:
                if config_path.exists():
                    shutil.copy2(config_path, tmp_path)
                update_config_toml(toml_data, tmp_path)
                AppConfig.load(tmp_path)
            except ConfigValidationError as exc:
                tmp_path.unlink(missing_ok=True)
                return {"ok": False, "error": str(exc)}, 400
            except Exception:
                tmp_path.unlink(missing_ok=True)
                raise
            else:
                tmp_path.unlink(missing_ok=True)
            update_config_toml(toml_data, config_path)

        with self.app_state._lock:
            reload_fn = self.app_state.reload_config
        if reload_fn:
            new_config = reload_fn()
        else:
            from dotenv import load_dotenv

            load_dotenv(override=True)
            new_config = AppConfig.load()
            with self.app_state._lock:
                self.app_state.config = new_config

        return {"ok": True, "config": config_to_dict(new_config)}, 200

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

    def _reload_config(self) -> tuple[dict, int]:
        from escriba.config import config_to_dict

        with self.app_state._lock:
            reload_fn = self.app_state.reload_config
        if reload_fn:
            new_config = reload_fn()
            return {"ok": True, "config": config_to_dict(new_config)}, 200

        from escriba.config import AppConfig
        from dotenv import load_dotenv

        load_dotenv(override=True)
        new_config = AppConfig.load()
        with self.app_state._lock:
            self.app_state.config = new_config
        return {"ok": True, "config": config_to_dict(new_config)}, 200

    def _save_notes(self, session_id: str, body: dict) -> tuple[dict, int]:
        db = self._require_db()
        if not db.get_session(session_id):
            return {"ok": False, "error": "Session not found"}, 404
        notes = body.get("notes_text", "")
        if notes is not None and not isinstance(notes, str):
            raise ApiError("notes_text must be a string", 400)
        if notes is None:
            notes = ""
        db.save_notes(session_id, notes)
        return {"ok": True}, 200

    # --- Folders ---

    def _create_folder(self, body: dict) -> tuple[dict, int]:
        db = self._require_db()
        name = (body.get("name") or "").strip()
        if not name:
            return {"ok": False, "error": "Folder name cannot be empty"}, 400
        folder_id = db.create_folder(name)
        return {"ok": True, "folder_id": folder_id}, 200

    def _rename_folder(self, folder_id: str, body: dict) -> tuple[dict, int]:
        db = self._require_db()
        name = (body.get("name") or "").strip()
        if not name:
            return {"ok": False, "error": "Name cannot be empty"}, 400
        folders = {f["id"] for f in db.list_folders()}
        if folder_id not in folders:
            return {"ok": False, "error": "Folder not found"}, 404
        db.rename_folder(folder_id, name)
        return {"ok": True}, 200

    def _delete_folder(self, folder_id: str) -> tuple[dict, int]:
        db = self._require_db()
        folders = {f["id"] for f in db.list_folders()}
        if folder_id not in folders:
            return {"ok": False, "error": "Folder not found"}, 404
        db.delete_folder(folder_id)
        return {"ok": True}, 200


def start_server(app_state: AppState, port: int = PORT) -> _ThreadingHTTPServer:
    """Start the HTTP server in a background thread. Returns the server instance."""
    _Handler.app_state = app_state
    server = _ThreadingHTTPServer(("127.0.0.1", port), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    logger.info("Web UI server started on http://127.0.0.1:%s", port)
    return server
