"""Lightweight HTTP server for the transcriber web UI."""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlparse, parse_qs

if TYPE_CHECKING:
    from escriba.app.database import Database
    from escriba.app.session import TranscriptionSession
    from escriba.config import AppConfig

logger = logging.getLogger(__name__)

if getattr(sys, "frozen", False):
    STATIC_DIR = Path(sys.executable).parent.parent / "Resources" / "static"
else:
    STATIC_DIR = Path(__file__).parent / "static"

PORT = 19876


def _concat_wav(
    sources: list[tuple[Path | None, float, float]], out_path: Path
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
        elif path == "/api/folders":
            self._json_response(self._get_folders())
        elif path == "/api/models":
            self._json_response(self._list_models())
        elif path == "/api/download-model/status":
            downloading = self.app_state.get("_downloading_model")
            result = self.app_state.pop("_download_result", None)
            self._json_response({"ok": True, "downloading": downloading, "result": result})
        elif path.startswith("/api/sessions/") and path.endswith("/audio"):
            session_id = path.split("/api/sessions/")[1].rsplit("/audio", 1)[0]
            self._serve_audio(session_id)
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
        elif path == "/api/sessions/move":
            body = self._read_body()
            self._json_response(self._move_sessions(body))
        elif path == "/api/folders":
            body = self._read_body()
            self._json_response(self._create_folder(body))
        elif path.startswith("/api/sessions/") and path.endswith("/retranscribe"):
            session_id = path.split("/api/sessions/")[1].rsplit("/retranscribe", 1)[0]
            self._json_response(self._retranscribe_session(session_id))
        elif path.startswith("/api/sessions/") and path.endswith("/split"):
            session_id = path.split("/api/sessions/")[1].rsplit("/split", 1)[0]
            body = self._read_body()
            self._json_response(self._split_session(session_id, body))
        elif path.startswith("/api/sessions/") and path.endswith("/generate-notes"):
            session_id = path.split("/api/sessions/")[1].rsplit("/generate-notes", 1)[0]
            body = self._read_body()
            self._json_response(self._generate_session_notes(session_id, body))
        elif path.startswith("/api/sessions/") and path.endswith("/notes"):
            session_id = path.split("/api/sessions/")[1].rsplit("/notes", 1)[0]
            body = self._read_body()
            self._json_response(self._save_notes(session_id, body))
        elif path.startswith("/api/sessions/") and path.endswith("/export"):
            session_id = path.split("/api/sessions/")[1].rsplit("/export", 1)[0]
            self._json_response(self._export_session(session_id))
        elif path == "/api/download-model":
            body = self._read_body()
            self._json_response(self._download_model(body))
        else:
            self._json_response({"ok": False, "error": "Not found"}, status=404)

    def do_PUT(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"

        if path == "/api/config":
            body = self._read_body()
            self._json_response(self._put_config(body))
        elif path.startswith("/api/sessions/") and path.endswith("/rename"):
            session_id = path.split("/api/sessions/")[1].rsplit("/rename", 1)[0]
            body = self._read_body()
            self._json_response(self._rename_session(session_id, body))
        elif path.startswith("/api/folders/") and path.endswith("/rename"):
            folder_id = path.split("/api/folders/")[1].rsplit("/rename", 1)[0]
            body = self._read_body()
            self._json_response(self._rename_folder(folder_id, body))
        else:
            self._json_response({"ok": False, "error": "Not found"}, status=404)

    def do_DELETE(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"

        if path.startswith("/api/folders/"):
            folder_id = path.split("/api/folders/")[1]
            if folder_id:
                self._json_response(self._delete_folder(folder_id))
            else:
                self._json_response({"ok": False, "error": "Not found"}, status=404)
        elif path.startswith("/api/sessions/"):
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

    def _export_session(self, session_id: str) -> dict:
        """Save transcript + notes to ~/Downloads as a text file."""
        db = self._get_db()
        if not db:
            return {"ok": False, "error": "Database not available"}
        session = db.get_session(session_id)
        if not session:
            return {"ok": False, "error": "Session not found"}
        segments = db.get_segments(session_id)

        lines: list[str] = []
        lines.append(f"# {session.get('name', 'Session')}")
        if session.get("started_at"):
            lines.append(f"Date: {session['started_at']}")
        lines.append("")

        if session.get("notes_text"):
            lines.append("## Notes")
            lines.append(session["notes_text"])
            lines.append("")

        lines.append("## Transcript")
        for seg in segments:
            ts = seg.get("start_time", 0)
            h, rem = divmod(int(ts), 3600)
            m, s = divmod(rem, 60)
            timestamp = f"{h:02d}:{m:02d}:{s:02d}"
            speaker = f"[{seg['speaker']}] " if seg.get("speaker") else ""
            lines.append(f"{timestamp}  {speaker}{seg.get('text', '')}")

        content = "\n".join(lines)
        safe_name = "".join(
            c if c.isalnum() or c in " -_" else "_"
            for c in session.get("name", "transcript")
        )
        filename = f"{safe_name.strip()}.txt"

        downloads = Path.home() / "Downloads"
        downloads.mkdir(exist_ok=True)
        out_path = downloads / filename
        # Avoid overwriting
        counter = 1
        while out_path.exists():
            out_path = downloads / f"{safe_name.strip()} ({counter}).txt"
            counter += 1
        out_path.write_text(content, encoding="utf-8")
        return {"ok": True, "path": str(out_path)}

    def _serve_audio(self, session_id: str):
        """Serve the WAV audio file for a session with HTTP Range support."""
        db = self._get_db()
        if not db:
            self._json_response({"ok": False, "error": "Database not available"}, status=500)
            return
        session = db.get_session(session_id)
        if not session or not session.get("audio_path"):
            self._json_response({"ok": False, "error": "No audio for this session"}, status=404)
            return
        audio_path = Path(session["audio_path"])
        if not audio_path.exists():
            self._json_response({"ok": False, "error": "Audio file not found"}, status=404)
            return

        file_size = audio_path.stat().st_size
        range_header = self.headers.get("Range")

        if range_header:
            range_spec = range_header.replace("bytes=", "")
            parts = range_spec.split("-")
            start = int(parts[0]) if parts[0] else 0
            end = int(parts[1]) if parts[1] else file_size - 1
            end = min(end, file_size - 1)
            length = end - start + 1

            self.send_response(206)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(length))
            self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()

            with open(audio_path, "rb") as f:
                f.seek(start)
                self.wfile.write(f.read(length))
        else:
            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(file_size))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()

            with open(audio_path, "rb") as f:
                while chunk := f.read(65536):
                    self.wfile.write(chunk)

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
            return {"ok": True, "sessions": [], "folders": []}
        sessions = db.list_sessions()
        folders = db.list_folders()
        return {"ok": True, "sessions": sessions, "folders": folders}

    def _get_folders(self) -> dict:
        db = self._get_db()
        if not db:
            return {"ok": True, "folders": []}
        return {"ok": True, "folders": db.list_folders()}

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

        from escriba.app.session import TranscriptionSession
        from escriba.config import AppConfig

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

        db_session_id = session.db_session_id
        session.stop()
        return {"ok": True, "session_id": db_session_id}

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

        try:
            merged_id, sources = db.merge_sessions(session_ids, name)
        except Exception as e:
            logger.error("Merge DB step failed: %s", e, exc_info=True)
            return {"ok": False, "error": "Merge failed"}

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

        return {"ok": True, "session_id": merged_id}

    def _split_session(self, session_id: str, body: dict) -> dict:
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
        db = self._get_db()
        if not db:
            return {"ok": False, "error": "Database not available"}

        segment_id = body.get("segment_id")
        if segment_id is None:
            return {"ok": False, "error": "segment_id is required"}
        try:
            segment_id = int(segment_id)
        except (TypeError, ValueError):
            return {"ok": False, "error": "segment_id must be an integer"}

        session = db.get_session(session_id)
        if not session:
            return {"ok": False, "error": "Session not found"}
        if session.get("status") == "active":
            return {
                "ok": False,
                "error": "Cannot split an active (recording) session",
            }

        orig_audio_path = session.get("audio_path")

        # Pre-compute split_time so we can slice the WAV *before* touching
        # the DB. The DB method re-validates this before it writes.
        segment_row = db.get_segment(segment_id)
        if not segment_row or segment_row.get("session_id") != session_id:
            return {"ok": False, "error": "Segment not in this session"}
        split_time = float(segment_row.get("start_time") or 0.0)
        if split_time <= 0:
            return {"ok": False, "error": "Cannot split at the first segment"}

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
                    return {"ok": False, "error": f"Audio split failed: {e}"}

        try:
            new_id, split_time, _ = db.split_session(session_id, segment_id)
        except ValueError as e:
            for p in (part1_tmp, part2_tmp):
                if p and p.exists():
                    p.unlink(missing_ok=True)
            return {"ok": False, "error": str(e)}
        except Exception as e:
            logger.error("Split DB step failed: %s", e, exc_info=True)
            for p in (part1_tmp, part2_tmp):
                if p and p.exists():
                    p.unlink(missing_ok=True)
            return {"ok": False, "error": "Split failed"}

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
        }

    def _regenerate_title(self, db, session_id: str) -> None:
        """Run `generate_session_title` against a session's transcript.

        Silent on failure — the caller doesn't block on the title and the
        session always has a valid fallback name ("(part 1)"/"(part 2)").
        """
        config: AppConfig | None = self.app_state.get("config")
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

    def _move_sessions(self, body: dict) -> dict:
        db = self._get_db()
        if not db:
            return {"ok": False, "error": "Database not available"}
        session_ids = body.get("session_ids", [])
        folder_id = body.get("folder_id")  # None means "unfiled"
        if not session_ids:
            return {"ok": False, "error": "No sessions specified"}
        db.move_sessions_to_folder(session_ids, folder_id)
        return {"ok": True}

    def _rename_session(self, session_id: str, body: dict) -> dict:
        db = self._get_db()
        if not db:
            return {"ok": False, "error": "Database not available"}
        name = body.get("name", "").strip()
        if not name:
            return {"ok": False, "error": "Name cannot be empty"}
        db.rename_session(session_id, name)
        return {"ok": True}

    def _delete_session(self, session_id: str) -> dict:
        db = self._get_db()
        if not db:
            return {"ok": False, "error": "Database not available"}
        db.delete_session(session_id)
        return {"ok": True}

    def _retranscribe_session(self, session_id: str) -> dict:
        """Re-transcribe a session from its saved WAV audio file."""
        db = self._get_db()
        if not db:
            return {"ok": False, "error": "Database not available"}
        session = db.get_session(session_id)
        if not session:
            return {"ok": False, "error": "Session not found"}
        if not session.get("audio_path"):
            return {"ok": False, "error": "No audio file for this session"}

        audio_path = Path(session["audio_path"])
        if not audio_path.exists():
            return {"ok": False, "error": "Audio file not found on disk"}

        active_session: TranscriptionSession | None = self.app_state.get("session")
        if active_session and active_session.is_active:
            return {"ok": False, "error": "Stop recording before re-transcribing"}

        try:
            from escriba.app.session import retranscribe_from_wav
            from escriba.config import AppConfig

            config: AppConfig = self.app_state.get("config", AppConfig.load())
            segments = retranscribe_from_wav(audio_path, config)

            db.delete_segments(session_id)
            if segments:
                db.add_segments(session_id, segments)

            return {"ok": True, "segment_count": len(segments)}
        except Exception as e:
            logger.error("Re-transcribe failed: %s", e, exc_info=True)
            return {"ok": False, "error": str(e)}

    def _generate_session_notes(self, session_id: str, body: dict) -> dict:
        db = self._get_db()
        if not db:
            return {"ok": False, "error": "Database not available"}
        segments = db.get_segments(session_id)
        if not segments:
            return {"ok": False, "error": "No segments in this session"}
        transcript = " ".join(s["text"] for s in segments)
        prompt = (body.get("prompt") or "").strip() or "Summarize the key points, decisions, and action items. Respond in the same language as the transcript."
        config = self.app_state.get("config")
        default_model = config.streaming.summary_model if config else "auto"
        model = body.get("model", default_model)
        try:
            from escriba.app.session import _generate_custom_notes
            notes = _generate_custom_notes(transcript, prompt, model=model)
            if notes:
                return {"ok": True, "notes": notes}
            return {"ok": False, "error": "Failed to generate notes"}
        except Exception as e:
            logger.error("Error generating session notes: %s", e, exc_info=True)
            return {"ok": False, "error": str(e)}

    def _list_models(self) -> dict:
        try:
            from escriba.summarize.llm_summary import list_available_models
            result = list_available_models()
            return {"ok": True, **result}
        except Exception as e:
            logger.error("Error listing models: %s", e, exc_info=True)
            return {"ok": False, "error": str(e)}

    def _download_model(self, body: dict) -> dict:
        """Download a local LLM model in the background."""
        from escriba.summarize.llm_summary import recommend_model

        model_id = (body.get("model") or "").strip()
        if not model_id or model_id == "auto":
            model_id = recommend_model()
        if not model_id:
            return {"ok": False, "error": "No local model available for this hardware"}

        # Track download state on the app_state so the UI can poll
        if self.app_state.get("_downloading_model"):
            return {"ok": False, "error": "A download is already in progress"}

        self.app_state["_downloading_model"] = model_id

        import threading

        def _do_download():
            try:
                from mlx_lm import load

                logger.info("Downloading model: %s", model_id)
                load(model_id)
                logger.info("Model download complete: %s", model_id)
                self.app_state["_download_result"] = {"ok": True, "model": model_id}
            except Exception as e:
                logger.error("Model download failed: %s", e, exc_info=True)
                self.app_state["_download_result"] = {"ok": False, "error": str(e)}
            finally:
                self.app_state["_downloading_model"] = None

        threading.Thread(target=_do_download, daemon=True).start()
        return {"ok": True, "message": f"Downloading {model_id}...", "model": model_id}

    def _get_config(self) -> dict:
        from escriba.config import AppConfig, config_to_dict

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

        from escriba.config import (
            AppConfig,
            config_to_dict,
            resolve_config_path,
            save_config_to_toml,
        )

        env_updates = {}
        env_key_names = ["GEMINI_API_KEY", "ANTHROPIC_API_KEY", "HUGGINGFACE_TOKEN"]
        for key in env_key_names:
            if key in body and body[key].strip():
                env_updates[key] = body[key].strip()

        if env_updates:
            self._update_env_file(env_updates)
            for k, v in env_updates.items():
                os.environ[k] = v

        toml_data = {k: v for k, v in body.items() if k not in env_key_names}
        if toml_data:
            current_config = self.app_state.get("config")
            config_path = (
                current_config.config_path
                if isinstance(current_config, AppConfig) and current_config.config_path
                else resolve_config_path() or Path("escriba.toml")
            )
            save_config_to_toml(toml_data, config_path)

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
        from escriba.config import config_to_dict

        reload_fn = self.app_state.get("reload_config")
        if reload_fn:
            new_config = reload_fn()
            return {"ok": True, "config": config_to_dict(new_config)}

        from escriba.config import AppConfig
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

    # --- Folders ---

    def _create_folder(self, body: dict) -> dict:
        db = self._get_db()
        if not db:
            return {"ok": False, "error": "Database not available"}
        name = body.get("name", "").strip()
        if not name:
            return {"ok": False, "error": "Folder name cannot be empty"}
        folder_id = db.create_folder(name)
        return {"ok": True, "folder_id": folder_id}

    def _rename_folder(self, folder_id: str, body: dict) -> dict:
        db = self._get_db()
        if not db:
            return {"ok": False, "error": "Database not available"}
        name = body.get("name", "").strip()
        if not name:
            return {"ok": False, "error": "Name cannot be empty"}
        db.rename_folder(folder_id, name)
        return {"ok": True}

    def _delete_folder(self, folder_id: str) -> dict:
        db = self._get_db()
        if not db:
            return {"ok": False, "error": "Database not available"}
        db.delete_folder(folder_id)
        return {"ok": True}


def start_server(app_state: dict, port: int = PORT) -> HTTPServer:
    """Start the HTTP server in a background thread. Returns the server instance."""
    _Handler.app_state = app_state
    server = HTTPServer(("127.0.0.1", port), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    logger.info("Web UI server started on http://127.0.0.1:%s", port)
    return server
