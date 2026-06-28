"""macOS menu bar app for Escriba."""

from __future__ import annotations

import logging
import plistlib
import stat
import subprocess
import time
from pathlib import Path

import rumps

from escriba.app.database import Database
from escriba.app.server import AppState, PORT, _ThreadingHTTPServer, start_server
from escriba.config import AppConfig

logger = logging.getLogger(__name__)

# Persistent location for the dashboard viewer .app
_DASHBOARD_APP_DIR = Path.home() / "Library" / "Application Support" / "Escriba"
_DASHBOARD_APP = _DASHBOARD_APP_DIR / "Escriba.app"


def _ensure_dashboard_app(icon_path: Path | None = None) -> Path:
    """Create (or update) a tiny .app bundle used to open the dashboard.

    macOS Cmd+Tab reads the app name from the bundle's Info.plist,
    so this is the only reliable way to show "Escriba" instead of "Python".
    """
    contents = _DASHBOARD_APP / "Contents"
    macos = contents / "MacOS"
    resources = contents / "Resources"

    # Rebuild if missing or if project directory changed
    needs_rebuild = not (macos / "open-dashboard").exists()
    if not needs_rebuild:
        existing = (macos / "open-dashboard").read_text()
        project_dir_check = str(Path(__file__).resolve().parent.parent.parent.parent)
        if project_dir_check not in existing:
            needs_rebuild = True

    if needs_rebuild:
        macos.mkdir(parents=True, exist_ok=True)
        resources.mkdir(parents=True, exist_ok=True)

        # Info.plist — this is what Cmd+Tab reads
        plist = {
            "CFBundleName": "Escriba",
            "CFBundleDisplayName": "Escriba",
            "CFBundleIdentifier": "com.escriba.dashboard",
            "CFBundleVersion": "1.0",
            "CFBundleExecutable": "open-dashboard",
            "CFBundlePackageType": "APPL",
            "NSHighResolutionCapable": True,
        }
        if icon_path and icon_path.exists():
            (resources / "Escriba.icns").write_bytes(icon_path.read_bytes())
            plist["CFBundleIconFile"] = "Escriba"

        with open(contents / "Info.plist", "wb") as f:
            plistlib.dump(plist, f)

        # Launcher script — uses the venv Python directly (no uv at runtime)
        # NOTE: project_dir is baked in at build time; if the project moves,
        # delete ~/Library/Application Support/Escriba/Escriba.app to rebuild.
        project_dir = Path(__file__).resolve().parent.parent.parent.parent
        venv_python = project_dir / ".venv" / "bin" / "python3"
        launcher = macos / "open-dashboard"
        launcher.write_text(f"""#!/usr/bin/env bash
cd "{project_dir}"
exec "{venv_python}" -c "
import webview, sys
url = sys.argv[1] if len(sys.argv) > 1 else 'http://127.0.0.1:{PORT}'
webview.create_window('Escriba', url, width=1060, height=720)
webview.start()
" "$@"
""")
        launcher.chmod(launcher.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        logger.info("Created dashboard app at %s", _DASHBOARD_APP)

    return _DASHBOARD_APP


def _find_icon() -> Path | None:
    """Find the Escriba.icns icon file."""
    # Check resources dir (source tree)
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    icon = project_root / "resources" / "Escriba.icns"
    if icon.exists():
        return icon
    # Check inside .app bundle
    import sys
    if getattr(sys, "frozen", False):
        bundle_icon = Path(sys.executable).parent.parent / "Resources" / "Escriba.icns"
        if bundle_icon.exists():
            return bundle_icon
    return None


def _notify(title: str, subtitle: str, message: str):
    """Send a notification, silently ignoring errors (e.g. missing Info.plist)."""
    try:
        rumps.notification(title, subtitle, message)
    except RuntimeError:
        logger.debug("Notification failed (missing Info.plist), skipping")


class TranscriberMenuBar(rumps.App):
    """Menu bar app that controls transcription sessions."""

    def __init__(self, config: AppConfig):
        super().__init__("\u3030", quit_button=None)
        self.config = config
        self.db = Database()
        self.app_state = AppState(config=config, db=self.db)
        self.app_state.reload_config = self._do_reload
        self.server: _ThreadingHTTPServer | None = None

        # Mic activation detection state
        self._prompt_cooldown_until: float = 0
        self._last_detected_app: str | None = None
        # session_id of the recording auto-started by call detection; only that
        # session is auto-stopped on hangup (a hand-started one keeps running).
        self._auto_started_session_id: str | None = None
        self._call_state = self._make_call_state_machine(config)
        self._call_item = rumps.MenuItem("", callback=self._record_detected_call)
        self._call_item.hidden = True

        self._recording_item = rumps.MenuItem("Start Recording", callback=self.toggle_recording)
        self.menu = [
            self._call_item,
            self._recording_item,
            None,
            rumps.MenuItem("Open Escriba", callback=self.open_dashboard),
            rumps.MenuItem("Reload Config", callback=self.reload_config),
            None,
            rumps.MenuItem("Quit", callback=self.quit_app),
        ]
        self._last_active = False
        self._mic_poll_timer: rumps.Timer | None = None

    @staticmethod
    def _make_call_state_machine(config: AppConfig):
        from escriba.audio.call_state import CallStateMachine

        ar = config.auto_record
        return CallStateMachine(
            start_debounce=ar.start_debounce_seconds,
            stop_debounce=ar.stop_debounce_seconds,
        )

    def _do_reload(self):
        from dotenv import load_dotenv

        load_dotenv(override=True)
        new_config = AppConfig.load()
        self.config = new_config
        with self.app_state._lock:
            self.app_state.config = new_config
        self._call_state = self._make_call_state_machine(new_config)
        if self._mic_poll_timer is not None:
            self._mic_poll_timer.interval = new_config.auto_record.poll_interval
        logger.info(
            "Config reloaded: backend=%s, model=%s",
            new_config.streaming.backend,
            new_config.streaming.model_size,
        )
        _notify("Escriba", "Config reloaded", f"model={new_config.streaming.model_size}")
        return new_config

    @rumps.timer(2)
    def _sync_ui_state(self, _):
        """Poll session state and keep menu bar icon/title in sync."""
        with self.app_state._lock:
            session = self.app_state.session
            is_active = session.is_active if session else False
        if is_active == self._last_active:
            return
        self._last_active = is_active
        if is_active:
            self._recording_item.title = "Stop Recording"
            self.title = "\u3030\u25cf"
        else:
            self._recording_item.title = "Start Recording"
            self.title = "\u3030"

    def _check_mic_activation(self, _):
        """Poll CoreAudio and drive debounced call start/end handling."""
        if not self.config.auto_record.enabled:
            return

        try:
            import threading

            from escriba.audio.call_detection import get_call_app_label_if_mic_active
            from escriba.audio.call_state import CallEvent, should_auto_stop
            from escriba.audio.mic_monitor import call_mic_active

            running = call_mic_active()
            event = self._call_state.update(running, time.monotonic())

            with self.app_state._lock:
                session = self.app_state.session
                is_recording = bool(session and session.is_active)
                current_id = session.session_id if session else None

            auto_started = (
                self._auto_started_session_id is not None
                and current_id == self._auto_started_session_id
            )
            if should_auto_stop(event, is_recording, auto_started):
                logger.info("Call ended, auto-stopping recording")
                self._begin_stop_recording_session()
                return

            if is_recording:
                if not self._call_item.hidden:
                    self._call_item.hidden = True
                return

            if event == CallEvent.CALL_ENDED:
                self._call_item.hidden = True
                return

            if time.time() < self._prompt_cooldown_until:
                return

            if event != CallEvent.CALL_STARTED:
                return

            app_name = get_call_app_label_if_mic_active(True)
            self._last_detected_app = app_name
            context = f" ({app_name})" if app_name else ""
            self._call_item.title = f"Record Call{context}"
            self._call_item.hidden = False

            if self.config.auto_record.start_mode == "auto":
                self._call_item.hidden = True
                self._begin_start_recording_session(detected_app=app_name, auto=True)
                return

            logger.info("Mic activation detected, prompting user")
            threading.Thread(
                target=self._show_call_dialog,
                args=(app_name,),
                daemon=True,
            ).start()
        except Exception:
            logger.debug("Mic activation check failed", exc_info=True)

    def _show_call_dialog(self, app_name: str | None):
        """Show a system dialog via osascript (runs in background thread)."""
        try:
            context = f" by {app_name}" if app_name else ""
            script = (
                f'display dialog "Mic activated{context}. Start recording?" '
                f'buttons {{"Dismiss", "Record"}} default button "Record" '
                f'with title "Escriba" giving up after 30'
            )
            result = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                text=True,
                timeout=35,
            )
            if "Record" in result.stdout:
                self._call_item.hidden = True
                self._begin_start_recording_session(
                    detected_app=app_name, auto=True
                )
            else:
                self._prompt_cooldown_until = (
                    time.time() + self.config.auto_record.cooldown_seconds
                )
                self._call_item.hidden = True
        except Exception:
            logger.debug("Call dialog failed, menu item still available", exc_info=True)

    def _record_detected_call(self, _):
        """User clicked the 'Record Call' menu item (responds to a detected call)."""
        self._call_item.hidden = True
        self._begin_start_recording_session(
            detected_app=self._last_detected_app, auto=True
        )

    def reload_config(self, _):
        self._do_reload()

    def _begin_stop_recording_session(self) -> bool:
        """Stop the active recording session. Returns True if stop was initiated."""
        import threading

        self._recording_item.title = "Start Recording"
        self.title = "\u3030"
        self._auto_started_session_id = None

        _data, status, session = self.app_state.begin_stop_recording()
        if status != 200 or session is None:
            return False

        def _stop_async():
            try:
                session.stop()
            except Exception:
                logger.exception("Session stop failed")
            finally:
                self.app_state.finish_stop_recording()
            _notify("Escriba", "Recording stopped", "Transcript saved.")

        threading.Thread(target=_stop_async, daemon=True).start()
        return True

    def _begin_start_recording_session(
        self, detected_app: str | None = None, auto: bool = False
    ) -> bool:
        """Start a recording session. Returns True if start succeeded.

        When ``auto`` is True the recording was initiated by call detection, so
        its session id is remembered and it will be auto-stopped on hangup; a
        hand-started recording (``auto`` False) is never auto-stopped.
        """
        data, _status = self.app_state.try_start_recording(detected_app=detected_app)
        self._last_detected_app = None

        if not data.get("ok"):
            _notify("Escriba", "Error", data.get("error", "Failed to start"))
            return False

        with self.app_state._lock:
            session = self.app_state.session
            self._auto_started_session_id = (
                session.session_id if (auto and session) else None
            )

        self._recording_item.title = "Stop Recording"
        self.title = "\u3030\u25cf"
        detail = f" ({detected_app})" if detected_app else ""
        _notify("Escriba", "Recording started", f"Capturing system audio...{detail}")
        return True

    def toggle_recording(self, sender):
        with self.app_state._lock:
            session = self.app_state.session
            is_active = session.is_active if session else False

        if is_active:
            self._begin_stop_recording_session()
        else:
            self._begin_start_recording_session(detected_app=self._last_detected_app)

    def run(self, **options):
        """Start the mic poll timer from config, then enter the rumps run loop."""
        self._mic_poll_timer = rumps.Timer(
            self._check_mic_activation,
            self.config.auto_record.poll_interval,
        )
        self._mic_poll_timer.start()
        super().run(**options)

    def open_dashboard(self, _):
        url = f"http://127.0.0.1:{PORT}"
        try:
            app_bundle = _ensure_dashboard_app(icon_path=_find_icon())
            if not app_bundle.exists():
                raise FileNotFoundError(f"Dashboard app not found: {app_bundle}")
            subprocess.Popen(["open", "-a", str(app_bundle), "--args", url])
        except Exception:
            logger.warning("Dashboard app launch failed, falling back to browser", exc_info=True)
            import webbrowser
            webbrowser.open(url)

    def _terminate_dashboard(self):
        """Terminate the dashboard viewer app if running."""
        try:
            subprocess.run(
                ["osascript", "-e", 'quit app id "com.escriba.dashboard"'],
                capture_output=True,
                timeout=5,
            )
        except Exception:
            logger.debug("Dashboard termination failed", exc_info=True)

    def quit_app(self, _):
        import threading

        self._terminate_dashboard()
        with self.app_state._lock:
            session = self.app_state.session
            active = session is not None and session.is_active
        if active and session is not None:
            stop_thread = threading.Thread(target=session.stop, daemon=True)
            stop_thread.start()
            stop_thread.join(timeout=5)
            if stop_thread.is_alive():
                logger.warning("Session stop still running at quit — proceeding anyway")
        if self.server:
            self.server.shutdown()
        self.db.close()
        rumps.quit_application()


def run_menubar_app(config: AppConfig | None = None):
    """Launch the menu bar app."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    from dotenv import load_dotenv

    load_dotenv()

    if config is None:
        config = AppConfig.load()

    logger.info(
        "Config: backend=%s, model=%s", config.streaming.backend, config.streaming.model_size
    )

    # Configure local LLM cache TTL from config
    try:
        from escriba.summarize.llm_summary import configure_local_cache

        configure_local_cache(ttl=config.local_llm.cache_ttl)
    except Exception:
        logger.debug("Could not configure local LLM cache", exc_info=True)

    # Pre-build dashboard app at startup (so first "Open Dashboard" is instant)
    try:
        _ensure_dashboard_app(icon_path=_find_icon())
    except Exception:
        logger.warning("Could not pre-build dashboard app", exc_info=True)

    app = TranscriberMenuBar(config)
    app.server = start_server(app.app_state)
    logger.info("Dashboard at http://127.0.0.1:%s", PORT)
    app.run()
