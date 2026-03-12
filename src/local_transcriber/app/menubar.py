"""macOS menu bar app for Escriba."""

from __future__ import annotations

import logging
import multiprocessing

import rumps

from local_transcriber.app.database import Database
from local_transcriber.app.server import PORT, start_server
from local_transcriber.app.session import TranscriptionSession
from local_transcriber.config import AppConfig

logger = logging.getLogger(__name__)


def _run_webview(url: str, title: str):
    """Open a native WebKit window (runs in a child process)."""
    import webview

    webview.create_window(title, url, width=1060, height=720)
    webview.start()


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
        self.app_state: dict = {"config": config, "session": None, "db": self.db}
        self.server = None

        self.app_state["reload_config"] = self._do_reload

        self.menu = [
            rumps.MenuItem("Start Recording", callback=self.toggle_recording),
            None,
            rumps.MenuItem("Open Dashboard", callback=self.open_dashboard),
            rumps.MenuItem("Reload Config", callback=self.reload_config),
            None,
            rumps.MenuItem("Quit", callback=self.quit_app),
        ]

    def _do_reload(self):
        from dotenv import load_dotenv

        load_dotenv(override=True)
        new_config = AppConfig.load()
        self.config = new_config
        self.app_state["config"] = new_config
        logger.info(
            "Config reloaded: backend=%s, model=%s",
            new_config.streaming.backend,
            new_config.streaming.model_size,
        )
        _notify("Escriba", "Config reloaded", f"model={new_config.streaming.model_size}")
        return new_config

    def reload_config(self, _):
        self._do_reload()

    def toggle_recording(self, sender):
        session: TranscriptionSession | None = self.app_state.get("session")

        if session and session.is_active:
            session.stop()
            sender.title = "Start Recording"
            self.title = "\u3030"
            _notify("Escriba", "Recording stopped", "Transcript saved.")
        else:
            session = TranscriptionSession(self.config, database=self.db)
            session.start()
            self.app_state["session"] = session

            if session.error:
                _notify("Escriba", "Error", session.error)
                return

            sender.title = "Stop Recording"
            self.title = "\u3030\u25cf"
            _notify("Escriba", "Recording started", "Capturing system audio...")

    def open_dashboard(self, _):
        url = f"http://127.0.0.1:{PORT}"
        try:
            ctx = multiprocessing.get_context("spawn")
            p = ctx.Process(target=_run_webview, args=(url, "Escriba"), daemon=True)
            p.start()
        except Exception:
            logger.warning("pywebview unavailable, falling back to browser")
            import webbrowser
            webbrowser.open(url)

    def quit_app(self, _):
        session: TranscriptionSession | None = self.app_state.get("session")
        if session and session.is_active:
            session.stop()
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
    app = TranscriberMenuBar(config)
    app.server = start_server(app.app_state)
    logger.info("Dashboard at http://127.0.0.1:%s", PORT)
    app.run()
