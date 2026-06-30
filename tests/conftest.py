"""Shared test helpers and fixtures."""
from __future__ import annotations

from io import BytesIO
from unittest.mock import MagicMock

from escriba.app.server import AppState, _Handler


def make_handler(app_state: AppState) -> _Handler:
    """Build a minimal _Handler wired to app_state, suitable for unit tests."""
    handler = _Handler.__new__(_Handler)
    handler.app_state = app_state
    handler.headers = {}
    handler.rfile = BytesIO()
    handler.wfile = BytesIO()
    handler.connection = MagicMock()
    handler.client_address = ("127.0.0.1", 12345)
    return handler
