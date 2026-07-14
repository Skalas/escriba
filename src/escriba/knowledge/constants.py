"""Shared constants for knowledge-store adapters."""
from __future__ import annotations

# Max seconds an export adapter may block its worker thread (stop path is off-thread).
EXPORT_TIMEOUT_CAP_SECONDS = 10.0
