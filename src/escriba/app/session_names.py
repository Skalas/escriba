"""Shared session naming helpers (no database or HTTP dependencies)."""

from __future__ import annotations

import re

_PART_SUFFIX_RE = re.compile(r"\s*\(part\s+\d+\)\s*$", re.IGNORECASE)


def strip_part_suffix(name: str) -> str:
    """Remove a trailing ``(part N)`` suffix from a session name."""
    stripped = _PART_SUFFIX_RE.sub("", name).strip()
    return stripped or name
