"""Presentation-layer export utilities: filesystem paths and Downloads writes."""
from __future__ import annotations

import os
from pathlib import Path


def safe_export_filename(name: str, ext: str) -> str:
    """Build a filesystem-safe export filename."""
    safe_name = (
        "".join(c if c.isalnum() or c in " -_" else "_" for c in name).strip()
        or "transcript"
    )
    return f"{safe_name}.{ext}"


def format_path_for_display(path: Path) -> str:
    """Return a user-friendly path (~-prefixed when under home)."""
    home = Path.home()
    try:
        return "~/" + str(path.relative_to(home))
    except ValueError:
        return str(path)


def reserve_export_path(directory: Path, filename: str) -> Path:
    """Atomically reserve a non-colliding export path under ``directory``."""
    stem = Path(filename).stem
    ext = Path(filename).suffix
    counter = 1
    while counter < 10_000:
        candidate_name = filename if counter == 1 else f"{stem} ({counter}){ext}"
        candidate = directory / candidate_name
        try:
            fd = os.open(candidate, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            return candidate
        except FileExistsError:
            counter += 1
    raise OSError(f"Could not reserve export path under {directory}")


def unique_export_path(directory: Path, filename: str) -> Path:
    """Return the next non-colliding export path (delegates to ``reserve_export_path``)."""
    return reserve_export_path(directory, filename)


def save_session_export_to_downloads(
    content: str,
    filename: str,
    downloads_dir: Path | None = None,
) -> Path:
    """Write export content to ~/Downloads with a de-duplicated filename."""
    directory = downloads_dir if downloads_dir is not None else Path.home() / "Downloads"
    directory.mkdir(parents=True, exist_ok=True)
    path = reserve_export_path(directory, filename)
    path.write_text(content, encoding="utf-8")
    return path
