"""Presentation-layer export utilities: filesystem paths and Downloads writes."""
from __future__ import annotations

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
