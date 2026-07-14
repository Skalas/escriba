"""Install and upgrade path inventory.

Single source for which steps each install entry point runs. ``install.sh`` and
``Makefile`` are documented here; only the in-app upgrade path is automated in
Python (``updates._execute_upgrade``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class InstallStep:
    """One install/upgrade step and which entry points include it."""

    id: str
    description: str
    fresh_install: bool
    make_install: bool
    in_app_upgrade: bool
    notes: str = ""


# Ordered steps shared conceptually across paths.
INSTALL_STEPS: tuple[InstallStep, ...] = (
    InstallStep(
        "precheck_macos",
        "Require macOS",
        fresh_install=True,
        make_install=False,
        in_app_upgrade=False,
        notes="upgrade runs this in _preflight_upgrade, not as a progress step",
    ),
    InstallStep(
        "install_uv",
        "Install uv if missing",
        fresh_install=True,
        make_install=False,
        in_app_upgrade=False,
    ),
    InstallStep(
        "clone_or_pull",
        "Clone repo or git pull --ff-only",
        fresh_install=True,
        make_install=False,
        in_app_upgrade=False,
        notes="upgrade uses git fetch + merge --ff-only <release-tag>",
    ),
    InstallStep(
        "git_fetch",
        "git fetch --tags origin",
        fresh_install=False,
        make_install=False,
        in_app_upgrade=True,
    ),
    InstallStep(
        "git_pull",
        "Fast-forward to release tag on default branch",
        fresh_install=False,
        make_install=False,
        in_app_upgrade=True,
    ),
    InstallStep(
        "uv_sync",
        "uv sync",
        fresh_install=True,
        make_install=False,
        in_app_upgrade=True,
    ),
    InstallStep(
        "env_template",
        "Copy .env.example when .env missing",
        fresh_install=True,
        make_install=False,
        in_app_upgrade=False,
    ),
    InstallStep(
        "swift_binary",
        "Download/extract audio-capture release asset",
        fresh_install=True,
        make_install=False,
        in_app_upgrade=True,
        notes="install.sh skips when binary exists; upgrade always refreshes",
    ),
    InstallStep(
        "build_app",
        "uv run python setup_app.py",
        fresh_install=True,
        make_install=True,
        in_app_upgrade=True,
    ),
    InstallStep(
        "install_app",
        "Copy/replace /Applications/Escriba.app",
        fresh_install=True,
        make_install=True,
        in_app_upgrade=True,
        notes="install.sh/Makefile use rm+cp; upgrade uses atomic staging replace",
    ),
)

UPGRADE_PROGRESS_STEP_IDS: tuple[str, ...] = tuple(
    step.id for step in INSTALL_STEPS if step.in_app_upgrade
)

UPGRADE_PROGRESS_MESSAGES: dict[str, str] = {
    "git_fetch": "Fetching latest changes…",
    "git_pull": "Updating source…",
    "uv_sync": "Installing dependencies…",
    "swift_binary": "Refreshing audio-capture binary…",
    "build_app": "Building Escriba.app…",
    "install_app": "Installing to /Applications…",
}


def upgrade_progress_message(step_id: str) -> str:
    """Return the user-facing message for an upgrade progress step."""
    return UPGRADE_PROGRESS_MESSAGES[step_id]


def install_path_inventory() -> list[dict[str, Any]]:
    """Return a JSON-serializable inventory for docs and tests."""
    return [
        {
            "id": step.id,
            "description": step.description,
            "fresh_install": step.fresh_install,
            "make_install": step.make_install,
            "in_app_upgrade": step.in_app_upgrade,
            "notes": step.notes,
        }
        for step in INSTALL_STEPS
    ]
