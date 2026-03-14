"""
Build a macOS .app bundle for Escriba.

Instead of py2app (which struggles with ML dependencies), this creates
a lightweight .app wrapper that launches the Python app via uv.
"""

import os
import plistlib
import stat
from pathlib import Path

APP_NAME = "Escriba"
BUNDLE_ID = "com.escriba.app"
VERSION = "0.1.0"

PROJECT_DIR = Path(__file__).parent.resolve()
DIST_DIR = PROJECT_DIR / "dist"
APP_DIR = DIST_DIR / f"{APP_NAME}.app"
CONTENTS = APP_DIR / "Contents"
MACOS_DIR = CONTENTS / "MacOS"
RESOURCES_DIR = CONTENTS / "Resources"


def build():
    # Clean previous build
    if APP_DIR.exists():
        import shutil
        shutil.rmtree(APP_DIR)

    # Create directory structure
    for d in (MACOS_DIR, RESOURCES_DIR):
        d.mkdir(parents=True)

    # Copy static files
    static_src = PROJECT_DIR / "src" / "escriba" / "app" / "static"
    static_dst = RESOURCES_DIR / "static"
    static_dst.mkdir()
    for f in static_src.iterdir():
        (static_dst / f.name).write_bytes(f.read_bytes())

    # Copy Swift audio-capture binary if available
    swift_bin = PROJECT_DIR / "swift-audio-capture" / ".build" / "release" / "audio-capture"
    if swift_bin.exists():
        dst = RESOURCES_DIR / "audio-capture"
        dst.write_bytes(swift_bin.read_bytes())
        dst.chmod(dst.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        print(f"Bundled Swift CLI: {swift_bin}")
    else:
        print(f"Warning: Swift audio-capture binary not found at {swift_bin}")

    # Copy icon
    icon_src = PROJECT_DIR / "resources" / "Escriba.icns"
    if icon_src.exists():
        (RESOURCES_DIR / "Escriba.icns").write_bytes(icon_src.read_bytes())
    else:
        print(f"Warning: icon not found at {icon_src}, run: uv run python scripts/generate_icon.py")

    # Write Info.plist
    plist = {
        "CFBundleName": APP_NAME,
        "CFBundleDisplayName": APP_NAME,
        "CFBundleIdentifier": BUNDLE_ID,
        "CFBundleVersion": VERSION,
        "CFBundleShortVersionString": VERSION,
        "CFBundleExecutable": "launcher",
        "CFBundleIconFile": "Escriba",
        "CFBundlePackageType": "APPL",
        "LSUIElement": True,
        "NSMicrophoneUsageDescription": (
            "Escriba needs microphone access to capture and transcribe audio."
        ),
        "NSScreenCaptureUsageDescription": (
            "Escriba needs screen recording permission to capture system audio."
        ),
    }
    with open(CONTENTS / "Info.plist", "wb") as f:
        plistlib.dump(plist, f)

    # Resolve uv path at build time so the launcher works from .app context
    import shutil as _shutil
    uv_path = _shutil.which("uv")
    if not uv_path:
        raise RuntimeError("uv not found on PATH")

    log_file = Path.home() / "Library" / "Logs" / "escriba" / "app.log"
    launcher = MACOS_DIR / "launcher"
    launcher.write_text(f"""#!/usr/bin/env bash
# Escriba launcher — runs the app via uv from the project directory.
export PATH="$HOME/.local/bin:/usr/local/bin:/opt/homebrew/bin:$PATH"
export ESCRIBA_PROJECT_ROOT="{PROJECT_DIR}"
unset VIRTUAL_ENV LOCAL_TRANSCRIBER_CONFIG 2>/dev/null
LOG_DIR="$(dirname "{log_file}")"
mkdir -p "$LOG_DIR"
cd "{PROJECT_DIR}"
exec "{uv_path}" run --no-sync escriba app >> "{log_file}" 2>&1
""")
    launcher.chmod(launcher.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    print(f"Built: {APP_DIR}")
    print(f"Install: cp -r '{APP_DIR}' /Applications/")


if __name__ == "__main__":
    build()
