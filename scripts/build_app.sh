#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "==> Building Swift audio-capture CLI..."
cd "$PROJECT_DIR/swift-audio-capture"
swift build -c release

echo "==> Building macOS .app bundle..."
cd "$PROJECT_DIR"
uv run python setup_app.py

echo "==> Bundling audio-capture binary..."
cp "$PROJECT_DIR/swift-audio-capture/.build/release/audio-capture" \
   "$PROJECT_DIR/dist/Local Transcriber.app/Contents/Resources/audio-capture"

echo "==> Done! App is at: dist/Local Transcriber.app"
