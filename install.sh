#!/usr/bin/env bash
# Escriba installer — sets up everything on a fresh Mac.
# Usage: curl -fsSL <raw-url>/install.sh | bash
set -euo pipefail

REPO="https://github.com/Skalas/escriba.git"
INSTALL_DIR="$HOME/.escriba"
APP_NAME="Escriba"

info()  { printf '\033[1;34m==> %s\033[0m\n' "$*"; }
ok()    { printf '\033[1;32m==> %s\033[0m\n' "$*"; }
err()   { printf '\033[1;31m==> %s\033[0m\n' "$*" >&2; }

# --- Pre-checks ---
if [[ "$(uname -s)" != "Darwin" ]]; then
  err "Escriba only runs on macOS."
  exit 1
fi

# --- Install uv if missing ---
if ! command -v uv &>/dev/null; then
  info "Installing uv (Python package manager)..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
  if ! command -v uv &>/dev/null; then
    err "Failed to install uv. Install it manually: https://docs.astral.sh/uv/"
    exit 1
  fi
  ok "uv installed"
else
  ok "uv already installed"
fi

# --- Clone or update repo ---
if [[ -d "$INSTALL_DIR/.git" ]]; then
  info "Updating existing installation..."
  git -C "$INSTALL_DIR" pull --ff-only
else
  info "Cloning Escriba..."
  git clone "$REPO" "$INSTALL_DIR"
fi
ok "Source ready at $INSTALL_DIR"

# --- Install dependencies ---
info "Installing Python dependencies (this may take a few minutes the first time)..."
cd "$INSTALL_DIR"
uv sync
ok "Dependencies installed"

# --- Copy .env template if no .env exists ---
if [[ ! -f "$INSTALL_DIR/.env" ]]; then
  if [[ -f "$INSTALL_DIR/.env.example" ]]; then
    cp "$INSTALL_DIR/.env.example" "$INSTALL_DIR/.env"
    info "Created .env from template — edit $INSTALL_DIR/.env to add API keys"
  fi
fi

# --- Download pre-built Swift audio-capture CLI ---
SWIFT_BIN_DIR="$INSTALL_DIR/swift-audio-capture/.build/release"
SWIFT_BIN="$SWIFT_BIN_DIR/audio-capture"

if [[ -x "$SWIFT_BIN" ]]; then
  ok "Swift audio-capture CLI already present"
else
  info "Downloading pre-built audio-capture CLI..."
  mkdir -p "$SWIFT_BIN_DIR"

  ASSET_URL=$(curl -fsSL \
    -H "Accept: application/vnd.github+json" \
    "https://api.github.com/repos/Skalas/escriba/releases/latest" \
    | grep -o '"browser_download_url":\s*"[^"]*audio-capture-arm64-darwin\.tar\.gz"' \
    | head -1 \
    | sed 's/"browser_download_url":\s*"//;s/"$//')

  if [[ -z "$ASSET_URL" ]]; then
    err "Could not find audio-capture binary in the latest GitHub release."
    err "You can build it manually: cd $INSTALL_DIR/swift-audio-capture && swift build -c release"
    exit 1
  fi

  curl -fsSL "$ASSET_URL" | tar xz -C "$SWIFT_BIN_DIR"
  chmod +x "$SWIFT_BIN"
  ok "Swift audio-capture CLI downloaded"
fi

# --- Build .app bundle ---
info "Building Escriba.app..."
uv run python setup_app.py
ok "App built"

# --- Install to /Applications ---
APP_SRC="$INSTALL_DIR/dist/$APP_NAME.app"
APP_DST="/Applications/$APP_NAME.app"

if [[ -d "$APP_DST" ]]; then
  info "Replacing existing $APP_NAME.app..."
  rm -rf "$APP_DST"
fi

cp -r "$APP_SRC" "$APP_DST"
ok "Installed to $APP_DST"

# --- Done ---
echo ""
ok "Escriba is ready!"
echo ""
echo "  Open from:  /Applications/$APP_NAME.app"
echo "  Or run:     open /Applications/$APP_NAME.app"
echo ""
echo "  Config:     $INSTALL_DIR/escriba.toml"
echo "  API keys:   $INSTALL_DIR/.env"
echo "  Logs:       ~/Library/Logs/escriba/app.log"
echo ""
echo "  To update later:  cd $INSTALL_DIR && git pull && uv sync && uv run python setup_app.py && cp -r dist/$APP_NAME.app /Applications/"
echo ""
