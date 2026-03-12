#!/usr/bin/env bash
# Escriba installer — sets up everything on a fresh Mac.
# Usage: curl -fsSL <raw-url>/install.sh | bash
set -euo pipefail

REPO="https://github.com/Skalas/local-transcriber.git"
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
echo "  Config:     $INSTALL_DIR/local-transcriber.toml"
echo "  API keys:   $INSTALL_DIR/.env"
echo "  Logs:       ~/Library/Logs/local-transcriber/app.log"
echo ""
echo "  To update later:  cd $INSTALL_DIR && git pull && uv sync && uv run python setup_app.py && cp -r dist/$APP_NAME.app /Applications/"
echo ""
