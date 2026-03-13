## Contributing to Escriba

Thanks for helping improve **Escriba** — local system + mic transcription for macOS.

- **Users**: start with the main `README.md`.
- **Contributors**: this guide covers setup, dev loop, style, and release basics.

### Prerequisites

- **macOS 13+ (Ventura)** with ScreenCaptureKit enabled
- **Python 3.10+**
- **uv** for Python dependency management ([docs](https://docs.astral.sh/uv/))
- **Swift toolchain** — Xcode or Xcode Command Line Tools (`xcode-select --install`), needed to build the audio capture helper
- Optionally **Docker / docker-compose** for any future dev tooling you wire up

### One‑liner install (for testing the full flow)

To validate the installer path the way users will experience it:

```bash
curl -fsSL https://raw.githubusercontent.com/Skalas/escriba/main/install.sh | bash
```

This will:

- Install `uv` if it is missing
- Clone/update the repo into `~/.escriba`
- Run `uv sync`
- Download the pre-built Swift audio-capture binary from the latest GitHub release
- Build the `.app` bundle and copy `Escriba.app` into `/Applications`

### Local development setup

Clone the repo and set up dependencies with `uv`:

```bash
git clone https://github.com/Skalas/escriba.git
cd escriba
uv sync
```

Build the Swift audio capture helper:

```bash
cd swift-audio-capture
swift build -c release
cd ..
```

Create a local `.env` from the example if you need API keys for AI notes:

```bash
cp .env.example .env  # then edit .env
```

### Running the app from source

For a dev build (no `.app` bundle):

```bash
uv run escriba app
```

Then open the dashboard at `http://127.0.0.1:19876` or use the menu bar app if configured.

To build the full macOS app locally:

```bash
uv run python setup_app.py
cp -r dist/Escriba.app /Applications/
open /Applications/Escriba.app
```

### Tests

Run the test suite with:

```bash
uv run pytest tests/
```

When adding new code, prefer writing tests alongside it and keep them fast enough to run frequently.

### Code style and conventions

Escriba follows a few core conventions (enforced by project rules and existing code):

- **Type hints** everywhere, with `from __future__ import annotations` at the top of Python files
- **Logging** via the standard `logging` module with module‑level `logger = logging.getLogger(__name__)`
- **Environment variables** accessed through helper functions with validation and sensible defaults
- **File paths** via `pathlib.Path`, not `os.path`
- **Threading** with explicit locks/events for shared state, and `daemon=True` for background threads as appropriate

Before sending a PR:

- Run `uv sync` (if dependencies changed)
- Run `uv run pytest`
- Manually smoke‑test the app (start/stop recording, basic dashboard flows)

### Project layout (quick reference)

- `src/escriba/cli.py` — CLI entry point (`escriba ...`)
- `src/escriba/app/` — menu bar, HTTP server, DB, session management, static dashboard assets
- `src/escriba/audio/` — audio capture and device detection
- `src/escriba/transcribe/` — streaming transcription backends and VAD / hallucination config
- `src/escriba/summarize/` — LLM‑based summary generation
- `swift-audio-capture/` — Swift ScreenCaptureKit helper

### Opening pull requests

- Keep PRs **small and focused** when possible.
- Include a short **“Why”** and **“What changed”** in the PR description.
- Call out any **breaking changes**, new environment variables, or migrations.

### Future improvements for contributors

Some ideas you (or others) might pick up:

- Add a `make dev` or `uv run` shortcut for the full dev loop
- Docker / docker‑compose recipes for backend‑only development
- Automated checks (lint, tests) in CI

If you are unsure about anything, open a draft PR or an issue with your proposal — contributions of ideas and feedback are welcome too.

