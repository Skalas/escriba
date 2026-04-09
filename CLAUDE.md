# Escriba

macOS menu bar app for local audio transcription (system audio + microphone) using Whisper on Apple Silicon.

## Installing

```bash
make install
```

This builds the `.app` bundle via `setup_app.py` and copies it to `/Applications/Escriba.app`. Do NOT use `pip install -e .` or `uv pip install` — the app runs through `uv` from the project directory.

## Running in development

```bash
uv run escriba app
```

## Key commands

- `make install` — build .app bundle and copy to /Applications
- `make download-model` — pre-download the local LLM model
- `uv run escriba app` — start menu bar app + HTTP server (dev)
- `uv run escriba live-stream` — CLI streaming transcription
- `uv run escriba list-devices` — list audio input devices
- `uv run escriba download-model` — download local LLM model
- `uv run escriba daemon start` — start background transcription daemon
- `uv run escriba watch-calendar` — auto-transcribe on calendar meetings
- `uv run escriba create-issues` — create GitHub issues from transcript

## Architecture

```
src/escriba/
├── app/                  # Menu bar app + web dashboard
│   ├── menubar.py        # rumps menu bar integration
│   ├── server.py         # stdlib HTTP server (no framework), port 19876
│   ├── session.py        # TranscriptionSession lifecycle
│   ├── database.py       # SQLite: folders, sessions, segments
│   └── static/index.html # Single-file SPA dashboard (HTML+CSS+JS)
├── audio/                # Audio capture
│   ├── screen_capture.py # ScreenCaptureKit system audio
│   ├── live_capture.py   # Live + streaming capture
│   ├── mic_monitor.py    # Microphone monitoring
│   ├── call_detection.py # Detect active calls
│   └── device_detection.py
├── transcribe/           # Whisper backends
│   ├── streaming.py      # Generic streaming transcription
│   ├── streaming_mlx.py  # MLX-whisper backend (Apple Silicon)
│   ├── streaming_mps.py  # MPS backend
│   ├── whisper.py        # faster-whisper / openai-whisper
│   ├── formats.py        # Export formats (txt, json)
│   └── config.py         # Transcription config
├── summarize/
│   └── llm_summary.py    # AI notes: local (mlx-lm), Gemini, Claude
├── speaker/              # Speaker detection/diarization
├── daemon/               # Background daemon mode
├── calendar/             # Apple Calendar integration
├── integrations/         # GitHub issues from transcripts
├── notify/               # Telegram notifications
├── watch/                # Folder watcher for audio files
├── config.py             # AppConfig from escriba.toml
└── cli.py                # Typer CLI entry point
```

## Tech stack

- **Python 3.10+** managed with **uv** (never use pip directly)
- **Whisper backends**: mlx-whisper (default on Apple Silicon), faster-whisper, openai-whisper
- **Audio capture**: ScreenCaptureKit (system) + sounddevice (mic), Swift CLI helper
- **Menu bar**: rumps
- **Web dashboard**: stdlib `http.server`, single HTML file SPA
- **Database**: SQLite via stdlib `sqlite3`
- **AI summaries**: mlx-lm (local), google-genai, anthropic
- **Config**: `escriba.toml` (TOML), `.env` for API keys

## Dashboard (index.html)

Single-file SPA at `src/escriba/app/static/index.html`. Contains all CSS, HTML, and JS inline. Key design decisions:

- **No build step** — plain HTML/CSS/JS, no bundler or framework
- **Apple HIG-inspired** — SF Pro font, system colors, 12px radius cards
- **Dark mode** — auto via `prefers-color-scheme`, separate token set
- **SVG icons** — all icons are inline SVGs (Lucide-style), no emoji icons
- **Accessibility** — `aria-label` on icon buttons, `:focus-visible` rings, `prefers-reduced-motion`
- **Design tokens** — CSS custom properties for all colors, shadows, radii, transitions

### Dashboard views

1. **Empty state** — microphone SVG illustration, prompt to start recording
2. **Live view** — real-time transcript during recording, AI notes generation
3. **Session detail** — past session transcript, notes (markdown), audio player with segment sync
4. **Settings** — audio, transcription, speaker detection, AI model, keyboard shortcuts

### API endpoints (server.py)

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/api/status` | Recording status |
| GET | `/api/sessions` | List all sessions + folders |
| GET | `/api/sessions/:id` | Session detail + segments |
| GET | `/api/sessions/:id/audio` | Serve WAV with Range support |
| GET | `/api/config` | Current config + env key status |
| GET | `/api/models` | Available AI models |
| POST | `/api/recording/start` | Start recording |
| POST | `/api/recording/stop` | Stop recording |
| POST | `/api/sessions/merge` | Merge multiple sessions |
| POST | `/api/sessions/move` | Move sessions to folder |
| POST | `/api/sessions/:id/generate-notes` | AI-generate notes |
| POST | `/api/sessions/:id/retranscribe` | Re-transcribe from WAV |
| PUT | `/api/config` | Update config + env keys |
| PUT | `/api/sessions/:id/rename` | Rename session |
| DELETE | `/api/sessions/:id` | Delete session |
| POST/PUT/DELETE | `/api/folders/...` | Folder CRUD |

## Config

- `escriba.toml` in project root — all app settings
- `.env` for API keys (`GEMINI_API_KEY`, `ANTHROPIC_API_KEY`, `HUGGINGFACE_TOKEN`)
- Never put inline comments on `.env` values — python-dotenv parses them as part of the value

## Dependencies

Managed via `uv sync`. To add extras: `uv sync --extra <name>`. Never use `uv pip install`.
