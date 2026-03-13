## Escriba

Local audio transcription app for macOS. Captures system audio + microphone, transcribes in real-time using Whisper, and provides a web dashboard for managing sessions, generating AI notes, and playing back recordings.

- **For users**: this README shows how to install and use Escriba.
- **For contributors**: see `CONTRIBUTING.md` for dev setup and guidelines.

### Features

- **Real-time transcription** — system audio + microphone capture via ScreenCaptureKit
- **Menu bar app** — start/stop recording from the macOS menu bar
- **Web dashboard** — manage sessions, view transcripts, generate AI notes
- **Audio recording & playback** — WAV files saved alongside transcripts with seek-to-segment
- **Re-transcribe** — retry failed transcriptions from saved audio
- **AI notes** — generate summaries, action items, meeting minutes via Gemini or Claude
- **Speaker detection** — simple energy-based or pyannote diarization
- **Settings UI** — configure everything from the dashboard
- **100% local transcription** — Whisper runs on-device (mlx-whisper on Apple Silicon)

### Requirements

- macOS 13.0+ (Ventura) for ScreenCaptureKit
- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager
 
### Installation

#### One-liner install (recommended)

On a supported macOS machine, you can install Escriba with:

```bash
curl -fsSL https://raw.githubusercontent.com/Skalas/escriba/main/install.sh | bash
```

This will:

- Install `uv` if needed
- Clone/update Escriba into `~/.escriba`
- Install Python dependencies with `uv sync`
- Download the pre-built audio-capture binary
- Build `Escriba.app` and copy it into `/Applications`

#### Manual install

```bash
# Clone and install
git clone https://github.com/Skalas/escriba.git
cd escriba
uv sync

# Build the Swift audio capture CLI
cd swift-audio-capture
swift build -c release
cd ..

# Build and install the macOS app
uv run python setup_app.py
cp -r dist/Escriba.app /Applications/
```

Open **Escriba** from `/Applications` or Spotlight. The menu bar icon `〰` appears — click it to start recording or open the dashboard.

### Permissions

On first run, macOS will prompt for:
- **Screen Recording** — required for system audio capture
- **Microphone** — required for mic capture

### Usage

#### Menu Bar

- `〰` — idle, `〰●` — recording
- **Start/Stop Recording** — toggle transcription
- **Open Dashboard** — open the web UI
- **Reload Config** — apply config changes without restart

#### Dashboard

Open from the menu bar or navigate to `http://127.0.0.1:19876`.

- **Sidebar** — session list with status badges, multi-select for merge/delete
- **Live view** — real-time transcript during recording
- **Session detail** — view transcript, play back audio, generate AI notes, re-transcribe
- **Settings** — configure audio, transcription, speaker detection, AI, and advanced options

### Audio Playback

Sessions recorded with v0.1.0+ save a WAV file alongside the transcript:
- Audio player appears in session detail when audio is available
- Click any transcript segment to seek to that point in the audio
- Current segment highlights during playback
- ~1.9 MB/min (16kHz mono WAV)

### AI Notes

Generate notes from transcripts using Gemini or Claude. Built-in prompt templates:
- Executive Summary, Action Items, Decisions, Open Questions
- Key Points, Risks & Blockers, Meeting Minutes, Follow-up Email

Set API keys in Settings or `.env`:
```bash
GEMINI_API_KEY=your-key
ANTHROPIC_API_KEY=your-key
```

### Re-transcribe

Sessions with saved audio but failed transcripts (0 segments) can be re-processed:
click **Re-transcribe** in the session detail view.

### Configuration

#### TOML (recommended)

Create `escriba.toml` in the project root:

```toml
[audio]
audio_source = "both"     # "system" | "mic" | "both"
mic_boost = 1.3
sample_rate = 16000

[streaming]
model_size = "medium"     # tiny | base | small | medium | large
language = "auto"         # auto | en | es | fr | ...
backend = "mlx-whisper"   # mlx-whisper | faster-whisper | openai-whisper
chunk_duration = 15
vad_enabled = true
summary_model = "gemini"  # gemini | claude

[speaker]
mode = "none"             # none | simple | pyannote
threshold = 0.3

[vad]
min_silence_ms = 500
threshold = 0.3

[whisper]
condition_on_previous_text = false
no_speech_threshold = 0.6
compression_ratio_threshold = 2.4
logprob_threshold = -1.0
```

Settings can also be changed from the dashboard UI (Settings gear icon).

#### Environment variables

Alternatively, use `.env` or environment variables. TOML takes precedence.

### CLI

The app also provides CLI commands:

```bash
# Run the menu bar app (same as opening Escriba.app)
escriba app

# Live streaming transcription (headless)
escriba live-stream --output-dir transcripts

# List audio devices
escriba list-devices
```

### Architecture

```
src/escriba/
├── cli.py                    # CLI entry point (typer)
├── config.py                 # TOML + env config (frozen dataclasses)
├── app/
│   ├── menubar.py            # macOS menu bar (rumps)
│   ├── server.py             # HTTP API (stdlib http.server)
│   ├── database.py           # SQLite session/segment storage
│   ├── session.py            # Recording session + audio WAV writer
│   └── static/index.html     # Dashboard SPA
├── audio/
│   ├── screen_capture.py     # System audio (ScreenCaptureKit via Swift CLI)
│   └── device_detection.py   # Auto-detect audio devices
├── transcribe/
│   ├── streaming_mlx.py      # MLX Whisper backend (Apple Silicon)
│   ├── streaming.py          # faster-whisper backend
│   └── config.py             # VAD + hallucination config
└── summarize/
    └── llm_summary.py        # Gemini/Claude summarization
```

#### Data storage

| What | Location |
|------|----------|
| Database | `~/Library/Application Support/Escriba/transcriber.db` |
| Audio files | `~/Library/Application Support/Escriba/audio/` |
| Logs | `~/Library/Logs/escriba/app.log` |
| Config | `./escriba.toml` (project root) |

### Development (for contributors)

For development setup, tests, and project conventions, see `CONTRIBUTING.md`.

### License

See `LICENSE` for details.
