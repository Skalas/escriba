# Changelog

All notable changes to Escriba are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html):
`feat` → minor, `fix` → patch, breaking → major.

## [Unreleased]

### Added
- **Reliable call detection / Notion-style auto-record** ([#39](https://github.com/Skalas/escriba/issues/39)–[#44](https://github.com/Skalas/escriba/issues/44)) — mic-activation auto-record is now usable end to end:
  - **Settings → Auto-record on call** toggle plus `start_mode` (Prompt / Automatic), cooldown, and start/stop debounces, all round-tripping through `[auto_record]` in `escriba.toml`. Still opt-in (`enabled = false` by default).
  - **Debounced detection** via a pure `CallStateMachine` fed by CoreAudio `is_mic_running()` — a sustained mic-on drives one start and a sustained mic-off one stop; brief blips (notifications, dictation) no longer trigger.
  - **Automatic mode** starts/stops recording without a blocking dialog and posts a non-blocking notification; both paths go through `try_start_recording` so there is never a second active session.
  - **Mic-gated app labels** — a meeting-app name is shown only when the mic is actually active and a known app matches, instead of guessing from background process presence.

### Fixed
- Auto-record was effectively dead: it defaulted to disabled with no `[auto_record]` config section and no Settings control, so the mic-activation loop never ran.
- `auto_record.poll_interval` now actually drives the menubar poll timer (was a hardcoded literal; the configured value was ignored).
- Auto start/stop no longer race the dashboard's HTTP start/stop (could previously start a recording while auto-stopping, or stop one while auto-starting).
- Debounce/cooldown values are clamped to non-negative on load.

## [0.6.1] - 2026-06-27

### Added
- **About / version panel** in Settings — version badge, git commit with a clean/uncommitted-changes indicator, active transcription backend/model, Python and platform info, the project directory, and a link to the repo. Backed by a new `GET /api/version` endpoint.

### Fixed
- Version is now sourced from `pyproject.toml` everywhere: `escriba.__version__` reads installed package metadata, and the `.app` bundle version is read from `pyproject.toml` at build time (both were stale at `0.1.0`).

## [0.6.0] - 2026-06-27

Three "better, not wider" features plus a rigorous interview-evaluation prompt.

### Added
- **Cross-session transcript search** ([#26](https://github.com/Skalas/escriba/issues/26)) — search across all sessions; click a result to open the session and jump to / highlight the segment.
- **Speaker labels** ([#27](https://github.com/Skalas/escriba/issues/27)) — name and edit speakers, persisted per session; renamed speakers appear in the transcript, AI notes, and exports.
- **Richer export / share** ([#28](https://github.com/Skalas/escriba/issues/28)) — Markdown bundle, copy-to-clipboard, save to `~/Downloads` (with the path reported), and per-segment deep links (`#session/<id>/seg/<id>`).
- **Interview Evaluation prompt** — a non-complacent quick-prompt template: evidence-demanding, demonstrated-vs-claimed, decisive hire/no-hire.
- **GFM table rendering** in AI notes.
- **Audio recovery** — sessions whose `audio_path` was empty are relinked to their on-disk WAV (`audio/<id>.wav`) on access and at startup; no recordings were lost.

### Fixed
- Notes generation is scoped to its session — a slow generation no longer bleeds into or sticks on another record after switching.
- Audio streaming handles client disconnect (seek/pause/close) quietly instead of logging `BrokenPipeError` and double-faulting on a 500.
- Re-transcribe shows a disabled state with a tooltip when a session has no audio (was hidden).
- Dark mode: clearer surface separation across all themes, and button/control text inherits the theme color (no near-black text on dark).
- Speaker display names are escaped (stored-XSS hardening on the rename feature); merge uses `INSERT OR IGNORE` against the segment-timing unique index.

## [0.5.0] - 2026-06-27

Transcription robustness ([#29](https://github.com/Skalas/escriba/issues/29)): the transcription path no longer silently degrades output.

### Added
- **Re-transcribe honors full config** — `retranscribe_from_wav` now applies the same dictionary/VAD/hallucination settings as live capture (via a shared `_build_transcriber`), so a re-transcription matches live behavior instead of using defaults.
- **Chunk-error resilience** — transient inference failures retry (bounded + backoff); persistent/fatal failures surface as `ChunkProcessingError` (logged) instead of being silently treated as silence.
- **Audio buffer backpressure** — the live, system, and mic PCM buffers are bounded (~2× chunk size); under load they drop oldest audio with a rate-limited warning instead of growing unbounded on long meetings.
- **Segment dedup** — a unique index on `(session_id, start_time, end_time)` + `INSERT OR IGNORE` prevents double-inserts on chunk retry; the migration de-duplicates any pre-existing rows before adding the index.

### Fixed
- "both" audio mode: system/mic buffers are now bounded and kept time-aligned (no source time-shift under backpressure), and a single empty source no longer discards the other source's audio.
- `merge_sessions` uses `INSERT OR IGNORE` so the new unique index can't roll back a merge on a colliding rebased timing.

## [0.4.0] - 2026-06-26

Reliability milestone ([Epic #12](https://github.com/Skalas/escriba/issues/12)): the core stops corrupting state under concurrent load and fails gracefully. See `docs/review/v0.4.0-review-findings.md` for the full review record.

### Added
- **Concurrency safety** — `AppState` guards shared state with an `RLock` (single-writer recording: exactly one active session, concurrent start → `409`); model-download state is atomic.
- **Threaded HTTP server** — `ThreadingHTTPServer` so long operations no longer block `/api/status` polling; recording start runs outside the lock.
- **Request hardening** — 1MB body cap (`413`), 30s socket timeout, JSON input validation, and correct status codes (`400`/`404`/`409`/`413`/`503`) with no stack traces leaked to clients.
- **LLM resilience** — 30s timeout on Gemini/Claude calls, retry with exponential backoff + jitter (3×) on `429`/`5xx` (never on 4xx auth), local-model cache eviction + retry on `MemoryError`/`RuntimeError`, and serialized `mlx-lm` calls (`Semaphore(1)`).
- **Database integrity** — `split_session`/`merge_sessions` run in a single transaction; **all** DB access is serialized on the shared connection so operations stay atomic during a live recording; `schema_version` table + idempotent migration runner; `idx_sessions_folder` / `idx_sessions_status` indexes.
- **Tests** — 84 tests (from 71), covering concurrency, atomicity, migration idempotency, HTTP status codes, and the record→stop lifecycle.
- Full mypy type-checking across the codebase; `ruff` + `mypy` added to the dev/CI gate.

### Fixed
- `{"prompt": null}` (and other null body fields) returned `500` instead of a structured `4xx`.
- "Generate Notes" with no prompt rendered raw JSON instead of formatted markdown.

## [0.3.0] - 2026-06-26

Cuts the accumulated feature work since `v0.2.0` as a proper minor release.

### Added
- **Theming system** — Ink Editorial (default), Indigo, and Graphite themes, with in-app modals and dashboard polish.
- **Customizable AI prompts** — user-editable system prompt, quick-prompt templates, and an Enhance-prompt action.
- **Local LLM provider** — on-device summaries and automatic session naming via `mlx-lm`.
- **Session split & merge** — with audio support across the operation.
- **Dynamic AI model selection** — model lists fetched from the provider APIs.
- **Custom dictionary** — improves transcription accuracy for domain terms.
- **Mic-activation detection** — auto-starts session naming on microphone activity.

### Fixed
- Recording-dependent UI now driven off tracked state instead of a stale badge.
- Responsive header buttons in the dashboard.
- A stack of launcher/spawn and dashboard UX fixes.

## [0.2.0]

Initial tracked release.

[Unreleased]: https://github.com/Skalas/escriba/compare/v0.6.0...HEAD
[0.6.0]: https://github.com/Skalas/escriba/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/Skalas/escriba/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/Skalas/escriba/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/Skalas/escriba/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/Skalas/escriba/releases/tag/v0.2.0
