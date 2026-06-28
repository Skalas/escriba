# Escriba Roadmap

> macOS menu bar app for local audio transcription (system audio + microphone) using Whisper on Apple Silicon.

This roadmap is a living document. It captures **where we are**, the **strategic priorities**, and the **planned milestones**. It is intentionally opinionated about sequencing: we harden the core before we widen the feature set.

_Last updated: 2026-06-28 · Current version: `0.7.0` (reliable call detection — Notion-style auto-record) · next up: `v0.8.0` (finish hardening + unblock local generation)_

---

## Guiding principles

1. **Reliability over reach.** Escriba runs live during real meetings. A dropped or corrupted recording is worse than a missing feature.
2. **Local-first.** On-device transcription and summarization are the default. Cloud providers are opt-in, never required.
3. **No build step, no framework creep.** The dashboard stays a single-file SPA; the server stays stdlib-first until concurrency forces an upgrade.
4. **Test the core loop.** `server.py`, `database.py`, and `session.py` are the spine — changes there ship with tests.
5. **Depth before breadth.** Improve the existing record → transcribe → summarize loop before adding adjacent features.

---

## Where we are

The app is feature-rich. Since `v0.2.0` we shipped (unreleased):

- Theming system (Ink Editorial default, Indigo, Graphite) + in-app modals + dashboard polish
- User-customizable AI system prompt + quick-prompt templates + Enhance-prompt
- Local LLM provider via `mlx-lm` for on-device summaries and session naming
- Session split & merge with audio support
- Dynamic AI model selection (API-fetched lists)
- Custom dictionary for transcription accuracy
- Mic-activation detection (shipped disabled — enable via Settings → Auto-record on call) + auto session naming
- A stack of dashboard UX and launcher/spawn fixes

**The gap (closed in `v0.4.0`):** core app modules had near-zero test coverage, shared state was largely unsynchronized, the HTTP server handled one request at a time, and LLM calls had no timeout/retry. Addressed under **[Epic #12: Backend hardening](https://github.com/Skalas/escriba/issues/12)** — the core loop is now concurrency-safe, the server is threaded with input validation, LLM calls time out/retry, and `server.py`/`database.py`/`session.py` have meaningful coverage (84 tests).

---

## Milestones

### `v0.3.0` — Ship what's done  ·  _shipped 2026-06-26_

Cut the accumulated feature work as a proper minor release.

- [x] Bump `pyproject.toml` → `0.3.0` (and any hardcoded version strings)
- [x] Update README / CHANGELOG with the feature list above
- [x] Commit `chore(release): v0.3.0`, tag `v0.3.0`, push tag

**Done when:** `v0.3.0` is tagged and pushed; `git describe` is clean. ✅

---

### `v0.4.0` — Reliability  ·  _Epic #12 (P0/P1) · shipped 2026-06-26_

The core stops corrupting state under concurrent load and fails gracefully. **This was the priority milestone.**

**Concurrency & thread safety (P0)**
- [x] Guard `app_state` with an `RLock` single-writer (`AppState`); `start()` runs outside the lock so audio init doesn't block `/api/status`
- [x] Wrap `split_session`/`merge_sessions` in a lock + single transaction (`database.py`)
- [x] Serialize all `mlx-lm` calls via a global `Semaphore(1)` (`llm_summary.py`)
- [x] Make model-download state atomic (`try_begin_model_download`/`finish`)
- [x] **DB1 (found in review):** serialize *all* DB access on the single shared connection — the per-op lock alone didn't make split/merge atomic vs concurrent `add_segments` during a live recording

**HTTP server (P0/P1)**
- [x] Move to `ThreadingHTTPServer` so long ops don't block `/api/status` polling
- [x] Enforce body size cap (1MB → 413) + socket timeout
- [x] Input validation layer → bad bodies return `400`/structured errors (incl. JSON-null fields, caught in smoke)
- [x] Correct status codes: `400`/`404`/`409`/`413`/`503`; no stack traces to clients

**LLM resilience (P0/P1)**
- [x] `timeout=30` on Gemini/Claude calls (SDK timeout + `concurrent.futures` backstop that actually unblocks the caller)
- [x] Retry with exponential backoff + jitter (3 tries) on `429`/`5xx`; never on 4xx auth
- [x] Evict local model cache on `MemoryError`/`RuntimeError`, retry once

**Tests (P0 — landed alongside the fixes)**
- [x] `test_database.py` — split atomicity, concurrent split+merge, idempotent migration, split‖add_segments
- [x] `test_server.py` — concurrent start → one session, oversized body rejected, bad input → 4xx (incl. null fields)
- [x] `test_session.py` — lifecycle, audio persisted on stop, mlx serialization, notes markdown

**Done when:** concurrent API calls during an active recording don't corrupt state or crash (proven by tests); LLM calls time out/retry instead of hanging the UI; the three core modules have meaningful coverage. ✅

**Also in this sprint:** whole-codebase mypy typing pass (0 errors); migration verified idempotent on a real 39-session/10.9k-segment DB; smoke fixes (`{"prompt": null}` → 500; raw-JSON notes → markdown). Review record: `docs/review/v0.4.0-review-findings.md`. Deferred polish filed as #30–#33.

---

### `v0.5.0` — Transcription robustness  ·  _shipped 2026-06-27 (#29)_

Hardened the transcription path so it never silently degrades output (the first "depth on the core loop" slice).

- [x] Re-transcribe respects full config (dictionary/VAD/hallucination) — shared `_build_transcriber` so live + re-transcribe can't drift
- [x] Recoverable vs fatal chunk errors — transient inference failures retry (bounded); fatal surface as `ChunkProcessingError` instead of silent silence
- [x] Audio buffer backpressure — live/system/mic buffers bounded (~2× chunk), drop oldest + warn instead of unbounded growth
- [x] Segment dedup — `(session_id, start_time, end_time)` unique index + `INSERT OR IGNORE`; migration de-dups existing rows safely

**Done when:** re-transcribe honors all config; transient chunk failures retry instead of silently dropping audio. ✅ (100 tests; review caught & fixed a merge/unique-index break, a both-mode mix desync, and an empty-source audio-discard.)

---

### `v0.6.0` — Search, speakers, export, interview prompt  ·  _shipped 2026-06-27_

Three "better, not wider" features + a rigorous interview-evaluation prompt.

- [x] Cross-session transcript search (#26) — search across all sessions, click-to-jump + highlight
- [x] Speaker-label naming & editing — persist names per session, reflected in transcript/notes/export (#27)
- [x] Richer export / share — Markdown bundle, copy-to-clipboard, save-to-Downloads, per-segment deep links (#28)
- [x] Non-complacent **Interview Evaluation** prompt (evidence-demanding, decisive hire/no-hire)

**Also (surfaced in smoke):** recovered orphaned audio (relink canonical WAV when `audio_path` was empty — no data lost); GFM table rendering in notes; notes-generation scoped to its session (no cross-record bleed); audio-stream client-disconnect handled (no BrokenPipe spam); dark-mode surface separation + button/control text inherit theme color; markdown-table/XSS escaping on speaker names.

**Done when:** the three features work end-to-end and the interview prompt yields a critical evaluation. ✅ (130 tests; review caught a stored-XSS via the rename feature.)

---

### `v0.7.0` — Reliable call detection (Notion-style auto-record)  ·  _shipped 2026-06-28 (#45)_

Make mic-activation auto-record actually usable: opt-in from Settings, debounced start/stop, and mic-gated app labels instead of background process heuristics. **Root cause it fixed:** auto-record shipped with `enabled` defaulting to `false`, no `[auto_record]` section in config, and no Settings toggle — so the detector never ran.

- [x] **Config + dashboard toggle** ([#39](https://github.com/Skalas/escriba/issues/39)) — `[auto_record]` keys round-trip through `escriba.toml` and Settings (`enabled`, `start_mode`, debounce/cooldown).
- [x] **Debounce state machine** ([#40](https://github.com/Skalas/escriba/issues/40)) — pure `CallStateMachine`; no raw edge flapping.
- [x] **Self-aware mic signal** ([#42](https://github.com/Skalas/escriba/issues/42)) — per-process audio API (macOS 14.4+) counts only *other* processes on the default input device, so Escriba's own capture doesn't pin auto-stop on and always-on daemons (`corespeechd`/Siri) are ignored; app label only when a meeting app actually holds the mic.
- [x] **Menubar integration** ([#41](https://github.com/Skalas/escriba/issues/41)) — prompt or auto-start via `try_start_recording` (single-writer); auto-stop on debounced call end, bound to the auto-started session (a hand-started recording keeps running); non-blocking notifications.
- [x] **Tests** ([#43](https://github.com/Skalas/escriba/issues/43)) — T1–T6 state machine, config round-trip, auto-stop gating, signal fallback.
- [x] **Docs** ([#44](https://github.com/Skalas/escriba/issues/44)) — roadmap, CLAUDE.md `[auto_record]` keys.

**Done when:** auto-record is enableable from Settings; sustained mic-on/off drives one start/stop cycle; `uv run pytest` green. ✅ (140 tests; auto-stop signal validated live. Real-call auto-start/stop confirmed in use.)

---

### `v0.8.0` — Finish hardening + unblock local generation  ·  _next up_

Close out **[Epic #12](https://github.com/Skalas/escriba/issues/12)** and fix the one reliability issue that's actually felt in use.

- [ ] **Subprocess inference (#36)** — move local `mlx-lm` generation to a subprocess so it can't starve the HTTP server; the dashboard stays responsive during note generation. _(headline)_
- [ ] **Observability — Epic #12 §8** — structured logging with `session_id`/durations, request correlation IDs, latency metrics (transcription, LLM-by-provider, handler P50/P99).
- [ ] **Config validation — Epic #12 §6** — `AppConfig.validate()`/bounds checks, hot-reload coordination, `prompts.templates` tuple/list consistency.
- [ ] **Remote model-probe hygiene (#33)** — stop error-spamming Gemini/Claude `/api/models` when keys are absent/invalid; cache results.

**Done when:** local note generation no longer blocks the dashboard; errors carry structured, traceable logs (the last unmet Epic #12 "Done when") → **Epic #12 closeable**.

---

### `v0.9.0` — Frontend quality + UX polish

Close the testing gap the single-file SPA exposed (this session's smoke caught XSS, table rendering, stale-state, dark-mode, and black-text bugs — none caught by the Python suite), then refine UX.

- [ ] **Frontend test harness (new)** — automated coverage for the SPA's risky logic (markdown/table rendering, `escHtml`/escaping, deep-link parsing, notes-generation state scoping). Highest-leverage quality investment given ~2,800 lines of untested JS.
- [ ] **Arrows / navigation (#37)** — keyboard nav + player controls (scope TBD with details).
- [ ] **Design cleanup (#31)** — encapsulation/DRY in `server.py`/`llm_summary.py` (D1–D4, D6).
- [ ] **Test depth (#32)** — lock-hold latency + on-the-wire HTTP dispatch (TG1/TG2).
- [ ] **Body-size cap for chunked requests (#30)** — close the W5 hardening loose end.

**Done when:** the SPA has a real test harness covering the bug classes smoke found; UX navigation is solid.

---

## Backlog (P2 — ride along opportunistically)

Not a milestone of its own; pull these in when adjacent work makes them cheap.

- **Persistence:** indexes (`idx_sessions_folder`, `idx_sessions_status`), `schema_version` table + migration runner, denormalized `segment_count`, batched segment writes
- **Config:** bounds checking (`__post_init__`/`validate()`), hot-reload coordination, `prompts.templates` tuple/list consistency
- **Observability:** structured logging (`structlog`) with `session_id`/durations, request correlation IDs, latency metrics
- **Type safety:** complete handler return types, typed response models, narrow broad `except Exception`
- **Streaming summaries** for long transcripts

---

## How we work

- Versioning follows **semver**: `feat` → minor, `fix` → patch, breaking → major.
- Each milestone's P0 items become standalone GitHub issues linked to Epic #12 (or its successor).
- Releases are cut with: bump `pyproject.toml` → `chore(release): vX.Y.Z` → tag → push.
- This document is updated when a milestone ships or priorities change.
