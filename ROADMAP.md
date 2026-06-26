# Escriba Roadmap

> macOS menu bar app for local audio transcription (system audio + microphone) using Whisper on Apple Silicon.

This roadmap is a living document. It captures **where we are**, the **strategic priorities**, and the **planned milestones**. It is intentionally opinionated about sequencing: we harden the core before we widen the feature set.

_Last updated: 2026-06-26 · Current version: `0.3.0` · next milestone: `v0.4.0` (Epic #12)_

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
- Mic-activation detection + auto session naming
- A stack of dashboard UX and launcher/spawn fixes

**The gap:** core app modules have near-zero test coverage, shared state is largely unsynchronized, the HTTP server handles one request at a time, and LLM calls have no timeout/retry. Tracked in **[Epic #12: Backend hardening](https://github.com/Skalas/escriba/issues/12)**.

---

## Milestones

### `v0.3.0` — Ship what's done  ·  _shipped 2026-06-26_

Cut the accumulated feature work as a proper minor release.

- [x] Bump `pyproject.toml` → `0.3.0` (and any hardcoded version strings)
- [x] Update README / CHANGELOG with the feature list above
- [x] Commit `chore(release): v0.3.0`, tag `v0.3.0`, push tag

**Done when:** `v0.3.0` is tagged and pushed; `git describe` is clean. ✅

---

### `v0.4.0` — Reliability  ·  _Epic #12 (P0/P1)_

The core stops corrupting state under concurrent load and fails gracefully. **This is the priority milestone.**

**Concurrency & thread safety (P0)**
- [ ] Guard `app_state` with an `RLock` / single-writer session manager (`server.py:129`)
- [ ] Wrap `split_session`/`merge_sessions` in a lock + single transaction (`database.py`)
- [ ] Serialize all `mlx-lm` calls via a global semaphore (or subprocess) (`llm_summary.py:461`)
- [ ] Make model-download state atomic (`threading.Event`/lock) (`server.py:820`)

**HTTP server (P0/P1)**
- [ ] Move to `ThreadingHTTPServer` so long ops don't block `/api/status` polling
- [ ] Enforce body size cap (~1MB) + socket timeout (`server.py:375`)
- [ ] Input validation layer → reject bad bodies with `400`
- [ ] Correct status codes: `400`/`404`/`409` (already recording)/`503`

**LLM resilience (P0/P1)**
- [ ] Add `timeout=30` to Gemini/Claude calls (`llm_summary.py:530`)
- [ ] Retry with exponential backoff + jitter (3 tries) on `429`/`5xx`
- [ ] Evict local model cache on `MemoryError`/`RuntimeError`, retry once

**Tests (P0 — land alongside the fixes)**
- [ ] `test_database.py` — split atomicity, concurrent split+merge, idempotent migration
- [ ] `test_server.py` — concurrent start → one session, oversized body rejected, bad input → 400
- [ ] `test_session.py` — lifecycle, audio persisted on stop, mlx serialization

**Done when:** concurrent API calls during an active recording don't corrupt state or crash (proven by tests); LLM calls time out/retry instead of hanging the UI; the three core modules have meaningful coverage.

---

### `v0.5.0` — Depth on the core loop

Make the existing record → transcribe → summarize flow better, not wider. _Candidates — to be scoped:_

- [ ] Cross-session transcript search
- [ ] Speaker-label naming & editing (persist names across a session)
- [ ] Richer export / share (Markdown bundle, copy-to-clipboard, per-segment links)
- [ ] Re-transcribe respects full config (dictionary/VAD/hallucination) (`session.py:624`)
- [ ] Transcription robustness: recoverable vs fatal chunk errors, audio buffer backpressure, segment dedup

**Done when:** _TBD at planning time._

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
