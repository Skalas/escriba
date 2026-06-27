# Escriba Roadmap

> macOS menu bar app for local audio transcription (system audio + microphone) using Whisper on Apple Silicon.

This roadmap is a living document. It captures **where we are**, the **strategic priorities**, and the **planned milestones**. It is intentionally opinionated about sequencing: we harden the core before we widen the feature set.

_Last updated: 2026-06-26 · Current version: `0.4.0` (Epic #12 shipped) · next milestone: `v0.5.0` (depth on the core loop)_

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

### `v0.5.0` — Depth on the core loop  ·  _next up_

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
