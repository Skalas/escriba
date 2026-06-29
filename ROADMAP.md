# Escriba Roadmap

> macOS menu bar app for local audio transcription (system audio + microphone) using Whisper on Apple Silicon.

This roadmap is a living document. It captures **where we are**, the **strategic priorities**, and the **planned milestones**. It is intentionally opinionated about sequencing: we harden the core before we widen the feature set.

_Last updated: 2026-06-28 · Current version: `0.10.1` (Notepad flow — jot & enhance) · next up: `v1.0.0` (release hardening)_

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

### `v0.8.0` — Finish hardening + unblock local generation  ·  _Epic #12 closeout · shipped 2026-06-28_

Closed out **[Epic #12](https://github.com/Skalas/escriba/issues/12)** and fixed the one reliability issue that was actually felt in use.

- [x] **Subprocess inference (#36)** — local `mlx-lm` generation runs in a dedicated single-worker subprocess, so it can't starve the threaded HTTP server; `/api/status` polling and navigation stay responsive during note generation. Crash/timeout degrades gracefully. _(headline)_
- [x] **Observability — Epic #12 §8** — structured logging with `session_id`/durations, per-request correlation IDs (`X-Correlation-ID` response header), latency metrics (transcription, LLM-by-provider, handler P50/P99) via new `app/observability.py`.
- [x] **Config validation — Epic #12 §6** — `AppConfig.validate()` bounds checks raising a field-named `ConfigValidationError`; `PUT /api/config` validates in a temp copy first so a rejected save can't corrupt `escriba.toml`; `prompts.templates` tuple/list consistency.
- [x] **Remote model-probe hygiene (#33)** — `/api/models` caches results, only probes a provider when its key is present, and downgrades invalid-key failures to `warning` instead of error-spamming.

**Done when:** local note generation no longer blocks the dashboard; errors carry structured, traceable logs (the last unmet Epic #12 "Done when"). ✅ (178 tests; review caught & fixed a config-corruption blocker, a log-injection vector, a traceback secret-leak, and a hollow subprocess-responsiveness test. Backend DoDs proven by live smoke; the real meeting loop stays the human UX check.)

**Deferred to v0.9.0 / [#31](https://github.com/Skalas/escriba/issues/31):** review leftovers — reuse `observability.timed()` for the three inline LLM-timing sites; make the module-level models cache lock-guarded (reset `_models_cache_time` on invalidate); hoist the deferred `observability` imports in `server.py`; `LatencyStore.snapshot()` per-key atomicity; cosmetic dead `future.cancel()` + stale cache-lock comment.

---

### `v0.9.0` — Frontend quality + UX polish  ·  _shipped 2026-06-28_

Closed the testing gap the single-file SPA exposed (earlier smoke caught XSS, table rendering, stale-state, dark-mode, and black-text bugs — none caught by the Python suite), then refined UX.

- [x] **Frontend test harness (new, #52)** — Playwright-driven pytest harness serving the real `index.html` against headless Chromium; 24 browser tests covering escaping/XSS, GFM tables, deep-link parsing, notes-generation session scoping, dark-mode legibility. No bundler, single-file SPA preserved; `playwright` is a dev-only dep.
- [x] **Arrows / navigation (#37)** — keyboard nav (Arrow Up/Down, Enter/Space) + player controls (Space play/pause, Arrow Left/Right seek ±5 s); focusable session items with `aria-label` + `:focus-visible`, ignored while typing.
- [x] **Design cleanup (#31)** — `observability.timed()` reused across LLM timing sites; atomic `snapshot()`; lock-guarded `/api/models` cache; hoisted imports.
- [x] **Test depth (#32)** — TG1 lock-hold latency + TG2 on-the-wire HTTP dispatch.
- [x] **Body-size cap for chunked requests (#30)** — `_read_body_bytes()` stream-and-counts → 413; closed the W5 loose end.

**Done when:** the SPA has a real test harness covering the bug classes smoke found; UX navigation is solid. ✅ (211 tests; review caught & fixed a Space-key double-fire and two hollow notes-scoping tests; T5 413 confirmed by live raw-socket smoke.)

**Deferred to a #31 follow-up:** migrate `_end_request` to `timed()`; dedup the percentile formula (`snapshot()` vs `percentile()`); extract shared `test_spa.py` helpers (session-setup + page fixtures) for v0.10.0 reuse; name the `SEEK_STEP_SECONDS` magic number.

---

### `v0.10.0` — Live Notepad + Export decoupling  ·  _shipped 2026-06-28_

Last feature sprint before 1.0. Two scoped features (HOLD mode).

- [x] **Live Notepad / note steering ([#53](https://github.com/Skalas/escriba/issues/53))** — notepad `<textarea>` captures notes live; `user_notes` persisted on the session and injected via a `{user_notes}` placeholder (back-compat fallback) on stop, re-generate, and live generation. Capture-during, inject-at-generation.
- [x] **Knowledge Adapters — port + `local-markdown` MVP ([#54](https://github.com/Skalas/escriba/issues/54))** — `KnowledgeStore` port + `local-markdown` default adapter via `[knowledge_store]`, reusing the v0.6.0 Markdown formatter; path-sanitized filenames, graceful export-failure degradation, default stays local. `webhook` + `custom-script` deferred to a fast-follow.

**Done when:** notes steer the generated summary and survive re-generation; sessions export to local Markdown via a pluggable adapter with the default staying fully local. ✅ (221 tests; review caught & fixed a live-path notes drop; path-traversal sanitization + graceful degradation confirmed by live smoke.)

**Deferred to a follow-up (design debt):** the `local-markdown` adapter imports the formatter from `app/server.py` (infra→presentation layer inversion) — move it to a shared module; route provider dispatch through a factory rather than a bare string check; remove the dead inner `except` in `_build_custom_prompt`; dedup `_make_handler` into `tests/conftest.py`. **Fast-follow feature:** `webhook` + `custom-script` knowledge adapters (env-var secrets, argv-not-shell, stdlib HTTP).

#### `v0.10.1` — Notepad flow (jot & enhance)  ·  _shipped 2026-06-28_

UX patch on top of 0.10.0 (#57): notepad-primary live view, transcript demoted to a collapsible toggle, a single "Enhance notes" action with optional instructions, in-place accessible provenance (rail + "AI" chip + label, not color-alone), and a unified saved-session note via a shared `buildProvenanceHtml`. Review fixed a non-closable disclosure, an AI-chip contrast failure across themes, a reduced-motion spinner gap, and a dropped-user-notes case. Visual fidelity confirmed light + dark.

**Deferred polish:** post-enhance, the editable notepad textarea and the rendered output both carry a "Your notes" heading (minor redundancy) — candidate for a quick follow-up (drop the in-output label, or transform the notepad in place like the mockup).

---

### `v1.0.0` — Release hardening  ·  _next up_

No new features — release-readiness only.

- [ ] Real-meeting soak across the record → transcribe → summarize loop.
- [ ] Clean install-from-scratch verification (one-liner installer → `/Applications`).
- [ ] Docs/onboarding pass; version-string + `uv.lock` audit.
- [ ] Triage the remaining P2 backlog (persistence indexes, schema versioning, typing) — pull in only what release quality demands.

**Done when:** a clean install runs a real meeting end-to-end without manual intervention; docs match behavior; version metadata is consistent.

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
