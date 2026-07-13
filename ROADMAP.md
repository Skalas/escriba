# Escriba Roadmap

> macOS menu bar app for local audio transcription (system audio + microphone) using Whisper on Apple Silicon.

This roadmap is a living document. It captures **where we are**, the **strategic priorities**, and the **planned milestones**. It is intentionally opinionated about sequencing: we harden the core before we widen the feature set.

_Last updated: 2026-07-13 · Current version: `1.2.0` (working tree: session-scoped notes + local-LLM timeout split + capture decomposition; closes #125/#108/#103 via #126–#135) · next up: real-meeting soak + clean-install verification, then calendar-driven recording product decision (#64)._

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

As of **`1.2.0`** (2026-07-08) the stop/finalization path is exception-safe, live capture buffer handling and faster-whisper resampling are hardened, dashboard note writes are guarded against stale async responses, daemon IPC is single-writer with owner-only socket permissions, and the previously silent automation paths now either work or fail honestly.

As of the **`1.3.0` sprint** (2026-07-13, unreleased until tagged) live notepad/notes-output are session-scoped across every start route and view transition (#125 / T1–T4); local inference separates model-load vs generation deadlines atomically (#108 / T5–T7); and `run_streaming_capture` is decomposed into `CaptureSupervisor` + `ChunkPump` with unit coverage (#103 / T8–T10). The remaining release-quality proof is still human-run: a real-meeting soak and a clean install-from-scratch.

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

### `v0.12.0` — Harden & shrink the core (pre-1.0)  ·  _shipped 2026-06-30_

A structural sprint before 1.0: shrink the surface and harden the request spine, with a batch of notes-flow fixes folded in from smoke. Mode HOLD (the `do_POST` spine governs the rigor).

- [x] **Decompose `do_POST` (#67)** into per-route dispatch; isolate route handlers. **Harden route parsing (#68)** → malformed paths return `404`/`400`, same two-segment guard applied to `do_PUT`. **Tests (#69).**
- [x] **Pay down layer-inversion + adapter debt (#70/#71/#72)** — export builders moved to `transcribe/formats.py`; `knowledge` no longer imports `app`; provider selection via a lazy factory; dead inner `except` removed; `_make_handler` deduped into `tests/conftest.py`.
- [x] **Remove dead code (#73/#74)** — deprecated MPS backend (~369 LOC) deleted; `openai-whisper` warns + falls back. _(The daemon signal handler flagged as "dead" was **not** — it's an OS callback; review caught the false positive and it was restored.)_
- [x] **Gate green (#75)** — ruff + mypy + pytest clean (276 tests).
- [x] **Folded in from smoke:** live-enhanced notes persist across stop; new recording resets the notes panel; backend-fallback no-transcriber bug fixed (review catch); **edit your own notes on a saved session** (new `POST /api/sessions/:id/user-notes` + dual-field editor); edit-box overflow contained.

**Done when:** the dispatcher is decomposed and malformed paths rejected; the layer inversion + adapter debt is paid down; dead code is gone; the full gate is green — all proven by tests. ✅ (Review caught a daemon-shutdown regression, a factory eager-construction crash, the layer inversion only being half-fixed, and a backend-fallback no-transcriber bug; smoke surfaced and folded in the notes-persistence, stale-panel, and edit-your-notes fixes. A claimed `null`-corruption blocker was adversarially rejected as a false positive.)

**Deferred (with triggers — read these at next discover):**
- **`/api/recording/start` failure handling** — the SPA's `if (!res.ok)` can throw if `apiCall` returns `null`/an HTTP error, bypassing the start flow. _Trigger:_ when `apiCall` is hardened to throw on 4xx/5xx, or a start failure is reported in use. → v1.0.0 hardening.
- **`saveNotes()` has no per-call error attribution** (`Promise.all` of the notes + user-notes POSTs). _Trigger:_ same `apiCall` hardening.
- **Saved-session generate path doesn't auto-persist** like the live path (`_generate_session_notes` returns without saving `notes_text`). _Trigger:_ users report regenerated notes lost on the saved-session view.
- **`/api/version` exposes the absolute `project_dir`** path. _Trigger:_ v1.0.0 security pass, or any plan to expose the dashboard beyond localhost.

---

### `v1.0.0` — Release hardening  ·  _shipped 2026-06-30 (#77–#85)_

No new features — release-readiness only. This sprint shipped the code-side hardening and cut the `1.0.0` version; the two manual-verification items remain as a `1.0.x` gate.

- [x] **Graceful failure spine (T1/T2/T3).** The dashboard's single `apiCall` chokepoint never throws — HTTP 4xx/5xx, network errors, and non-JSON bodies become a structured `{ok,error,status}` result; success is normalized so existing `res.ok` sites are unaffected. Start-recording failures can't leave a half-started UI; saving notes attributes which write failed.
- [x] **Path-disclosure pass (T4).** `/api/version` dropped the absolute `project_dir`; the export endpoint dropped its absolute `path`; model-load / mic / split / model-download / export errors return static text to the UI (detail logged server-side only). (CWE-209.)
- [x] **Notes single-writer (T5).** Saved-session generate no longer double-persists (the SPA is the sole writer there, preserving existing notes); a duplication/clobber race is gone.
- [x] **CI-safe tests (T6).** The swift `audio-capture` integration tests skip gracefully with no input device / no built executable — `uv run pytest` no longer hangs headless.
- [x] **Version + lock audit (T7/T9).** `1.0.0` unified across `pyproject.toml`, `__init__.py`, `uv.lock`; install docs verified against `install.sh` / Makefile / `setup_app.py` (T8).
- [ ] Real-meeting soak across the record → transcribe → summarize loop. _(manual; 1.0.x gate)_
- [ ] Clean install-from-scratch verification (one-liner installer → `/Applications`). _(manual; 1.0.x gate)_
- [ ] Triage the remaining P2 backlog (persistence indexes, schema versioning, typing) — pull in only what release quality demands.

**Done when:** a clean install runs a real meeting end-to-end without manual intervention; docs match behavior; version metadata is consistent. ✅ for the code/metadata half (283 tests; review caught & rejected a false-positive "live-path duplication" blocker after confirming the live path is single-writer, and folded in five extra path-disclosure fixes); the soak + clean-install proofs remain a human-run 1.0.x gate.

**Deferred (with triggers — read these at next discover):**
- **Concurrent background note-generation race** — `appendNotesToSession` does a read-then-write (read existing `notes_text`, write the concatenation); two rapid background "Enhance notes" calls on the same session can lose one append. Mitigated today by the per-session in-flight button guard. _Trigger:_ users report lost/overwritten notes from rapid re-generation, or any move to generate notes for multiple sessions concurrently. → needs a server-side atomic `append-notes` endpoint.
- **Config-validation errors echo the submitted value** — `PUT /api/config` returns `ConfigValidationError` text verbatim (e.g. the rejected value). Acceptable for a localhost validation API and useful UX; the path-embedding parse-error subcase already falls through to logged-only. _Trigger:_ any plan to expose the dashboard beyond localhost.

---

### `v1.0.1` — Release-blocker hardening  ·  _shipped 2026-07-04 (#88–#104)_

A post-1.0.0 full-repo review filed 6 P0 + 6 P1 correctness/reliability/security bugs the release-readiness sprint hadn't covered. HOLD mode — fix the release-blocker surface, don't widen it. Three strands:

- [x] **Audio-capture correctness.** 32-bit PCM scaled by the correct bit-depth divisor (#88); the transcript clock advances on a failed chunk so later timestamps don't drift (#90); the Swift `PCMConverter` guards NaN/Inf before `Int16()` (#92); `both`-mode chunks each stream by its own rate and resamples system→mic before mixing (#104).
- [x] **Recording lifecycle.** `TranscriptionSession.start()` failure releases the subprocess/WAV and errors the DB row (#89); `monitor_swift_cli` stops capture before dropping it on give-up (#91); `ScreenCaptureAudioCapture` start/restart is single-spawn and `stop()`/`restart()` always reap the child (#102); a timed-out local inference kills the GPU worker (#100); the daemon proves socket liveness and the CLI stops throwing tracebacks after a crash (#98).
- [x] **Web-security pass.** CSRF guard (Content-Type + Origin + Host) on all mutating routes (#93); `escJsAttr` closes the inline-handler stored-XSS (#95); `.env` newline injection rejected (#94); Telegram token redacted from logs (#96); whisper argv-injection hardening for watched filenames (#101).

**Done when:** the core loop produces correct, in-sync output; start/stop/failure leaks nothing; no state-changing endpoint accepts a cross-origin request — all proven by tests. ✅ for the code half (307 tests; review applied a whisper double-substitution fix + DRY/naming cleanups and re-verified with 0 blockers; a live-server smoke confirmed the CSRF guard returns 415/403/421 end-to-end). The soak + clean-install proofs remain the same human-run 1.0.x gate.

**Deferred (with triggers — read these at next discover):**
- **No Swift unit-test target.** T3 (#92 NaN guard) is covered by a source-regression guard + `swift build`, not a real XCTest. _Trigger:_ any further Swift change to `PCMConverter`/`CoreAudioTap`, or a second Swift bug — stand up an XCTest target (needs the converter moved to a testable library target).
- **`_abort_start` uses an exception-funnel for cleanup.** Adds one indentation level to `start()`. Acceptable for now (reviewer: no change needed). _Trigger:_ if `start()` grows another capture mode, extract `_start_transcriber()`/`_start_system_capture()`/`_start_mic_capture_guarded()` so the body flattens.

---

### `v1.2.0` — Reliability, correctness, and automation hardening  ·  _shipped 2026-07-08 (#97/#99/#110–#123)_

Merged the fresh full-repo defect slate into one HOLD sprint so the real-meeting loop is safer before the manual soak.

- [x] **Stop/finalization data safety.** `TranscriptionSession.stop()` now isolates cleanup-step failures, avoids concurrent flush/close when the process thread outlives the stop timeout, skips title refinement while a prior local generation is still alive, and app quit no longer closes the DB under an in-flight stop (#114/#115).
- [x] **Live audio correctness.** System/mic buffers are lock-protected, Swift restart backoff is stop-aware, faster-whisper resamples ndarray input to 16 kHz, malformed WAV parameters are rejected, unsupported MLX sample widths fail closed, and SRT cue numbers stay contiguous (#110/#111/#113/#116/#123).
- [x] **Dashboard note safety.** Saved-session selection, note generation, re-transcription, transcript polling, and pending highlight state now guard against stale async completions and resets (#120/#121/#123).
- [x] **Daemon and automation.** Daemon start/stop is single-writer, command reads are complete, socket permissions are owner-only, GUI MLX fallback degrades to faster-whisper, Calendar watch parses events while `--auto-start` fails clearly, watch-folder handles atomic moves/retries, Telegram sends plain text, and menubar auto-stop only clears tracking after stop begins (#64/#97/#99/#117/#118/#119/#122).
- [x] **API/config polish.** Range, chunked body, move-folder, merge-missing-session, TOML `null`, and local-summary-timeout edge cases now return clear behavior instead of hidden 500s or stale state (#123).

**Done when:** `uv run ruff check .`, `uv run mypy .`, and `uv run pytest` are green; smoke launches `uv run escriba app` and confirms `/api/status`, `/api/version`, and `/api/sessions` return valid JSON. ✅ (315 tests; smoke passed on localhost. Human real-meeting soak and clean-install verification remain manual gates.)

**Deferred (with triggers — read these at next discover):**
- **Calendar auto-start remains intentionally unavailable.** Calendar event detection now parses upcoming events, but automatic recording from calendar events is still blocked with a clear CLI error. _Trigger:_ product decision to support calendar-driven recording beyond mic-activation auto-record.
- **Swift unit-test target remains absent.** Same trigger as v1.0.1: any further Swift `PCMConverter`/`CoreAudioTap` change or another Swift bug should justify moving the converter into a testable library target.

---

### `v1.3.0` — Session-scoped notes + local-LLM timeouts + capture decomposition  ·  _unreleased (#126–#135; parents #125/#108/#103)_

HOLD sprint (REDUCE on the capture spine). No product-surface expansion.

- [x] **Session-scoped live notes (#125 / T1–T4).** Notepad + notes-output reconcile on every displayed `session_id` change (poll, showLiveView, all start routes); autosave posts `session_id` and the server rejects mismatches; Enhance/generate refuse cross-session writes; debounced-save and slow-fetch races closed.
- [x] **Local inference load vs generation timeout (#108 / T5–T7).** Atomic `_subprocess_run_inference` worker job with separate load/generation deadlines, parent grace for IPC skew, RLock to avoid timeout-reset deadlock.
- [x] **Decompose `run_streaming_capture` (#103 / T8–T10).** `CaptureSupervisor` + `ChunkPump` extracted; mic-only pop retains audio when partial system data is buffered; unit tests on the new seams; behavior-preserving.

**Done when:** T1–T10 green; `uv run ruff check .` + `uv run pytest` green (340 tests); smoke probes `/api/status`, `/api/version`, `/api/sessions`. ✅ code half. Human soak + clean-install remain the 1.0.x gate.

**Deferred (with triggers — read these at next discover):**
- **Calendar auto-start / Up-next spike (#64).** Still intentionally unavailable beyond mic-activation auto-record. _Trigger:_ product decision after soak.
- **Swift unit-test target.** Unchanged. _Trigger:_ further Swift `PCMConverter`/`CoreAudioTap` change or another Swift bug.
- **Further split `CaptureSupervisor` stderr-drainer.** Review declined this sprint (REDUCE scope). _Trigger:_ adding another process-lifecycle concern to `CaptureSupervisor`, or a bug that needs isolated stderr-drain tests.
- **Server-side atomic `append-notes`.** Concurrent background Enhance still mitigated by the in-flight button guard. _Trigger:_ users report lost/overwritten notes from rapid re-generation, or multi-session concurrent generate.

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
