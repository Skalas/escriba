# Changelog

All notable changes to Escriba are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html):
`feat` → minor, `fix` → patch, breaking → major.

## [Unreleased]

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

[Unreleased]: https://github.com/Skalas/escriba/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/Skalas/escriba/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/Skalas/escriba/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/Skalas/escriba/releases/tag/v0.2.0
