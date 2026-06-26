# Changelog

All notable changes to Escriba are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html):
`feat` → minor, `fix` → patch, breaking → major.

## [Unreleased]

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

[Unreleased]: https://github.com/Skalas/escriba/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/Skalas/escriba/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/Skalas/escriba/releases/tag/v0.2.0
