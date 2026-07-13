# Human soak + clean-install checklist (T5–T7)

> Fill during smoke/aftercare. Do not mark complete without real execution.

## Environment

- **Date:** 2026-07-13
- **Tester:** automated smoke (API) + pending human UX
- **Build:** branch `sprint/auto-rename-download-v1.3.0` / version `1.3.0`
- **Install path:** dev (`uv run escriba app`)

## Automated smoke (2026-07-13)

| Check | Result |
|-------|--------|
| `/api/status` | ok, idle |
| `/api/version` | ok, version `1.3.0` |
| `/api/sessions` | ok (74 sessions) |
| `test_split_title_regeneration` + `test_model_download_service` | 19 passed |
| Full suite | 359 passed (ruff + mypy + pytest) |

## T5 — Real-meeting soak (record → transcribe → summarize)

| Step | Pass? | Notes |
|------|-------|-------|
| Start recording (menu bar or dashboard) | ☑ | Human signed off 2026-07-13 |
| System + mic audio captured | ☑ | |
| Live transcript updates during meeting | ☑ | |
| Stop recording; session appears in sidebar | ☑ | |
| Auto-name applied (if enabled) | ☑ | |
| Generate AI notes completes without manual rescue | ☑ | |
| Split at segment boundary; both halves auto-rename when auto-name on | ☑ | |
| Audio playback syncs with transcript | ☑ | |

**Blockers filed:** none

## T6 — Clean install-from-scratch

| Step | Pass? | Notes |
|------|-------|-------|
| Fresh machine or removed prior `/Applications/Escriba.app` | ☑ | `make install` → `/Applications` 2026-07-13 |
| One-liner / documented install path succeeds | ☑ | |
| App launches from Applications | ☑ | |
| Permissions (mic, screen capture) granted | ☑ | |
| First recording + transcription works | ☑ | |
| Model download (if local LLM) succeeds or degrades clearly | ☑ | |

**Blockers filed:** none

## T7 — Soak/install summary

- **Overall:** ☑ Pass · ☐ Fail
- **Issues to file:** none

## Sign-off

- [x] Automated API + unit DoD recorded
- [x] Human T5/T6 results pasted into aftercare / issue #142–#144
- [x] No P0 blockers remain for ship (or ship deferred with explicit waiver)
