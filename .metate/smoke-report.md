# Smoke report — calendar-picker-prune-watch

Sprint: `calendar-picker-prune-watch` · branch same · 2026-07-14

## Automated evidence

- `uv run ruff check .` — green
- `uv run mypy .` — green (97 files)
- `uv run pytest` — **470 passed**
- Focused calendar suite — **33 passed** (`-k calendar`)

`smoke.command` is `uv run escriba app` (manual launch). No Playwright e2e suite; DoD mapped to unit/SPA tests below.

## DoD matrix (T1–T10)

| ID | Result | Evidence |
|----|--------|----------|
| T1 | ✅ | `test_calendar_allowlist_round_trip_from_toml` |
| T2 | ✅ | `test_get_upcoming_events_allowlist_filters_calendars`, `test_calendar_upcoming_passes_config_allowlist` |
| T3 | ✅ | `test_empty_calendar_allowlist_means_all_non_skipped`, `test_get_upcoming_events_empty_allowlist_queries_all_non_skipped` |
| T4 | ✅ | SPA calendar load/save tests; `test_calendar_calendars_endpoint_*` |
| T5 | ✅ | `test_calendar_upcoming_passes_config_allowlist` |
| T6 | ✅ | `test_get_upcoming_events_skip_list_applies_on_allowlist` |
| T7 | ✅ | `test_watch_calendar_command_removed` |
| T8 | ✅ | Same; auto-start remains product-parked (no CLI) |
| T9 | ✅ | README/AGENTS/CLAUDE updated; CLI help has no watch-calendar |
| T10 | ⏳ human | ROADMAP notes H1–H3 pending; auto-start parked |

## Human gates (plan prose — no `smoke.humanGates` ledger)

No profile `humanGates` block; checklist only:

### H1 — UX (Settings picker + Up next)
**Why:** Confirm the picker is usable and Up next still readable with a narrowed set.  
**Do:** Open dashboard → Settings → Calendar → enable only the calendar that has your next event → Save → Home.  
**Approved means:** Selected calendars toggle cleanly; Up next shows the expected event (or a clear hint).

### H2 — Live Calendar permission
**Why:** osascript + multi-account Mac behavior isn’t fully covered by mocks.  
**Do:** On a real Mac with Calendar access granted to Terminal/Escriba, confirm Settings lists calendars and Up next returns within ~30s when scoped.  
**Approved means:** No stuck “Checking Calendar…” / false unavailable after allowlist narrows the set.

### H3 — Graduation
**Why:** Full auto-start (#64) stays parked until you choose.  
**Do:** Decide spike-only (picker enough) vs schedule a calendar auto-start sprint.  
**Approved means:** Decision recorded (comment on #64 / ROADMAP); no silent enablement of `--auto-start`.

## Verdict

Automated smoke **green**. Human H1–H3 still open for product soak — do not block merge of picker work; T10 closes when you record the call.
