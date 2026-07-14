# Sprint plan — calendar picker + prune watch-calendar orphans (+ human gates)

> Entry doc for `metate-prep`. Selected from discover: **merge 1 + 3**, with **human
> gates from candidate 2** (H1–H3 as soak/product checklist — not a full auto-start build).
> Mode hint: **HOLD** on calendar selection; **REDUCE** on orphan CLI/calendar path.

## Goal

Let users pick which Apple Calendar calendars Escriba reads for Up next (and related
calendar reads), persist that choice, and stop scanning every synced account by default.
While in the calendar module, prune or clearly deprecate the orphan `watch_calendar` /
meeting-link CLI surface that Up next does not use. Keep `--auto-start` blocked; use the
H-matrix for UX / permission / graduation sign-off only.

## Why now

- **#1 / issue #181** — Multi-account Calendar scans timed out Up next; the event lived on
  one personal calendar. User-requested for this cycle (P1 / performance).
- **#3** — Graph: `watch_calendar` in-degree 0; `has_meeting_link` unused on the Up-next
  path. Cheap REDUCE ride-along while the calendar module is open.
- **Human gates (from #2 / #64)** — Thin Up next already shipped; full auto-start stays
  parked until H1–H3. Include the checklist in this sprint’s smoke/aftercare — do not
  implement calendar-driven auto-record.

## Scope note

Parents / links: [#181](https://github.com/Skalas/escriba/issues/181),
[#64](https://github.com/Skalas/escriba/issues/64) (gates only).

**In scope**

- Settings multi-select of Calendar.app calendars
- Persist selection in `escriba.toml` (e.g. `[calendar]` / allowlist)
- `get_upcoming_events` / `GET /api/calendar/upcoming` honor the selection
- Holiday/birthday skip list remains a safety net
- Document empty-selection default (all non-skipped vs require ≥1 — prep picks one)
- Prune or deprecate `watch_calendar` / unused meeting-link CLI path so public surface matches behavior
- H1–H3 human validation (plan prose; profile has no `smoke.humanGates`)

**Out of scope**

- Full calendar auto-start / menubar scheduling product (#64 build)
- Server `do_GET` decomposition (discover #4)
- #105 P2 pile (MLX resample, Swift XCTest, schema_version runner, etc.)
- Sparkle / bash install single-sourcing

## Definition of Done

Done when: user can enable/disable calendars in Settings; selection survives restart;
Up next only queries the selected set (faster when scoped); holiday skip still applies;
orphan watch-calendar surface is gone or explicitly deprecated with matching CLI/docs;
`--auto-start` remains blocked with a clear error; H1–H3 recorded (approve / live
permission / graduation call); ship gate green.

## Seed test matrix

### Strand A — Calendar picker (#181) · HOLD

| ID | Criterion |
|----|-----------|
| T1 | Config: `[calendar]` (or equivalent) loads/saves allowlist; round-trip via `PUT /api/config` / `update_config_toml` |
| T2 | `get_upcoming_events` (or server wrapper) only queries selected calendars; unlisted calendars never scanned |
| T3 | Empty selection behaves per documented default (all non-skipped **or** require ≥1 — one rule, tested) |
| T4 | Settings UI lists available calendars; toggle persists; permission_denied / unavailable surfaces a soft hint |
| T5 | Home Up next uses the filtered API; SPA regression or server test covers selection → fewer calendars queried |
| T6 | Holiday/birthday skip list still applied on top of the allowlist |

### Strand B — Prune orphan calendar CLI (#3) · REDUCE

| ID | Criterion |
|----|-----------|
| T7 | `watch_calendar` / `cmd_watch_calendar` / unused meeting-link path: delete **or** deprecate with clear docs — no silent dead public API |
| T8 | `--auto-start` stays blocked with clear error (existing test remains or is updated) |
| T9 | CLI help / README matches the remaining calendar surface; no broken imports |

### Strand C — Decision hygiene · HOLD

| ID | Criterion |
|----|-----------|
| T10 | ROADMAP / #64 comment: record H1–H3 outcomes; auto-start remains parked unless H3 explicitly schedules a follow-up sprint |

## Seed H-matrix (plan prose only — no `smoke.humanGates` in profile)

Prep will not seed an H ledger unless the profile gains `smoke.humanGates`; treat these as
a human checklist in smoke/aftercare.

| ID | Type | What the human does |
|----|------|---------------------|
| H1 | ux | Approve calendar-picker Settings + Up next still readable with a narrowed set |
| H2 | live | On a real Mac: Calendar permission works; selected calendar(s) show the expected event |
| H3 | graduation | Product call: stay spike-only (picker only) vs schedule a full calendar auto-start sprint later |

## Mode

Prep finalizes mode. Hint: **HOLD** overall (calendar selection is product-facing but
behavior-bounded); Strand B is **REDUCE**.

## Next ceremony

Hand off to **`metate-prep`**, which reads this file as its entry doc, files the issue
ledger from the T-matrix, and cuts the working branch from `main`.
