# Smoke report — appstate-mic-activation-seam

Sprint: `appstate-mic-activation-seam` · 2026-07-21

## Automated evidence

- `uv run ruff check .` — green
- `uv run pytest` — **481 passed**
- Focused: `test_recording_seam`, `test_mic_poll_decide`, `test_menubar_quit`, `test_call_state` — green

## DoD matrix (T1–T9)

| ID | Result | Evidence |
|----|--------|----------|
| T1 | ✅ | Menubar + HTTP use `try_start_recording`; concurrent start tests |
| T2 | ✅ | Both use `begin_stop` + `complete_stop_recording` |
| T3 | ✅ | Idempotent no-session / already-stopping / claim-required tests |
| T4 | ✅ | Concurrent stop-claim race + start-while-stop + stop-while-start tests |
| T5 | ✅ | `MicPollSnapshot` / `MicPollAction` / detect·decide·act |
| T6 | ✅ | `test_call_state` + decide auto_stop / hide_while_recording |
| T7 | ✅ | decide prompt vs auto + cooldown tests |
| T8 | ✅ | No calendar auto-start in menubar (grep) |
| T9 | ✅ | ROADMAP “Where we are” note |

## Human gates (plan prose)

| ID | What to do |
|----|------------|
| H1 | Mic auto-record: one real call-ish start/stop |
| H2 | Manual Record + dashboard Stop still clean |
| H3 | Confirm no surprise calendar auto-starts (#193 still off) |

## Verdict

Automated smoke **green**. Human H1–H3 optional soak before relying on mic auto-record in production use.
