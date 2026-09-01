# Sprint plan — survive a Bluetooth profile switch, and make a degraded capture visible

> Entry doc for `metate-prep`.
> Mode: **HOLD** — this is a correctness sprint on the capture path. No new surface,
> with one folded-in debt item (the Swift XCTest target, whose trigger has fired).

## Goal

Make system-audio capture survive a mid-session output route/format change
(A2DP ↔ HFP), detect a microphone that has degraded rather than failed, and
surface both to the user instead of letting Whisper paper over them with
hallucinated filler.

## Why now

A live call on 2026-08-21 was recorded for 888 seconds and produced 273
segments of pure hallucination. Measured evidence:

| Session | Tap created | Level |
|---|---|---|
| `c0746d0e` (broken) | while AirPods in A2DP (stereo/48k) | **-47 to -50 dBFS** for 888s |
| `240db51a` (restart, same routing) | while AirPods in HFP (mono/24k) | **-16.6 to -18.6 dBFS** |

Identical hardware, identical routing, ~30 dB apart. The only variable was
*when* the tap was created — which rules out a platform limitation and points
at tap lifetime. Confirmed root cause: `CoreAudioTapStart` built the tap,
aggregate device and IO proc once against the format in force at start, and
`grep -c AudioObjectAddPropertyListener` over the bridge returned **0** — the
chain was never rebuilt when the device changed underneath it.

Two findings shape the design and are worth recording, because they invalidate
the obvious fix:

1. A global tap reports a **constant** `rate=48000 channels=2` regardless of
   what the output device is doing. So `kAudioTapPropertyFormat` never changes.
2. The AirPods remain the **same** default output device across an A2DP→HFP
   switch. So `kAudioHardwarePropertyDefaultOutputDevice` never fires either.

Property listeners on the tap itself therefore cannot detect this failure.
The original design made dead-sample detection the load-bearing signal.

**Design correction (2026-08-22).** Live testing invalidated that design:
silence is *not* a tap failure. A quiet room still clocks the IO proc, and
rebuilding the chain reconfigures the default output — four rebuilds in one
3-minute session, including a 61s-silence fire, each one audible. Rebuild on
loss of *clock*, not loss of signal. Do not listen on the tap's own
`kAudioTapPropertyFormat` either: creating the tap emits that notification
and the rebuild loops.

That correction removed the only detector that would have caught the recorded
failure, so the sprint now has two independent paths and it is worth being
exact about what each one covers:

- **Prevention — `keep_bluetooth_playback`.** Escriba does not open the
  headset mic while it is also the output, so Escriba never triggers the
  profile switch itself. This does **not** cover the recorded failure: in
  session `c0746d0e` the call app held the AirPods mic and forced HFP, and
  Escriba's own mic had already moved to the built-in. Prevention covers the
  case where Escriba is the trigger, which is a real case but not that one.
- **Recovery — output-device route/format listeners.** With dead-sample
  detection gone, this is the *only* path that can catch the recorded
  failure, because the tap in that session kept clocking: 888s of audio
  reached Whisper, which is why it hallucinated instead of producing nothing.
  The 8s clock-stall watchdog returns "no rebuild" for that entire session.
  **Verified on hardware 2026-09-01 (T12 / #217): the listener fires within
  seconds of a live A2DP→HFP switch and the rebuilt chain captures real
  audio.** This is the path that fixes the reported bug. The clock-stall
  watchdog has never fired on real hardware (A3) — it is a correct guard for a
  failure that does not occur here, and the sprint's headline belongs to the
  listener, not the watchdog.

## Scope

- **W1 — Tap chain is disposable.** Rebuild tap + aggregate + IO proc on
  route change, output-device format/rate/channel change, or a clock-stall
  watchdog verdict. Not on tap format (self-triggering) and not on silence
  (see the design correction).
- **W2 — Do not rebuild a chain that is still clocking.** *(Replaces the
  original W2, which tightened the silence thresholds to ~4s and exempted the
  first `sawSignal` recovery from the cooldown. Both were removed on
  2026-08-22 — they made the churn worse, not better.)* A rebuild is
  triggered only by 8s without an IO-proc callback, and self-inflicted HAL
  chatter is suppressed by a 2s settle window.
- **W8 — Keep the Bluetooth link in A2DP (added 2026-08-22).** In `both`
  mode, when the default input is also the default output, record the
  built-in mic instead so macOS is never asked to flip the headset to HFP.
  User-visible setting `audio.keep_bluetooth_playback`, default on, with a
  banner naming the device actually opened. *(Covered by T11.)*
- **W3 — Stream the Swift CLI's stderr into the logger.** `screen_capture.py`
  reads stderr only when the process fails, so every `[tap] rebuilt …` line is
  lost. The rebuild had to be inferred from the waveform. Unacceptable for a
  failure mode this quiet.
- **W4 — Level-based mic degradation.** `_track_system_silence` tests
  `max(system_pcm) != 0` — exact zeros, system buffer only. The observed failure
  was a mic sitting at -50 dBFS, which is never zero and never inspected. Detect
  sustained sub-threshold RMS on the mic buffer too.
- **W5 — Re-arm persistent warnings.** `consume_warning()` is one-shot and
  `_silence_warned` latches, so a condition still true 16 minutes later shows a
  banner once and never again — and one stray dismiss silences it permanently.
  A still-true degradation must keep the banner asserted.
- **W7 — Stand up the Swift XCTest target (folded-in debt).** `ROADMAP.md` has
  carried this since `v1.0.1` with the trigger *"any further Swift change to
  `PCMConverter`/`CoreAudioTap`, or a second Swift bug"*. This sprint is both.
  Requires moving the testable logic into a library target, and wiring
  `swift build` + `swift test` into `fastGate`/`shipGate` — a Swift test suite
  that no gate runs is decoration.
- **W6 — Reconsider the `vad_enabled = false` default.** VAD-off is what lets a
  -50 dBFS noise floor become `Thanks for watching!` × 164. Decide and record.

## Out of scope

- Anything that forces a Bluetooth profile. HFP persisting after hangup is held
  by the call app, not Escriba (Escriba's mic had already moved to the built-in).
  With W1 in place it no longer affects capture.
- The ScreenCaptureKit fallback path (`--use-screen-capture`).
- The leaked mlx-lm inference worker (separate, tracked).

## Load-bearing assumptions

Required by `assumptions.require` in `.metate/profile.yml`. Every row is
behaviour the design *rests on* and that only real hardware, a signed bundle or
a TCC grant can exhibit. An unprobed row blocks entry to review — this sprint is
the reason the rule exists, and the rows are filled in retroactively from the
2026-09-01 log analysis.

| # | Assumption | Probe | Verdict |
|---|---|---|---|
| A1 | Rebuilding the chain on sustained silence is safe for playback | one live 3-min recording, listen | **FALSE** — 4 rebuilds, audible glitches. Found 2026-08-22, *after* 9 review rounds. Had this been probed first, rounds 4–9 would not have happened. |
| A2 | The output-device format listener fires on an A2DP→HFP switch **and the rebuild recovers audio** | `.metate/t12-listener-check.md` | **TRUE** — verified on hardware 2026-09-01 (#217, closed). Rebuild fires within seconds of the switch, audio after it is real speech, no rebuild storm. |
| A3 | The failure presents as a *stalled* IO proc | `grep -c "rebuilt after IO proc stall" app.log` | **FALSE** — 0 firings, ever. The tap keeps clocking and delivers silence. The clock-stall watchdog guards a failure that does not occur on this hardware; the listener does all the real work. |
| A4 | 120 s of system silence indicates a fault | count warnings against real sessions | **FALSE** — fired 123 s into a session where nothing had played yet. Fixed by restoring the warm/cold split (`SYSTEM_SILENCE_COLD_WARN_SECONDS`). |
| A5 | Opening the headset mic is what forces HFP, so avoiding it prevents the switch | log line at capture start | **TRUE for Escriba-caused switches**, and confirmed working on 2026-08-31. Does **not** cover the recorded failure, where the call app forced HFP. |
| A6 | A terminal-launched tap can validate capture | run the CLI, read levels | **FALSE** — TCC yields digital zeros to non-app processes. Known before the sprint; it is why A1–A4 were unprobeable in review. |

Four of six load-bearing assumptions were false. Every one of them was cheap to
probe and none of them was probeable by code review.

## DoD test matrix

> **Re-derived 2026-09-01.** T1–T10 were filed on 2026-08-21 (issues
> #205–#214) against the silence-based design that the 2026-08-22 correction
> replaced. T1 and T2 asserted the *opposite* of what the code now does and
> have been rewritten; the sprint's largest behaviour change (W8) and its one
> load-bearing unknown had no criterion at all and are now T11 and T12.
> Nothing below is closed by a test that proves the contrary.

- **T1** *(inverted 2026-09-01; was "a silent tap is rebuilt within ~5s")* — A
  tap that is still clocking is **not** rebuilt, however quiet it is: zeros
  from a muted call or an empty room produce zero rebuilds.
- **T2** *(replaced 2026-09-01; was the ~30s cold threshold, which no longer
  exists)* — 8s with no IO-proc callback triggers exactly one rebuild, and a
  chain that never clocks keeps retrying rather than latching dead.
- **T3** — Repeated rebuild *failures* are spaced by the exponential backoff,
  and self-inflicted HAL chatter within the 2s settle window never triggers a
  rebuild — including when the watchdog and a listener event coincide.
- **T4** — A failed rebuild self-heals: the next watchdog tick retries rather
  than leaving the session permanently without a chain.
- **T5** — `CoreAudioTapStop` during an in-flight listener hand-off does not
  use freed memory (registry lookup returns nothing to rebuild).
- **T6** — Swift CLI stderr appears in `app.log` during a live session, not
  only on process failure.
- **T7** — A mic delivering sustained low-level noise (≈-50 dBFS) raises a
  warning; brief quiet does not.
- **T8** *(scoped down 2026-09-01)* — A still-true degradation keeps the
  banner asserted across polls without the 3s poll resetting its dismiss
  timer. Dismissal is **deliberately sticky per source for the rest of the
  recording** and is cleared on the next recording; re-showing a dismissed
  banner on a TTL is deferred to `ROADMAP.md`. The original wording ("does
  not permanently silence a still-true condition") described behaviour that
  was never built.
- **T9** — No regression: a healthy 60s capture produces zero rebuilds.
- **T10** *(scoped down 2026-09-01)* — The testable tap logic lives in a
  library target with a standalone `watchdog-tests` executable, run by
  `fastGate` and `shipGate`. **Not XCTest**: `xcode-select -p` is CLT-only on
  this machine. Coverage is the pure verdict and backoff arithmetic only —
  tap creation, the aggregate device, IO-proc delivery and listener dispatch
  are not headlessly reachable (see `ROADMAP.md`).
- **T11** *(new 2026-09-01 — covers W8, the sprint's largest behaviour
  change, default-on for every `audio_source = "both"` user)* — With
  `keep_bluetooth_playback` on and the default input also the default output,
  capture opens the built-in mic; with it off, the headset mic. The banner
  names the device **actually opened**, so a built-in mic that fails to open
  never produces a banner claiming it is recording.
- **T12** *(new 2026-09-01 — the load-bearing unknown; hardware-only)* — On a
  live A2DP→HFP switch forced by another app, the output-device format
  listener fires and `app.log` shows `[tap] rebuilt after route/format
  change`. This is the only path that covers the recorded failure, in which
  the tap kept clocking; it cannot be verified headlessly (TCC yields digital
  zeros to non-app processes) and must be walked through the app bundle.

## Risks

- **Rebuild churn during genuine silence.** Mitigated by the warm/cold
  threshold split and the cooldown; T2/T3 pin it.
- **`exclusive = YES` tap recreation may fail transiently.** Must self-heal
  rather than latch dead — T4.
- **The XCTest target may not reach the failure mode.** Most of the bridge is
  Core Audio I/O that XCTest cannot exercise; the watchdog *verdict* (silent
  seconds + `sawSignal` + cooldown → rebuild?) is the part worth extracting and
  testing. Be explicit about which of T1–T4 is genuinely unit-covered vs proven
  at the CLI boundary and in smoke — do not claim coverage the target cannot
  support.
- **Folding in W7 widens a HOLD sprint.** Accepted deliberately: the trigger has
  fired twice and the bridge is now the highest-risk file in the repo. If the
  library-target extraction starts reshaping capture behaviour, stop and split it
  into its own sprint.
- **TCC.** A terminal-launched tap reads digital zeros regardless of the code,
  so signal can only be validated through the app bundle. Terminal runs are
  still useful for exercising the *cold* watchdog path, which is what zeros are.
- **Two-tree install.** The app runs from the dev checkout via `uv run`, but
  `~/.escriba` holds a separate installed copy. A bridge fix needs `make install`
  to reach the installed bundle; call this out in aftercare.
