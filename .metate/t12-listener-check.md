# T12 — does the output-device format listener fire on a live A2DP→HFP switch?

Issue: [#217](https://github.com/Skalas/escriba/issues/217) · Sprint: `tap-profile-switch-recovery`

## Why this gate exists

The 2026-08-22 design correction removed silence-based rebuild — correctly, it
glitched playback — but that was the only detector that could have caught the
recorded failure. In session `c0746d0e` **the tap kept clocking**: 888 s of audio
reached Whisper, which is why it hallucinated 273 segments instead of producing
nothing. The 8 s clock-stall watchdog returns "no rebuild" for that whole session.

So the route/format listener is now the *only* path covering the reported bug.
It is unverified. `keep_bluetooth_playback` does not cover it either — the call
app held the AirPods mic and forced HFP, and Escriba's mic had already moved to
the built-in.

**This cannot be verified from a terminal.** TCC denies the audio grant to
non-app processes, so a terminal-launched tap reads digital zeros regardless of
the code. It must run through the app bundle.

## Status: partially verified from the log, 2026-09-01

Mined from `~/Library/Logs/escriba/app.log` (recorded on issue #217):

| Signal | Count | Reading |
|---|---|---|
| `rebuilt after route/format change` | 6 (Aug 22 ×3, Aug 31 ×3) | listener **fires** on real hardware |
| `rebuilt after IO proc stall` | **0, ever** | the clock-stall watchdog has never triggered |
| `[tap] dead` | **0, ever** | the failure-backoff path is untriggered in practice |
| `Default mic … is also the output device` | Aug 31 | prevention **works** |

The failure mode here is a tap that keeps *clocking* while delivering silence,
so the listener is doing all the recovery work and the watchdog guards a
failure that does not appear to occur on this hardware.

**Resolved 2026-09-01: PASS.** The walkthrough below bound a rebuild to a
deliberate A2DP→HFP switch and confirmed the audio after it was real. See the
verdict at the bottom.

## Preconditions

- [ ] `cd swift-audio-capture && swift build -c release` is current
      (`.build/release/audio-capture` newer than the last bridge edit).
- [ ] Escriba **restarted** after the branch's Python changes — a bundle started
      earlier is running the old `session.py`. `pgrep -fl "escriba app"`, quit
      from the menu bar, relaunch from `/Applications/Escriba.app`.
- [ ] AirPods connected, playing audio, confirmed A2DP:
      `system_profiler SPBluetoothDataType | grep -iA3 "AirPods"`

## Run

Terminal 1 — watch the bridge talk:

```sh
./scripts/watch-tap-log.sh
```

Terminal 2 / GUI:

1. **Start continuous audio playing and keep it playing for the whole test** —
   a YouTube video, a podcast, anything with speech. This is not optional: the
   whole verdict is "did real audio survive the switch", and with nothing
   playing, silence after the switch is indistinguishable from silence before
   it. Speech is better than music (you can hear whether it is intelligible).
2. Start a recording in `both` mode from the dashboard.
3. Let it run ~20 s so the chain settles past the 2 s window, then check the
   dashboard shows a live transcript of the playing audio. **If the transcript
   is empty or garbage now, stop — the tap is broken before any switch and this
   gate has nothing to measure.**
4. **Force HFP from another app** — join a Meet/Zoom call, or open the AirPods
   mic in QuickTime (New Audio Recording → select AirPods). You should hear the
   playback quality drop to mono; that is the switch happening.
5. Watch Terminal 1, and keep the audio playing another ~60 s.
6. Stop the recording. Play the session back in the dashboard and listen to the
   audio *after* the switch, and read the transcript across that boundary.

## Verdict

**PASS** — all three:
- `[tap] rebuilt after route/format change` appears within a few seconds of the
  switch.
- Segments after the switch contain real speech, not filler.
- No repeated rebuild storm: at most 1–2 rebuild lines for one switch. More than
  that means the settle window or the M9 stale-request guard is not holding.

**FAIL** — no rebuild line, and levels stay around −47 dBFS.
Then the branch has **no working recovery path** for the reported bug, only the
warning banner. That has to be said plainly in the release notes rather than
implied fixed, and #217 stays open as the next sprint's headline.

**PARTIAL** — rebuild fires but audio is still silent afterwards: the listener
works and the rebuild does not recover the format. Record the log excerpt on
#217; the tap is being rebuilt against a stale format.

## Record the result

Paste the log excerpt into #217 and check the box here:

- [x] **PASS** — 2026-09-01, verified by the user through the app bundle.
- [ ] FAIL
- [ ] PARTIAL

Observed on: 2026-09-01  macOS: 26.6.2 (25G83)  Device: Mac16,5 / M4 Max / AirPods Pro

All three criteria met: the rebuild fired within seconds of a deliberate
A2DP→HFP switch, audio after the switch was real speech, and there was no
rebuild storm. Recorded on #217, which is closed.

**Re-run this gate when:** the bridge's listener set changes, the settle window
or the stale-request guard changes, or macOS changes how it reports Bluetooth
device formats. It is the only check that covers the recorded failure.
