// Watchdog verdict tests — standalone executable (no Xcode/XCTest required).
//
// Tests the pure CoreAudioTapWatchdogShouldRebuild and
// CoreAudioTapListenerShouldIgnore C functions. This is the only part of the
// bridge that can be unit-tested without real Core Audio I/O or TCC grants.
//
// What is NOT covered here (requires app bundle + TCC + real hardware):
//   - Tap creation, aggregate device setup, IO proc
//   - Property listener dispatch (ChainChangedListener)
//   - Actual audio data flowing through the chain
// Those scenarios are validated through smoke testing the live app.
//
// Run (after release build):  .build/release/watchdog-tests

import CoreAudioTapBridge
import Foundation

var failures = 0

func expect(_ condition: Bool, _ label: String) {
    if condition {
        print("  ✓ \(label)")
    } else {
        print("  ✗ FAIL: \(label)")
        failures += 1
    }
}

func shouldRebuild(stalledSeconds: Double, stall: Double = 8.0) -> Bool {
    CoreAudioTapWatchdogShouldRebuild(stalledSeconds, stall) != 0
}

print("CoreAudioTap watchdog verdict tests")
print("")

// --- T1: a clocking tap is healthy even when every sample is zero ----------
print("T1 — clocking tap, zeros are not a failure")
expect(!shouldRebuild(stalledSeconds: 0.0),
       "IO proc just fired → no rebuild")
expect(!shouldRebuild(stalledSeconds: 0.02),
       "IO proc firing every 20ms → no rebuild (quiet room / muted call)")
expect(!shouldRebuild(stalledSeconds: 7.9),
       "7.9s since last callback → still below 8s stall threshold")

// --- T2: missing callbacks are a dead tap ----------------------------------
print("")
print("T2 — IO proc stall")
expect(shouldRebuild(stalledSeconds: 8.0),
       "8s without a callback → rebuild")
expect(shouldRebuild(stalledSeconds: 60.0),
       "long stall still rebuilds (threshold is a floor, not a window)")

// --- T9: healthy signal never triggers a rebuild ---------------------------
print("")
print("T9 — no spurious rebuilds while the clock is alive")
expect(!shouldRebuild(stalledSeconds: 1.0),
       "1s gap (HAL hiccup) → no rebuild")
expect(!shouldRebuild(stalledSeconds: 4.0),
       "4s gap → no rebuild")

// --- S3: failure backoff and dead-tap detection ----------------------------
print("")
print("S3 — failure backoff and dead-tap detection")
expect(CoreAudioTapFailureBackoffSeconds(1) == 40.0,
       "1 failure → 20s * 2 = 40s backoff")
expect(CoreAudioTapFailureBackoffSeconds(2) == 80.0,
       "2 failures → 20s * 4 = 80s backoff")
expect(CoreAudioTapFailureBackoffSeconds(3) == 160.0,
       "3 failures → 20s * 8 = 160s backoff")
expect(CoreAudioTapFailureBackoffSeconds(4) == 320.0,
       "4 failures → 20s * 16 = 320s (clamp floor)")
expect(CoreAudioTapFailureBackoffSeconds(5) == 320.0,
       "5 failures → clamped to 4 → still 320s")
expect(CoreAudioTapFailureBackoffSeconds(99) == 320.0,
       "99 failures → still clamped at 320s")

expect(CoreAudioTapShouldMarkDead(4) == 0,
       "4 failures → not yet dead (threshold is 5)")
expect(CoreAudioTapShouldMarkDead(5) != 0,
       "5 failures → mark dead (kMaxConsecutiveRebuildFailures = 5)")
expect(CoreAudioTapShouldMarkDead(10) != 0,
       "10 failures → still dead")

// --- settle window: our own rebuild must not retrigger ---------------------
print("")
print("L1 — listener settle ignore")
expect(CoreAudioTapListenerShouldIgnore(10.5, 10.0, 2.0) != 0,
       "0.5s after our rebuild → ignore HAL chatter")
expect(CoreAudioTapListenerShouldIgnore(11.9, 10.0, 2.0) != 0,
       "1.9s after our rebuild → still inside 2s settle")
expect(CoreAudioTapListenerShouldIgnore(12.0, 10.0, 2.0) == 0,
       "2.0s after our rebuild → settle expired, listener may run")
expect(CoreAudioTapListenerShouldIgnore(10.0, 0.0, 2.0) == 0,
       "no prior rebuild (lastRebuild=0) → do not ignore")
expect(CoreAudioTapListenerShouldIgnore(10.0, 10.0, 0.0) == 0,
       "zero settle window → do not ignore")

// --- M5: state machine — stalledSeconds from lastCallback / chainBuilt -----
//
// WatchdogState mirrors WatchdogMain:
//   lastActive = lastCallbackTime > 0 ? lastCallbackTime : chainBuiltTime
//   stalledSeconds = now - lastActive
// callbackArrived(at:) is IOProcCallback writing lastCallbackTime on every
// invocation, silent or not.
print("")
print("M5 — state-machine: clocking vs stalled")

struct WatchdogState {
    var rebuildCount: Int = 0
    var consecutiveFailures: Int = 0
    var chainBuiltTime: Double
    var lastCallbackTime: Double = 0.0
    var lastRebuildAttempt: Double = 0.0

    mutating func callbackArrived(at t: Double) {
        lastCallbackTime = t
    }

    mutating func tick(now: Double,
                       stall: Double = 8.0,
                       maxBackoff: Double = 60.0,
                       rebuildSucceeds: Bool = true) -> Bool {
        let lastActive = lastCallbackTime > 0.0 ? lastCallbackTime : chainBuiltTime
        let stalledSeconds = (lastActive > 0.0 && now > lastActive) ? (now - lastActive) : 0.0

        if consecutiveFailures > 0 {
            let backoff = min(CoreAudioTapFailureBackoffSeconds(Int32(consecutiveFailures)),
                              maxBackoff)
            if (now - lastRebuildAttempt) < backoff { return false }
        }

        let verdict = CoreAudioTapWatchdogShouldRebuild(stalledSeconds, stall) != 0
        guard verdict else { return false }

        lastRebuildAttempt = now
        if rebuildSucceeds {
            rebuildCount += 1
            consecutiveFailures = 0
            chainBuiltTime = now
            lastCallbackTime = 0.0
        } else {
            consecutiveFailures += 1
        }
        return true
    }
}

var state = WatchdogState(chainBuiltTime: 10.0)
expect(!state.tick(now: 15),
       "fresh chain, 5s since build, no callbacks yet → no rebuild (below 8s)")

// IO proc starts clocking with digital zeros at t=12. Quiet room for a minute.
state.callbackArrived(at: 12)
for t in stride(from: 13.0, through: 72.0, by: 1.0) {
    state.callbackArrived(at: t)
}
expect(!state.tick(now: 72),
       "IO proc still firing after 60s of zeros → no rebuild")

// Clock dies at t=72. 8s later the stall verdict fires.
expect(!state.tick(now: 79.9),
       "7.9s after last callback → no rebuild")
expect(state.tick(now: 80),
       "8s after last callback → rebuild")
expect(state.rebuildCount == 1, "exactly 1 rebuild after the stall")

// Fresh chain after rebuild: lastCallbackTime=0, measured from chainBuilt=80.
expect(!state.tick(now: 86),
       "6s after rebuild, aggregate not yet clocking → no rebuild")
expect(state.tick(now: 88),
       "8s after rebuild with no callbacks → second rebuild (chain never clocked)")
expect(state.rebuildCount == 2, "exactly 2 rebuilds total")

// --- M5b: failure-backoff gate ---------------------------------------------
print("")
print("M5b — failure-backoff gate: failed rebuild suppresses next attempt")

var bState = WatchdogState(chainBuiltTime: 10.0)
bState.callbackArrived(at: 50.0)

let firstAttempt = bState.tick(now: 58, rebuildSucceeds: false)
expect(firstAttempt, "backoff: 8s stall triggers rebuild attempt")
expect(bState.consecutiveFailures == 1, "backoff: failed rebuild → consecutiveFailures=1")

expect(!bState.tick(now: 59),
       "backoff: 1s after failure → inside 40s window → suppressed")
expect(bState.tick(now: 99),
       "backoff: 41s elapsed → gate passes, rebuild attempt fires")

// --- T3 — a request overtaken by another path is dropped -------------------
// The watchdog does not run on gRebuildQueue, so a listener block can clear its
// settle check, block on ctx->lock while the watchdog rebuilds, and then tear
// down a chain that is milliseconds old. This predicate is what stops it.
print("")
print("T3 — stale rebuild requests are dropped")

func dropsRequest(requestedAt: Double, lastRebuild: Double) -> Bool {
    CoreAudioTapShouldDropStaleRequest(requestedAt, lastRebuild) != 0
}

expect(dropsRequest(requestedAt: 100.0, lastRebuild: 100.5),
       "listener decided at t=100, watchdog finished a rebuild at t=100.5 → drop")
expect(!dropsRequest(requestedAt: 100.0, lastRebuild: 99.5),
       "last rebuild predates the request → the request still stands")
expect(!dropsRequest(requestedAt: 100.0, lastRebuild: 100.0),
       "exactly equal → not overtaken, the request proceeds")
expect(!dropsRequest(requestedAt: 0.0, lastRebuild: 500.0),
       "requestedAt=0 means unconditional → never dropped")
expect(!dropsRequest(requestedAt: 100.0, lastRebuild: 0.0),
       "no rebuild has ever completed → nothing to be overtaken by")
expect(dropsRequest(requestedAt: 0.001, lastRebuild: 0.002),
       "the guard is a strict ordering, not a threshold — sub-ms still counts")

print("")
if failures == 0 {
    print("All tests passed.")
    exit(0)
} else {
    print("\(failures) test(s) FAILED.")
    exit(1)
}
