#import "CoreAudioTapBridge.h"
#import <CoreAudio/AudioHardware.h>
#import <CoreAudio/AudioHardwareTapping.h>
#import <CoreAudio/CATapDescription.h>
#import <pthread.h>
#import <stdarg.h>
#import <unistd.h>
#import <stdio.h>
#import <stdlib.h>
#import <string.h>
#import <time.h>
#import <math.h>

/* A tap is bound to the audio format that was in force when it was created.
 * A Bluetooth headset moving between A2DP (stereo, 48 kHz) and HFP call mode
 * (mono, 24 kHz) stays the same default output device the whole time and only
 * changes format underneath. Opening that headset as a microphone is what
 * forces the switch — Escriba avoids that in mix mode (keep_bluetooth_playback).
 * When something else still flips the profile, device-format listeners are the
 * fast path. The watchdog is only for a chain that has stopped clocking.
 *
 * Do not rebuild on digital silence. Quiet rooms, muted calls and pauses all
 * produce zeros while the IO proc keeps firing; tearing the chain down then
 * reconfigures the default output and is what the user hears. Rebuild when
 * the IO proc stops being called, or when the output device's route/format
 * actually changes. Never listen on the tap's own kAudioTapPropertyFormat:
 * creating the tap emits that notification and the rebuild loops. */
static const double kClockStallRebuildSeconds = 8.0;
/* Ignore route/format listener events this long after we ourselves rebuilt.
 * HAL chatters while the new aggregate starts; acting on that chatter is the
 * self-trigger that produced back-to-back rebuilds at session start. */ 
static const double kRebuildSettleSeconds = 2.0;
/* Base multiplier for exponential failure backoff (separate from the inter-rebuild
 * floor even though they share the same value today, to prevent accidental coupling). */
static const double kFailureBackoffBaseSeconds = 20.0;
/* Watchdog wakeup interval — interruptible via pthread_cond_timedwait. */
static const long kWatchdogIntervalSec = 2;
/* After this many consecutive rebuild failures, latch tapDead, log "[tap] dead"
 * for Python to propagate, and fall back to kMaxBackoffSeconds retry cadence. */
static const int kMaxConsecutiveRebuildFailures = 5;
/* Maximum retry interval after repeated failures (replaces the old permanent
 * tapDead latch — W3: the watchdog keeps trying at this cadence). */
static const double kMaxBackoffSeconds = 60.0;

typedef struct {
    CoreAudioTapSamplesCallback callback;
    void *callbackContext;
    int targetSampleRate;
    int targetChannels;

    AudioStreamBasicDescription tapFormat;

    AudioObjectID tapObjectID;
    AudioDeviceID aggregateDeviceID;
    AudioDeviceIOProcID procID;
    AudioDeviceID listenedDeviceID;

    volatile int running;
    volatile uint64_t callbackCount;
    volatile uint64_t totalFrames;
    volatile uint64_t rebuildCount;

    /* Clock-loss detection. chainBuiltTime is set in BuildChainLocked on
     * success; lastCallbackTime is updated on every IO-proc invocation, silent
     * or not. A tap that is still being called is alive. */
    volatile double chainBuiltTime;
    volatile double lastCallbackTime;

    /* M2: removed lastWatchdogRebuild — both paths now share gLastRebuildTime
     * under gRegistryLock for cross-path failure backoff. */

    /* B2: consecutive failures + dead-tap state.
     * tapDead is NOT a permanent latch (W3): the watchdog retries at
     * kMaxBackoffSeconds once set, and BuildChainLocked clears it on success. */
    volatile int consecutiveRebuildFailures;
    volatile int tapDead;

    pthread_mutex_t lock;
    pthread_cond_t stopCond;
    pthread_t watchdog;
    int watchdogStarted;
} TapContext;

/* B2: gRebuildQueue is created at the TOP of CoreAudioTapStart, BEFORE any
 * AudioObjectAddPropertyListener call, so the queue is never NULL when a
 * listener fires. ChainChangedListener also guards with `if (!gRebuildQueue)`.
 *
 * W5: instead of DROPPING the newest request when it falls inside the 1s
 * debounce window, we dispatch_after the remaining interval so the settled
 * format always wins. */
static pthread_mutex_t gRegistryLock = PTHREAD_MUTEX_INITIALIZER;
static TapContext *gActiveCtx = NULL;
static dispatch_queue_t gRebuildQueue = NULL;
static uint64_t gRebuildGeneration = 0;
/* Debounce clock for listener-triggered rebuilds (1s settle window). */
static double gLastListenerRebuildTime = 0.0;
/* M2: shared failure-backoff clock. Both the watchdog and the listener stamp
 * this after every RebuildActiveChain call. Both paths read it when computing
 * whether consecutiveRebuildFailures warrants a backoff delay. Initialized to
 * NowSeconds() at session start so zero cannot mean "infinitely long ago". */
static double gLastRebuildTime = 0.0;

static double NowSeconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

static void LogTap(const char *fmt, ...) {
    char buf[512];
    va_list args;
    va_start(args, fmt);
    /* M1: clamp both offsets so buf[n] = '\n' is always in bounds even if a
     * format expands past the buffer (vsnprintf returns would-have-written). */
    int n = snprintf(buf, sizeof buf, "[tap] ");
    int m = vsnprintf(buf + n, sizeof buf - (size_t)n - 1, fmt, args);
    va_end(args);
    if (m < 0) m = 0;
    if (m > (int)(sizeof buf - (size_t)n - 1)) m = (int)(sizeof buf - (size_t)n - 1);
    n += m;
    buf[n++] = '\n';
    fwrite(buf, 1, (size_t)n, stderr);
    fflush(stderr);
}

static AudioDeviceID CurrentDefaultOutputDevice(void) {
    AudioDeviceID device = kAudioObjectUnknown;
    UInt32 size = sizeof(device);
    AudioObjectPropertyAddress addr = {
        kAudioHardwarePropertyDefaultOutputDevice,
        kAudioObjectPropertyScopeGlobal,
        kAudioObjectPropertyElementMain
    };
    AudioObjectGetPropertyData(kAudioObjectSystemObject, &addr, 0, NULL, &size, &device);
    return device;
}

static OSStatus ChainChangedListener(AudioObjectID inObjectID,
                                     UInt32 inNumberAddresses,
                                     const AudioObjectPropertyAddress *inAddresses,
                                     void *inClientData);

static void RebuildActiveChain(const char *reason, double requestedAt);

/* --- listener registration ------------------------------------------------ */

static const AudioObjectPropertyAddress kDeviceFormatProps[] = {
    { kAudioDevicePropertyNominalSampleRate,  kAudioObjectPropertyScopeGlobal, kAudioObjectPropertyElementMain },
    { kAudioDevicePropertyStreamFormat,       kAudioDevicePropertyScopeOutput, kAudioObjectPropertyElementMain },
    { kAudioDevicePropertyStreamConfiguration, kAudioDevicePropertyScopeOutput, kAudioObjectPropertyElementMain },
};
static const size_t kDeviceFormatPropCount =
    sizeof(kDeviceFormatProps) / sizeof(kDeviceFormatProps[0]);

static void RegisterDeviceListenersLocked(TapContext *ctx) {
    AudioDeviceID device = CurrentDefaultOutputDevice();
    if (device == kAudioObjectUnknown) return;
    for (size_t i = 0; i < kDeviceFormatPropCount; i++) {
        AudioObjectAddPropertyListener(device, &kDeviceFormatProps[i], ChainChangedListener, NULL);
    }
    ctx->listenedDeviceID = device;
}

static void UnregisterDeviceListenersLocked(TapContext *ctx) {
    if (ctx->listenedDeviceID == kAudioObjectUnknown) return;
    for (size_t i = 0; i < kDeviceFormatPropCount; i++) {
        AudioObjectRemovePropertyListener(ctx->listenedDeviceID, &kDeviceFormatProps[i],
                                          ChainChangedListener, NULL);
    }
    ctx->listenedDeviceID = kAudioObjectUnknown;
}

/* --- IO proc -------------------------------------------------------------- */

static OSStatus IOProcCallback(AudioObjectID inDevice,
                               const AudioTimeStamp *inNow,
                               const AudioBufferList *inInputData,
                               const AudioTimeStamp *inInputTime,
                               AudioBufferList *outOutputData,
                               const AudioTimeStamp *inOutputTime,
                               void *inClientData) {
    TapContext *ctx = (TapContext *)inClientData;
    if (!ctx || !ctx->callback || !inInputData || inInputData->mNumberBuffers == 0) {
        return noErr;
    }

    const AudioBuffer *buf = &inInputData->mBuffers[0];
    UInt32 numFrames = buf->mDataByteSize / (sizeof(float) * (buf->mNumberChannels > 0 ? buf->mNumberChannels : 1));
    if (numFrames == 0 || !buf->mData) {
        return noErr;
    }

    if (inInputData->mNumberBuffers == 1) {
        const float *samples = (const float *)buf->mData;
        ctx->lastCallbackTime = NowSeconds();
        ctx->callbackCount++;
        ctx->totalFrames += numFrames;
        ctx->callback(ctx->callbackContext,
                      samples,
                      numFrames,
                      buf->mNumberChannels,
                      ctx->tapFormat.mSampleRate);
        return noErr;
    }

    UInt32 numChannels = inInputData->mNumberBuffers;
    float *mix = (float *)malloc(numFrames * sizeof(float));
    if (!mix) return noErr;
    memset(mix, 0, numFrames * sizeof(float));
    for (UInt32 b = 0; b < numChannels; b++) {
        const AudioBuffer *bptr = &inInputData->mBuffers[b];
        UInt32 bFrames = bptr->mDataByteSize / sizeof(float);
        if (bptr->mData && bFrames >= numFrames) {
            const float *src = (const float *)bptr->mData;
            for (UInt32 i = 0; i < numFrames; i++) mix[i] += src[i];
        }
    }
    for (UInt32 i = 0; i < numFrames; i++) mix[i] /= (float)numChannels;
    ctx->lastCallbackTime = NowSeconds();
    ctx->callbackCount++;
    ctx->totalFrames += numFrames;
    ctx->callback(ctx->callbackContext, mix, numFrames, 1, ctx->tapFormat.mSampleRate);
    free(mix);
    return noErr;
}

/* --- chain build / teardown ----------------------------------------------- */

static void TeardownChainLocked(TapContext *ctx) {
    if (ctx->aggregateDeviceID != kAudioObjectUnknown) {
        if (ctx->procID) {
            AudioDeviceStop(ctx->aggregateDeviceID, ctx->procID);
            AudioDeviceDestroyIOProcID(ctx->aggregateDeviceID, ctx->procID);
            ctx->procID = NULL;
        }
        AudioHardwareDestroyAggregateDevice(ctx->aggregateDeviceID);
        ctx->aggregateDeviceID = kAudioObjectUnknown;
    }
    if (ctx->tapObjectID != kAudioObjectUnknown) {
        AudioHardwareDestroyProcessTap(ctx->tapObjectID);
        ctx->tapObjectID = kAudioObjectUnknown;
    }
}

/* S2: single goto-fail unwind path. */
static OSStatus BuildChainLocked(TapContext *ctx, char *errorOut, size_t errorOutSize) {
    OSStatus status = noErr;

    @autoreleasepool {
        NSArray<NSNumber *> *excludeProcesses = @[];
        CATapDescription *tapDesc = [[CATapDescription alloc] initStereoGlobalTapButExcludeProcesses:excludeProcesses];
        if (!tapDesc) {
            if (errorOut && errorOutSize > 0) snprintf(errorOut, errorOutSize, "failed to create CATapDescription");
            return kAudioHardwareUnspecifiedError;
        }

        NSUUID *tapUUID = [NSUUID UUID];
        tapDesc.UUID = tapUUID;
        tapDesc.name = [NSString stringWithFormat:@"escriba-tap-%@", [tapUUID UUIDString]];
        tapDesc.privateTap = NO;
        tapDesc.muteBehavior = CATapUnmuted;
        tapDesc.exclusive = YES;

        AudioObjectID tapObjectID = 0;
        status = AudioHardwareCreateProcessTap(tapDesc, &tapObjectID);
        if (status != noErr) {
            if (errorOut && errorOutSize > 0) snprintf(errorOut, errorOutSize, "AudioHardwareCreateProcessTap failed: %d", (int)status);
            goto fail;
        }
        ctx->tapObjectID = tapObjectID;

        UInt32 formatSize = sizeof(AudioStreamBasicDescription);
        AudioObjectPropertyAddress propAddr = {
            kAudioTapPropertyFormat,
            kAudioObjectPropertyScopeGlobal,
            kAudioObjectPropertyElementMain
        };
        status = AudioObjectGetPropertyData(tapObjectID, &propAddr, 0, NULL, &formatSize, &ctx->tapFormat);
        if (status != noErr) {
            if (errorOut && errorOutSize > 0) snprintf(errorOut, errorOutSize, "failed to get tap format: %d", (int)status);
            goto fail;
        }

        /* Do not listen on kAudioTapPropertyFormat. Creating this tap makes
         * HAL publish a format, which fired ChainChangedListener and rebuilt
         * the chain we had just started. Device-route listeners below are the
         * ones that mean a real output change. */

        NSString *tapUIDString = [tapUUID UUIDString];
        NSArray *tapList = @[ @{
            [NSString stringWithUTF8String:kAudioSubTapUIDKey]: tapUIDString,
            [NSString stringWithUTF8String:kAudioSubTapDriftCompensationKey]: @YES,
        } ];

        NSString *aggUID = [NSString stringWithFormat:@"com.escriba.aggregate.%@", [[NSUUID UUID] UUIDString]];
        NSDictionary *aggProps = @{
            [NSString stringWithUTF8String:kAudioAggregateDeviceNameKey]: @"EscribaAggregate",
            [NSString stringWithUTF8String:kAudioAggregateDeviceUIDKey]: aggUID,
            [NSString stringWithUTF8String:kAudioAggregateDeviceTapListKey]: tapList,
            [NSString stringWithUTF8String:kAudioAggregateDeviceTapAutoStartKey]: @NO,
            [NSString stringWithUTF8String:kAudioAggregateDeviceIsPrivateKey]: @YES
        };

        AudioDeviceID aggregateDeviceID = 0;
        status = AudioHardwareCreateAggregateDevice((__bridge CFDictionaryRef)aggProps, &aggregateDeviceID);
        if (status != noErr) {
            if (errorOut && errorOutSize > 0) snprintf(errorOut, errorOutSize, "AudioHardwareCreateAggregateDevice failed: %d", (int)status);
            goto fail;
        }
        ctx->aggregateDeviceID = aggregateDeviceID;

        AudioDeviceIOProcID procID = NULL;
        status = AudioDeviceCreateIOProcID(aggregateDeviceID, IOProcCallback, ctx, &procID);
        if (status != noErr) {
            if (errorOut && errorOutSize > 0) snprintf(errorOut, errorOutSize, "AudioDeviceCreateIOProcID failed: %d", (int)status);
            goto fail;
        }
        ctx->procID = procID;

        status = AudioDeviceStart(aggregateDeviceID, procID);
        if (status != noErr) {
            if (errorOut && errorOutSize > 0) snprintf(errorOut, errorOutSize, "AudioDeviceStart failed: %d", (int)status);
            goto fail;
        }

        /* Reset per-chain clock on success only (T4). lastCallbackTime is
         * zeroed so a fresh chain that never clocks is measured from
         * chainBuiltTime; IOProcCallback advances it on every invocation. */
        ctx->chainBuiltTime = NowSeconds();
        ctx->lastCallbackTime = 0.0;
        ctx->tapDead = 0;
    }

    return noErr;

fail:
    TeardownChainLocked(ctx);
    return status;
}

static void RebuildActiveChain(const char *reason, double requestedAt) {
    /* S2: hold gRegistryLock only long enough to read gActiveCtx, then release
     * it before any blocking HAL work. The HAL notification thread
     * (ChainChangedListener) bumps gRebuildGeneration under gRegistryLock; if
     * we kept the lock for the entire rebuild it would stall for seconds.
     * ctx->lock is sufficient to serialise rebuild vs. TeardownChainLocked in
     * CoreAudioTapStop. CoreAudioTapStop drains gRebuildQueue before free(ctx),
     * so ctx cannot be freed while this function is executing. */
    pthread_mutex_lock(&gRegistryLock);
    TapContext *ctx = gActiveCtx;
    pthread_mutex_unlock(&gRegistryLock);
    if (!ctx) return;

    pthread_mutex_lock(&ctx->lock);
    /* CoreAudioTapStop may have won the race and set running=0. */
    if (!ctx->running) {
        pthread_mutex_unlock(&ctx->lock);
        return;
    }

    /* M9: drop a request that another path already satisfied.
     *
     * The watchdog calls this on its own thread, NOT on gRebuildQueue, so a
     * listener block can clear its settle check against a stale
     * gLastRebuildTime, then block here on ctx->lock while the watchdog
     * rebuilds, then rebuild again the moment the lock is released — one
     * spurious teardown of a chain that is three milliseconds old, which is
     * exactly the audible glitch this sprint exists to remove. The generation
     * counter cannot catch it: this function bumps the generation on entry, so
     * HAL chatter provoked by our own rebuild gets a HIGHER generation and
     * reads as fresh.
     *
     * requestedAt is when the caller decided a rebuild was needed. If a
     * rebuild has completed since then, the caller's evidence is stale. */
    pthread_mutex_lock(&gRegistryLock);
    double completedSince = gLastRebuildTime;
    pthread_mutex_unlock(&gRegistryLock);
    if (CoreAudioTapShouldDropStaleRequest(requestedAt, completedSince)) {
        pthread_mutex_unlock(&ctx->lock);
        LogTap("skipped rebuild after %s — already rebuilt since the request", reason);
        return;
    }

    /* Cancel in-flight deferred listener timers so HAL chatter from this
     * teardown/rebuild cannot run after we return. */
    pthread_mutex_lock(&gRegistryLock);
    ++gRebuildGeneration;
    pthread_mutex_unlock(&gRegistryLock);

    TeardownChainLocked(ctx);
    UnregisterDeviceListenersLocked(ctx);
    RegisterDeviceListenersLocked(ctx);

    char err[256] = {0};
    OSStatus status = BuildChainLocked(ctx, err, sizeof(err));
    if (status != noErr) {
        /* T4: lastCallbackTime is left alone on failure so the stall clock
         * still shows the chain is dead and the watchdog can retry. */
        ctx->consecutiveRebuildFailures++;
        if (CoreAudioTapShouldMarkDead(ctx->consecutiveRebuildFailures)) {
            if (!ctx->tapDead) {
                ctx->tapDead = 1;
                /* Emit the machine-readable "[tap] dead" token (LogTap prepends
                 * "[tap] " so the call site must not repeat it). screen_capture.py
                 * matches the exact prefix to set a session warning. */
                LogTap("dead: %d consecutive rebuild failures (last: %s) — retrying every %.0fs",
                       ctx->consecutiveRebuildFailures, err[0] ? err : "unknown",
                       kMaxBackoffSeconds);
            }
        } else {
            LogTap("rebuild after %s failed (attempt %d/%d, OSStatus=%d): %s",
                   reason, ctx->consecutiveRebuildFailures, kMaxConsecutiveRebuildFailures,
                   (int)status, err[0] ? err : "unknown error");
        }
    } else {
        ctx->consecutiveRebuildFailures = 0;
        ctx->rebuildCount++;
        LogTap("rebuilt after %s (rate=%.0f channels=%u, rebuild #%llu)",
               reason, ctx->tapFormat.mSampleRate, (unsigned)ctx->tapFormat.mChannelsPerFrame,
               (unsigned long long)ctx->rebuildCount);
    }
    pthread_mutex_unlock(&ctx->lock);

    /* M2: stamp the shared failure-backoff clock so both the watchdog and the
     * listener path see the same "last attempt" time regardless of which path
     * triggered this rebuild. Stamped on every attempt (success AND failure)
     * so the backoff correctly measures time since the most recent attempt. */
    pthread_mutex_lock(&gRegistryLock);
    gLastRebuildTime = NowSeconds();
    pthread_mutex_unlock(&gRegistryLock);
}

/* W2/W5: coalesced listener → serial queue (generation counter).
 * W5: DEFER instead of DROP — if the debounce window is still active, schedule
 * a dispatch_after for the remaining interval, re-checking the generation,
 * so the LAST notification (the settled format) always eventually runs.
 * B2: guard NULL queue so a notification arriving before the queue is created
 * in CoreAudioTapStart does not crash libdispatch. */
static OSStatus ChainChangedListener(AudioObjectID inObjectID,
                                     UInt32 inNumberAddresses,
                                     const AudioObjectPropertyAddress *inAddresses,
                                     void *inClientData) {
    /* B2 NULL guard */
    if (!gRebuildQueue) return noErr;

    pthread_mutex_lock(&gRegistryLock);
    uint64_t gen = ++gRebuildGeneration;
    pthread_mutex_unlock(&gRegistryLock);

    dispatch_async(gRebuildQueue, ^{
        pthread_mutex_lock(&gRegistryLock);
        int stale = (gRebuildGeneration != gen);
        double lastRebuildTime = gLastListenerRebuildTime;
        pthread_mutex_unlock(&gRegistryLock);

        if (stale) return;

        double now = NowSeconds();
        pthread_mutex_lock(&gRegistryLock);
        double lastRebuild = gLastRebuildTime;
        pthread_mutex_unlock(&gRegistryLock);
        if (CoreAudioTapListenerShouldIgnore(now, lastRebuild, kRebuildSettleSeconds)) {
            return;
        }

        /* W4: honour tapDead and failure backoff on the listener path too. */
        pthread_mutex_lock(&gRegistryLock);
        TapContext *ctx = gActiveCtx;
        pthread_mutex_unlock(&gRegistryLock);
        if (ctx) {
            pthread_mutex_lock(&ctx->lock);
            int dead = ctx->tapDead;
            int failures = ctx->consecutiveRebuildFailures;
            pthread_mutex_unlock(&ctx->lock);
            if (dead) return;
            if (failures > 0) {
                double backoff = CoreAudioTapFailureBackoffSeconds(failures);
                if (backoff > kMaxBackoffSeconds) backoff = kMaxBackoffSeconds;
                /* M2: read the SHARED clock so watchdog-path failures gate the
                 * listener path too (and vice versa). gLastListenerRebuildTime is
                 * kept only for the 1s debounce below. */
                pthread_mutex_lock(&gRegistryLock);
                double lastAttempt = gLastRebuildTime;
                pthread_mutex_unlock(&gRegistryLock);
                if ((now - lastAttempt) < backoff) return;
            }
        }

        double remainingDebounce = 0.0;
        if (lastRebuildTime > 0.0 && (now - lastRebuildTime) < 1.0) {
            remainingDebounce = 1.0 - (now - lastRebuildTime);
        }

        if (remainingDebounce > 0.0) {
            /* W5: defer, don't drop — schedule after the debounce expires. */
            uint64_t delay = (uint64_t)(remainingDebounce * NSEC_PER_SEC);
            dispatch_after(dispatch_time(DISPATCH_TIME_NOW, delay), gRebuildQueue, ^{
                pthread_mutex_lock(&gRegistryLock);
                int stillFresh = (gRebuildGeneration == gen);
                pthread_mutex_unlock(&gRegistryLock);
                if (!stillFresh) return;

                /* M9: the settle window is re-checked at fire time, not only
                 * when the notification arrived — a rebuild may have landed
                 * during the deferral. */
                double firedAt = NowSeconds();
                pthread_mutex_lock(&gRegistryLock);
                double settledAgainst = gLastRebuildTime;
                gLastListenerRebuildTime = firedAt;
                pthread_mutex_unlock(&gRegistryLock);
                if (CoreAudioTapListenerShouldIgnore(firedAt, settledAgainst,
                                                     kRebuildSettleSeconds)) {
                    return;
                }

                RebuildActiveChain("route/format change (deferred)", firedAt);
            });
            return;
        }

        pthread_mutex_lock(&gRegistryLock);
        gLastListenerRebuildTime = NowSeconds();
        pthread_mutex_unlock(&gRegistryLock);

        /* M9: `now` is when this block cleared the settle check. If the
         * watchdog rebuilds while we wait on ctx->lock, that check is void. */
        RebuildActiveChain("route/format change", now);
    });
    return noErr;
}

/* --- pure functions (testable without Core Audio I/O) --------------------- */

int CoreAudioTapWatchdogShouldRebuild(double stalledSeconds,
                                      double stallThreshold) {
    return stalledSeconds >= stallThreshold;
}

int CoreAudioTapListenerShouldIgnore(double now,
                                     double lastRebuildTime,
                                     double settleSeconds) {
    return lastRebuildTime > 0.0 && settleSeconds > 0.0 && (now - lastRebuildTime) < settleSeconds;
}

/* M9: a rebuild request is stale when some other path completed a rebuild
 * after the request was made. requestedAt <= 0 means "unconditional". */
int CoreAudioTapShouldDropStaleRequest(double requestedAt, double lastRebuildTime) {
    return requestedAt > 0.0 && lastRebuildTime > requestedAt;
}

/* S3: extracted pure functions so test assertions can cover the math without
 * Core Audio I/O. */
double CoreAudioTapFailureBackoffSeconds(int consecutiveFailures) {
    int clampedFailures = consecutiveFailures < 4 ? consecutiveFailures : 4;
    return kFailureBackoffBaseSeconds * (double)(1 << clampedFailures);
}

int CoreAudioTapShouldMarkDead(int consecutiveFailures) {
    return consecutiveFailures >= kMaxConsecutiveRebuildFailures;
}

/* --- watchdog ------------------------------------------------------------- */

static void *WatchdogMain(void *arg) {
    TapContext *ctx = (TapContext *)arg;

    while (1) {
        struct timespec deadline;
        clock_gettime(CLOCK_REALTIME, &deadline);
        deadline.tv_sec += kWatchdogIntervalSec;

        pthread_mutex_lock(&ctx->lock);
        if (ctx->running) {
            pthread_cond_timedwait(&ctx->stopCond, &ctx->lock, &deadline);
        }
        if (!ctx->running) {
            pthread_mutex_unlock(&ctx->lock);
            break;
        }

        /* Clock-loss: lastCallbackTime advances on every IO-proc call, zeros
         * included. A missing callback is the only signal that the aggregate
         * has stopped clocking. Fresh chains start lastCallbackTime at 0 so
         * the stall is measured from chainBuiltTime. */
        double chainBuilt = ctx->chainBuiltTime;
        double lastCallback = ctx->lastCallbackTime;
        double now = NowSeconds();
        double lastActive = (lastCallback > 0.0) ? lastCallback : chainBuilt;
        double stalledSeconds = (lastActive > 0.0 && now > lastActive) ? (now - lastActive) : 0.0;

        int failures = ctx->consecutiveRebuildFailures;
        pthread_mutex_unlock(&ctx->lock);

        /* M2: read the shared failure-backoff clock that both this path and the
         * listener path stamp inside RebuildActiveChain. */
        pthread_mutex_lock(&gRegistryLock);
        double lastRebuild = gLastRebuildTime;
        pthread_mutex_unlock(&gRegistryLock);

        /* tapDead is not a permanent latch — use kMaxBackoffSeconds cadence. */
        double effectiveBackoff = 0.0;
        if (failures > 0) {
            effectiveBackoff = CoreAudioTapFailureBackoffSeconds(failures);
            if (effectiveBackoff > kMaxBackoffSeconds) effectiveBackoff = kMaxBackoffSeconds;
        }
        if (effectiveBackoff > 0.0 && (now - lastRebuild) < effectiveBackoff) continue;

        /* The dead flag means "we already logged the dead notice"; the watchdog
         * still evaluates the verdict and may attempt another rebuild. */
        if (CoreAudioTapWatchdogShouldRebuild(stalledSeconds, kClockStallRebuildSeconds)) {
            LogTap("IO proc stalled for %.0fs — rebuilding", stalledSeconds);
            RebuildActiveChain("IO proc stall", now);
            /* gLastRebuildTime is stamped inside RebuildActiveChain (M2). */
        }
    }
    return NULL;
}

/* --- public API ----------------------------------------------------------- */

CoreAudioTapHandle CoreAudioTapStart(CoreAudioTapSamplesCallback callback,
                                     void * _Nullable callbackContext,
                                     int targetSampleRate,
                                     int targetChannels,
                                     char * _Nullable errorOut,
                                     size_t errorOutSize) {
    if (!callback) {
        if (errorOut && errorOutSize > 0) snprintf(errorOut, errorOutSize, "callback is NULL");
        return NULL;
    }

    /* B2: create the serial rebuild queue HERE, before any listener is
     * registered, so ChainChangedListener never sees a NULL queue. */
    static dispatch_once_t once;
    dispatch_once(&once, ^{
        gRebuildQueue = dispatch_queue_create("com.escriba.tap.rebuild", DISPATCH_QUEUE_SERIAL);
    });

    TapContext *ctx = (TapContext *)calloc(1, sizeof(TapContext));
    if (!ctx) {
        if (errorOut && errorOutSize > 0) snprintf(errorOut, errorOutSize, "out of memory");
        return NULL;
    }
    ctx->callback = callback;
    ctx->callbackContext = callbackContext;
    ctx->targetSampleRate = targetSampleRate;
    ctx->targetChannels = targetChannels;
    ctx->running = 1;
    ctx->tapObjectID = kAudioObjectUnknown;
    ctx->aggregateDeviceID = kAudioObjectUnknown;
    ctx->listenedDeviceID = kAudioObjectUnknown;
    ctx->chainBuiltTime = 0.0;
    ctx->lastCallbackTime = 0.0;
    pthread_mutex_init(&ctx->lock, NULL);
    pthread_cond_init(&ctx->stopCond, NULL);

    pthread_mutex_lock(&ctx->lock);
    OSStatus status = BuildChainLocked(ctx, errorOut, errorOutSize);
    if (status != noErr) {
        pthread_mutex_unlock(&ctx->lock);
        pthread_cond_destroy(&ctx->stopCond);
        pthread_mutex_destroy(&ctx->lock);
        free(ctx);
        return NULL;
    }
    RegisterDeviceListenersLocked(ctx);
    pthread_mutex_unlock(&ctx->lock);

    AudioObjectPropertyAddress defaultOutAddr = {
        kAudioHardwarePropertyDefaultOutputDevice,
        kAudioObjectPropertyScopeGlobal,
        kAudioObjectPropertyElementMain
    };
    AudioObjectAddPropertyListener(kAudioObjectSystemObject, &defaultOutAddr,
                                   ChainChangedListener, NULL);

    pthread_mutex_lock(&gRegistryLock);
    gActiveCtx = ctx;
    /* W2: reset the debounce clock so the new session doesn't inherit a stale
     * timestamp from the previous one (which could misfire the 1s debounce). */
    gLastListenerRebuildTime = 0.0;
    /* M2: initialize the shared backoff clock to the chain-build time so a zero
     * value cannot mean "infinitely long ago" if failures arrive immediately. */
    gLastRebuildTime = NowSeconds();
    pthread_mutex_unlock(&gRegistryLock);

    if (pthread_create(&ctx->watchdog, NULL, WatchdogMain, ctx) == 0) {
        ctx->watchdogStarted = 1;
    } else {
        LogTap("could not start clock-loss watchdog; relying on route listeners only");
    }

    return (CoreAudioTapHandle)ctx;
}

void CoreAudioTapGetStats(CoreAudioTapHandle handle, uint64_t *outCallbacks, uint64_t *outFrames) {
    if (!handle || !outCallbacks || !outFrames) return;
    TapContext *ctx = (TapContext *)handle;
    *outCallbacks = ctx->callbackCount;
    *outFrames = ctx->totalFrames;
}

void CoreAudioTapStop(CoreAudioTapHandle handle) {
    if (!handle) return;
    TapContext *ctx = (TapContext *)handle;

    /* Stop ordering (T5 + B2 NULL-queue safety):
     * 1. Remove ctx from registry → no new dispatch blocks touch it.
     * 2. Remove system default-output listener → no new blocks dispatched.
     * 3. Lock ctx, set running=0, signal stopCond → watchdog wakes immediately.
     * 4. Remove per-device listeners and tear down chain (under ctx->lock).
     * 5. Join watchdog — safe because it checks running==0 under the lock.
     * 6. Destroy lock, cond, free ctx. */
    pthread_mutex_lock(&gRegistryLock);
    if (gActiveCtx == ctx) gActiveCtx = NULL;
    /* W2: bump generation so any pending dispatch_after timers from
     * ChainChangedListener see a mismatch and self-cancel. dispatch_sync only
     * drains already-queued blocks — unfired timers are still in-flight. */
    ++gRebuildGeneration;
    pthread_mutex_unlock(&gRegistryLock);

    AudioObjectPropertyAddress defaultOutAddr = {
        kAudioHardwarePropertyDefaultOutputDevice,
        kAudioObjectPropertyScopeGlobal,
        kAudioObjectPropertyElementMain
    };
    AudioObjectRemovePropertyListener(kAudioObjectSystemObject, &defaultOutAddr,
                                      ChainChangedListener, NULL);

    pthread_mutex_lock(&ctx->lock);
    ctx->running = 0;
    pthread_cond_signal(&ctx->stopCond);
    UnregisterDeviceListenersLocked(ctx);
    TeardownChainLocked(ctx);
    pthread_mutex_unlock(&ctx->lock);

    if (ctx->watchdogStarted) {
        pthread_join(ctx->watchdog, NULL);
    }

    /* Drain gRebuildQueue before freeing ctx. RebuildActiveChain no longer holds
     * gRegistryLock across the rebuild, so an in-flight dispatch_async block may
     * still hold ctx->lock when we reach here. dispatch_sync blocks until all
     * currently-enqueued blocks finish, after which no block can acquire ctx->lock.
     * NOTE: dispatch_sync does NOT drain unfired dispatch_after timers — those are
     * neutralised by the ++gRebuildGeneration bump above (W2), which causes them to
     * bail on the stillFresh check before touching gActiveCtx or ctx. */
    if (gRebuildQueue) {
        dispatch_sync(gRebuildQueue, ^{});
    }

    pthread_cond_destroy(&ctx->stopCond);
    pthread_mutex_destroy(&ctx->lock);
    free(ctx);
}

int CoreAudioTapCheckPermission(void) {
    @autoreleasepool {
        NSArray<NSNumber *> *exclude = @[];
        CATapDescription *tapDesc = [[CATapDescription alloc] initStereoGlobalTapButExcludeProcesses:exclude];
        if (!tapDesc) return 0;
        NSUUID *uuid = [NSUUID UUID];
        tapDesc.UUID = uuid;
        tapDesc.privateTap = YES;
        tapDesc.muteBehavior = CATapUnmuted;
        tapDesc.exclusive = YES;

        AudioObjectID tapID = 0;
        OSStatus status = AudioHardwareCreateProcessTap(tapDesc, &tapID);
        if (status == noErr) {
            AudioHardwareDestroyProcessTap(tapID);
            return 1;
        }
        return 0;
    }
}
