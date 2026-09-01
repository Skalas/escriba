#import <Foundation/Foundation.h>
#import <CoreAudio/CoreAudio.h>

NS_ASSUME_NONNULL_BEGIN

/// Callback type: (context, float32_data, num_frames, num_channels, sample_rate)
typedef void (*CoreAudioTapSamplesCallback)(void * _Nullable context,
                                            const float * _Nonnull data,
                                            UInt32 numFrames,
                                            UInt32 numChannels,
                                            Float64 sampleRate);

/// Opaque handle for the tap session
typedef void * CoreAudioTapHandle;

/// Create process tap and aggregate device, start capture. Returns handle or NULL on error.
/// Callback is invoked on the audio IO thread - do minimal work, copy data if needed.
CoreAudioTapHandle _Nullable CoreAudioTapStart(CoreAudioTapSamplesCallback callback,
                                     void * _Nullable callbackContext,
                                     int targetSampleRate,
                                     int targetChannels,
                                     char * _Nullable errorOut,
                                     size_t errorOutSize);

/// Stop capture and destroy tap. Handle becomes invalid.
void CoreAudioTapStop(CoreAudioTapHandle handle);

/// Get diagnostic stats (callback count, total frames). Set AUDIO_TAP_DEBUG=1 when running to see if tap is receiving data.
void CoreAudioTapGetStats(CoreAudioTapHandle handle, uint64_t * _Nonnull outCallbacks, uint64_t * _Nonnull outFrames);

/// Check if Audio Capture permission is available (macOS shows dialog on first create attempt).
int CoreAudioTapCheckPermission(void);

/// Pure watchdog verdict: returns 1 if a rebuild should be triggered, 0 otherwise.
/// Extracted for unit testing — this is the only part of the watchdog that can be
/// exercised without real Core Audio I/O or TCC grants.
///
/// Rebuild on loss of clock, not loss of signal. A tap that is still invoking
/// its IO proc is healthy even when every sample is zero (muted meeting, pause
/// between speakers). Tearing that chain down reconfigures the default output
/// and is what the user hears as a glitch.
///
/// @param stalledSeconds  Seconds since the last IO-proc callback, or since
///                        the chain was built if no callback has arrived yet.
/// @param stallThreshold  Seconds of missing callbacks that count as dead.
int CoreAudioTapWatchdogShouldRebuild(double stalledSeconds,
                                      double stallThreshold);

/// Returns 1 when a route/format listener should drop this notification
/// because we just rebuilt the chain ourselves. Creating or destroying the
/// aggregate device makes HAL emit format events; acting on those retriggers
/// the rebuild and glitches playback.
int CoreAudioTapListenerShouldIgnore(double now,
                                     double lastRebuildTime,
                                     double settleSeconds);

/// M9: returns non-zero when a rebuild request has been overtaken — some other
/// path completed a rebuild after this request was made, so the evidence behind
/// it (a settle check, a stall reading) no longer describes the live chain.
/// The watchdog does not run on the rebuild queue, so this is the only thing
/// standing between a coincident listener event and a redundant teardown.
/// @param requestedAt      When the caller decided a rebuild was needed. <= 0 disables the check.
/// @param lastRebuildTime  When the most recent rebuild attempt completed.
int CoreAudioTapShouldDropStaleRequest(double requestedAt, double lastRebuildTime);

/// S3: exponential backoff for consecutive rebuild failures.
/// Returns kRebuildCooldownSeconds * 2^min(failures, 4). Capped by the caller at kMaxBackoffSeconds.
double CoreAudioTapFailureBackoffSeconds(int consecutiveFailures);

/// S3: returns non-zero when failures have reached the dead-tap threshold.
int CoreAudioTapShouldMarkDead(int consecutiveFailures);

NS_ASSUME_NONNULL_END
