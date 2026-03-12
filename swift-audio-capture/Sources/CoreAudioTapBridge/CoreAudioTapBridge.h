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

NS_ASSUME_NONNULL_END
