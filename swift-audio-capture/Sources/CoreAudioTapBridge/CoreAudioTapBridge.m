#import "CoreAudioTapBridge.h"
#import <CoreAudio/AudioHardware.h>
#import <CoreAudio/AudioHardwareTapping.h>
#import <CoreAudio/CATapDescription.h>
#import <stdlib.h>
#import <string.h>

typedef struct {
    CoreAudioTapSamplesCallback callback;
    void *callbackContext;
    int targetSampleRate;
    int targetChannels;
    AudioStreamBasicDescription tapFormat;
    volatile int running;
    volatile uint64_t callbackCount;
    volatile uint64_t totalFrames;
} TapContext;

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

    /* Single buffer: interleaved (e.g. L,R,L,R). Use as-is. */
    if (inInputData->mNumberBuffers == 1) {
        ctx->callbackCount++;
        ctx->totalFrames += numFrames;
        ctx->callback(ctx->callbackContext,
                      (const float *)buf->mData,
                      numFrames,
                      buf->mNumberChannels,
                      ctx->tapFormat.mSampleRate);
        return noErr;
    }

    /* Non-interleaved: one buffer per channel. Mix to mono and pass (e.g. buffer 0 = L, buffer 1 = R). */
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
    ctx->callbackCount++;
    ctx->totalFrames += numFrames;
    ctx->callback(ctx->callbackContext, mix, numFrames, 1, ctx->tapFormat.mSampleRate);
    free(mix);
    return noErr;
}

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

    AudioObjectID tapObjectID = 0;
    AudioDeviceID aggregateDeviceID = 0;
    AudioDeviceIOProcID procID = NULL;

    @autoreleasepool {
        NSArray<NSNumber *> *excludeProcesses = @[];
        /* Use global tap only: tap on default output device often delivers silence
         * (e.g. Bluetooth). Global tap captures system-wide mix. */
        CATapDescription *tapDesc = [[CATapDescription alloc] initStereoGlobalTapButExcludeProcesses:excludeProcesses];
        if (!tapDesc) {
            if (errorOut && errorOutSize > 0) snprintf(errorOut, errorOutSize, "failed to create CATapDescription");
            free(ctx);
            return NULL;
        }

        NSUUID *tapUUID = [NSUUID UUID];
        tapDesc.UUID = tapUUID;
        tapDesc.name = [NSString stringWithFormat:@"local-transcriber-tap-%@", [tapUUID UUIDString]];
        /* Public tap: on some macOS versions private tap receives no audio; public can fix it. */
        tapDesc.privateTap = NO;
        tapDesc.muteBehavior = CATapUnmuted;
        tapDesc.exclusive = YES;

        OSStatus status = AudioHardwareCreateProcessTap(tapDesc, &tapObjectID);
        if (status != noErr) {
            if (errorOut && errorOutSize > 0) snprintf(errorOut, errorOutSize, "AudioHardwareCreateProcessTap failed: %d", (int)status);
            free(ctx);
            return NULL;
        }

        UInt32 formatSize = sizeof(AudioStreamBasicDescription);
        AudioObjectPropertyAddress propAddr = {
            kAudioTapPropertyFormat,
            kAudioObjectPropertyScopeGlobal,
            kAudioObjectPropertyElementMain
        };
        status = AudioObjectGetPropertyData(tapObjectID, &propAddr, 0, NULL, &formatSize, &ctx->tapFormat);
        if (status != noErr) {
            AudioHardwareDestroyProcessTap(tapObjectID);
            if (errorOut && errorOutSize > 0) snprintf(errorOut, errorOutSize, "failed to get tap format: %d", (int)status);
            free(ctx);
            return NULL;
        }

        NSString *tapUIDString = [tapUUID UUIDString];
        NSArray *tapList = @[
            @{
                [NSString stringWithUTF8String:kAudioSubTapUIDKey]: tapUIDString,
                [NSString stringWithUTF8String:kAudioSubTapDriftCompensationKey]: @YES
            }
        ];
        NSString *aggUID = [NSString stringWithFormat:@"com.local-transcriber.aggregate.%@", [[NSUUID UUID] UUIDString]];
        NSDictionary *aggProps = @{
            [NSString stringWithUTF8String:kAudioAggregateDeviceNameKey]: @"LocalTranscriberAggregate",
            [NSString stringWithUTF8String:kAudioAggregateDeviceUIDKey]: aggUID,
            [NSString stringWithUTF8String:kAudioAggregateDeviceTapListKey]: tapList,
            [NSString stringWithUTF8String:kAudioAggregateDeviceTapAutoStartKey]: @NO,
            [NSString stringWithUTF8String:kAudioAggregateDeviceIsPrivateKey]: @YES
        };

        status = AudioHardwareCreateAggregateDevice((__bridge CFDictionaryRef)aggProps, &aggregateDeviceID);
        if (status != noErr) {
            AudioHardwareDestroyProcessTap(tapObjectID);
            if (errorOut && errorOutSize > 0) snprintf(errorOut, errorOutSize, "AudioHardwareCreateAggregateDevice failed: %d", (int)status);
            free(ctx);
            return NULL;
        }

        status = AudioDeviceCreateIOProcID(aggregateDeviceID, IOProcCallback, ctx, &procID);
        if (status != noErr) {
            AudioHardwareDestroyAggregateDevice(aggregateDeviceID);
            AudioHardwareDestroyProcessTap(tapObjectID);
            if (errorOut && errorOutSize > 0) snprintf(errorOut, errorOutSize, "AudioDeviceCreateIOProcID failed: %d", (int)status);
            free(ctx);
            return NULL;
        }

        status = AudioDeviceStart(aggregateDeviceID, procID);
        if (status != noErr) {
            AudioDeviceDestroyIOProcID(aggregateDeviceID, procID);
            AudioHardwareDestroyAggregateDevice(aggregateDeviceID);
            AudioHardwareDestroyProcessTap(tapObjectID);
            if (errorOut && errorOutSize > 0) snprintf(errorOut, errorOutSize, "AudioDeviceStart failed: %d", (int)status);
            free(ctx);
            return NULL;
        }
    }

    typedef struct {
        TapContext *ctx;
        AudioObjectID tapObjectID;
        AudioDeviceID aggregateDeviceID;
        AudioDeviceIOProcID procID;
    } HandleStorage;
    HandleStorage *storage = (HandleStorage *)calloc(1, sizeof(HandleStorage));
    if (!storage) {
        AudioDeviceStop(aggregateDeviceID, procID);
        AudioDeviceDestroyIOProcID(aggregateDeviceID, procID);
        AudioHardwareDestroyAggregateDevice(aggregateDeviceID);
        AudioHardwareDestroyProcessTap(tapObjectID);
        free(ctx);
        return NULL;
    }
    storage->ctx = ctx;
    storage->tapObjectID = tapObjectID;
    storage->aggregateDeviceID = aggregateDeviceID;
    storage->procID = procID;
    return (CoreAudioTapHandle)storage;
}

void CoreAudioTapGetStats(CoreAudioTapHandle handle, uint64_t *outCallbacks, uint64_t *outFrames) {
    if (!handle || !outCallbacks || !outFrames) return;
    typedef struct { TapContext *ctx; AudioObjectID tapObjectID; AudioDeviceID aggregateDeviceID; AudioDeviceIOProcID procID; } HandleStorage;
    HandleStorage *storage = (HandleStorage *)handle;
    *outCallbacks = storage->ctx->callbackCount;
    *outFrames = storage->ctx->totalFrames;
}

void CoreAudioTapStop(CoreAudioTapHandle handle) {
    if (!handle) return;
    typedef struct {
        TapContext *ctx;
        AudioObjectID tapObjectID;
        AudioDeviceID aggregateDeviceID;
        AudioDeviceIOProcID procID;
    } HandleStorage;
    HandleStorage *storage = (HandleStorage *)handle;
    TapContext *ctx = storage->ctx;
    AudioObjectID tapObjectID = storage->tapObjectID;
    AudioDeviceID aggregateDeviceID = storage->aggregateDeviceID;
    AudioDeviceIOProcID procID = storage->procID;

    ctx->running = 0;
    AudioDeviceStop(aggregateDeviceID, procID);
    AudioDeviceDestroyIOProcID(aggregateDeviceID, procID);
    AudioHardwareDestroyAggregateDevice(aggregateDeviceID);
    AudioHardwareDestroyProcessTap(tapObjectID);
    free(ctx);
    free(storage);
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
