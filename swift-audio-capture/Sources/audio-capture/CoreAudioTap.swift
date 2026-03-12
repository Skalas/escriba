import Foundation
import CoreAudioTapBridge

/// Captures system audio using Core Audio Taps API (macOS 14.2+).
/// Uses Audio Capture permission only — no Screen Recording required.
@available(macOS 14.2, *)
class CoreAudioTapCapture {
    private var handle: CoreAudioTapHandle?
    private let sampleRate: Int
    private let channelCount: Int
    private let outputHandler: (Data) -> Void
    private let outputQueue: DispatchQueue
    private var isCapturing = false

    init(sampleRate: Int = 16000, channelCount: Int = 1, outputHandler: @escaping (Data) -> Void) {
        self.sampleRate = sampleRate
        self.channelCount = channelCount
        self.outputHandler = outputHandler
        self.outputQueue = DispatchQueue(label: "com.audio-capture.output-queue", qos: .userInitiated)
    }

    /// Check if Audio Capture permission is granted (no public API; attempt create and destroy).
    static func checkPermission() -> Bool {
        return CoreAudioTapCheckPermission() != 0
    }

    /// Permission is requested by macOS when first creating a tap (no explicit request API).
    static func requestPermission() {
        _ = CoreAudioTapCheckPermission()
    }

    /// Start capturing audio.
    func start() throws {
        guard !isCapturing else {
            throw CoreAudioTapError.alreadyCapturing
        }

        var errBuf = [CChar](repeating: 0, count: 256)
        let ctx = Unmanaged.passUnretained(self).toOpaque()

        let callback: @convention(c) (UnsafeMutableRawPointer?, UnsafePointer<Float>, UInt32, UInt32, Float64) -> Void = { context, data, numFrames, numChannels, sampleRate in
            guard let context = context, numFrames > 0 else { return }
            let capture = Unmanaged<CoreAudioTapCapture>.fromOpaque(context).takeUnretainedValue()
            capture.handleSamples(data: data, numFrames: numFrames, numChannels: numChannels, sampleRate: sampleRate)
        }

        guard let h = CoreAudioTapStart(
            callback,
            ctx,
            Int32(sampleRate),
            Int32(channelCount),
            &errBuf,
            errBuf.count
        ) else {
            let msg = String(cString: errBuf, encoding: .utf8) ?? "unknown error"
            throw CoreAudioTapError.startFailed(msg)
        }

        handle = h
        isCapturing = true
    }

    /// Called from C callback on audio thread — copy data and dispatch to output queue.
    func handleSamples(data: UnsafePointer<Float>, numFrames: UInt32, numChannels: UInt32, sampleRate: Float64) {
        let frameCount = Int(numFrames)
        let chCount = Int(numChannels)
        let floatCount = frameCount * chCount

        let floats = Array(UnsafeBufferPointer(start: data, count: floatCount))
        let pcmData = PCMConverter.convertFloat32ToInt16(floats, channelCount: chCount)

        var resampled = pcmData
        let inputRate = Int(sampleRate)
        if inputRate != self.sampleRate {
            resampled = PCMConverter.resample(pcmData, fromRate: inputRate, toRate: self.sampleRate)
        }

        outputQueue.async { [weak self] in
            self?.outputHandler(resampled)
        }
    }

    /// Diagnostic: number of IO callbacks and total frames received. Use AUDIO_TAP_DEBUG=1 when running to see if tap receives data.
    func getStats() -> (callbacks: UInt64, frames: UInt64) {
        var c: UInt64 = 0, f: UInt64 = 0
        if let h = handle {
            CoreAudioTapGetStats(h, &c, &f)
        }
        return (c, f)
    }

    /// Stop capturing.
    func stop() {
        guard isCapturing else { return }
        isCapturing = false
        if let h = handle {
            CoreAudioTapStop(h)
            handle = nil
        }
    }
}

enum CoreAudioTapError: Error, LocalizedError {
    case alreadyCapturing
    case startFailed(String)

    var errorDescription: String? {
        switch self {
        case .alreadyCapturing:
            return "Audio capture is already running"
        case .startFailed(let details):
            return "Failed to start capture: \(details)"
        }
    }
}
