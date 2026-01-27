import Foundation
import ScreenCaptureKit
import CoreMedia
import CoreAudio

/// Delegate for receiving audio samples from ScreenCaptureKit
@available(macOS 13.0, *)
class AudioStreamDelegate: NSObject, SCStreamOutput {
    var onAudioSample: ((CMSampleBuffer) -> Void)?
    
    func stream(_ stream: SCStream, didOutputSampleBuffer sampleBuffer: CMSampleBuffer, of type: SCStreamOutputType) {
        if type == .audio {
            onAudioSample?(sampleBuffer)
        }
    }
}

/// Captures system audio using ScreenCaptureKit
@available(macOS 13.0, *)
class AudioCapture {
    private var stream: SCStream?
    private var delegate: AudioStreamDelegate?
    private var isCapturing = false
    private let sampleRate: Int
    private let channelCount: Int
    private let outputHandler: (Data) -> Void
    
    init(sampleRate: Int = 16000, channelCount: Int = 1, outputHandler: @escaping (Data) -> Void) {
        self.sampleRate = sampleRate
        self.channelCount = channelCount
        self.outputHandler = outputHandler
    }
    
    /// Check if Screen Recording permission is granted
    static func checkPermission() -> Bool {
        // Try to get shareable content - this will fail if permission is not granted
        let semaphore = DispatchSemaphore(value: 0)
        var hasPermission = false
        
        SCShareableContent.getWithCompletionHandler { content, error in
            hasPermission = error == nil && content != nil
            semaphore.signal()
        }
        
        _ = semaphore.wait(timeout: .now() + 2.0)
        return hasPermission
    }
    
    /// Request Screen Recording permission (macOS will show dialog automatically)
    static func requestPermission() {
        SCShareableContent.getWithCompletionHandler { _, _ in
            // Permission dialog will be shown by macOS
        }
    }
    
    /// List available displays
    static func listDisplays() {
        let semaphore = DispatchSemaphore(value: 0)
        var displays: [SCDisplay] = []
        
        SCShareableContent.getWithCompletionHandler { content, error in
            if let error = error {
                print("Error getting displays: \(error.localizedDescription)")
                semaphore.signal()
                return
            }
            
            if let content = content {
                displays = content.displays
            }
            semaphore.signal()
        }
        
        _ = semaphore.wait(timeout: .now() + 5.0)
        
        if displays.isEmpty {
            print("No displays found")
        } else {
            print("Available displays:")
            for (index, display) in displays.enumerated() {
                print("  [\(index)] Display ID: \(display.displayID), Width: \(display.width), Height: \(display.height)")
            }
        }
    }
    
    /// Start capturing audio
    func start() throws {
        guard !isCapturing else {
            throw AudioCaptureError.alreadyCapturing
        }
        
        // Get shareable content
        let semaphore = DispatchSemaphore(value: 0)
        var shareableContent: SCShareableContent?
        var contentError: Error?
        
        SCShareableContent.getWithCompletionHandler { content, error in
            shareableContent = content
            contentError = error
            semaphore.signal()
        }
        
        _ = semaphore.wait(timeout: .now() + 5.0)
        
        guard let content = shareableContent, contentError == nil else {
            throw AudioCaptureError.permissionDenied(contentError?.localizedDescription ?? "Unknown error")
        }
        
        guard let display = content.displays.first else {
            throw AudioCaptureError.noDisplays
        }
        
        // Create content filter
        let filter = SCContentFilter(display: display, excludingWindows: [])
        
        // Create stream configuration
        let config = SCStreamConfiguration()
        config.capturesAudio = true
        config.excludesCurrentProcessAudio = false
        config.sampleRate = sampleRate
        config.channelCount = channelCount
        
        // Create delegate
        delegate = AudioStreamDelegate()
        delegate?.onAudioSample = { [weak self] sampleBuffer in
            self?.processAudioSample(sampleBuffer)
        }
        
        // Create stream (delegate is optional, we use SCStreamOutput instead)
        stream = SCStream(filter: filter, configuration: config, delegate: nil)
        
        // Create a queue for audio sample handling
        let audioQueue = DispatchQueue(label: "com.audio-capture.audio-queue", qos: .userInitiated)
        
        // Add audio output (keep strong reference to delegate)
        do {
            try stream?.addStreamOutput(delegate!, type: .audio, sampleHandlerQueue: audioQueue)
        } catch {
            throw AudioCaptureError.startFailed("Failed to add stream output: \(error.localizedDescription)")
        }
        
        // Start capture
        let startSemaphore = DispatchSemaphore(value: 0)
        var startError: Error?
        
        stream?.startCapture { error in
            startError = error
            startSemaphore.signal()
        }
        
        _ = startSemaphore.wait(timeout: .now() + 5.0)
        
        if let error = startError {
            throw AudioCaptureError.startFailed(error.localizedDescription)
        }
        
        isCapturing = true
    }
    
    /// Process audio sample buffer
    private func processAudioSample(_ sampleBuffer: CMSampleBuffer) {
        guard let formatDescription = CMSampleBufferGetFormatDescription(sampleBuffer) else {
            return
        }
        
        // Get audio format description
        guard let formatPtr = CMAudioFormatDescriptionGetStreamBasicDescription(formatDescription) else {
            return
        }
        let format = formatPtr.pointee
        
        // Get audio data
        guard let dataBuffer = CMSampleBufferGetDataBuffer(sampleBuffer) else {
            return
        }
        
        // Get total length
        let totalLength = dataBuffer.dataLength
        guard totalLength > 0 else {
            return
        }
        
        // Copy data from CMBlockBuffer
        var buffer = [UInt8](repeating: 0, count: totalLength)
        let copyStatus = CMBlockBufferCopyDataBytes(
            dataBuffer,
            atOffset: 0,
            dataLength: totalLength,
            destination: &buffer
        )
        
        guard copyStatus == noErr else {
            return
        }
        
        let audioData = Data(buffer)
        
        // Convert to PCM int16
        let pcmData = convertToPCM(audioData, format: format)
        outputHandler(pcmData)
    }
    
    /// Convert audio data to PCM int16
    private func convertToPCM(_ data: Data, format: AudioStreamBasicDescription) -> Data {
        // ScreenCaptureKit typically provides float32 audio
        if format.mFormatID == kAudioFormatLinearPCM && format.mBitsPerChannel == 32 {
            // Float32 format
            let sampleCount = data.count / MemoryLayout<Float>.size
            let floatData = data.withUnsafeBytes { bytes -> [Float] in
                guard let baseAddress = bytes.baseAddress else { return [] }
                let floatPointer = baseAddress.assumingMemoryBound(to: Float.self)
                return Array(UnsafeBufferPointer(start: floatPointer, count: sampleCount))
            }
            
            guard !floatData.isEmpty else { return Data() }
            
            let inputChannels = Int(format.mChannelsPerFrame)
            return PCMConverter.convertFloat32ToInt16(floatData, channelCount: inputChannels)
        } else if format.mFormatID == kAudioFormatLinearPCM && format.mBitsPerChannel == 16 {
            // Already int16, but may need resampling or channel conversion
            var pcmData = data
            
            // Resample if needed
            let inputRate = Int(format.mSampleRate)
            if inputRate != sampleRate {
                pcmData = PCMConverter.resample(pcmData, fromRate: inputRate, toRate: sampleRate)
            }
            
            // Convert channels if needed
            let inputChannels = Int(format.mChannelsPerFrame)
            if inputChannels != channelCount && inputChannels > 1 {
                // Mix to mono
                let sampleCount = pcmData.count / MemoryLayout<Int16>.size / inputChannels
                let int16Data = pcmData.withUnsafeBytes { bytes -> [Int16] in
                    guard let baseAddress = bytes.baseAddress else { return [] }
                    let int16Pointer = baseAddress.assumingMemoryBound(to: Int16.self)
                    return Array(UnsafeBufferPointer(start: int16Pointer, count: pcmData.count / MemoryLayout<Int16>.size))
                }
                
                guard !int16Data.isEmpty else { return pcmData }
                
                var monoData = [Int16](repeating: 0, count: sampleCount)
                for i in 0..<sampleCount {
                    var sum = 0
                    for ch in 0..<inputChannels {
                        sum += Int(int16Data[i * inputChannels + ch])
                    }
                    monoData[i] = Int16(sum / inputChannels)
                }
                
                pcmData = Data(bytes: monoData, count: monoData.count * MemoryLayout<Int16>.size)
            }
            
            return pcmData
        }
        
        // Unknown format, return as-is (may cause issues)
        return data
    }
    
    /// Stop capturing
    func stop() {
        guard isCapturing else {
            return
        }
        
        isCapturing = false
        
        let semaphore = DispatchSemaphore(value: 0)
        stream?.stopCapture { error in
            if let error = error {
                let errorMsg = "Error stopping capture: \(error.localizedDescription)\n"
                try? FileHandle.standardError.write(contentsOf: errorMsg.data(using: .utf8)!)
            }
            semaphore.signal()
        }
        
        _ = semaphore.wait(timeout: .now() + 2.0)
        
        stream = nil
        delegate = nil
    }
}

enum AudioCaptureError: Error, LocalizedError {
    case alreadyCapturing
    case permissionDenied(String)
    case noDisplays
    case startFailed(String)
    
    var errorDescription: String? {
        switch self {
        case .alreadyCapturing:
            return "Audio capture is already running"
        case .permissionDenied(let details):
            return "Screen Recording permission denied: \(details)"
        case .noDisplays:
            return "No displays found"
        case .startFailed(let details):
            return "Failed to start capture: \(details)"
        }
    }
}
