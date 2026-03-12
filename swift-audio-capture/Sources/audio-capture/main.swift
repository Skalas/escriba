import Foundation
import Darwin

// Helper for stderr output
var stderr = FileHandle.standardError

// Global variables for signal handling (Any? to avoid @available on global)
var audioCapture: Any?
var shouldStop = false

// Signal handler (wrapper so we can register at top level)
func signalHandler(signal: Int32) {
    shouldStop = true
    if #available(macOS 14.2, *) {
        (audioCapture as? CoreAudioTapCapture)?.stop()
    }
    if #available(macOS 13.0, *) {
        (audioCapture as? AudioCapture)?.stop()
    }
    exit(0)
}

// Setup signal handlers
signal(SIGINT, signalHandler)
signal(SIGTERM, signalHandler)

// Parse command line arguments
var sampleRate = 16000
var channelCount = 1
var listMode = false
var useScreenCapture = ProcessInfo.processInfo.environment["USE_SCREEN_CAPTURE_KIT"] == "1"

var args = CommandLine.arguments
args.removeFirst() // Remove program name

var i = 0
while i < args.count {
    let arg = args[i]
    
    switch arg {
    case "--sample-rate":
        if i + 1 < args.count, let rate = Int(args[i + 1]) {
            sampleRate = rate
            i += 2
        } else {
            try? stderr.write(contentsOf: "Error: --sample-rate requires a value\n".data(using: .utf8)!)
            exit(1)
        }
    case "--channels":
        if i + 1 < args.count, let channels = Int(args[i + 1]) {
            channelCount = channels
            i += 2
        } else {
            try? stderr.write(contentsOf: "Error: --channels requires a value\n".data(using: .utf8)!)
            exit(1)
        }
    case "--list":
        listMode = true
        i += 1
    case "--use-screen-capture":
        useScreenCapture = true
        i += 1
    case "--help", "-h":
        print("""
        audio-capture - Capture system audio (Core Audio Taps or ScreenCaptureKit)
        
        Usage:
          audio-capture [options]
          audio-capture --list
          audio-capture --use-screen-capture   # Fallback: Screen Recording permission, captures system audio reliably
        
        Options:
          --sample-rate <rate>    Sample rate in Hz (default: 16000)
          --channels <count>      Number of channels (default: 1, mono)
          --use-screen-capture    Use ScreenCaptureKit (requires Screen Recording permission; use if Core Audio Taps gives no system audio)
          --list                  Check permission
          --help, -h              Show this help message
        
        Output:
          Writes raw PCM audio (int16, little-endian) to stdout.
          Press Ctrl+C to stop.
        
        Modes:
          - Default: Core Audio Taps (macOS 14.2+), Audio Capture permission only. On some Macs/system versions no system audio is captured; use --use-screen-capture then.
          - --use-screen-capture: ScreenCaptureKit (macOS 13+), requires Screen Recording permission. Captures system audio reliably.
        """)
        exit(0)
    default:
        try? stderr.write(contentsOf: "Unknown option: \(arg)\n".data(using: .utf8)!)
        try? stderr.write(contentsOf: "Use --help for usage information\n".data(using: .utf8)!)
        exit(1)
    }
}

// Validate arguments
if sampleRate < 8000 || sampleRate > 48000 {
    try? stderr.write(contentsOf: "Error: Sample rate must be between 8000 and 48000 Hz\n".data(using: .utf8)!)
    exit(1)
}

if channelCount < 1 || channelCount > 2 {
    try? stderr.write(contentsOf: "Error: Channel count must be 1 (mono) or 2 (stereo)\n".data(using: .utf8)!)
    exit(1)
}

// Handle list mode (permission check)
if listMode {
    if useScreenCapture {
        if #available(macOS 13.0, *) {
            if AudioCapture.checkPermission() {
                print("Screen Recording permission: granted (ScreenCaptureKit)")
            } else {
                print("Screen Recording permission: not granted")
                try? stderr.write(contentsOf: "Grant permission in: System Settings > Privacy & Security > Screen Recording. Add your terminal app.\n".data(using: .utf8)!)
                exit(1)
            }
        } else {
            try? stderr.write(contentsOf: "Error: macOS 13+ required for ScreenCaptureKit\n".data(using: .utf8)!)
            exit(1)
        }
    } else if #available(macOS 14.2, *) {
        if CoreAudioTapCapture.checkPermission() {
            print("Audio Capture permission: granted (Core Audio Taps)")
        } else {
            print("Audio Capture permission: not granted")
            try? stderr.write(contentsOf: "Grant permission in: System Settings > Privacy & Security > Screen & System Audio Recording\n".data(using: .utf8)!)
            exit(1)
        }
    } else {
        try? stderr.write(contentsOf: "Error: macOS 14.2+ required for Core Audio Taps (or use --use-screen-capture on macOS 13+)\n".data(using: .utf8)!)
        exit(1)
    }
    exit(0)
}

// Check permissions and start capture
if useScreenCapture {
    if #available(macOS 13.0, *) {
        if !AudioCapture.checkPermission() {
            try? stderr.write(contentsOf: "Error: Screen Recording permission required for --use-screen-capture. Grant in System Settings > Privacy & Security > Screen Recording.\n".data(using: .utf8)!)
            exit(1)
        }
        let stdout = FileHandle.standardOutput
        let capture = AudioCapture(sampleRate: sampleRate, channelCount: channelCount) { pcmData in
            do {
                try stdout.write(contentsOf: pcmData)
            } catch {
                try? stderr.write(contentsOf: "Error writing to stdout: \(error.localizedDescription)\n".data(using: .utf8)!)
                shouldStop = true
            }
        }
        audioCapture = capture
        do {
            try capture.start()
            try? stderr.write(contentsOf: "Started audio capture (ScreenCaptureKit). Sample rate: \(sampleRate) Hz, Channels: \(channelCount)\n".data(using: .utf8)!)
            try? stderr.write(contentsOf: "Writing PCM data to stdout. Press Ctrl+C to stop.\n".data(using: .utf8)!)
        } catch {
            try? stderr.write(contentsOf: "Error starting capture: \(error.localizedDescription)\n".data(using: .utf8)!)
            exit(1)
        }
    } else {
        try? stderr.write(contentsOf: "Error: macOS 13+ required for --use-screen-capture\n".data(using: .utf8)!)
        exit(1)
    }
} else if #available(macOS 14.2, *) {
    if !CoreAudioTapCapture.checkPermission() {
        let errorMsg = """
        Error: Audio Capture permission is required.
        
        Please grant permission in:
          System Settings > Privacy & Security > Screen & System Audio Recording
          Add your terminal app (Terminal, iTerm, etc.)
        
        If system audio is not captured, try: audio-capture --use-screen-capture
        (requires Screen Recording permission instead.)
        
        Attempting to request permission (run without --list to trigger dialog)...
        """
        try? stderr.write(contentsOf: errorMsg.data(using: .utf8)!)
        
        CoreAudioTapCapture.requestPermission()
        
        Thread.sleep(forTimeInterval: 2.0)
        
        if !CoreAudioTapCapture.checkPermission() {
            try? stderr.write(contentsOf: "Error: Audio Capture permission not granted. Please grant permission and try again.\n".data(using: .utf8)!)
            exit(1)
        }
    }
    
    let stdout = FileHandle.standardOutput
    
    let capture = CoreAudioTapCapture(sampleRate: sampleRate, channelCount: channelCount) { pcmData in
        do {
            try stdout.write(contentsOf: pcmData)
        } catch {
            try? stderr.write(contentsOf: "Error writing to stdout: \(error.localizedDescription)\n".data(using: .utf8)!)
            shouldStop = true
        }
    }
    audioCapture = capture
    
    do {
        try capture.start()
        try? stderr.write(contentsOf: "Started audio capture (Core Audio Taps). Sample rate: \(sampleRate) Hz, Channels: \(channelCount)\n".data(using: .utf8)!)
        try? stderr.write(contentsOf: "Writing PCM data to stdout. Press Ctrl+C to stop.\n".data(using: .utf8)!)
        if ProcessInfo.processInfo.environment["AUDIO_TAP_DEBUG"] == "1" {
            try? stderr.write(contentsOf: "AUDIO_TAP_DEBUG: will log tap stats every 2s to stderr. Play system audio to test.\n".data(using: .utf8)!)
            DispatchQueue.global(qos: .utility).async {
                while !shouldStop {
                    Thread.sleep(forTimeInterval: 2.0)
                    if shouldStop { break }
                    let (c, f) = capture.getStats()
                    let line = "Core Audio Taps: \(c) callbacks, \(f) frames\n"
                    try? stderr.write(contentsOf: line.data(using: .utf8)!)
                }
            }
        }
    } catch {
        try? stderr.write(contentsOf: "Error starting capture: \(error.localizedDescription)\n".data(using: .utf8)!)
        exit(1)
    }
} else {
    try? stderr.write(contentsOf: "Error: macOS 14.2+ required for Core Audio Taps (or use --use-screen-capture on macOS 13+)\n".data(using: .utf8)!)
    exit(1)
}

// Run until stopped
// NOTE: runLoop.run() returns false if no events were processed.
// We must NOT use it as a loop condition, or the loop exits immediately
// if there are no pending events (race condition with audio capture startup).
let runLoop = RunLoop.current
while !shouldStop {
    // Process run loop events with a short timeout
    // This keeps the process alive while Core Audio Taps captures audio
    runLoop.run(until: Date(timeIntervalSinceNow: 0.1))
}

// Cleanup
if #available(macOS 14.2, *) {
    (audioCapture as? CoreAudioTapCapture)?.stop()
}
if #available(macOS 13.0, *) {
    (audioCapture as? AudioCapture)?.stop()
}
exit(0)
