import Foundation
import Darwin

// Helper for stderr output
var stderr = FileHandle.standardError

// Global variables for signal handling
var audioCapture: AudioCapture?
var shouldStop = false

// Signal handler
func signalHandler(signal: Int32) {
    shouldStop = true
    audioCapture?.stop()
    exit(0)
}

// Setup signal handlers
signal(SIGINT, signalHandler)
signal(SIGTERM, signalHandler)

// Parse command line arguments
var sampleRate = 16000
var channelCount = 1
var listMode = false

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
    case "--help", "-h":
        print("""
        audio-capture - Capture system audio using ScreenCaptureKit
        
        Usage:
          audio-capture [options]
          audio-capture --list
        
        Options:
          --sample-rate <rate>    Sample rate in Hz (default: 16000)
          --channels <count>       Number of channels (default: 1, mono)
          --list                  List available displays
          --help, -h              Show this help message
        
        Output:
          Writes raw PCM audio (int16, little-endian) to stdout.
          Press Ctrl+C to stop.
        
        Requirements:
          - macOS 12.3+ (ScreenCaptureKit)
          - Screen Recording permission
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

// Handle list mode
if listMode {
    if #available(macOS 13.0, *) {
        AudioCapture.listDisplays()
    } else {
        try? stderr.write(contentsOf: "Error: macOS 13.0+ required for ScreenCaptureKit\n".data(using: .utf8)!)
        exit(1)
    }
    exit(0)
}

// Check permissions
if #available(macOS 13.0, *) {
    if !AudioCapture.checkPermission() {
        let errorMsg = """
        Error: Screen Recording permission is required.
        
        Please grant permission in:
          System Settings > Privacy & Security > Screen Recording
          Add your terminal app (Terminal, iTerm, etc.)
        
        Attempting to request permission...
        """
        try? stderr.write(contentsOf: errorMsg.data(using: .utf8)!)
        
        AudioCapture.requestPermission()
        
        // Wait a bit and check again
        Thread.sleep(forTimeInterval: 2.0)
        
        if !AudioCapture.checkPermission() {
            try? stderr.write(contentsOf: "Error: Screen Recording permission not granted. Please grant permission and try again.\n".data(using: .utf8)!)
            exit(1)
        }
    }
    
    // Create audio capture
    let stdout = FileHandle.standardOutput
    
    audioCapture = AudioCapture(sampleRate: sampleRate, channelCount: channelCount) { pcmData in
        // Write PCM data to stdout
        do {
            try stdout.write(contentsOf: pcmData)
        } catch {
            try? stderr.write(contentsOf: "Error writing to stdout: \(error.localizedDescription)\n".data(using: .utf8)!)
            shouldStop = true
        }
    }
    
    // Start capture
    do {
        try audioCapture?.start()
        try? stderr.write(contentsOf: "Started audio capture. Sample rate: \(sampleRate) Hz, Channels: \(channelCount)\n".data(using: .utf8)!)
        try? stderr.write(contentsOf: "Writing PCM data to stdout. Press Ctrl+C to stop.\n".data(using: .utf8)!)
    } catch {
        try? stderr.write(contentsOf: "Error starting capture: \(error.localizedDescription)\n".data(using: .utf8)!)
        exit(1)
    }
} else {
    try? stderr.write(contentsOf: "Error: macOS 13.0+ required for audio capture\n".data(using: .utf8)!)
    exit(1)
}

// Run until stopped
let runLoop = RunLoop.current
while !shouldStop && runLoop.run(mode: .default, before: Date(timeIntervalSinceNow: 0.1)) {
    // Keep running
}

// Cleanup
audioCapture?.stop()
exit(0)
