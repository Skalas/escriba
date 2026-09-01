// swift-tools-version: 5.9
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "audio-capture",
    platforms: [
        .macOS(.v14)  // Core Audio Taps require macOS 14.2+
    ],
    products: [
        .executable(
            name: "audio-capture",
            targets: ["audio-capture"]
        ),
        // T10: test runner for the watchdog verdict function. Run as the
        // gates do, so a local pass means what CI means:
        //   swift build -c release && ./.build/release/watchdog-tests
        // Full XCTest requires Xcode (not CLT-only). This executable is the
        // test artifact that fastGate/shipGate can invoke without Xcode.
        .executable(
            name: "watchdog-tests",
            targets: ["WatchdogTests"]
        ),
    ],
    dependencies: [],
    targets: [
        .target(
            name: "CoreAudioTapBridge",
            path: "Sources/CoreAudioTapBridge",
            sources: ["CoreAudioTapBridge.m"],
            publicHeadersPath: ".",
            cSettings: [
                .headerSearchPath("."),
                .unsafeFlags(["-Wno-unguarded-availability-new"]),
            ],
            linkerSettings: [
                .linkedFramework("CoreAudio"),
                .linkedFramework("AudioToolbox"),
                .linkedFramework("Foundation"),
            ]
        ),
        .executableTarget(
            name: "audio-capture",
            dependencies: ["CoreAudioTapBridge"],
            path: "Sources/audio-capture",
            sources: [
                "main.swift",
                "CoreAudioTap.swift",
                "PCMConverter.swift",
                "AudioCapture.swift",
            ],
            linkerSettings: [
                .linkedFramework("Accelerate"),
                .linkedFramework("ScreenCaptureKit"),
                .linkedFramework("CoreMedia"),
            ]
        ),
        // T10: watchdog verdict tests (standalone, no Xcode required).
        // Covers clock-loss verdict, listener settle ignore, and failure backoff.
        // What this cannot cover without TCC + real hardware:
        //   tap creation, aggregate device, IO proc, listener dispatch, and any
        //   audio data flowing through the chain.
        .executableTarget(
            name: "WatchdogTests",
            dependencies: ["CoreAudioTapBridge"],
            path: "Tests/CoreAudioTapBridgeTests",
            linkerSettings: [
                .linkedFramework("CoreAudio"),
                .linkedFramework("AudioToolbox"),
                .linkedFramework("Foundation"),
            ]
        ),
    ]
)
