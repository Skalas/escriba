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
    ]
)
