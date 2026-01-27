// swift-tools-version: 5.7
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "audio-capture",
    platforms: [
        .macOS(.v13)  // Audio capture requires macOS 13.0+
    ],
    products: [
        .executable(
            name: "audio-capture",
            targets: ["audio-capture"]
        ),
    ],
    dependencies: [],
    targets: [
        .executableTarget(
            name: "audio-capture",
            dependencies: [],
            linkerSettings: [
                .linkedFramework("ScreenCaptureKit"),
                .linkedFramework("CoreMedia"),
                .linkedFramework("AVFoundation"),
                .linkedFramework("Accelerate"),  // For vDSP resampling
            ]
        ),
    ]
)
