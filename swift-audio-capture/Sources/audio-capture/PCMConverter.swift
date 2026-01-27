import Foundation
import Accelerate

/// Utilities for converting audio formats to PCM int16
struct PCMConverter {
    /// Convert float32 audio data to int16 PCM
    /// - Parameters:
    ///   - floatData: Audio data as float32 array (values typically -1.0 to 1.0)
    ///   - channelCount: Number of channels in the input data
    /// - Returns: PCM int16 data (little-endian)
    static func convertFloat32ToInt16(_ floatData: [Float], channelCount: Int) -> Data {
        let sampleCount = floatData.count / channelCount
        var int16Data = [Int16](repeating: 0, count: sampleCount)
        
        // Convert each channel separately, then mix if needed
        if channelCount == 1 {
            // Mono: direct conversion
            for i in 0..<sampleCount {
                let value = floatData[i]
                int16Data[i] = Int16(max(-32768, min(32767, value * 32767.0)))
            }
        } else {
            // Multi-channel: mix to mono
            for i in 0..<sampleCount {
                var sum: Float = 0.0
                for ch in 0..<channelCount {
                    sum += floatData[i * channelCount + ch]
                }
                let avg = sum / Float(channelCount)
                int16Data[i] = Int16(max(-32768, min(32767, avg * 32767.0)))
            }
        }
        
        return Data(bytes: int16Data, count: int16Data.count * MemoryLayout<Int16>.size)
    }
    
    /// Convert float32 audio data to int16 PCM using vDSP for better performance
    /// - Parameters:
    ///   - floatData: Audio data as float32 array
    ///   - channelCount: Number of channels in the input data
    /// - Returns: PCM int16 data (little-endian)
    static func convertFloat32ToInt16VDSP(_ floatData: [Float], channelCount: Int) -> Data {
        let sampleCount = floatData.count / channelCount
        
        // If mono, direct conversion using vDSP
        if channelCount == 1 {
            var scaledFloat = [Float](repeating: 0.0, count: sampleCount)
            vDSP_vsmul(floatData, 1, [32767.0], &scaledFloat, 1, vDSP_Length(sampleCount))
            
            // Convert to int16 and clamp
            var int16Data = [Int16](repeating: 0, count: sampleCount)
            for i in 0..<sampleCount {
                int16Data[i] = Int16(max(-32768, min(32767, scaledFloat[i])))
            }
            return Data(bytes: int16Data, count: int16Data.count * MemoryLayout<Int16>.size)
        }
        
        // Multi-channel: mix to mono first using vDSP
        var monoData = [Float](repeating: 0.0, count: sampleCount)
        for ch in 0..<channelCount {
            var channelData = [Float](repeating: 0.0, count: sampleCount)
            // Extract channel (stride by channelCount)
            for i in 0..<sampleCount {
                channelData[i] = floatData[i * channelCount + ch]
            }
            vDSP_vadd(monoData, 1, channelData, 1, &monoData, 1, vDSP_Length(sampleCount))
        }
        // Average
        vDSP_vsdiv(monoData, 1, [Float(channelCount)], &monoData, 1, vDSP_Length(sampleCount))
        
        // Convert to int16
        var scaledFloat = [Float](repeating: 0.0, count: sampleCount)
        vDSP_vsmul(monoData, 1, [32767.0], &scaledFloat, 1, vDSP_Length(sampleCount))
        
        // Clamp and convert to int16
        var int16Data = [Int16](repeating: 0, count: sampleCount)
        for i in 0..<sampleCount {
            int16Data[i] = Int16(max(-32768, min(32767, scaledFloat[i])))
        }
        
        return Data(bytes: int16Data, count: int16Data.count * MemoryLayout<Int16>.size)
    }
    
    /// Resample audio data from one sample rate to another
    /// - Parameters:
    ///   - data: Input PCM int16 data
    ///   - fromRate: Source sample rate
    ///   - toRate: Target sample rate
    /// - Returns: Resampled PCM int16 data
    static func resample(_ data: Data, fromRate: Int, toRate: Int) -> Data {
        if fromRate == toRate {
            return data
        }
        
        let ratio = Double(toRate) / Double(fromRate)
        let inputSamples = data.count / MemoryLayout<Int16>.size
        let outputSamples = Int(Double(inputSamples) * ratio)
        
        // Simple linear interpolation resampling
        let input = data.withUnsafeBytes { bytes -> [Int16] in
            guard let baseAddress = bytes.baseAddress else { return [] }
            let int16Pointer = baseAddress.assumingMemoryBound(to: Int16.self)
            return Array(UnsafeBufferPointer(start: int16Pointer, count: inputSamples))
        }
        
        guard !input.isEmpty else { return data }
        
        var output = [Int16](repeating: 0, count: outputSamples)
        
        for i in 0..<outputSamples {
            let srcIndex = Double(i) / ratio
            let index1 = Int(srcIndex)
            let index2 = min(index1 + 1, inputSamples - 1)
            let fraction = srcIndex - Double(index1)
            
            let value1 = Double(input[index1])
            let value2 = Double(input[index2])
            output[i] = Int16(value1 + (value2 - value1) * fraction)
        }
        
        return Data(bytes: output, count: output.count * MemoryLayout<Int16>.size)
    }
}
