import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLMCommon

private final class SanitizeQuantizedModel: Module, BaseLanguageModel {
    @ModuleInfo(key: "layer") var layer: Linear

    override init() {
        _layer.wrappedValue = Linear(64, 8, bias: false)
    }

    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        let quantized = QuantizedLinear(
            weight: layer.weight, bias: nil, groupSize: 16, bits: 4, mode: .nvfp4,
            globalScale: MLXArray(Float(1)))
        update(modules: ModuleChildren.unflattened([("layer", quantized)]))
        return weights
    }
}

@Suite("Modules quantized by sanitize")
struct PrequantizedLoadTests {
    @Test("An nvfp4 module with global_scale built by sanitize survives an affine default")
    func sanitizeQuantizedNVFP4Survives() throws {
        let weight = MLXRandom.normal([8, 64])
        let (wq, scales, _) = MLX.quantized(weight, groupSize: 16, bits: 4, mode: .nvfp4)
        let directory = URL(filePath: NSTemporaryDirectory())
            .appending(component: "prequantized-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try save(
            arrays: [
                "layer.weight": wq, "layer.scales": scales,
                "layer.global_scale": MLXArray(Float(0.5)),
            ],
            url: directory.appending(component: "model.safetensors"))

        let model = SanitizeQuantizedModel()
        try loadWeights(
            modelDirectory: directory, model: model,
            quantization: .init(groupSize: 64, bits: 4))

        let layer = try #require(model.layer as? QuantizedLinear)
        #expect(layer.mode == .nvfp4)
        #expect(layer.groupSize == 16)
        #expect(layer.globalScale?.item(Float.self) == 0.5)
    }
}
