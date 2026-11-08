import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLMCommon

@Suite("NF4 checkpoints load as stock QuantizedLinear and QuantizedSwitchLinear in mode nf4")
struct NF4LoadTests {
    final class Block: Module {
        @ModuleInfo var proj: Linear
        @ModuleInfo var expert: Linear

        override init() {
            _proj.wrappedValue = Linear(64, 16, bias: true)
            _expert.wrappedValue = Linear(64, 16, bias: false)
        }
    }

    static func relativeError(_ a: MLXArray, _ b: MLXArray) -> Float {
        sqrt((a.asType(.float32) - b.asType(.float32)).square().sum() / b.asType(.float32).square().sum()).item(Float.self)
    }

    @Test(
        "mode nf4 and the older quant_method nf4 both give nf4 layers with the group read from the scales; a layer with biases stays affine",
        arguments: [
            #"{"model_type": "t", "quantization": {"group_size": 64, "bits": 4, "mode": "affine", "quant_method": "nf4"}}"#,
            #"{"model_type": "t", "quantization": {"group_size": 32, "bits": 4, "mode": "nf4"}}"#,
        ])
    func loads(json: String) throws {
        let config = try JSONDecoder().decode(BaseConfiguration.self, from: Data(json.utf8))
        let dense = MLXRandom.normal([16, 64], key: MLXRandom.key(1))
        let (codes, scales, none) = quantized(dense, groupSize: 32, bits: 4, mode: .nf4)
        #expect(none == nil)
        let (wq, affineScales, biases) = quantized(dense, groupSize: 64, bits: 4)
        let weights: [String: MLXArray] = [
            "proj.weight": codes, "proj.scales": scales, "proj.bias": MLXArray.zeros([16]),
            "expert.weight": wq, "expert.scales": affineScales, "expert.biases": try #require(biases),
        ]
        let block = Block()
        quantize(model: block) { path, _ in
            layerQuantization(of: path, weights: weights, quantization: nil, perLayerQuantization: config.perLayerQuantization)
        }
        try block.update(parameters: ModuleParameters.unflattened(weights), verify: [.all])
        let proj = try #require(block.proj as? QuantizedLinear)
        let expert = try #require(block.expert as? QuantizedLinear)
        #expect(proj.mode == .nf4 && proj.groupSize == 32 && proj.bits == 4)
        #expect(expert.mode == .affine && expert.groupSize == 64)
        let x = MLXRandom.normal([2, 3, 64], key: MLXRandom.key(2))
        let reference = matmul(x, dequantized(codes, scales: scales, biases: nil, groupSize: 32, bits: 4, mode: .nf4).T)
        #expect(Self.relativeError(proj(x), reference) < 1e-3)
        #expect(Self.relativeError(proj(x), Device.withDefaultDevice(.cpu) { proj(x) }) < 1e-5)
    }

    @Test("Stacked NF4 experts run through gather_qmm like the dequantized experts")
    func experts() throws {
        let dense = MLXRandom.normal([4, 16, 64], key: MLXRandom.key(3))
        let (codes, scales, _) = quantized(dense, groupSize: 64, bits: 4, mode: .nf4)
        let x = MLXRandom.normal([5, 1, 1, 64], key: MLXRandom.key(4))
        let indices = MLXArray([Int32(0), 3, 1, 2, 3]).reshaped(5, 1)
        let out = gatherQuantizedMM(x, codes, scales: scales, biases: nil, rhsIndices: indices, transpose: true, groupSize: 64, bits: 4, mode: .nf4)
        let experts = dequantized(codes, scales: scales, biases: nil, groupSize: 64, bits: 4, mode: .nf4)
        let reference = matmul(x, experts[indices].swappedAxes(-1, -2))
        #expect(out.shape == reference.shape)
        #expect(Self.relativeError(out, reference) < 1e-3)
        let cpu = gatherQuantizedMM(x, codes, scales: scales, biases: nil, rhsIndices: indices, transpose: true, groupSize: 64, bits: 4, mode: .nf4, stream: .cpu)
        #expect(Self.relativeError(out, cpu) < 1e-5)
    }

    @Test("An affine checkpoint keeps its configured mode and group")
    func affine() {
        let weights = ["proj.weight": MLXArray.zeros([16, 8], dtype: .uint32), "proj.scales": MLXArray.zeros([16, 1]), "proj.biases": MLXArray.zeros([16, 1])]
        let tuple = layerQuantization(of: "proj", weights: weights, quantization: BaseConfiguration.Quantization(groupSize: 64, bits: 4), perLayerQuantization: nil)
        #expect(tuple?.groupSize == 64 && tuple?.mode == .affine)
        #expect(layerQuantization(of: "other", weights: weights, quantization: BaseConfiguration.Quantization(groupSize: 64, bits: 4), perLayerQuantization: nil) == nil)
    }
}
