import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@Suite("NF4: codebook codec, decode kernel and NF4Linear loading")
struct NF4Tests {
    final class Holder: Module {
        @ModuleInfo var proj: Linear
        init(_ proj: Linear) { self._proj.wrappedValue = proj }
    }

    static func relativeError(_ a: MLXArray, _ b: MLXArray) -> Float {
        let difference = (a.asType(.float32) - b.asType(.float32)).square().sum()
        let norm = b.asType(.float32).square().sum()
        return sqrt(difference / norm).item(Float.self)
    }

    @Test(
        "The qmv kernel matches dequantize-then-matmul for 1 to 8 rows, a K that is not a multiple of 256 and an N that is not a multiple of 4",
        arguments: [(1, 32, DType.float16), (3, 64, .float32), (8, 32, .bfloat16), (5, 64, .float16)])
    func kernelMatchesReference(rows: Int, groupSize: Int, dtype: DType) throws {
        let dense = MLXRandom.normal([70, 320], key: MLXRandom.key(1))
        let (weight, scales) = try NF4.quantized(dense, groupSize: groupSize)
        let x = MLXRandom.normal([1, rows, 320], key: MLXRandom.key(2)).asType(dtype)
        let reference = matmul(
            x.asType(.float32), NF4.dequantized(weight, scales: scales, groupSize: groupSize).asType(.float32).T)
        let kernel = NF4.qmv(x, weight: weight, scales: scales, groupSize: groupSize)
        #expect(kernel.shape == [1, rows, 70])
        #expect(kernel.dtype == dtype)
        #expect(Self.relativeError(kernel, reference) < (dtype == .float32 ? 1e-5 : 1e-2))
    }

    @Test("Codebook values round-trip exactly with a power-of-two scale, and Gaussian weights land within the NF4 error band")
    func roundTrip() throws {
        let row = MLXArray(NF4.codebook + NF4.codebook.reversed()).reshaped(1, 32) * 4
        let (weight, scales) = try NF4.quantized(row, groupSize: 32)
        #expect(weight.dtype == .uint32 && weight.shape == [1, 4] && scales.shape == [1, 1])
        let restored = NF4.dequantized(weight, scales: scales, groupSize: 32).asType(.float32)
        #expect(restored.asArray(Float.self) == row.asType(.float16).asType(.float32).asArray(Float.self))
        let gaussian = MLXRandom.normal([256, 1024], key: MLXRandom.key(3))
        let (codes, groupScales) = try NF4.quantized(gaussian, groupSize: 64)
        let error = Self.relativeError(NF4.dequantized(codes, scales: groupScales, groupSize: 64), gaussian)
        #expect(error > 0.05 && error < 0.1)
    }

    @Test("A group size other than 32 or 64, an input dimension that is not a multiple of the group and mismatched scales are typed errors")
    func refusesBadShapes() throws {
        #expect(throws: NF4Error.groupSize(128)) { try NF4.quantized(MLXArray.zeros([4, 128]), groupSize: 128) }
        #expect(throws: NF4Error.self) { try NF4.quantized(MLXArray.zeros([4, 100]), groupSize: 32) }
        let (weight, _) = try NF4.quantized(MLXArray.zeros([4, 128]), groupSize: 32)
        #expect(throws: NF4Error.groupSize(16)) { try NF4Linear(weight: weight, scales: MLXArray.zeros([4, 8])) }
        #expect(throws: NF4Error.self) { try NF4Linear(weight: weight, scales: MLXArray.zeros([5, 4])) }
    }

    @Test("quant_method nf4 in config.json reaches the quantization, and install swaps only bias-free scaled linears for NF4Linear that computes the dequantized product")
    func installFromCheckpoint() throws {
        let json = #"{"model_type": "t", "quantization": {"group_size": 64, "bits": 4, "mode": "affine", "quant_method": "nf4"}}"#
        let config = try JSONDecoder().decode(BaseConfiguration.self, from: Data(json.utf8))
        #expect(config.perLayerQuantization?.quantization?.quantMethod == NF4.method)
        let dense = MLXRandom.normal([16, 64], key: MLXRandom.key(4))
        let (weight, scales) = try NF4.quantized(dense, groupSize: 32)
        let weights = ["proj.weight": weight, "proj.scales": scales, "proj.bias": MLXArray.zeros([16])]
        let holder = Holder(Linear(64, 16))
        try NF4Linear.install(in: holder, weights: weights)
        try holder.update(parameters: ModuleParameters.unflattened(weights), verify: [.all])
        let layer = try #require(holder.proj as? NF4Linear)
        #expect(layer.groupSize == 32 && layer.shape.0 == 16 && layer.shape.1 == 64)
        let x = MLXRandom.normal([2, 9, 64], key: MLXRandom.key(5))
        let reference = matmul(x, NF4.dequantized(weight, scales: scales, groupSize: 32).asType(.float32).T)
        #expect(Self.relativeError(layer(x), reference) < 1e-5)
        let affine = Holder(Linear(64, 16))
        try NF4Linear.install(in: affine, weights: weights.merging(["proj.biases": scales]) { $1 })
        #expect(!(affine.proj is NF4Linear))
    }
}
