import Foundation
import MLX
import MLXNN

public enum NF4Error: Error, Equatable, Sendable {
    case groupSize(Int)
    case shape(String)
}

public enum NF4 {
    public static let method = "nf4"
    public static let groupSizes: Set<Int> = [32, 64]
    public static let decodeRows = 1 ... 8
    public static let kernelEnabled = ProcessInfo.processInfo.environment["MLXLM_NF4_KERNEL"] == "1"

    public static let codebook: [Float] = [
        -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
        -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
        0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
        0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0,
    ]

    public static func quantized(_ weight: MLXArray, groupSize: Int) throws(NF4Error) -> (
        weight: MLXArray, scales: MLXArray
    ) {
        guard groupSizes.contains(groupSize) else { throw .groupSize(groupSize) }
        let columns = weight.dim(-1)
        guard weight.ndim >= 2, columns % groupSize == 0 else {
            throw .shape("\(weight.shape) has an input dimension that is not a multiple of the group size \(groupSize)")
        }
        let groups = weight.asType(.float32).reshaped(-1, groupSize)
        let scales = abs(groups).max(axis: -1, keepDims: true).asType(.float16)
        let scale = scales.asType(.float32)
        let normalized = groups / which(scale .> 0, scale, 1)
        var codes = MLXArray.zeros(normalized.shape, dtype: .uint32)
        for (low, high) in zip(codebook, codebook.dropFirst()) {
            codes = codes + (normalized .> ((low + high) / 2)).asType(.uint32)
        }
        let shifts = MLXArray((0 ..< 8).map { UInt32(1) << UInt32(4 * $0) })
        let packed = (codes.reshaped(-1, 8) * shifts).sum(axis: -1).asType(.uint32)
        let leading = Array(weight.shape.dropLast())
        return (packed.reshaped(leading + [columns / 8]), scales.reshaped(leading + [columns / groupSize]))
    }

    public static func dequantized(_ weight: MLXArray, scales: MLXArray, groupSize: Int) -> MLXArray {
        let shifts = MLXArray((0 ..< 8).map { UInt32(4 * $0) })
        let codes = (weight.expandedDimensions(axis: -1) >> shifts) & UInt32(15)
        let values = MLXArray(codebook).take(codes.reshaped(-1))
        let grouped = values.reshaped(-1, groupSize) * scales.reshaped(-1, 1).asType(.float32)
        return grouped.reshaped(Array(weight.shape.dropLast()) + [weight.dim(-1) * 8]).asType(scales.dtype)
    }

    public static func matmul(_ x: MLXArray, weight: MLXArray, scales: MLXArray, groupSize: Int) -> MLXArray {
        if kernelEnabled, decodeRows.contains(x.size / x.dim(-1)) {
            return qmv(x, weight: weight, scales: scales, groupSize: groupSize)
        }
        return MLX.matmul(x, dequantized(weight, scales: scales, groupSize: groupSize).asType(x.dtype).T)
    }

    public static func qmv(_ x: MLXArray, weight: MLXArray, scales: MLXArray, groupSize: Int) -> MLXArray {
        let columns = x.dim(-1)
        let outputs = weight.dim(0)
        let rows = x.size / columns
        let result = qmvKernel(
            [x.reshaped(rows, columns), weight, scales],
            template: [("T", x.dtype), ("K", columns), ("N", outputs), ("M", rows), ("G", groupSize)],
            grid: (32, (outputs + qmvRowsPerSimdgroup - 1) / qmvRowsPerSimdgroup, 1),
            threadGroup: (32, 2, 1),
            outputShapes: [[rows, outputs]],
            outputDTypes: [x.dtype])[0]
        return result.reshaped(Array(x.shape.dropLast()) + [outputs])
    }

    static let qmvRowsPerSimdgroup = 4

    static let header = "constant float NF4_LUT[16] = {"
        + codebook.map { String(format: "%.9ff", $0) }.joined(separator: ", ") + "};\n"

    static let source = """
        constexpr int R = 4;
        const uint KW = uint(K) / 8u;
        const uint KG = uint(K) / uint(G);
        uint lane = thread_index_in_simdgroup;
        uint n0 = thread_position_in_grid.y * uint(R);
        float acc[M][R];
        for (int m = 0; m < M; m++) {
            for (int r = 0; r < R; r++) {
                acc[m][r] = 0.0f;
            }
        }
        for (uint k8 = lane; k8 < KW; k8 += 32u) {
            float xv[M][8];
            for (int m = 0; m < M; m++) {
                for (int j = 0; j < 8; j++) {
                    xv[m][j] = float(x[uint(m) * uint(K) + k8 * 8u + uint(j)]);
                }
            }
            uint g = (k8 * 8u) / uint(G);
            for (int r = 0; r < R; r++) {
                uint n = n0 + uint(r);
                if (n >= uint(N)) {
                    break;
                }
                uint word = w[n * KW + k8];
                float s = float(scales[n * KG + g]);
                float d[8];
                for (int j = 0; j < 8; j++) {
                    d[j] = NF4_LUT[(word >> (4u * uint(j))) & 15u];
                }
                for (int m = 0; m < M; m++) {
                    float p = 0.0f;
                    for (int j = 0; j < 8; j++) {
                        p += d[j] * xv[m][j];
                    }
                    acc[m][r] += p * s;
                }
            }
        }
        for (int m = 0; m < M; m++) {
            for (int r = 0; r < R; r++) {
                float v = simd_sum(acc[m][r]);
                uint n = n0 + uint(r);
                if (lane == 0u && n < uint(N)) {
                    out[uint(m) * uint(N) + n] = static_cast<T>(v);
                }
            }
        }
        """
}

private let qmvKernel = MLXFast.metalKernel(
    name: "nf4_qmv", inputNames: ["x", "w", "scales"], outputNames: ["out"], source: NF4.source,
    header: NF4.header)

open class NF4Linear: Linear {
    public let groupSize: Int
    @ParameterInfo public private(set) var scales: MLXArray

    open override var shape: (Int, Int) { (weight.dim(0), weight.dim(1) * 8) }

    public init(weight: MLXArray, bias: MLXArray? = nil, scales: MLXArray) throws(NF4Error) {
        let columns = weight.dim(-1) * 8
        guard weight.ndim == 2, scales.ndim == 2, scales.dim(0) == weight.dim(0), scales.dim(1) > 0,
            columns % scales.dim(1) == 0
        else { throw .shape("NF4 codes \(weight.shape) do not fit scales \(scales.shape)") }
        let groupSize = columns / scales.dim(1)
        guard NF4.groupSizes.contains(groupSize) else { throw .groupSize(groupSize) }
        self.groupSize = groupSize
        self._scales.wrappedValue = scales
        super.init(weight: weight, bias: bias)
        self.freeze()
    }

    public override func unfreeze(recursive: Bool = true, keys: [String]? = nil, strict: Bool = false) throws {
        try super.unfreeze(recursive: recursive, keys: keys, strict: strict)
        self.freeze(recursive: false)
    }

    open override func callAsFunction(_ x: MLXArray) -> MLXArray {
        let result = NF4.matmul(x, weight: weight, scales: scales, groupSize: groupSize)
        guard let bias else { return result }
        return result + bias
    }

    public static func install(in model: Module, weights: [String: MLXArray]) throws(NF4Error) {
        var updates: [(String, Module)] = []
        for (path, module) in model.leafModules().flattened() {
            guard ObjectIdentifier(type(of: module)) == ObjectIdentifier(Linear.self),
                let weight = weights[path + ".weight"], let scales = weights[path + ".scales"],
                weights[path + ".biases"] == nil
            else { continue }
            updates.append((path, try NF4Linear(weight: weight, bias: weights[path + ".bias"], scales: scales)))
        }
        model.update(modules: ModuleChildren.unflattened(updates))
    }
}
