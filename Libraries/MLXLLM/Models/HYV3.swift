import Foundation
import MLX
import MLXLMCommon
import MLXNN

// Port of hy_v3.py (Tencent HYV3, Hy-MT3-30B-A3B): GQA with q/k RMSNorm, a dense first layer, then
// sigmoid-routed MoE layers with an expert selection bias and an always-on shared expert.

class HYV3Attention: Module {
    let heads: Int
    let kvHeads: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    let rope: RoPE

    init(_ args: HYV3Configuration) {
        let dim = args.hiddenSize
        let headDim = args.headDim
        self.heads = args.attentionHeads
        self.kvHeads = args.kvHeads
        self.scale = pow(Float(headDim), -0.5)
        _wq.wrappedValue = Linear(dim, heads * headDim, bias: false)
        _wk.wrappedValue = Linear(dim, kvHeads * headDim, bias: false)
        _wv.wrappedValue = Linear(dim, kvHeads * headDim, bias: false)
        _wo.wrappedValue = Linear(heads * headDim, dim, bias: false)
        _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)
        _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)
        self.rope = RoPE(dimensions: headDim, traditional: false, base: args.ropeTheta)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?,
        probe: (any ForwardProbe)? = nil, layer: Int = 0
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))
        var queries = qNorm(wq(x).reshaped(B, L, heads, -1)).transposed(0, 2, 1, 3)
        var keys = kNorm(wk(x).reshaped(B, L, kvHeads, -1)).transposed(0, 2, 1, 3)
        let values = wv(x).reshaped(B, L, kvHeads, -1).transposed(0, 2, 1, 3)

        let offset = cache?.ropeOffset
        queries = applyRotaryPosition(rope, to: queries, offset: offset)
        keys = applyRotaryPosition(rope, to: keys, offset: offset)

        let output = attentionWithCacheUpdate(
            queries: queries, keys: keys, values: values, cache: cache, scale: scale, mask: mask,
            probe: probe, layer: layer
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)
        return wo(output)
    }
}

class HYV3MLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    @ModuleInfo(key: "up_proj") var up: Linear

    init(dimensions: Int, hiddenDimensions: Int) {
        _gate.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        _down.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
        _up.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        down(silu(gate(x)) * up(x))
    }
}

class HYV3Router: Module {
    @ModuleInfo(key: "gate") var gate: Linear

    init(dimensions: Int, numExperts: Int) {
        _gate.wrappedValue = Linear(dimensions, numExperts, bias: false)
    }
}

class HYV3MoE: Module, UnaryLayer {
    let topK: Int
    let routeNorm: Bool
    let routerScale: Float

    @ModuleInfo(key: "router") var router: HYV3Router
    @ParameterInfo(key: "expert_bias") var expertBias: MLXArray
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
    @ModuleInfo(key: "shared_mlp") var sharedMLP: HYV3MLP

    init(_ args: HYV3Configuration) {
        self.topK = args.numExpertsPerToken
        self.routeNorm = args.routeNorm
        self.routerScale = args.routerScalingFactor
        _router.wrappedValue = HYV3Router(dimensions: args.hiddenSize, numExperts: args.numExperts)
        _expertBias.wrappedValue = MLXArray.zeros([args.numExperts], dtype: .float32)
        _switchMLP.wrappedValue = SwitchGLU(
            inputDims: args.hiddenSize, hiddenDims: args.moeIntermediateSize,
            numExperts: args.numExperts)
        _sharedMLP.wrappedValue = HYV3MLP(
            dimensions: args.hiddenSize,
            hiddenDimensions: args.moeIntermediateSize * args.numSharedExperts)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        callAsFunction(x, probe: nil, layer: 0)
    }

    func callAsFunction(_ x: MLXArray, probe: (any ForwardProbe)?, layer: Int) -> MLXArray {
        // The router runs in float32, as in the reference: its logits decide which experts run.
        var logits = x.asType(.float32).matmul(router.gate.weight.asType(.float32).T)
        if let probe { logits = probe.maskedRouterLogits(layer: layer, logits) }
        let scores = sigmoid(logits)
        let selection = scores + expertBias.asType(.float32)

        let k = topK
        // Expert weights come from the unbiased scores; the bias only steers the selection.
        let inds = MLX.stopGradient(
            MLX.argPartition(-selection, kth: k - 1, axis: -1)[.ellipsis, ..<k])
        var weights = MLX.takeAlong(scores, inds, axis: -1)
        if routeNorm {
            weights = weights / (MLX.sum(weights, axis: -1, keepDims: true) + 1e-20)
        }
        weights = (weights * routerScale).asType(x.dtype)
        probe?.routerTopK(layer: layer, indices: inds, gates: weights)

        let y = switchMLP(x, inds)
        return weightedExpertSum(y, weights).asType(y.dtype) + sharedMLP(x)
    }
}

class HYV3DecoderLayer: Module {
    let layerIdx: Int

    @ModuleInfo(key: "self_attn") var selfAttn: HYV3Attention
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    fileprivate let mlp: UnaryLayer

    init(_ args: HYV3Configuration, layerIdx: Int) {
        self.layerIdx = layerIdx
        _selfAttn.wrappedValue = HYV3Attention(args)
        _inputLayerNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        if layerIdx < args.firstKDenseReplace {
            self.mlp = HYV3MLP(dimensions: args.hiddenSize, hiddenDimensions: args.intermediateSize)
        } else {
            self.mlp = HYV3MoE(args)
        }
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?,
        probe: (any ForwardProbe)? = nil
    ) -> MLXArray {
        let h =
            x + selfAttn(inputLayerNorm(x), mask: mask, cache: cache, probe: probe, layer: layerIdx)
        let normed = postAttentionLayerNorm(h)
        if let probe, let moe = mlp as? HYV3MoE {
            return h + moe(normed, probe: probe, layer: layerIdx)
        }
        return h + mlp(normed)
    }
}

public class HYV3ModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    fileprivate let layers: [HYV3DecoderLayer]
    let norm: RMSNorm

    init(_ args: HYV3Configuration) {
        precondition(args.vocabularySize > 0)
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)
        self.layers = (0 ..< args.hiddenLayers).map { HYV3DecoderLayer(args, layerIdx: $0) }
        self.norm = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var h = embedTokens(inputs)
        let mask = createAttentionMask(h: h, cache: cache?.first)
        let probe = currentForwardProbe
        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i], probe: probe)
            if let probe { h = probe.residual(layer: i, h) }
        }
        return norm(h)
    }
}

public class HYV3Model: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    public let model: HYV3ModelInner
    let configuration: HYV3Configuration

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ args: HYV3Configuration) {
        self.configuration = args
        self.vocabularySize = args.vocabularySize
        self.kvHeads = (0 ..< args.hiddenLayers).map { _ in args.kvHeads }
        self.model = HYV3ModelInner(args)
        if !args.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let out = model(inputs, cache: cache)
        guard let lmHead else { return model.embedTokens.asLinear(out) }
        // The checkpoint asks for the head in float32; that applies to an unquantized head only.
        if configuration.enableLMHeadFP32, !(lmHead is Quantized) {
            return out.asType(.float32).matmul(lmHead.weight.asType(.float32).T)
        }
        return lmHead(out)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        let layerCount = configuration.hiddenLayers
        var sanitized: [String: MLXArray] = [:]
        for (key, value) in weights {
            let parts = key.split(separator: ".")
            if key.hasPrefix("model.layers."), parts.count > 2, let index = Int(parts[2]),
                index >= layerCount
            {
                continue
            }
            // Some conversions store the selection bias under the router.
            sanitized[key.replacingOccurrences(of: ".mlp.router.expert_bias", with: ".mlp.expert_bias")] =
                value
        }
        sanitized = filterLMHeadWeights(
            from: sanitized, tiedWordEmbeddings: configuration.tieWordEmbeddings)

        for l in configuration.firstKDenseReplace ..< layerCount {
            let prefix = "model.layers.\(l).mlp"
            guard sanitized["\(prefix).experts.0.up_proj.weight"] != nil else { continue }
            for name in ["up_proj", "down_proj", "gate_proj"] {
                for suffix in ["weight", "scales", "biases"]
                where sanitized["\(prefix).experts.0.\(name).\(suffix)"] != nil {
                    let parts = (0 ..< configuration.numExperts).map { e in
                        sanitized.removeValue(forKey: "\(prefix).experts.\(e).\(name).\(suffix)")!
                    }
                    sanitized["\(prefix).switch_mlp.\(name).\(suffix)"] = MLX.stacked(parts)
                }
            }
        }
        return sanitized
    }
}

public struct HYV3Configuration: Codable, Sendable {
    var modelType: String = "hy_v3"
    var hiddenSize: Int
    var hiddenLayers: Int
    var intermediateSize: Int
    var attentionHeads: Int
    var kvHeads: Int
    var headDim: Int
    var numExperts: Int
    var numExpertsPerToken: Int
    var moeIntermediateSize: Int
    var numSharedExperts: Int
    var firstKDenseReplace: Int
    var rmsNormEps: Float
    var vocabularySize: Int
    var ropeTheta: Float
    var tieWordEmbeddings: Bool
    var routeNorm: Bool = true
    var enableLMHeadFP32: Bool = false
    var routerScalingFactor: Float = 1.0
    var maxPositionEmbeddings: Int = 262144

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case numExperts = "num_experts"
        case numExpertsPerToken = "num_experts_per_tok"
        case moeIntermediateSize = "moe_intermediate_size"
        case numSharedExperts = "num_shared_experts"
        case firstKDenseReplace = "first_k_dense_replace"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case ropeTheta = "rope_theta"
        case tieWordEmbeddings = "tie_word_embeddings"
        case routeNorm = "route_norm"
        case enableLMHeadFP32 = "enable_lm_head_fp32"
        case routerScalingFactor = "router_scaling_factor"
        case maxPositionEmbeddings = "max_position_embeddings"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "hy_v3"
        self.hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        self.hiddenLayers = try c.decode(Int.self, forKey: .hiddenLayers)
        self.intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        self.attentionHeads = try c.decode(Int.self, forKey: .attentionHeads)
        self.kvHeads = try c.decode(Int.self, forKey: .kvHeads)
        self.headDim = try c.decode(Int.self, forKey: .headDim)
        self.numExperts = try c.decode(Int.self, forKey: .numExperts)
        self.numExpertsPerToken = try c.decode(Int.self, forKey: .numExpertsPerToken)
        self.moeIntermediateSize = try c.decode(Int.self, forKey: .moeIntermediateSize)
        self.numSharedExperts = try c.decode(Int.self, forKey: .numSharedExperts)
        self.firstKDenseReplace = try c.decode(Int.self, forKey: .firstKDenseReplace)
        self.rmsNormEps = try c.decode(Float.self, forKey: .rmsNormEps)
        self.vocabularySize = try c.decode(Int.self, forKey: .vocabularySize)
        self.ropeTheta = try c.decode(Float.self, forKey: .ropeTheta)
        self.tieWordEmbeddings = try c.decode(Bool.self, forKey: .tieWordEmbeddings)
        self.routeNorm = try c.decodeIfPresent(Bool.self, forKey: .routeNorm) ?? true
        self.enableLMHeadFP32 =
            try c.decodeIfPresent(Bool.self, forKey: .enableLMHeadFP32) ?? false
        self.routerScalingFactor =
            try c.decodeIfPresent(Float.self, forKey: .routerScalingFactor) ?? 1.0
        self.maxPositionEmbeddings =
            try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 262144
    }
}

extension HYV3Model: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}
