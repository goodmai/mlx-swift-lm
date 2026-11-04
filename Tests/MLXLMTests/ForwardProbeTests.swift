// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import Testing

/// Pins the ``ForwardProbe`` hooks: they fire once per layer, and a probe that changes
/// nothing leaves the logits bit-identical.
struct ForwardProbeTests {

    final class RecordingProbe: ForwardProbe {
        var residualLayers: [Int] = []
        var routerLayers: [Int] = []
        var attention: [Int: MLXArray] = [:]
        var attentionLayers: Set<Int> = []
        var heads: [Int: MLXArray] = [:]
        var experts: MLXArray?

        func residual(layer: Int, _ hidden: MLXArray) -> MLXArray {
            residualLayers.append(layer)
            return hidden
        }
        func attentionWeights(layer: Int, _ weights: MLXArray) { attention[layer] = weights }
        func wantsAttention(layer: Int) -> Bool { attentionLayers.contains(layer) }
        func headMask(layer: Int) -> MLXArray? { heads[layer] }
        func routerLogits(layer: Int, _ logits: MLXArray) { routerLayers.append(layer) }
        func expertMask(layer: Int) -> MLXArray? { experts }
    }

    let tokens = MLXArray([1, 2, 3, 4, 5]).reshaped(1, 5)

    static func qwen3() throws -> Qwen3Model {
        MLXRandom.seed(7)
        return Qwen3Model(
            try JSONDecoder().decode(
                Qwen3Configuration.self,
                from: Data(
                    """
                    {
                        "hidden_size": 16, "num_hidden_layers": 3, "intermediate_size": 32,
                        "num_attention_heads": 2, "rms_norm_eps": 1e-5, "vocab_size": 32,
                        "num_key_value_heads": 1, "head_dim": 8
                    }
                    """.utf8)))
    }

    static func qwen3MoE() throws -> Qwen3MoEModel {
        MLXRandom.seed(7)
        return Qwen3MoEModel(
            try JSONDecoder().decode(
                Qwen3MoEConfiguration.self,
                from: Data(
                    """
                    {
                        "hidden_size": 16, "num_hidden_layers": 2, "intermediate_size": 32,
                        "num_attention_heads": 2, "num_experts": 4, "num_experts_per_tok": 2,
                        "decoder_sparse_step": 1, "mlp_only_layers": [],
                        "moe_intermediate_size": 12, "rms_norm_eps": 1e-5, "vocab_size": 32,
                        "num_key_value_heads": 1, "head_dim": 8
                    }
                    """.utf8)))
    }

    static func nemotronH() -> NemotronHModel {
        MLXRandom.seed(7)
        return NemotronHModel(
            NemotronHConfiguration(
                vocabSize: 32, hiddenSize: 64, numHiddenLayers: 3, numAttentionHeads: 4,
                numKeyValueHeads: 2, mambaNumHeads: 4, mambaHeadDim: 16, ssmStateSize: 16,
                convKernel: 4, nGroups: 2, intermediateSize: 128, moeIntermediateSize: 64,
                moeSharedExpertIntermediateSize: 64, nRoutedExperts: 4, numExpertsPerTok: 2,
                hybridOverridePattern: "M*E", layerNormEpsilon: 1e-5, nGroup: 2, topkGroup: 1))
    }

    func logits(_ model: some LanguageModel, _ probe: (any ForwardProbe)?) -> MLXArray {
        let out = withForwardProbe(probe) { model(tokens, cache: nil) }
        eval(out)
        return out
    }

    func identical(_ a: MLXArray, _ b: MLXArray) -> Bool {
        a.dtype == b.dtype && a.shape == b.shape
            && a.asArray(Float.self).map(\.bitPattern) == b.asArray(Float.self).map(\.bitPattern)
    }

    @Test func `residual fires once per layer in order`() throws {
        let qwen = RecordingProbe()
        _ = try logits(Self.qwen3(), qwen)
        #expect(qwen.residualLayers == [0, 1, 2])

        let hybrid = RecordingProbe()
        _ = logits(Self.nemotronH(), hybrid)
        #expect(hybrid.residualLayers == [0, 1, 2])
        #expect(hybrid.routerLayers == [2])
    }

    @Test func `an identity probe leaves logits bit-identical`() throws {
        let qwen = try Self.qwen3()
        #expect(identical(logits(qwen, RecordingProbe()), logits(qwen, nil)))

        let hybrid = Self.nemotronH()
        #expect(identical(logits(hybrid, RecordingProbe()), logits(hybrid, nil)))
    }

    @Test func `a head mask of ones is identical and a zeroed head is not`() throws {
        let model = try Self.qwen3()
        let baseline = logits(model, nil)

        let ones = RecordingProbe()
        ones.heads[0] = MLXArray([Float(1), 1])
        #expect(identical(logits(model, ones), baseline))

        let zeroed = RecordingProbe()
        zeroed.heads[0] = MLXArray([Float(0), 1])
        #expect(!identical(logits(model, zeroed), baseline))
    }

    @Test func `an expert mask of ones is identical`() throws {
        let model = try Self.qwen3MoE()
        let baseline = logits(model, nil)

        let ones = RecordingProbe()
        ones.experts = MLXArray([Float(1), 1, 1, 1])
        #expect(identical(logits(model, ones), baseline))
        #expect(ones.routerLayers == [0, 1])
    }

    @Test func `requested attention weights are causal softmax rows and match fused attention`()
        throws
    {
        let model = try Self.qwen3()
        let probe = RecordingProbe()
        probe.attentionLayers = [1]
        let probed = logits(model, probe)

        #expect(probe.attention.keys.sorted() == [1])
        let weights = try #require(probe.attention[1])
        #expect(weights.shape == [1, 2, 5, 5])
        let rowSums = weights.sum(axis: -1)
        #expect(allClose(rowSums, MLXArray.ones([1, 2, 5]), atol: 1e-5).item(Bool.self))
        #expect(weights[0, 0, 0, 1 ..< 5].sum().item(Float.self) == 0)
        #expect(allClose(probed, logits(model, nil), atol: 1e-4).item(Bool.self))
    }

    @Test func `performEvaluating binds the context's probe`() async throws {
        let processor = TestInputProcessor()
        let model = try Self.qwen3()
        var context = ModelContext(
            configuration: processor.configuration, model: model,
            processor: processor, tokenizer: processor.tokenizer)
        context.probe = RecordingProbe()
        let container = ModelContainer(context: context)

        let layers = await container.performEvaluating(nonSendable: tokens) { context, tokens in
            eval(context.model(tokens, cache: nil))
            return (context.probe as? RecordingProbe)?.residualLayers ?? []
        }
        #expect(layers == [0, 1, 2])
    }
}
