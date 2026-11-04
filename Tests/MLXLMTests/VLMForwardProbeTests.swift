// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXVLM
import Testing

/// The ``ForwardProbe`` hooks in the text towers of Qwen3-VL, Qwen3-VL-MoE and LFM2-VL.
struct VLMForwardProbeTests {

    typealias RecordingProbe = ForwardProbeTests.RecordingProbe

    let tokens = MLXArray([Int32(1), 2, 3, 4, 5]).reshaped(1, 5)

    static let qwen3VLText = """
        "hidden_size": 16, "num_hidden_layers": 3, "intermediate_size": 32,
        "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 8, "vocab_size": 32,
        "max_position_embeddings": 512, "rms_norm_eps": 1e-6, "rope_theta": 10000.0,
        "rope_scaling": { "type": "default", "mrope_section": [2, 1, 1] }
        """

    static let qwen3VLVision = """
        "depth": 1, "hidden_size": 8, "intermediate_size": 16, "out_hidden_size": 16,
        "num_heads": 1, "patch_size": 16, "spatial_merge_size": 1, "temporal_patch_size": 1,
        "num_position_embeddings": 8
        """

    static func qwen3VL() throws -> Qwen3VL {
        let json = """
            {
                "model_type": "qwen3_vl", "vocab_size": 32,
                "text_config": { "model_type": "qwen3_vl", \(qwen3VLText) },
                "vision_config": { "model_type": "qwen3_vl", \(qwen3VLVision) }
            }
            """
        let config = try JSONDecoder().decode(Qwen3VLConfiguration.self, from: Data(json.utf8))
        return withRandomState(MLXRandom.RandomState(seed: 7)) { Qwen3VL(config) }
    }

    static func qwen3VLMoE() throws -> Qwen3VLMoE {
        let json = """
            {
                "model_type": "qwen3_vl_moe",
                "text_config": {
                    "model_type": "qwen3_vl_moe", \(qwen3VLText),
                    "num_experts": 4, "num_experts_per_tok": 2, "decoder_sparse_step": 1,
                    "mlp_only_layers": [1], "moe_intermediate_size": 12
                },
                "vision_config": { "model_type": "qwen3_vl_moe", \(qwen3VLVision) }
            }
            """
        let config = try JSONDecoder().decode(
            Qwen3VLMoEConfiguration.self, from: Data(json.utf8))
        return withRandomState(MLXRandom.RandomState(seed: 7)) { Qwen3VLMoE(config) }
    }

    static func lfm2VL() throws -> LFM2VL {
        let json = """
            {
                "model_type": "lfm2-vl", "downsample_factor": 1, "projector_hidden_size": 32,
                "projector_use_layernorm": false,
                "text_config": {
                    "model_type": "lfm2", "hidden_size": 32, "num_hidden_layers": 3,
                    "num_attention_heads": 2, "num_key_value_heads": 1, "vocab_size": 32,
                    "full_attn_idxs": [1]
                },
                "vision_config": {
                    "model_type": "siglip2_navit", "hidden_size": 32, "intermediate_size": 64,
                    "num_hidden_layers": 1, "num_attention_heads": 2, "patch_size": 8,
                    "image_size": 32
                }
            }
            """
        let config = try JSONDecoder().decode(LFM2VLConfiguration.self, from: Data(json.utf8))
        return withRandomState(MLXRandom.RandomState(seed: 7)) { LFM2VL(config) }
    }

    func logits(_ model: some LanguageModel, _ probe: (any ForwardProbe)?) -> MLXArray {
        let out = withForwardProbe(probe) {
            model(LMInput.Text(tokens: tokens), cache: nil, state: nil).logits
        }
        eval(out)
        return out
    }

    func identical(_ a: MLXArray, _ b: MLXArray) -> Bool {
        a.dtype == b.dtype && a.shape == b.shape
            && a.asArray(Float.self).map(\.bitPattern) == b.asArray(Float.self).map(\.bitPattern)
    }

    func expectHooked(_ model: some LanguageModel) {
        let probe = RecordingProbe()
        let probed = logits(model, probe)
        #expect(probe.residualLayers == [0, 1, 2])
        #expect(identical(probed, logits(model, nil)))
    }

    @Test func `Qwen3-VL sees one residual per layer and an identity probe is exact`() throws {
        expectHooked(try Self.qwen3VL())
    }

    @Test func `LFM2-VL sees one residual per layer and an identity probe is exact`() throws {
        expectHooked(try Self.lfm2VL())
    }

    @Test func `Qwen3-VL-MoE routes sparse layers through the probe`() throws {
        let model = try Self.qwen3VLMoE()
        expectHooked(model)

        let ones = RecordingProbe()
        ones.experts = MLXArray([Float(1), 1, 1, 1])
        #expect(identical(logits(model, ones), logits(model, nil)))
        #expect(ones.routerLayers == [0, 2])
    }

    @Test func `requested attention weights come from the attention layers`() throws {
        let probe = RecordingProbe()
        probe.attentionLayers = [0, 1, 2]
        _ = logits(try Self.lfm2VL(), probe)
        #expect(probe.attention.keys.sorted() == [1])
        #expect(probe.attention[1]?.shape == [1, 2, 5, 5])

        let qwen = RecordingProbe()
        qwen.attentionLayers = [1]
        _ = logits(try Self.qwen3VL(), qwen)
        #expect(qwen.attention[1]?.shape == [1, 2, 5, 5])
    }
}
