import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXLLM

final class Qwen3NextRouterParityTests: XCTestCase {
    private func configuration(experts: Int, topK: Int) throws -> Qwen3NextConfiguration {
        let json = """
            {
                "hidden_size": 16, "num_hidden_layers": 4, "intermediate_size": 32,
                "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 8,
                "linear_num_value_heads": 2, "linear_num_key_heads": 2,
                "linear_key_head_dim": 8, "linear_value_head_dim": 8, "linear_conv_kernel_dim": 4,
                "num_experts": \(experts), "num_experts_per_tok": \(topK), "decoder_sparse_step": 1,
                "shared_expert_intermediate_size": 16, "mlp_only_layers": [],
                "moe_intermediate_size": 16, "rms_norm_eps": 1e-6, "vocab_size": 32,
                "rope_theta": 10000.0, "partial_rotary_factor": 0.25,
                "max_position_embeddings": 128, "norm_topk_prob": true,
                "tie_word_embeddings": true, "attention_bias": false, "full_attention_interval": 2
            }
            """
        return try JSONDecoder().decode(Qwen3NextConfiguration.self, from: Data(json.utf8))
    }

    private func chain(_ block: Qwen3NextSparseMoeBlock, _ x: MLXArray, topK: Int) -> (MLXArray, MLXArray) {
        let gates = MLX.softmax(block.gate(x), axis: -1, precise: true)
        let inds = MLX.argSort(-gates, axis: -1)[.ellipsis, ..<topK]
        var scores = MLX.takeAlong(gates, inds, axis: -1)
        scores = scores / scores.sum(axis: -1, keepDims: true)
        let routed = (block.switchMLP(x, inds) * scores[.ellipsis, .newAxis]).sum(axis: -2)
        let shared = sigmoid(block.sharedExpertGate(x)) * block.sharedExpert(x)
        return (inds, routed + shared)
    }

    func testFusedRouterMatchesTheStockChain() throws {
        for (experts, topK) in [(16, 4), (512, 10)] {
            let block = Qwen3NextSparseMoeBlock(try configuration(experts: experts, topK: topK))
            for tokens in [1, 5] {
                let x = MLXRandom.normal([1, tokens, 16], key: MLXRandom.key(UInt64(experts + tokens)))
                let gates = MLX.softmax(block.gate(x), axis: -1, precise: true)
                let (fusedIndices, _) = moeRouterTopK(gates, k: topK, normalize: true)
                let (chainIndices, expected) = chain(block, x, topK: topK)
                let actual = block(x)
                eval(fusedIndices, chainIndices, expected, actual)
                XCTAssertEqual(
                    sorted(fusedIndices, axis: -1).asArray(Int32.self),
                    sorted(chainIndices.asType(.int32), axis: -1).asArray(Int32.self),
                    "experts \(experts) tokens \(tokens)")
                let gap = abs(actual - expected).max().item(Float.self)
                XCTAssertLessThanOrEqual(gap, 1e-5, "experts \(experts) tokens \(tokens)")
            }
        }
    }
}
