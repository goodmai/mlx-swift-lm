import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXVLM

struct Gemma4SplitFeedForwardPrefillTests {
    static func model() throws -> Gemma4Unified {
        let json = """
            {
              "model_type": "gemma4_unified", "vocab_size": 32, "image_token_id": 31, "audio_token_id": 30, "video_token_id": 29,
              "text_config": {
                "model_type": "gemma4_unified_text", "hidden_size": 16, "num_hidden_layers": 4, "intermediate_size": 32,
                "num_attention_heads": 2, "num_key_value_heads": 1, "num_global_key_value_heads": 1, "head_dim": 8, "global_head_dim": 8,
                "vocab_size": 32, "vocab_size_per_layer_input": 32, "num_kv_shared_layers": 1, "hidden_size_per_layer_input": 4,
                "sliding_window": 4, "sliding_window_pattern": 2, "attention_k_eq_v": true, "use_double_wide_mlp": false,
                "layer_types": ["sliding_attention", "full_attention", "sliding_attention", "full_attention"], "tie_word_embeddings": true
              },
              "vision_config": null, "audio_config": null
            }
            """
        let config = try JSONDecoder.json5().decode(Gemma4UnifiedConfiguration.self, from: Data(json.utf8))
        let model = withRandomState(MLXRandom.RandomState(seed: 7)) { Gemma4Unified(config) }
        model.apply { $0.dtype.isFloatingPoint ? $0.asType(.float16) : $0 }
        return model
    }

    @Test("Prefill with each dense feed-forward awaited from the layer's own MLP equals the stock forward, logits and cache")
    func matchesStockForward() async throws {
        let model = try Self.model()
        let tokens = MLXArray([0, 2, 3, 4, 5, 1, 7, 9, 11, 6].map(Int32.init)).expandedDimensions(axis: 0)
        let stockCache = try model.newCache(parameters: nil)
        let stock = model(tokens, cache: stockCache)
        let modules = Dictionary(model.namedModules(), uniquingKeysWith: { first, _ in first })
        #expect(model.denseFeedForwardLayers == [0, 1, 2, 3])
        var called: [Int] = []
        let splitCache = try model.newCache(parameters: nil)
        let split = try await model.prefill(text: tokens, cache: splitCache) { layer, x in
            called.append(layer)
            let mlp = try #require(modules[model.feedForwardPath(layer: layer)] as? any UnaryLayer)
            return mlp(x)
        }
        #expect(called == [0, 1, 2, 3])
        #expect(split.shape == stock.shape)
        #expect(arrayEqual(split, stock).item(Bool.self))
        #expect(zip(stockCache, splitCache).allSatisfy { $0.offset == $1.offset })
        for (a, b) in zip(stockCache, splitCache) {
            for (x, y) in zip(a.state, b.state) { #expect(arrayEqual(x, y).item(Bool.self)) }
        }
    }
}
