import MLX

/// A model whose text prefill can take each dense layer's feed-forward output from the caller.
///
/// A prefill that computes part of the feed-forward elsewhere (another device, another
/// process) awaits it per layer. The model hands `denseFeedForward` the layer index and the
/// layer's pre-feed-forward-normalized input and continues with what it returns, which must
/// equal the layer's own `mlp` of that input. Layers outside `denseFeedForwardLayers` (MoE)
/// run their own feed-forward.
public protocol SplitFeedForwardPrefill {
    /// Indices of the decoder layers whose feed-forward is one dense MLP.
    var denseFeedForwardLayers: [Int] { get }

    /// Module path of a dense layer's MLP (`gate_proj`, `up_proj`, `down_proj` under it).
    func feedForwardPath(layer: Int) -> String

    /// Runs `tokens` (`[L]` or `[1, L]`) through the model into `cache` and returns the logits.
    func prefill(
        text tokens: MLXArray, cache: [any KVCache],
        denseFeedForward: (Int, MLXArray) async throws -> MLXArray
    ) async throws -> MLXArray
}
