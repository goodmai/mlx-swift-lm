import MLX

/// Observes and edits a model forward from inside its layers.
///
/// Bind one with ``withForwardProbe(_:_:)`` (or set ``ModelContext/probe`` and use
/// ``ModelContainer/performEvaluating(nonSendable:_:)``). Models that support probing read the
/// bound probe once per forward; with none bound they pay one nil check per hook site.
public protocol ForwardProbe: AnyObject {
    /// Observe and/or replace the residual stream after decoder layer `layer`.
    func residual(layer: Int, _ hidden: MLXArray) -> MLXArray
    /// Softmax attention weights `[B, heads, queries, keys]`, computed only when
    /// ``wantsAttention(layer:)`` is true.
    func attentionWeights(layer: Int, _ weights: MLXArray)
    func wantsAttention(layer: Int) -> Bool
    /// `[heads]` of 0/1, multiplied into the attention output per head before the output projection.
    func headMask(layer: Int) -> MLXArray?
    /// MoE router logits before top-k.
    func routerLogits(layer: Int, _ logits: MLXArray)
    /// `[experts]` of 0/1; masked experts get `-inf` router logits before top-k.
    func expertMask(layer: Int) -> MLXArray?
}

private struct BoundForwardProbe: @unchecked Sendable {
    // Read only by forwards that run synchronously inside `withForwardProbe` on the binding task.
    let probe: any ForwardProbe
}

private enum ForwardProbeBinding {
    @TaskLocal static var bound: BoundForwardProbe?
}

/// Runs `body` with `probe` bound for the model forwards it runs synchronously on this task.
/// A nil `probe` keeps any outer binding.
public func withForwardProbe<R>(_ probe: (any ForwardProbe)?, _ body: () throws -> R) rethrows
    -> R
{
    guard let probe else { return try body() }
    return try ForwardProbeBinding.$bound.withValue(
        BoundForwardProbe(probe: probe), operation: body)
}

/// The probe bound on this task, if any. Models read it once per forward.
package var currentForwardProbe: (any ForwardProbe)? {
    ForwardProbeBinding.bound?.probe
}

extension ForwardProbe {
    /// Reports `logits` and returns them with masked experts set to `-inf`.
    package func maskedRouterLogits(layer: Int, _ logits: MLXArray) -> MLXArray {
        routerLogits(layer: layer, logits)
        guard let mask = expertMask(layer: layer) else { return logits }
        return MLX.where(mask .> 0, logits, -Float.infinity)
    }

    /// Applies ``headMask(layer:)`` to an attention output `[B, heads, L, D]`.
    package func maskedHeads(layer: Int, _ output: MLXArray) -> MLXArray {
        guard let mask = headMask(layer: layer) else { return output }
        return output * mask.asType(output.dtype).reshaped(1, -1, 1, 1)
    }

    /// Attention as explicit softmax(QKᵀ·scale + mask)·V, reporting the weights.
    /// Keys and values are the full (cached) `[B, kvHeads, S, D]`.
    fileprivate func explicitAttention(
        layer: Int, queries: MLXArray, keys: MLXArray, values: MLXArray, scale: Float,
        mask: MLXFast.ScaledDotProductAttentionMaskMode
    ) -> MLXArray {
        let groups = queries.dim(1) / keys.dim(1)
        let k = groups > 1 ? repeated(keys, count: groups, axis: 1) : keys
        let v = groups > 1 ? repeated(values, count: groups, axis: 1) : values
        var scores = matmul(
            queries.asType(.float32) * scale, k.asType(.float32).transposed(0, 1, 3, 2))
        switch mask {
        case .causal:
            let (l, s) = (scores.dim(-2), scores.dim(-1))
            let allowed =
                MLXArray.arange(s).reshaped(1, s) .<= (MLXArray.arange(l) + (s - l)).reshaped(l, 1)
            scores = MLX.where(allowed, scores, -Float.infinity)
        default:
            if let m = mask.mask {
                scores =
                    m.dtype == .bool
                    ? MLX.where(m, scores, -Float.infinity) : scores + m.asType(.float32)
            }
        }
        let weights = softmax(scores, axis: -1)
        attentionWeights(layer: layer, weights)
        return matmul(weights, v.asType(.float32)).asType(values.dtype)
    }
}

/// ``attentionWithCacheUpdate(queries:keys:values:cache:scale:mask:)`` with probe hooks for
/// `layer`: explicit attention when the probe wants the weights and the cache holds dense keys,
/// then the probe's head mask. With no probe it is the plain call.
package func attentionWithCacheUpdate(
    queries: MLXArray,
    keys: MLXArray,
    values: MLXArray,
    cache: KVCache?,
    scale: Float,
    mask: MLXFast.ScaledDotProductAttentionMaskMode,
    probe: (any ForwardProbe)?,
    layer: Int
) -> MLXArray {
    guard let probe else {
        return attentionWithCacheUpdate(
            queries: queries, keys: keys, values: values, cache: cache, scale: scale, mask: mask)
    }
    let denseKeys = cache.map {
        usesPlainAttentionCacheRoute($0) && !($0 is KVCacheAttentionProtocol)
    }
    let output: MLXArray
    if probe.wantsAttention(layer: layer), denseKeys ?? true {
        let (k, v) = cache?.update(keys: keys, values: values) ?? (keys, values)
        output = probe.explicitAttention(
            layer: layer, queries: queries, keys: k, values: v, scale: scale, mask: mask)
    } else {
        output = attentionWithCacheUpdate(
            queries: queries, keys: keys, values: values, cache: cache, scale: scale, mask: mask)
    }
    return probe.maskedHeads(layer: layer, output)
}
