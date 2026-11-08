// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

final class Qwen35MTPPredictor: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding?
    @ModuleInfo(key: "fc") var fc: Linear
    @ModuleInfo(key: "layers") var layers: [Qwen35DecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm
    @ModuleInfo(key: "pre_fc_norm_embedding") var preFCNormEmbedding: RMSNorm
    @ModuleInfo(key: "pre_fc_norm_hidden") var preFCNormHidden: RMSNorm

    init(_ args: Qwen35TextConfiguration) {
        var mtpArgs = args
        mtpArgs.hiddenLayers = max(args.mtpNumHiddenLayers, 1)
        mtpArgs.fullAttentionInterval = 1

        if args.mtpUseDedicatedEmbeddings {
            _embedTokens.wrappedValue = Embedding(
                embeddingCount: args.vocabularySize,
                dimensions: args.hiddenSize
            )
        }
        _fc.wrappedValue = Linear(args.hiddenSize * 2, args.hiddenSize, bias: false)
        _layers.wrappedValue = (0 ..< mtpArgs.hiddenLayers).map {
            Qwen35DecoderLayer(mtpArgs, layerIdx: $0, forceFullAttention: true)
        }
        _norm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _preFCNormEmbedding.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _preFCNormHidden.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        super.init()
    }

    func newCache() -> [KVCache] {
        layers.map { _ in KVCacheSimple() }
    }

    func callAsFunction(
        inputsEmbeds: MLXArray,
        hiddenStates previousHidden: MLXArray,
        cache: [KVCache],
        positionOffset: Int
    ) -> MLXArray {
        var hiddenStates = concatenated(
            [preFCNormEmbedding(inputsEmbeds), preFCNormHidden(previousHidden)], axis: -1)
        hiddenStates = fc(hiddenStates)

        precondition(cache.count == layers.count, "Qwen MTP cache/layer count mismatch")
        for (layer, layerCache) in zip(layers, cache) {
            let faMask = createAttentionMask(h: hiddenStates, cache: layerCache)
            hiddenStates = layer(
                hiddenStates,
                attentionMask: faMask,
                ssmMask: nil,
                cache: layerCache,
                positionOffset: positionOffset)
        }

        return norm(hiddenStates)
    }
}

public final class Qwen35MTPDraftModel: Module, StatefulMTPDrafterModel {
    public let configuration: Qwen35TextConfiguration
    public let maximumBlockSize: Int? = 8
    public let requiresSharedTargetKV = false
    public let requiresPromptPrefill = true
    public let requiresGreedySampling = true
    private let preconvertedNorms: Bool
    public private(set) var draftHead: Linear?

    @ModuleInfo(key: "mtp") var mtp: Qwen35MTPPredictor

    public init(
        _ configuration: Qwen35TextConfiguration,
        preconvertedNorms: Bool = false
    ) {
        self.configuration = configuration
        self.preconvertedNorms = preconvertedNorms
        _mtp.wrappedValue = Qwen35MTPPredictor(configuration)
        super.init()
    }

    public convenience init(
        _ configuration: Qwen35Configuration,
        preconvertedNorms: Bool = false
    ) {
        self.init(configuration.textConfig, preconvertedNorms: preconvertedNorms)
    }

    public func makeState(parameters: GenerateParameters?) -> MTPDrafterState {
        MTPDrafterState(cache: mtp.newCache())
    }

    public func prepareDrafterState(
        target: any LanguageModel,
        input: MTPPrefillInput,
        state: inout MTPDrafterState,
        sampler: any LogitSampler
    ) {
        let (targetEmbedTokens, lmHead) = targetEmbeddingAndHead(target)
        let inputEmbedding = mtp.embedTokens ?? targetEmbedTokens
        let prompt = normalizedMTPTokenBatch(input.promptTokens)
        guard prompt.dim(-1) > 0 else { return }

        let firstBonus = input.firstBonus
        let rows = input.targetHidden[0..., ..<prompt.dim(-1), 0...]
        let hiddenRows = state.carryHidden.map { concatenated([$0, rows], axis: 1) } ?? rows
        let nextTokens = state.carryHidden == nil ? prompt[0..., 1...] : prompt
        let shifted = firstBonus.map { concatenated([nextTokens, normalizedMTPColumn($0)], axis: 1) } ?? nextTokens
        state.carryHidden = firstBonus == nil ? hiddenRows[0..., (-1)..., 0...] : nil
        guard shifted.dim(1) > 0 else { return }
        let hidden = hiddenRows[0..., ..<shifted.dim(1), 0...]
        let offset = state.nextPosition
        let mtpHidden = mtp(
            inputsEmbeds: inputEmbedding(shifted),
            hiddenStates: hidden,
            cache: state.cache,
            positionOffset: offset)
        state.nextPosition = offset + shifted.dim(1)
        guard firstBonus != nil else { return }
        state.seedHidden = mtpHidden[0..., (-1)..., 0...]
        state.seedToken = sampleMTPSeed(
            hidden: state.seedHidden!, targetEmbedTokens: targetEmbedTokens,
            lmHead: lmHead, sampler: sampler)
        state.proposalAppended = 0
    }

    public func draftBlock(
        target: any LanguageModel,
        input: MTPDraftInput,
        sampler: any LogitSampler
    ) -> MLXArray {
        var state = makeState(parameters: nil)
        return draftBlock(target: target, input: input, state: &state, sampler: sampler)
    }

    public func draftBlock(
        target: any LanguageModel,
        input: MTPDraftInput,
        state: inout MTPDrafterState,
        sampler: any LogitSampler
    ) -> MLXArray {
        let blockSize = input.blockSize
        let (targetEmbedTokens, lmHead) = targetEmbeddingAndHead(target)
        let inputEmbedding = mtp.embedTokens ?? targetEmbedTokens

        if let seed = state.seedToken {
            let seedHidden = state.seedHidden
            state.seedToken = nil
            state.seedHidden = nil
            state.proposalAppended = 0
            guard blockSize > 2, let seedHidden else { return seed }
            let first = normalizedMTPColumn(seed)
            state.proposalAppended = blockSize - 2
            let rest = draftMTPTokenBlock(
                targetEmbedTokens: targetEmbedTokens,
                lmHead: lmHead,
                inputEmbedding: inputEmbedding,
                lastToken: first,
                lastHidden: seedHidden,
                queryOffset: state.nextPosition,
                blockSize: blockSize - 1,
                sampler: sampler,
                cache: state.cache
            ) { inputsEmbeds, hiddenStates, cache, positionOffset in
                mtp(
                    inputsEmbeds: inputsEmbeds,
                    hiddenStates: hiddenStates,
                    cache: cache,
                    positionOffset: positionOffset)
            }
            state.nextPosition += state.proposalAppended
            return concatenated([first, rest], axis: 1)
        }

        state.proposalAppended = blockSize - 1
        let proposed = draftMTPTokenBlock(
            targetEmbedTokens: targetEmbedTokens,
            lmHead: lmHead,
            inputEmbedding: inputEmbedding,
            lastToken: input.lastToken,
            lastHidden: input.lastHidden,
            queryOffset: input.queryOffset,
            blockSize: blockSize,
            sampler: sampler,
            cache: state.cache
        ) { inputsEmbeds, hiddenStates, cache, positionOffset in
            mtp(
                inputsEmbeds: inputsEmbeds,
                hiddenStates: hiddenStates,
                cache: cache,
                positionOffset: positionOffset)
        }
        state.nextPosition += state.proposalAppended
        return proposed
    }

    public func commitDrafterState(
        target: any LanguageModel,
        input: MTPCommitInput,
        state: inout MTPDrafterState,
        sampler: any LogitSampler
    ) {
        let acceptedCount = input.acceptedCount
        let keepAppended = Swift.min(acceptedCount, state.proposalAppended)
        let trim = state.proposalAppended - keepAppended
        if trim > 0 {
            trimPromptCache(state.cache, numTokens: trim)
            state.nextPosition -= trim
        }

        var tokens = [MLXArray]()
        var hiddens = [MLXArray]()
        for index in keepAppended ..< acceptedCount {
            tokens.append(input.draftTokens[0..., index ..< (index + 1)])
            hiddens.append(input.targetHidden[0..., index ..< (index + 1), 0...])
        }
        tokens.append(normalizedMTPColumn(input.finalToken))
        hiddens.append(input.targetHidden[0..., acceptedCount ..< (acceptedCount + 1), 0...])

        let (targetEmbedTokens, lmHead) = targetEmbeddingAndHead(target)
        let inputEmbedding = mtp.embedTokens ?? targetEmbedTokens
        let committedTokens = concatenated(tokens, axis: 1)
        let committedHidden = concatenated(hiddens, axis: 1)
        let mtpHidden = mtp(
            inputsEmbeds: inputEmbedding(committedTokens),
            hiddenStates: committedHidden,
            cache: state.cache,
            positionOffset: state.nextPosition)
        state.nextPosition += committedTokens.dim(1)
        state.seedHidden = mtpHidden[0..., (-1)..., 0...]
        state.seedToken = sampleMTPSeed(
            hidden: state.seedHidden!, targetEmbedTokens: targetEmbedTokens,
            lmHead: lmHead, sampler: sampler)
        state.proposalAppended = 0
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        qwenMTPSanitizeWeights(
            weights: weights,
            mtpNumHiddenLayers: configuration.mtpNumHiddenLayers,
            numExperts: configuration.numExperts,
            shiftNormWeights: !preconvertedNorms
        )
    }

    private func targetHead(_ target: any LanguageModel) -> (Embedding, Linear?) {
        if let model = target as? Qwen35Model {
            return (model.languageModel.model.embedTokens, model.languageModel.lmHead)
        }
        if let model = target as? Qwen35TextModel {
            return (model.model.embedTokens, model.lmHead)
        }
        fatalError(
            "Qwen35MTPDraftModel requires a Qwen35 target, got \(type(of: target))")
    }

    private func targetEmbeddingAndHead(_ target: any LanguageModel) -> (Embedding, Linear?) {
        let (embedding, head) = targetHead(target)
        return (embedding, draftHead ?? head)
    }

    @discardableResult
    public func useDraftHead(of target: any LanguageModel, bits: Int = 4, groupSize: Int = 64) -> Bool {
        let (embedding, head) = targetHead(target)
        let dense: MLXArray
        if let quantized = head as? QuantizedLinear {
            guard quantized.bits > bits else { return false }
            dense = dequantized(
                quantized.weight, scales: quantized.scales, biases: quantized.biases, groupSize: quantized.groupSize,
                bits: quantized.bits, mode: quantized.mode)
        } else if let head {
            dense = head.weight
        } else if let quantized = embedding as? QuantizedEmbedding {
            guard quantized.bits > bits else { return false }
            dense = dequantized(
                quantized.weight, scales: quantized.scales, biases: quantized.biases, groupSize: quantized.groupSize,
                bits: quantized.bits, mode: quantized.mode)
        } else {
            dense = embedding.weight
        }
        guard dense.dim(-1) % groupSize == 0 else { return false }
        let copy = QuantizedLinear(weight: dense, bias: head?.bias, groupSize: groupSize, bits: bits)
        eval(copy)
        draftHead = copy
        return true
    }
}
