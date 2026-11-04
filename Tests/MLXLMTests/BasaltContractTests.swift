// Copyright © 2026 Apple Inc.

import Dispatch
import Foundation
import MLX
import MLXLLM
import MLXNN
import Testing

@testable import MLXLMCommon

/// Pins the surface goodmai/basalt builds on: a change that fails here breaks basalt.
struct BasaltContractTests {

    final class Counter {
        var value = 0
    }

    final class Token: Sendable {}

    final class BatchState {
        let token = Token()
    }

    private final class StubDrafter: Module, MTPDrafterModel {
        func draftBlock(target: any LanguageModel, input: MTPDraftInput, sampler: any LogitSampler)
            -> MLXArray
        {
            MLXArray.zeros([input.lastToken.dim(0), input.blockSize - 1], dtype: .int32)
        }
    }

    /// An engine actor that hands the container values from its own region, as basalt's does.
    actor Engine {
        let container: ModelContainer
        var counter = Counter()

        init(container: ModelContainer) {
            self.container = container
        }

        func send() async -> Int {
            let returned = await container.performSending(nonSendable: counter) { _, counter in
                counter.value += 1
                return counter
            }
            returned.value += 1
            counter = returned
            return counter.value
        }

        func evaluate() async -> (value: Int, held: Bool) {
            let (returned, held) = await container.performEvaluating(nonSendable: counter) {
                context, counter in
                let lock = ModelEvaluationLock.lock(for: context.model)
                let held = !BasaltContractTests.acquiredElsewhere(lock, within: .milliseconds(200))
                let nested = context.model.evaluating { BasaltContractTests.nested(context.model) }
                counter.value += nested
                return (counter, held)
            }
            counter = returned
            return (counter.value, held)
        }
    }

    static func tinyContext() -> ModelContext {
        let processor = TestInputProcessor()
        let config = Gemma3TextConfiguration(
            modelType: "text",
            hiddenSize: 64, hiddenLayers: 1, intermediateSize: 64,
            attentionHeads: 4, headDim: 64,
            rmsNormEps: 0.00001, vocabularySize: 100, kvHeads: 4,
            ropeTheta: 1_000_000, ropeLocalBaseFreq: 10_000,
            ropeTraditional: false, queryPreAttnScalar: 256,
            slidingWindow: 512, slidingWindowPattern: 6,
            maxPositionEmbeddings: 32768
        )
        return ModelContext(
            configuration: processor.configuration, model: Gemma3TextModel(config),
            processor: processor, tokenizer: processor.tokenizer)
    }

    static func nested(_ model: any LanguageModel) -> Int {
        model.evaluating { 1 }
    }

    /// Whether another thread takes `lock` within `timeout`.
    static func acquiredElsewhere(_ lock: FairRecursiveLock, within timeout: DispatchTimeInterval)
        -> Bool
    {
        let acquired = DispatchSemaphore(value: 0)
        Thread { _ = lock.withLock { acquired.signal() } }.start()
        return acquired.wait(timeout: .now() + timeout) == .success
    }

    @Test func `performSending takes and returns non-Sendable values on an actor`() async {
        let engine = Engine(container: ModelContainer(context: Self.tinyContext()))
        #expect(await engine.send() == 2)
        #expect(await engine.send() == 4)
    }

    @Test func `performEvaluating holds the model's lock around a reentrant body`() async {
        let container = ModelContainer(context: Self.tinyContext())
        let engine = Engine(container: container)
        let first = await engine.evaluate()
        #expect(first.value == 1)
        #expect(first.held)
        #expect(await engine.evaluate().value == 2)
        let lock = await container.perform { ModelEvaluationLock.lock(for: $0.model) }
        #expect(Self.acquiredElsewhere(lock, within: .seconds(5)))
    }

    @Test func `evaluating runs its body under the model's lock and is reentrant`() {
        let model = Self.tinyContext().model
        let other = Self.tinyContext().model
        let lock = ModelEvaluationLock.lock(for: model)
        let (held, otherFree, nested) = model.evaluating {
            (
                !Self.acquiredElsewhere(lock, within: .milliseconds(200)),
                Self.acquiredElsewhere(ModelEvaluationLock.lock(for: other), within: .seconds(5)),
                model.evaluating { Self.nested(model) }
            )
        }
        #expect(held)
        #expect(otherFree)
        #expect(nested == 1)
        #expect(Self.acquiredElsewhere(lock, within: .seconds(5)))
    }

    @Test func `mtpDrafter and batch slots survive update and perform`() async {
        let container = ModelContainer(context: Self.tinyContext())
        let empty = await container.perform { $0.mtpDrafter == nil && $0.batch == nil }
        #expect(empty)

        await container.update {
            $0.mtpDrafter = StubDrafter()
            if $0.batch == nil { $0.batch = BatchState() }
        }
        let drafter = await container.perform { $0.mtpDrafter.map { ObjectIdentifier($0) } }
        let token = await container.perform { ($0.batch as? BatchState)?.token }
        #expect(drafter != nil)
        #expect(token != nil)

        await container.update { if $0.batch == nil { $0.batch = BatchState() } }
        await container.update { _ in }
        let seen = await container.performSending(nonSendable: ()) { context, _ in
            (
                context.mtpDrafter.map { ObjectIdentifier($0) },
                (context.batch as? BatchState)?.token
            )
        }
        #expect(seen.0 == drafter)
        #expect(seen.1 === token)
        let after = await container.perform { ($0.batch as? BatchState)?.token }
        #expect(after === token)
    }

    @Test func `MTP input values keep the fields basalt passes through`() {
        let token = MLXArray([Int32(7)])
        let hidden = MLXArray.zeros([1, 1, 4])
        let keys = MLXArray.zeros([1, 1, 8, 4])
        let values = MLXArray.zeros([1, 1, 8, 4])

        var draft = MTPDraftInput(
            lastToken: token, lastHidden: hidden, sharedKV: ["full_attention": (keys, values)],
            queryOffset: 5, blockSize: 4)
        #expect(draft.lastToken === token)
        #expect(draft.lastHidden === hidden)
        #expect(draft.sharedKV["full_attention"]?.0 === keys)
        #expect(draft.sharedKV["full_attention"]?.1 === values)
        #expect(draft.positionDeltas == nil)
        #expect(draft.queryOffset == 5)
        #expect(draft.blockSize == 4)
        draft.queryOffset = 6
        draft.positionDeltas = token
        #expect(draft.queryOffset == 6)
        #expect(draft.positionDeltas === token)

        let prefill = MTPPrefillInput(promptTokens: token, targetHidden: hidden, firstBonus: nil)
        #expect(prefill.promptTokens === token)
        #expect(prefill.targetHidden === hidden)
        #expect(prefill.firstBonus == nil)
        #expect(prefill.positionDeltas == nil)

        let commit = MTPCommitInput(
            targetHidden: hidden, draftTokens: token, acceptedCount: 2, finalToken: token)
        #expect(commit.targetHidden === hidden)
        #expect(commit.draftTokens === token)
        #expect(commit.acceptedCount == 2)
        #expect(commit.finalToken === token)
        #expect(commit.positionDeltas == nil)

        let cache = KVCacheSimple()
        var state = MTPDrafterState(cache: [cache], nextPosition: 3, carryHidden: hidden)
        #expect(state.cache.first as AnyObject === cache as AnyObject)
        #expect(state.nextPosition == 3)
        #expect(state.carryHidden === hidden)
        #expect(state.seedToken == nil)
        #expect(state.seedHidden == nil)
        #expect(state.proposalAppended == 0)
        state.nextPosition = 4
        #expect(state.nextPosition == 4)
    }
}
