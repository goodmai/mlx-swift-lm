// Copyright © 2025 Apple Inc.

import Foundation
import MLX
import MLXLLM
import Testing

@testable import MLXLMCommon

struct ModelContainerPerformSendingTests {

    final class Counter {
        var value = 0
    }

    @Test func `performSending hands a non-Sendable value in and back out`() async throws {
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
        let container = ModelContainer(
            context: ModelContext(
                configuration: processor.configuration, model: Gemma3TextModel(config),
                processor: processor, tokenizer: processor.tokenizer))
        let returned = await container.performSending(nonSendable: Counter()) { _, counter in
            counter.value += 1
            return counter
        }
        returned.value += 1
        #expect(returned.value == 2)
    }
}
