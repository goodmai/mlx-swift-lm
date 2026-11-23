// Copyright © 2026 Apple Inc.

import Foundation

/// Chooses how many drafts a speculative round verifies, from what the target accepted so far
/// and what a wider verify pass costs.
///
/// Keeps a decaying count, per draft position, of rounds that reached the position and rounds
/// that accepted it, so its estimate of each position's acceptance follows the content: code
/// accepts deep blocks, prose few. A round with `k` drafts yields `1 + Σ_{j≤k} Π_{i≤j} a_i`
/// tokens on average and costs `1 + widthCost · k` single-token steps; the policy picks the `k`
/// with the most tokens per step.
public struct MTPDraftDepthPolicy: Sendable {
    /// The extra cost of one more verified position, as a fraction of a single-token step.
    public let widthCost: Double
    public let decay: Double
    private var reached: [Double]
    private var accepted: [Double]
    private let priorAcceptance: Double
    private let priorWeight: Double

    public init(
        widthCost: Double, maximumDepth: Int, decay: Double = 0.95, priorAcceptance: Double = 0.7,
        priorWeight: Double = 2
    ) {
        precondition(widthCost > 0 && maximumDepth > 0 && (0 ..< 1).contains(decay))
        self.widthCost = widthCost
        self.decay = decay
        self.priorAcceptance = priorAcceptance
        self.priorWeight = priorWeight
        reached = Array(repeating: 0, count: maximumDepth)
        accepted = Array(repeating: 0, count: maximumDepth)
    }

    /// The estimated chance that position `index` (0-based) is accepted given the ones before it.
    public func acceptance(at index: Int) -> Double {
        (accepted[index] + priorAcceptance * priorWeight) / (reached[index] + priorWeight)
    }

    /// The depth, from 1 to `maximum`, with the most expected tokens per unit of verify cost.
    public func depth(maximum: Int) -> Int {
        let limit = Swift.min(maximum, reached.count)
        guard limit > 1 else { return Swift.max(1, limit) }
        var (best, bestScore) = (1, 0.0)
        var (expected, chain) = (1.0, 1.0)
        for k in 1 ... limit {
            chain *= acceptance(at: k - 1)
            expected += chain
            let score = expected / (1 + widthCost * Double(k))
            if score > bestScore { (best, bestScore) = (k, score) }
        }
        return best
    }

    /// Records a round that verified `drafted` drafts and accepted the first `acceptedCount`.
    public mutating func record(drafted: Int, accepted acceptedCount: Int) {
        for index in reached.indices {
            reached[index] *= decay
            accepted[index] *= decay
        }
        for index in 0 ..< Swift.min(drafted, reached.count) where index <= acceptedCount {
            reached[index] += 1
            if index < acceptedCount { accepted[index] += 1 }
        }
    }
}
