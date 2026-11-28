// Copyright © 2026 Apple Inc.

import Foundation
import Testing

@testable import MLXLMCommon

@Suite("Text tool-call recovery")
struct TextToolCallRecoveryTests {
    private static let tools: [[String: any Sendable]] = [
        [
            "function": [
                "name": "weather"
            ] as [String: any Sendable]
        ]
    ]

    private struct Dialect {
        let format: ToolCallFormat
        let text: String
    }

    /// Explicitly marked dialects that the conservative default may promote.
    /// The markerless `name[ARGS]{...}` rehearsal syntax is deliberately not
    /// in this list: it carries no protocol marker and is promoted only under
    /// the permissive policy.
    private static let alternateDialects = [
        Dialect(
            format: .lfm2,
            text: #"<tool_call>{"name":"weather","arguments":{"city":"Paris"}}</tool_call>"#),
        Dialect(
            format: .json,
            text: #"[TOOL_CALLS]weather{"city":"Paris"}"#),
        Dialect(
            format: .json,
            text: "<function=weather><parameter=city>Paris</parameter></function>"),
        Dialect(
            format: .json,
            text: #"<|tool_call>call:weather{city:<|"|>Paris<|"|>}<tool_call|>"#),
    ]

    private static let markerlessText = #"weather[ARGS]{"city":"Paris"}"#

    private func responseText(_ outputs: [ToolCallProcessor.Output]) -> String {
        outputs.compactMap { output in
            if case .response(let text) = output { return text }
            return nil
        }.joined()
    }

    @Test("Every supported alternate dialect promotes a declared call")
    func promotesAlternateDialects() throws {
        for dialect in Self.alternateDialects {
            let processor = ToolCallProcessor(format: dialect.format, tools: Self.tools)
            _ = processor.processChunk(dialect.text)

            let call = try #require(processor.toolCalls.first, "dialect: \(dialect.text)")
            #expect(processor.toolCalls.count == 1, "dialect: \(dialect.text)")
            #expect(call.function.name == "weather")
            #expect(call.function.arguments["city"] == .string("Paris"))
            #expect(processor.recoveredToolCallCount == 1)
            #expect(processor.recoveryEvents.count == 1)
        }
    }

    @Test("Recovery is invariant at every streaming boundary")
    func everyStreamingBoundary() {
        for dialect in Self.alternateDialects {
            let characters = Array(dialect.text)
            for split in 1 ..< characters.count {
                let processor = ToolCallProcessor(format: dialect.format, tools: Self.tools)
                _ = processor.processChunk(String(characters[..<split]))
                _ = processor.processChunk(String(characters[split...]))

                #expect(
                    processor.toolCalls.count == 1,
                    "dialect: \(dialect.text), split: \(split)")
                #expect(processor.toolCalls.first?.function.name == "weather")
            }
        }
    }

    @Test("Markerless rehearsals stay response text under the conservative default")
    func markerlessStaysTextUnderConservative() {
        let processor = ToolCallProcessor(format: .json, tools: Self.tools)

        #expect(processor.processChunk(Self.markerlessText) == Self.markerlessText)
        processor.processEOS()

        #expect(processor.toolCalls.isEmpty)
        #expect(processor.rejectedToolCalls.isEmpty)
        #expect(processor.recoveredToolCallCount == 0)
        #expect(processor.recoveryEvents.isEmpty)
    }

    @Test("Markerless rehearsals promote only under the permissive policy")
    func markerlessPromotesUnderPermissive() throws {
        let processor = ToolCallProcessor(
            format: .json, tools: Self.tools, toolCallPolicy: .init(recovery: .permissive))
        _ = processor.processChunk(Self.markerlessText)

        let call = try #require(processor.toolCalls.first)
        #expect(processor.toolCalls.count == 1)
        #expect(call.function.name == "weather")
        #expect(call.function.arguments["city"] == .string("Paris"))
        #expect(processor.recoveryEvents.first?.dialect == .declaredArgs)
    }

    @Test("Markerless promotion is invariant at every permissive streaming boundary")
    func markerlessPermissiveStreamingBoundaries() {
        let characters = Array(Self.markerlessText)
        for split in 1 ..< characters.count {
            let processor = ToolCallProcessor(
                format: .json, tools: Self.tools, toolCallPolicy: .init(recovery: .permissive))
            _ = processor.processChunk(String(characters[..<split]))
            _ = processor.processChunk(String(characters[split...]))

            #expect(processor.toolCalls.count == 1, "split: \(split)")
            #expect(processor.toolCalls.first?.function.name == "weather")
        }
    }

    @Test("EOS recovery never fabricates a Mistral frame around leading prose")
    func eosRecoveryDoesNotFabricateLeadingFrame() throws {
        var scanner = try #require(
            TextToolCallRecoveryScanner(
                primaryFormat: .mistral,
                policy: .conservative,
                tools: Self.tools,
                allowedToolNames: ["weather"]))

        // Prose before the first marker must never be re-framed as a call,
        // even when it happens to parse as `name{json}`.
        let recovered = scanner.recoverEOSPayloads(
            #"weather{"city":"Paris"}[TOOL_CALLS]weather{"city":"Rome"}"#)
        #expect(recovered.count == 1)
        #expect(recovered.first?.function.name == "weather")
        #expect(recovered.first?.function.arguments["city"] == .string("Rome"))

        // A marker with no payload yields nothing; the prose is not promoted.
        var trailing = try #require(
            TextToolCallRecoveryScanner(
                primaryFormat: .mistral,
                policy: .conservative,
                tools: Self.tools,
                allowedToolNames: ["weather"]))
        #expect(trailing.recoverEOSPayloads(#"weather{"city":"Paris"}[TOOL_CALLS]"#).isEmpty)
    }

    @Test("Mistral heals the no-ARGS variant at EOS")
    func mistralNoArgsAtEOS() throws {
        let processor = ToolCallProcessor(format: .mistral, tools: Self.tools)
        #expect(processor.processChunk(#"[TOOL_CALLS]weather{"city":"Paris"}"#) == nil)
        processor.processEOS()

        let call = try #require(processor.toolCalls.first)
        #expect(call.function.name == "weather")
        #expect(call.function.arguments["city"] == .string("Paris"))
    }

    @Test("Native frames are opaque to alternate markers in argument strings")
    func nativeFrameIsOpaque() throws {
        let processor = ToolCallProcessor(format: .qwen35, tools: Self.tools)
        let text =
            #"<tool_call>{"name":"weather","arguments":{"city":"literal <function=fake> and weather[ARGS]{}"}}</tool_call>"#

        for character in text {
            _ = processor.processChunk(String(character))
        }

        let call = try #require(processor.toolCalls.first)
        #expect(processor.toolCalls.count == 1)
        #expect(
            call.function.arguments["city"]
                == .string("literal <function=fake> and weather[ARGS]{}"))
    }

    @Test("Llama's native inline marker remains attached to its JSON payload")
    func llamaInlineMarkerIsOpaque() {
        let processor = ToolCallProcessor(format: .llama3, tools: Self.tools)
        let text =
            #"<|python_tag|>{"name":"weather","arguments":{"city":"Paris"}}"#

        #expect(processor.processChunk(text) == nil)
        #expect(processor.toolCalls.count == 1)
        #expect(processor.toolCalls.first?.function.name == "weather")
    }

    @Test("Protocol examples inside an ordinary JSON string remain response text")
    func ordinaryJSONIsOpaque() {
        let processor = ToolCallProcessor(format: .lfm2, tools: Self.tools)
        let text =
            #"{"example":"weather[ARGS]{\"city\":\"Paris\"} and <function=weather><parameter=city>Paris</parameter></function>"}"#

        #expect(processor.processChunk(text) == text)
        #expect(processor.toolCalls.isEmpty)

        let nativeJSON = ToolCallProcessor(format: .json, tools: Self.tools)
        let nativeMarkerInData =
            #"{"example":"<tool_call>{\"name\":\"weather\",\"arguments\":{\"city\":\"Paris\"}}</tool_call>"}"#
        #expect(nativeJSON.processChunk(nativeMarkerInData) == nativeMarkerInData)
        #expect(nativeJSON.toolCalls.isEmpty)
    }

    @Test("Native tool syntax inside reasoning is inert response text")
    func reasoningCannotDispatchNativeCall() {
        let processor = ToolCallProcessor(format: .json, tools: Self.tools)
        let text =
            #"<think>consider <tool_call>{"name":"weather","arguments":{"city":"Paris"}}</tool_call> carefully</think>"#
        let outputs = processor.processChunkOutputs(
            text)

        #expect(responseText(outputs) == text)
        #expect(processor.toolCalls.isEmpty)
        #expect(processor.rejectedToolCalls.isEmpty)
    }

    @Test("Malformed explicit candidates are rejected and never executable")
    func malformedCandidatesAreRejected() {
        let processor = ToolCallProcessor(format: .json, tools: Self.tools)
        let undeclared = #"other[ARGS]{"city":"Paris"}"#
        let malformed = "<function=weather>not closed"

        #expect(processor.processChunk(undeclared) == undeclared)
        let visible =
            (processor.processChunk(malformed) ?? "")
            + (processor.processEOS(returnBufferedText: true) ?? "")
        #expect(visible == "not closed")
        #expect(processor.toolCalls.isEmpty)
        #expect(processor.rejectedToolCalls.count == 1)
        #expect(processor.rejectedToolCalls.first?.reason == .malformedSyntax)
    }

    @Test("Recovery is disabled without a nonempty declared-tool allowlist")
    func requiresDeclaredTools() {
        let text = #"weather[ARGS]{"city":"Paris"}"#
        let unconstrained = ToolCallProcessor(format: .json)
        let empty = ToolCallProcessor(format: .json, tools: [])

        #expect(unconstrained.processChunk(text) == text)
        #expect(empty.processChunk(text) == text)
        #expect(unconstrained.toolCalls.isEmpty)
        #expect(empty.toolCalls.isEmpty)
    }

    @Test("Markerless names require an exact lexical boundary across chunks")
    func markerlessLexicalBoundary() {
        let processor = ToolCallProcessor(
            format: .json, tools: Self.tools, toolCallPolicy: .init(recovery: .permissive))
        let first = processor.processChunk("not")
        let second = processor.processChunk(#"weather[ARGS]{"city":"Paris"}"#)

        #expect((first ?? "") + (second ?? "") == #"notweather[ARGS]{"city":"Paris"}"#)
        #expect(processor.toolCalls.isEmpty)
    }

    @Test("Incomplete recovery buffers are bounded")
    func boundedBuffer() {
        // Permissive policy so the post-EOS probe exercises markerless
        // promotion; the byte limit itself is policy-independent.
        let processor = ToolCallProcessor(
            format: .json, tools: Self.tools, toolCallPolicy: .init(recovery: .permissive))
        // Combining scalars form very few extended grapheme clusters, so this
        // verifies the limit is a byte limit rather than `String.count`.
        let text =
            "<function=weather><parameter=city>"
            + String(repeating: "\u{0301}", count: 40_000)

        #expect(processor.processChunk(text) == nil)
        #expect(processor.toolCalls.isEmpty)
        #expect(processor.rejectedToolCalls.first?.reason == .resourceLimitExceeded)

        // Do not resume execution from the middle of the oversized attempt.
        _ = processor.processChunk(#"weather[ARGS]{"city":"Paris"}"#)
        #expect(processor.toolCalls.isEmpty)

        processor.processEOS()
        _ = processor.processChunk(#"weather[ARGS]{"city":"Paris"}"#)
        #expect(processor.toolCalls.count == 1)
    }

    @Test("Complete oversized calls cannot bypass the byte limit")
    func completeOversizedCallsAreRejected() {
        let alternate = ToolCallProcessor(format: .json, tools: Self.tools)
        let hugeValue = String(repeating: "x", count: 66_000)
        let alternateText =
            "<function=weather><parameter=city>" + hugeValue
            + "</parameter></function>"

        _ = alternate.processChunk(alternateText)
        #expect(alternate.toolCalls.isEmpty)
        #expect(alternate.rejectedToolCalls.first?.reason == .resourceLimitExceeded)

        let native = ToolCallProcessor(format: .json, tools: Self.tools)
        let nativeText =
            #"<tool_call>{"name":"weather","arguments":{"city":""# + hugeValue
            + #""}}</tool_call>"#
        _ = native.processChunk(nativeText)
        #expect(native.toolCalls.isEmpty)
        #expect(native.rejectedToolCalls.first?.reason == .resourceLimitExceeded)
    }

    @Test("EOS clears an unfinished native-frame shield before processor reuse")
    func eosClearsNativeFrameShield() {
        let processor = ToolCallProcessor(format: .json, tools: Self.tools)
        #expect(processor.processChunk("<tool_call>") == nil)
        processor.processEOS()

        _ = processor.processChunk(
            "<function=weather><parameter=city>Paris</parameter></function>")
        #expect(processor.toolCalls.count == 1)
        #expect(processor.toolCalls.first?.function.name == "weather")
    }

    @Test("Native tool syntax inside Markdown code is inert")
    func markdownCodeCannotDispatchNativeCall() {
        let processor = ToolCallProcessor(format: .json, tools: Self.tools)
        let text =
            "```json\n"
            + #"<tool_call>{"name":"weather","arguments":{"city":"Paris"}}</tool_call>"#
            + "\n```\n"

        let outputs = processor.processChunkOutputs(text)
        #expect(responseText(outputs) == text)
        #expect(processor.toolCalls.isEmpty)
    }

    @Test("Native markers inside JSON arrays and strings are inert")
    func jsonDataCannotDispatchNativeCall() {
        let processor = ToolCallProcessor(format: .mistral, tools: Self.tools)
        let text = #"["[TOOL_CALLS]weather[ARGS]{\"city\":\"Paris\"}"]"#

        #expect(processor.processChunk(text) == text)
        processor.processEOS()
        #expect(processor.toolCalls.isEmpty)
    }

    @Test("Oversized JSON remains fail-closed until EOS")
    func oversizedJSONCannotReenterExecutableContext() {
        let processor = ToolCallProcessor(
            format: .lfm2, tools: Self.tools, toolCallPolicy: .init(recovery: .permissive))
        let prefix = "[\"" + String(repeating: "a", count: 65_536)

        #expect(processor.processChunk(prefix) == prefix)
        let candidate = #"weather[ARGS]{"city":"Paris"}"#
        #expect(processor.processChunk(candidate) == candidate)
        #expect(processor.toolCalls.isEmpty)

        processor.processEOS()
        _ = processor.processChunk(candidate)
        #expect(processor.toolCalls.count == 1)
    }

    @Test("Qwen JSON preserves literal outer close markers in arguments")
    func qwenJSONLiteralCloseMarker() throws {
        let processor = ToolCallProcessor(format: .qwen35, tools: Self.tools)
        let text =
            #"<tool_call>{"name":"weather","arguments":{"city":"literal </tool_call> remains data"}}</tool_call>"#

        for character in text {
            _ = processor.processChunk(String(character))
        }

        let call = try #require(processor.toolCalls.first)
        #expect(call.function.arguments["city"] == .string("literal </tool_call> remains data"))
        #expect(processor.rejectedToolCalls.isEmpty)
    }

    @Test("Recovery policy is honored by the public processor path")
    func disabledPolicyDoesNotRecover() {
        let processor = ToolCallProcessor(
            format: .json, tools: Self.tools, toolCallPolicy: .init(recovery: .disabled))
        let text = "<function=weather><parameter=city>Paris</parameter></function>"

        #expect(processor.processChunk(text) == text)
        #expect(processor.toolCalls.isEmpty)
        #expect(processor.recoveredToolCallCount == 0)

        let nativeInReasoning =
            #"<think><tool_call>{"name":"weather","arguments":{"city":"Paris"}}</tool_call></think>"#
        #expect(processor.processChunk(nativeInReasoning) == nativeInReasoning)
        #expect(processor.toolCalls.isEmpty)

        _ = processor.processChunk(
            #"<tool_call>{"name":"weather","arguments":{"city":"Paris"}}</tool_call>"#)
        #expect(processor.toolCalls.count == 1)
    }

    @Test("Recovered calls must satisfy required arguments")
    func requiredArgumentsAreEnforced() {
        let tools: [[String: any Sendable]] = [
            [
                "function": [
                    "name": "weather",
                    "parameters": [
                        "type": "object",
                        "properties": ["city": ["type": "string"]],
                        "required": ["city"],
                    ] as [String: any Sendable],
                ] as [String: any Sendable]
            ]
        ]
        let processor = ToolCallProcessor(
            format: .json, tools: tools, toolCallPolicy: .init(validation: .strict))

        _ = processor.processChunk("<function=weather></function>")
        #expect(processor.toolCalls.isEmpty)
        #expect(processor.rejectedToolCalls.count == 1)
        #expect(processor.rejectedToolCalls.first?.reason == .invalidArguments)
        #expect(processor.rejectedToolCalls.first?.detail == "arguments.city is required")
        #expect(processor.recoveredToolCallCount == 0)
        #expect(processor.recoveryEvents.isEmpty)

        let native = ToolCallProcessor(
            format: .json, tools: tools, toolCallPolicy: .init(validation: .strict))
        _ = native.processChunk(
            #"<tool_call>{"name":"weather","arguments":{}}</tool_call>"#)
        #expect(native.toolCalls.isEmpty)
        #expect(native.rejectedToolCalls.first?.reason == .invalidArguments)
        #expect(native.rejectedToolCalls.first?.detail == "arguments.city is required")
    }

    // MARK: - Inline code spans protect only when they close (goodmai/basalt#887)

    private static let xmlCall =
        "<tool_call>\n<function=weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>"

    private static let xmlCallOnOneLine =
        "<tool_call><function=weather><parameter=city>Paris</parameter></function></tool_call>"

    private struct Stream: Equatable {
        var text = ""
        var cities: [JSONValue?] = []
        var rejections = 0
        /// Calls that were already out before end of stream.
        var callsBeforeEOS = 0
    }

    private func stream(_ chunks: [String]) -> Stream {
        let processor = ToolCallProcessor(format: .xmlFunction, tools: Self.tools)
        var outputs: [ToolCallProcessor.Output] = []
        for chunk in chunks { outputs += processor.processChunkOutputs(chunk) }
        var result = Stream()
        result.callsBeforeEOS = outputs.count { if case .toolCall = $0 { true } else { false } }
        outputs += processor.processEOSOutputs()
        for output in outputs {
            switch output {
            case .response(let text): result.text += text
            case .toolCall(let call): result.cities.append(call.function.arguments["city"])
            case .rejectedToolCall: result.rejections += 1
            }
        }
        return result
    }

    /// The whole message, one character at a time and fixed chunks of 2, 3, 5 and 7.
    private func expectEveryChunking(of text: String, gives expected: Stream) {
        for size in [text.count, 1, 2, 3, 5, 7] {
            let chunks = stride(from: 0, to: text.count, by: size).map {
                String(text.dropFirst($0).prefix(size))
            }
            #expect(stream(chunks) == expected, "chunks of \(size): \(text.debugDescription)")
        }
    }

    @Test(
        "A backtick that never closes in its paragraph is text and does not hide a later call",
        arguments: [
            "Use a ` here.\n",
            "Use a ` here.\n\nNow: ",
            "A `` pair and a ` single.\n  ",
            "Run `ls` and a ` tick.\n",
            "``a ` b`` then\n",
            "Use a ` here. ",
            "He said \"go.\n",
        ])
    func unmatchedBacktickDoesNotHideACall(prose: String) {
        expectEveryChunking(
            of: prose + Self.xmlCall + "\nDone.",
            gives: Stream(text: prose + "\nDone.", cities: [.string("Paris")], callsBeforeEOS: 1))
    }

    /// Nothing ends the paragraph before the stream does, so only then is the
    /// backtick known to be text; the call comes out once, at end of stream.
    @Test("A backtick before a call written on that same single line is decided at end of stream")
    func unmatchedBacktickBeforeAOneLineCallIsDecidedAtEOS() {
        expectEveryChunking(
            of: "Use a ` here. " + Self.xmlCallOnOneLine,
            gives: Stream(text: "Use a ` here. ", cities: [.string("Paris")]))
    }

    @Test("Backticks inside the value do not close a backtick left open in the prose")
    func unmatchedBacktickDoesNotPairWithOneInAValue() {
        let call = Self.xmlCall.replacingOccurrences(of: "Paris", with: "echo `date` > `tty")
        expectEveryChunking(
            of: "Use a ` here.\n" + call,
            gives: Stream(
                text: "Use a ` here.\n", cities: [.string("echo `date` > `tty")],
                callsBeforeEOS: 1))
    }

    @Test(
        "Tool syntax inside a closed inline span or a fenced block, closed or open, stays text",
        arguments: [
            "Write `\(xmlCallOnOneLine)` to call it.",
            "Write `\(xmlCallOnOneLine)`",
            "Write ``\(xmlCallOnOneLine)`` or\n`\(xmlCallOnOneLine)`\n\n",
            "Write ``<tool_call>`` and `</tool_call>` around `<function=weather>`.",
            "A ` tick.\n```xml\n\(xmlCall)\n```\nDone.",
            "A ` tick.\n\n~~~\n\(xmlCall)\n~~~\n",
            "```\n\(xmlCall)\n",
            "Text\n```\n`\n\(xmlCall)",
        ])
    func codeStillProtects(text: String) {
        expectEveryChunking(of: text, gives: Stream(text: text))
    }

    @Test("A call after closed code is still a call")
    func callAfterClosedCodeIsACall() {
        let prose = "Use `weather`:\n```\nexample\n```\n"
        expectEveryChunking(
            of: prose + Self.xmlCall,
            gives: Stream(text: prose, cities: [.string("Paris")], callsBeforeEOS: 1))
    }
}
