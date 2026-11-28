// Copyright © 2026 Apple Inc.

import Foundation
import MLXLMCommon
import Testing

/// Structural strictness of the native `.xmlFunction` protocol.
///
/// `XMLFunctionParser` previously accepted a structurally degraded payload and
/// returned a call with silently missing arguments. The only thing preventing
/// execution was schema-level required-argument validation, so a tool whose
/// schema omitted `required` could be invoked with no arguments at all from a
/// truncated frame. These tests pin the parser to the shared
/// `QwenXMLPayloadScanner` grammar and deliberately declare **no** `required`
/// list, so structural validity alone must carry the rejection.
struct XMLFunctionParserStrictnessTests {

    private let parser = XMLFunctionParser(startTag: "<tool_call>", endTag: "</tool_call>")

    /// Schemas without a `required` list: required-argument validation cannot
    /// mask a structural parser flaw here.
    private func toolSchemas(_ names: String...) -> [[String: any Sendable]] {
        names.map { name in
            ["function": ["name": name] as [String: any Sendable]]
        }
    }

    // MARK: - Structural rejection

    @Test("An unclosed <parameter> never becomes a call")
    func unclosedParameterIsRejected() {
        let payloads = [
            "<function=weather><parameter=city>Paris</function>",
            "<tool_call><function=weather><parameter=city>Paris</function></tool_call>",
            """
            <tool_call>
            <function=weather>
            <parameter=city>Paris
            </function>
            </tool_call>
            """,
            // First parameter closes, second does not: partial success is still
            // a structurally invalid payload.
            "<function=weather><parameter=city>Paris</parameter><parameter=unit>c</function>",
        ]

        for payload in payloads {
            #expect(
                parser.parse(content: payload, tools: nil) == nil,
                "expected structural rejection of: \(payload)")
        }
    }

    @Test("A missing </function> never becomes a call")
    func missingFunctionCloseIsRejected() {
        let payloads = [
            "<function=weather>",
            "<function=weather><parameter=city>Paris</parameter>",
            "<tool_call><function=weather><parameter=city>Paris</parameter></tool_call>",
        ]

        for payload in payloads {
            #expect(
                parser.parse(content: payload, tools: nil) == nil,
                "expected structural rejection of: \(payload)")
        }
    }

    @Test("Malformed function and parameter names never become a call")
    func malformedNamesAreRejected() {
        let payloads = [
            "<function=><parameter=city>Paris</parameter></function>",
            "<function=get weather><parameter=city>Paris</parameter></function>",
            "<function=weather><parameter=>Paris</parameter></function>",
        ]

        for payload in payloads {
            #expect(
                parser.parse(content: payload, tools: nil) == nil,
                "expected structural rejection of: \(payload)")
        }
    }

    @Test("Content beyond the payload is never silently discarded")
    func trailingContentIsRejected() {
        let payloads = [
            // A second call in one frame must not silently collapse to the first.
            "<function=a><parameter=x>1</parameter></function><function=b></function>",
            "<function=weather></function>trailing text",
            // A complete frame followed by text: the streaming processor splits
            // this, so a direct parse must not quietly drop the remainder.
            "<tool_call><function=weather></function></tool_call>trailing",
        ]

        for payload in payloads {
            #expect(
                parser.parse(content: payload, tools: nil) == nil,
                "expected rejection of payload with trailing content: \(payload)")
        }
    }

    // MARK: - Literal markers inside values are data

    @Test("A literal </function> inside a parameter value is preserved")
    func literalFunctionCloseIsPreserved() throws {
        let content = """
            <tool_call>
            <function=weather>
            <parameter=city>
            literal </function> text
            </parameter>
            </function>
            </tool_call>
            """

        let call = try #require(parser.parse(content: content, tools: nil))

        #expect(call.function.name == "weather")
        #expect(call.function.arguments["city"] == .string("literal </function> text"))
    }

    @Test("A literal </tool_call> inside a parameter value is preserved")
    func literalFrameCloseIsPreserved() throws {
        let content =
            "<tool_call><function=weather><parameter=city>a</tool_call>b</parameter>"
            + "</function></tool_call>"

        let call = try #require(parser.parse(content: content, tools: nil))

        #expect(call.function.arguments["city"] == .string("a</tool_call>b"))
    }

    // MARK: - Canonical payloads keep working

    @Test("Canonical payloads still parse unchanged")
    func canonicalPayloadsStillParse() throws {
        let unframed = try #require(
            parser.parse(
                content:
                    "<function=get_weather><parameter=location>Tokyo</parameter>"
                    + "<parameter=unit>celsius</parameter></function>",
                tools: nil))
        #expect(unframed.function.name == "get_weather")
        #expect(unframed.function.arguments["location"] == .string("Tokyo"))
        #expect(unframed.function.arguments["unit"] == .string("celsius"))

        let framedMultiline = try #require(
            parser.parse(
                content: """
                    <tool_call>
                    <function=get_current_datetime>
                    </function>
                    </tool_call>
                    """,
                tools: nil))
        #expect(framedMultiline.function.name == "get_current_datetime")
        #expect(framedMultiline.function.arguments.isEmpty)

        let multilineValue = try #require(
            parser.parse(
                content: """
                    <function=get_weather>
                    <parameter=location>
                    Tokyo
                    </parameter>
                    </function>
                    """,
                tools: nil))
        #expect(multilineValue.function.arguments["location"] == .string("Tokyo"))
    }

    @Test("Schema-driven type conversion is unchanged")
    func typeConversionIsUnchanged() throws {
        let tools: [[String: any Sendable]] = [
            [
                "function": [
                    "name": "search",
                    "parameters": [
                        "properties": [
                            "page": ["type": "integer"],
                            "filters": ["type": "object"],
                        ]
                    ],
                ] as [String: any Sendable]
            ]
        ]
        let content =
            "<tool_call><function=search><parameter=page>1</parameter>"
            + #"<parameter=filters>{"archived": false, "limit": 1}</parameter></function></tool_call>"#

        let call = try #require(parser.parse(content: content, tools: tools))

        #expect(call.function.arguments["page"] == .int(1))
        #expect(
            call.function.arguments["filters"]
                == .object(["archived": .bool(false), "limit": .int(1)]))
    }

    // MARK: - End-to-end through ToolCallProcessor

    @Test("Native XML rejects an unclosed parameter at EOS")
    func nativeXMLRejectsUnclosedParameterAtEOS() {
        let processor = ToolCallProcessor(
            format: .xmlFunction,
            tools: toolSchemas("weather"))

        _ = processor.processChunk(
            "<tool_call><function=weather><parameter=city>Paris</function></tool_call>")
        processor.processEOS()

        #expect(processor.toolCalls.isEmpty)
        #expect(processor.rejectedToolCalls.count == 1)
    }

    @Test("A truncated frame is rejected at every chunk boundary")
    func truncatedFrameRejectedAtEveryChunkBoundary() {
        let payload = """
            <tool_call>
            <function=weather>
            <parameter=city>Paris
            </function>
            </tool_call>
            """

        for splitOffset in 0 ... payload.count {
            let split = payload.index(payload.startIndex, offsetBy: splitOffset)
            let processor = ToolCallProcessor(
                format: .xmlFunction,
                tools: toolSchemas("weather"))

            _ = processor.processChunk(String(payload[..<split]))
            _ = processor.processChunk(String(payload[split...]))
            processor.processEOS()

            #expect(
                processor.toolCalls.isEmpty,
                "executed a truncated frame when split at \(splitOffset)")
            #expect(
                processor.rejectedToolCalls.count == 1,
                "expected exactly one rejection when split at \(splitOffset)")
        }
    }

    @Test("A canonical call still executes at every chunk boundary")
    func canonicalCallExecutesAtEveryChunkBoundary() throws {
        let payload = """
            <tool_call>
            <function=weather>
            <parameter=city>
            Paris
            </parameter>
            </function>
            </tool_call>
            """

        for splitOffset in 0 ... payload.count {
            let split = payload.index(payload.startIndex, offsetBy: splitOffset)
            let processor = ToolCallProcessor(
                format: .xmlFunction,
                tools: toolSchemas("weather"))

            _ = processor.processChunk(String(payload[..<split]))
            _ = processor.processChunk(String(payload[split...]))
            processor.processEOS()

            #expect(
                processor.rejectedToolCalls.isEmpty,
                "rejected a canonical call when split at \(splitOffset)")
            #expect(
                processor.toolCalls.count == 1,
                "lost a canonical call when split at \(splitOffset)")
            let call = try #require(processor.toolCalls.first)
            #expect(call.function.name == "weather")
            #expect(
                call.function.arguments["city"] == .string("Paris"),
                "dropped arguments when split at \(splitOffset)")
        }
    }

    @Test("A literal </function> argument survives streaming at every chunk boundary")
    func literalFunctionCloseSurvivesStreaming() throws {
        let payload = """
            <tool_call>
            <function=weather>
            <parameter=city>
            literal </function> text
            </parameter>
            </function>
            </tool_call>
            """

        for splitOffset in 0 ... payload.count {
            let split = payload.index(payload.startIndex, offsetBy: splitOffset)
            let processor = ToolCallProcessor(
                format: .xmlFunction,
                tools: toolSchemas("weather"))

            _ = processor.processChunk(String(payload[..<split]))
            _ = processor.processChunk(String(payload[split...]))
            processor.processEOS()

            #expect(
                processor.toolCalls.count == 1,
                "lost the call when split at \(splitOffset)")
            let call = try #require(processor.toolCalls.first)
            #expect(
                call.function.arguments["city"] == .string("literal </function> text"),
                "corrupted the argument when split at \(splitOffset)")
        }
    }

    @Test("Text following a canonical call is still emitted")
    func trailingTextAfterCallIsPreserved() {
        let processor = ToolCallProcessor(
            format: .xmlFunction,
            tools: toolSchemas("weather"))

        var response = ""
        response +=
            processor.processChunk(
                "<tool_call><function=weather><parameter=city>Paris</parameter>"
                    + "</function></tool_call>done") ?? ""
        response += processor.processEOS(returnBufferedText: true) ?? ""

        #expect(processor.toolCalls.count == 1)
        #expect(processor.rejectedToolCalls.isEmpty)
        #expect(response.contains("done"))
    }

    @Test("Consecutive canonical calls are all executed")
    func consecutiveCallsAreExecuted() {
        let processor = ToolCallProcessor(
            format: .xmlFunction,
            tools: toolSchemas("weather", "clock"))

        _ = processor.processChunk(
            "<tool_call><function=weather><parameter=city>Paris</parameter></function></tool_call>"
                + "<tool_call><function=clock></function></tool_call>")
        processor.processEOS()

        #expect(processor.rejectedToolCalls.isEmpty)
        #expect(processor.toolCalls.map(\.function.name) == ["weather", "clock"])
    }

    // MARK: - Cross-parser parity

    @Test("Both Qwen XML dialect parsers agree on structural validity")
    func dialectParsersAgree() {
        let qwen35 = Qwen35ToolCallParser(startTag: "<tool_call>", endTag: "</tool_call>")
        let tools = toolSchemas("weather")

        let malformed = [
            "<tool_call><function=weather><parameter=city>Paris</function></tool_call>",
            "<tool_call><function=weather><parameter=city>Paris</parameter></tool_call>",
            "<tool_call><function=></function></tool_call>",
        ]
        for payload in malformed {
            #expect(parser.parse(content: payload, tools: tools) == nil, "xmlFunction: \(payload)")
            #expect(qwen35.parse(content: payload, tools: tools) == nil, "qwen35: \(payload)")
        }

        let canonical =
            "<tool_call><function=weather><parameter=city>Paris</parameter></function></tool_call>"
        #expect(parser.parse(content: canonical, tools: tools) != nil)
        #expect(qwen35.parse(content: canonical, tools: tools) != nil)
    }

    /// Before this fix the schema's `required` list was the only defense: the
    /// same truncated frame was executed as `weather()` without one and
    /// rejected with one. Both must now be rejected structurally.
    @Test("Rejection no longer depends on a declared required list")
    func rejectionIsIndependentOfRequiredList() {
        let withoutRequired: [[String: any Sendable]] = toolSchemas("weather")
        let withRequired: [[String: any Sendable]] = [
            [
                "function": [
                    "name": "weather",
                    "parameters": [
                        "type": "object",
                        "required": ["city"],
                    ] as [String: any Sendable],
                ] as [String: any Sendable]
            ]
        ]

        for tools in [withoutRequired, withRequired] {
            let processor = ToolCallProcessor(format: .xmlFunction, tools: tools)
            _ = processor.processChunk(
                "<tool_call><function=weather><parameter=city>Paris</function></tool_call>")
            processor.processEOS()

            #expect(processor.toolCalls.isEmpty)
            #expect(processor.rejectedToolCalls.count == 1)
        }
    }

    // MARK: - A literal </parameter> inside a value (goodmai/basalt#887)

    /// What one stream produced, in a form two chunkings can be compared by.
    private struct Parsed: Equatable {
        var text = ""
        var calls: [ToolCall.Function] = []
        var rejections: [RejectedToolCall.Reason] = []
    }

    private func parsed(
        _ chunks: [String], format: ToolCallFormat = .xmlFunction, tools: String...
    ) -> Parsed {
        let processor = ToolCallProcessor(
            format: format,
            tools: tools.map { ["function": ["name": $0] as [String: any Sendable]] })
        var outputs: [ToolCallProcessor.Output] = []
        for chunk in chunks { outputs += processor.processChunkOutputs(chunk) }
        outputs += processor.processEOSOutputs()
        var parsed = Parsed()
        for output in outputs {
            switch output {
            case .response(let text): parsed.text += text
            case .toolCall(let call): parsed.calls.append(call.function)
            case .rejectedToolCall(let rejection): parsed.rejections.append(rejection.reason)
            }
        }
        return parsed
    }

    /// One character at a time, fixed chunks of 2, 3, 5 and 7, and two chunks
    /// split at every offset.
    private func chunkings(of text: String) -> [[String]] {
        [1, 2, 3, 5, 7].map { size in
            stride(from: 0, to: text.count, by: size).map {
                String(text.dropFirst($0).prefix(size))
            }
        } + (0 ... text.count).map { [String(text.prefix($0)), String(text.dropFirst($0))] }
    }

    /// The form the Qwen3-Coder chat template writes: every tag on its own line.
    private func bashCall(_ command: String, newline: String = "\n") -> String {
        [
            "<tool_call>", "<function=Bash>", "<parameter=command>", command, "</parameter>",
            "</function>", "</tool_call>",
        ].joined(separator: newline)
    }

    @Test(
        "A value keeps a literal </parameter> that is not followed by <parameter= or </function>",
        arguments: [
            "echo '</parameter>' done",
            "first line\n</parameter>\nsecond line",
            "the value ends with </parameter>",
            "</parameter>",
            "</parameter></parameter>\n</parameter>",
            "a </function> b </tool_call> c </parameter> d <parameter=x> e",
        ])
    func literalParameterCloseIsPreserved(command: String) {
        let expected = Parsed(calls: [.init(name: "Bash", arguments: ["command": command])])

        for newline in ["\n", "\r\n"] {
            let payload = bashCall(command, newline: newline)
            #expect(
                parser.parse(content: payload, tools: nil)?.function == expected.calls[0],
                "newline: \(newline.debugDescription)")
            for chunks in chunkings(of: payload) {
                #expect(
                    parsed(chunks, tools: "Bash") == expected,
                    "newline: \(newline.debugDescription), chunks: \(chunks.prefix(2))")
            }
        }
    }

    @Test("A single-line value keeps a literal </parameter> and still closes without a newline")
    func literalParameterCloseInSingleLineValues() {
        let payload =
            "<tool_call><function=Bash><parameter=command>echo '</parameter>' done</parameter>"
            + "<parameter=cwd>/tmp</parameter></function></tool_call>"
        let expected = Parsed(calls: [
            .init(name: "Bash", arguments: ["command": "echo '</parameter>' done", "cwd": "/tmp"])
        ])

        #expect(parser.parse(content: payload, tools: nil)?.function == expected.calls[0])
        for chunks in chunkings(of: payload) {
            #expect(parsed(chunks, tools: "Bash") == expected, "chunks: \(chunks.prefix(2))")
        }
    }

    /// Values are not escaped, so this text has two readings. The earliest
    /// legal close wins: the alternative, the latest one, would merge every
    /// parameter of an ordinary multi-parameter call into the first.
    @Test("A value holding </parameter> and then <parameter=…> splits at the earliest legal close")
    func literalCloseFollowedByParameterOpenSplitsTheValue() {
        let payload = bashCall("a\n</parameter>\n<parameter=x>\nb")
        let expected = Parsed(calls: [.init(name: "Bash", arguments: ["command": "a", "x": "b"])])

        for chunks in chunkings(of: payload) {
            #expect(parsed(chunks, tools: "Bash") == expected, "chunks: \(chunks.prefix(2))")
        }
    }

    @Test("A stream that ends inside or right after a value with a literal </parameter> terminates")
    func truncatedLiteralParameterCloseTerminates() {
        let payload = bashCall("echo '</parameter>' done")
        let cuts = ["' done", "' done\n</parameter>", "' done\n</parameter>\n</func"]

        for cut in cuts {
            let prefix = String(payload[..<payload.range(of: cut)!.upperBound])
            for chunks in chunkings(of: prefix) {
                #expect(
                    parsed(chunks, tools: "Bash") == Parsed(rejections: [.incompleteOutput]),
                    "cut after \(cut.debugDescription), chunks: \(chunks.prefix(2))")
            }
        }
    }

    @Test("The recovered bare <function= dialect keeps a literal </parameter> at every boundary")
    func recoveredDialectKeepsLiteralParameterClose() {
        let payload = "<function=Bash><parameter=command>a</parameter>b</parameter></function>"
        let expected = Parsed(
            calls: [.init(name: "Bash", arguments: ["command": "a</parameter>b"])])

        for chunks in chunkings(of: payload) {
            #expect(
                parsed(chunks, format: .json, tools: "Bash") == expected,
                "chunks: \(chunks.prefix(2))")
        }
    }
}
