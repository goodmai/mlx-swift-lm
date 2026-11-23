// Copyright © 2025 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import XCTest

public class BaseConfigurationTests: XCTestCase {

    func testQuantization() throws {
        let json =
            """
            {
                "model_type": "Test",
                "quantization": {
                    "group_size": 128,
                    "bits": 4
                }
            }
            """

        let config = try JSONDecoder().decode(
            BaseConfiguration.self, from: json.data(using: .utf8)!)

        XCTAssertEqual(
            config.perLayerQuantization?.quantization(layer: "x"), .init(groupSize: 128, bits: 4))
    }

    func testHeterogenousQuantization() throws {
        // from https://huggingface.co/mlx-community/Qwen3-1.7B-4bit-AWQ/blob/main/config.json#L20
        let json =
            """
            {
                "model_type": "Test",
                "quantization": {
                    "group_size": 64,
                    "bits": 4,
                    "model.embed_tokens": {
                        "group_size": 32,
                        "bits": 4
                    },
                    "model.layers.0.self_attn.q_norm": false,
                    "true_layer": true
                }
            }
            """

        let config = try JSONDecoder().decode(
            BaseConfiguration.self, from: json.data(using: .utf8)!)

        // a random layer -- no specific configuration gets default
        XCTAssertEqual(
            config.perLayerQuantization?.quantization(layer: "x"),
            .init(groupSize: 64, bits: 4))

        // layer with an override
        XCTAssertEqual(
            config.perLayerQuantization?.quantization(layer: "model.embed_tokens"),
            .init(groupSize: 32, bits: 4))

        // layer with an override -- not quant
        XCTAssertNil(
            config.perLayerQuantization?.quantization(layer: "model.layers.0.self_attn.q_norm"))

        // layer with an override -- true, use the default
        XCTAssertEqual(
            config.perLayerQuantization?.quantization(layer: "true_layer"),
            .init(groupSize: 64, bits: 4))
    }

    func testNestedTextConfigurationEOSTokenIds() throws {
        let json =
            """
            {
                "model_type": "qwen3_5",
                "text_config": {
                    "eos_token_id": 248044
                }
            }
            """

        let config = try JSONDecoder().decode(
            BaseConfiguration.self, from: json.data(using: .utf8)!)

        XCTAssertNil(config.eosTokenIds)
        XCTAssertEqual(config.effectiveEOSTokenIds, [248044])
    }

    func testRootEOSTokenIdsTakePrecedenceOverNestedValues() throws {
        let json =
            """
            {
                "model_type": "qwen3_5",
                "eos_token_id": 248046,
                "text_config": {
                    "eos_token_id": [248044, 248046]
                }
            }
            """

        let config = try JSONDecoder().decode(
            BaseConfiguration.self, from: json.data(using: .utf8)!)

        XCTAssertEqual(config.eosTokenIds?.values, [248046])
        XCTAssertEqual(config.effectiveEOSTokenIds, [248046])
    }

    func testScalarArrayNullAndNonLayerDictMetadataAreIgnored() throws {
        // Mirrors real configs like hermitdave/LFM2.5-24B-A2B-MLX-DWQ-3bit
        let json =
            """
            {
                "model_type": "lfm2",
                "quantization": {
                    "group_size": 64,
                    "bits": 3,
                    "method": "dwq",
                    "calibration": "allenai/tulu-3-sft-mixture",
                    "quantization_backend": "mx.quantize",
                    "tied_embedding": "fp16_passthrough",
                    "mtp": "none",
                    "recipe": "experts_only",
                    "engram_bits": 4,
                    "num_samples": 1024,
                    "quantized_components": ["mlp", "self_attn"],
                    "targets": ["layers"],
                    "modules": {"note": "not a layer override"},
                    "nullable_meta": null,
                    "model.embed_tokens": {"bits": 6, "group_size": 64},
                    "model.layers.0.norm": false
                }
            }
            """

        let config = try JSONDecoder().decode(
            BaseConfiguration.self, from: json.data(using: .utf8)!)
        let perLayer = try XCTUnwrap(config.perLayerQuantization)

        XCTAssertEqual(perLayer.quantization(layer: "model.layers.1.mlp"), .init(groupSize: 64, bits: 3))
        XCTAssertEqual(perLayer.quantization(layer: "model.embed_tokens"), .init(groupSize: 64, bits: 6))
        XCTAssertNil(perLayer.quantization(layer: "model.layers.0.norm"))
        XCTAssertEqual(
            perLayer.ignoredKeys,
            [
                "calibration",
                "engram_bits",
                "method",
                "modules",
                "mtp",
                "nullable_meta",
                "num_samples",
                "quantization_backend",
                "quantized_components",
                "recipe",
                "targets",
                "tied_embedding",
            ]
        )
    }

    func testDefaultGroupSizeByMode() throws {
        // Mirrors OsaurusAI/gemma-4-12B-it-qat-MXFP4 and nvfp4/affine omitted group_size
        let cases: [(String, Int, QuantizationMode, Int)] = [
            (#"{"model_type":"gemma4","quantization":{"bits":4,"mode":"mxfp4","model.head":{"bits":8,"mode":"mxfp8"}}}"#, 4, .mxfp4, 32),
            (#"{"model_type":"gemma4","quantization":{"bits":8,"mode":"mxfp8"}}"#, 8, .mxfp8, 32),
            (#"{"model_type":"qwen3","quantization":{"bits":4,"mode":"nvfp4"}}"#, 4, .nvfp4, 16),
            (#"{"model_type":"llama","quantization":{"bits":4}}"#, 4, .affine, 64),
        ]

        for (json, bits, mode, expectedGroupSize) in cases {
            let config = try JSONDecoder().decode(
                BaseConfiguration.self, from: json.data(using: .utf8)!)
            let q = try XCTUnwrap(config.perLayerQuantization?.quantization(layer: "other"))
            XCTAssertEqual(q, .init(groupSize: expectedGroupSize, bits: bits, mode: mode))
        }

        let mxfp = try JSONDecoder().decode(
            BaseConfiguration.self, from: cases[0].0.data(using: .utf8)!)
        XCTAssertEqual(
            mxfp.perLayerQuantization?.quantization(layer: "model.head"),
            .init(groupSize: 32, bits: 8, mode: .mxfp8))
    }

    func testUnknownQuantizationModeFailsNamingMode() {
        for badMode in ["turboquant", "jangtq2", "fp8_e4m3"] {
            let json =
                """
                {
                    "model_type": "qwen3",
                    "quantization": {
                        "bits": 4,
                        "mode": "\(badMode)"
                    }
                }
                """
            XCTAssertThrowsError(
                try JSONDecoder().decode(BaseConfiguration.self, from: json.data(using: .utf8)!)
            ) { error in
                XCTAssertTrue(
                    String(describing: error).contains(badMode),
                    "Expected error to name mode '\(badMode)', got: \(error)")
            }
        }
    }

    func testUnknownModeInPerLayerOverrideFailsNamingMode() {
        let json =
            """
            {
                "model_type": "qwen3",
                "quantization": {
                    "bits": 4,
                    "group_size": 64,
                    "model.layers.0.mlp": {"bits": 4, "mode": "turboquant"}
                }
            }
            """
        XCTAssertThrowsError(
            try JSONDecoder().decode(BaseConfiguration.self, from: json.data(using: .utf8)!)
        ) { error in
            XCTAssertTrue(String(describing: error).contains("turboquant"))
        }
    }

    func testQuantizationConfigFallbackForMlxOnly() throws {
        // Mirrors inferencerlabs/gemma-4-31B-MLX-Q9 and rapid-mlx/Ling-3.0-tiny-MLX-4bit
        let implicitMlx =
            """
            {
                "model_type": "bailing_moe",
                "quantization_config": {
                    "bits": 4,
                    "group_size": 64,
                    "model.embed_tokens": {"bits": 8, "group_size": 64}
                }
            }
            """
        let decodedImplicit = try JSONDecoder().decode(
            BaseConfiguration.self, from: implicitMlx.data(using: .utf8)!)
        XCTAssertEqual(
            decodedImplicit.perLayerQuantization?.quantization(layer: "model.layers.0.mlp"),
            .init(groupSize: 64, bits: 4))
        XCTAssertEqual(
            decodedImplicit.perLayerQuantization?.quantization(layer: "model.embed_tokens"),
            .init(groupSize: 64, bits: 8))

        let explicitMlx =
            """
            {
                "model_type": "gemma4",
                "quantization_config": {
                    "quant_method": "mlx",
                    "bits": 4,
                    "mode": "mxfp4"
                }
            }
            """
        let decodedExplicit = try JSONDecoder().decode(
            BaseConfiguration.self, from: explicitMlx.data(using: .utf8)!)
        XCTAssertEqual(
            decodedExplicit.perLayerQuantization?.quantization(layer: "x"),
            .init(groupSize: 32, bits: 4, mode: .mxfp4))
    }

    func testForeignQuantMethodNeverFallsBack() throws {
        for method in [
            "awq", "gptq", "modelopt", "bitsandbytes", "fp8",
            "compressed-tensors", "mxfp4", "gemma", "paroquant",
        ] {
            let json =
                """
                {
                    "model_type": "qwen3",
                    "quantization_config": {
                        "quant_method": "\(method)",
                        "bits": 4,
                        "group_size": 128
                    }
                }
                """
            let config = try JSONDecoder().decode(
                BaseConfiguration.self, from: json.data(using: .utf8)!)
            XCTAssertNil(
                config.perLayerQuantization,
                "Foreign quant_method '\(method)' must not fall back to MLX quantization")
        }
    }

}
