// Copyright © 2025 Apple Inc.

import Foundation
import MLX

private let baseConfigLogger = Logger(subsystem: "mlx-swift-lm", category: "BaseConfiguration")

/// The fundamental configuration for any MLX-based model.
///
/// `BaseConfiguration` provides the metadata necessary to identify the model architecture
/// (`modelType`) and describes the quantization parameters used to compress the model's weights.
/// It is designed to be decoded directly from a model repository's `config.json`.
///
/// Typically used the ``GenericModelFactory`` implementations during load.
public struct BaseConfiguration: Codable, Sendable {

    /// The architecture identifier (e.g., "bert", "roberta", "xlm-roberta").
    public let modelType: String

    /// Configuration parameters for weight quantization.
    ///
    /// MLX uses group-wise quantization to reduce memory footprint. This struct
    /// defines how weights are grouped and the precision (bits) used for each group.
    public struct Quantization: Codable, Sendable, Equatable {

        /// Initializes a new quantization configuration.
        /// - Parameters:
        ///   - groupSize: The number of weights that share the same scale and bias.
        ///   - bits: The bit-depth of the quantized weights (e.g., 4 or 8).
        ///   - mode: The quantization mode (defaults to `.affine`).
        public init(groupSize: Int, bits: Int, mode: QuantizationMode = .affine) {
            self.groupSize = groupSize
            self.bits = bits
            self._mode = mode == .affine ? nil : mode
        }

        /// Default group size for a quantization mode when `group_size` is omitted from `config.json`.
        public static func defaultGroupSize(for mode: QuantizationMode) -> Int {
            switch mode {
            case .affine: 64
            case .mxfp4, .mxfp8: 32
            case .nvfp4: 16
            }
        }

        public init(from decoder: any Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            self.bits = try container.decode(Int.self, forKey: .bits)
            if let rawMode = try container.decodeIfPresent(String.self, forKey: ._mode) {
                guard let parsed = QuantizationMode(rawValue: rawMode) else {
                    throw DecodingError.dataCorruptedError(
                        forKey: ._mode, in: container,
                        debugDescription: "Unsupported quantization mode '\(rawMode)'")
                }
                self._mode = parsed
            } else {
                self._mode = nil
            }
            self.groupSize =
                try container.decodeIfPresent(Int.self, forKey: .groupSize)
                ?? Self.defaultGroupSize(for: self._mode ?? .affine)
        }

        public static func == (lhs: Quantization, rhs: Quantization) -> Bool {
            lhs.groupSize == rhs.groupSize && lhs.bits == rhs.bits && lhs.mode == rhs.mode
        }

        /// The size of the quantization group.
        public let groupSize: Int

        /// The number of bits per weight.
        public let bits: Int

        /// Internal storage for the quantization mode.
        private var _mode: QuantizationMode? = nil

        /// The quantization method to use (defaults to `.affine`).
        ///
        /// Affine quantization (asymmetric) uses both a scale and a zero-point
        /// to map floating point values to integers.
        public var mode: QuantizationMode { _mode ?? .affine }

        /// Converts the configuration into a tuple format compatible with `MLX.quantize`.
        public var asTuple: (Int, Int, QuantizationMode) { (groupSize, bits, mode) }

        enum CodingKeys: String, CodingKey {
            case groupSize = "group_size"
            case bits = "bits"
            case _mode = "mode"
        }
    }

    /// handling instructions for ``PerLayerQuantization``
    public enum QuantizationOption: Sendable {
        /// Do not quantize this specific layer (keep it in high precision).
        case skip
        /// Quantize this layer using the provided parameters.
        case quantize(Quantization)
    }

    /// A container for per-layer ``Quantization`` settings.
    ///
    /// This allows for "Mixed-Precision" or "Heterogeneous" quantization, where
    /// sensitive layers (like the embedding head) can be kept at higher precision
    /// while the rest of the model is compressed.
    public struct PerLayerQuantization: Sendable {
        /// The default quantization for any layer not explicitly named in `perLayerQuantization`.
        public var quantization: Quantization? = nil

        /// A dictionary mapping layer paths (e.g., "model.embed_tokens") to their quantization options.
        public var perLayerQuantization: [String: QuantizationOption]

        /// Sorted list of keys in the quantization block that were not valid per-layer instructions
        /// and were ignored during decoding.
        public var ignoredKeys: [String] = []

        public init(
            quantization: BaseConfiguration.Quantization? = nil,
            perLayerQuantization: [String: BaseConfiguration.QuantizationOption],
            ignoredKeys: [String] = []
        ) {
            self.quantization = quantization
            self.perLayerQuantization = perLayerQuantization
            self.ignoredKeys = ignoredKeys
        }

        /// Resolves the quantization parameters for a specific layer.
        /// - Parameter layer: The path/name of the layer.
        /// - Returns: The `Quantization` settings to apply, or `nil` if the layer should be skipped.
        public func quantization(layer: String) -> Quantization? {
            if let perLayer = perLayerQuantization[layer] {
                switch perLayer {
                case .skip:
                    return nil
                case .quantize(let quantization):
                    return quantization
                }
            } else {
                return quantization
            }
        }
    }

    /// An internal container designed to handle the mixed JSON structure found in `config.json`.
    ///
    /// ```
    /// "quantization": {
    ///     "group_size": 64,
    ///     "bits": 4,
    ///     "model.embed_tokens": {
    ///         "group_size": 32,
    ///         "bits": 4
    ///     },
    ///     "model.layers.0.self_attn.q_norm": false,
    /// ```
    ///
    /// Quantization configs in MLX often interleave global keys (like `bits`) with
    /// specific layer keys (like `model.layers.0...`). This container uses manual
    /// decoding to separate these interleaved values.
    struct QuantizationContainer: Codable, Sendable {
        var quantization: Quantization
        var perLayerQuantization: PerLayerQuantization

        /// A custom CodingKey used to iterate over arbitrary layer names in JSON.
        internal struct _DictionaryCodingKey: CodingKey {
            internal let stringValue: String
            internal let intValue: Int?

            internal init(stringValue: String) {
                self.stringValue = stringValue
                self.intValue = Int(stringValue)
            }

            internal init(intValue: Int) {
                self.stringValue = "\(intValue)"
                self.intValue = intValue
            }
        }

        init(from decoder: any Decoder) throws {
            // handle the embedded Quantization
            self.quantization = try Quantization(from: decoder)

            // and the interleaved per-layer values
            var perLayerQuantization = [String: QuantizationOption]()
            var ignoredKeys = [String]()
            let container = try decoder.container(keyedBy: _DictionaryCodingKey.self)
            let bitsKey = _DictionaryCodingKey(stringValue: Quantization.CodingKeys.bits.rawValue)
            for key in container.allKeys {
                switch key.stringValue {
                case Quantization.CodingKeys.groupSize.rawValue: continue
                case Quantization.CodingKeys.bits.rawValue: continue
                case Quantization.CodingKeys._mode.rawValue: continue

                // additional keys that are not layer instructions, see
                // mlx-community/bitnet-b1.58-2B-4T-4bit
                case "quant_method", "linear_class", "quantization_mode": continue

                default:
                    // A per-layer instruction is a Bool or a dict containing "bits".
                    // Every other value (string, number, array, null, dict without "bits") is ignored.
                    if let f = try? container.decode(Bool.self, forKey: key) {
                        if !f {
                            perLayerQuantization[key.stringValue] = .skip
                        }
                    } else if let nested = try? container.nestedContainer(
                        keyedBy: _DictionaryCodingKey.self, forKey: key),
                        nested.contains(bitsKey)
                    {
                        perLayerQuantization[key.stringValue] = .quantize(
                            try container.decode(Quantization.self, forKey: key))
                    } else {
                        ignoredKeys.append(key.stringValue)
                    }
                }
            }
            ignoredKeys.sort()
            if !ignoredKeys.isEmpty {
                let joined = ignoredKeys.joined(separator: ", ")
                baseConfigLogger.info("config.json quantization keys ignored: \(joined)")
            }
            self.perLayerQuantization = PerLayerQuantization(
                quantization: quantization,
                perLayerQuantization: perLayerQuantization,
                ignoredKeys: ignoredKeys)
        }

        func encode(to encoder: any Encoder) throws {
            try quantization.encode(to: encoder)

            var container = encoder.container(keyedBy: _DictionaryCodingKey.self)
            for (key, value) in perLayerQuantization.perLayerQuantization {
                switch value {
                case .skip:
                    try container.encode(false, forKey: .init(stringValue: key))
                case .quantize(let q):
                    try container.encode(q, forKey: .init(stringValue: key))
                }
            }
        }
    }

    private struct TextConfiguration: Codable, Sendable {
        var eosTokenIds: IntOrIntArray?

        enum CodingKeys: String, CodingKey {
            case eosTokenIds = "eos_token_id"
        }
    }

    /// Internal storage for quantization details extracted from `config.json`.
    var quantizationContainer: QuantizationContainer?

    /// Text-model metadata nested by composite model configurations.
    private var textConfiguration: TextConfiguration?

    /// EOS token IDs from config.json. Can be a single Int or an array of Ints.
    public var eosTokenIds: IntOrIntArray?

    /// EOS token IDs declared at either the model root or in `text_config`.
    public var effectiveEOSTokenIds: Set<Int> {
        Set(eosTokenIds?.values ?? textConfiguration?.eosTokenIds?.values ?? [])
    }

    /// The default quantization settings.
    @available(*, deprecated, message: "Please use perLayerQuantization instead")
    public var quantization: Quantization? {
        quantizationContainer?.quantization
    }

    /// The per-layer quantization settings, including the default fallback.
    public var perLayerQuantization: PerLayerQuantization? {
        quantizationContainer?.perLayerQuantization
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case quantizationContainer = "quantization"
        case quantizationConfig = "quantization_config"
        case textConfiguration = "text_config"
        case eosTokenIds = "eos_token_id"
    }

    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.modelType = try container.decode(String.self, forKey: .modelType)
        self.textConfiguration = try container.decodeIfPresent(
            TextConfiguration.self, forKey: .textConfiguration)
        self.eosTokenIds = try container.decodeIfPresent(IntOrIntArray.self, forKey: .eosTokenIds)

        if container.contains(.quantizationContainer),
            !(try container.decodeNil(forKey: .quantizationContainer))
        {
            self.quantizationContainer = try container.decode(
                QuantizationContainer.self, forKey: .quantizationContainer)
        } else if container.contains(.quantizationConfig),
            !(try container.decodeNil(forKey: .quantizationConfig)),
            let nested = try? container.nestedContainer(
                keyedBy: QuantizationContainer._DictionaryCodingKey.self,
                forKey: .quantizationConfig)
        {
            let quantMethod = try? nested.decodeIfPresent(
                String.self, forKey: .init(stringValue: "quant_method"))
            let hasBits = nested.contains(
                .init(stringValue: Quantization.CodingKeys.bits.rawValue))
            if (quantMethod == nil || quantMethod == "mlx"), hasBits {
                self.quantizationContainer = try container.decode(
                    QuantizationContainer.self, forKey: .quantizationConfig)
            } else {
                self.quantizationContainer = nil
            }
        } else {
            self.quantizationContainer = nil
        }
    }

    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(modelType, forKey: .modelType)
        try container.encodeIfPresent(quantizationContainer, forKey: .quantizationContainer)
        try container.encodeIfPresent(textConfiguration, forKey: .textConfiguration)
        try container.encodeIfPresent(eosTokenIds, forKey: .eosTokenIds)
    }
}
