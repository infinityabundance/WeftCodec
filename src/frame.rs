//! WFT1 frame layout (WEFTCODEC-FORMAT §2–§3, §5–§6).
//!
//! ```text
//! header (96) || dependency table (48/entry) || chunk table (48/entry) || body || trailer (4)
//! ```
//!
//! All lengths are checked with checked arithmetic. Parsing never panics on
//! untrusted bytes; every structural violation returns a typed error.

use crate::error::{Error, ErrorClass, Result};
use crate::hash::{Hash32, HASH_LEN};
use crate::limits::{check_len, Limits};

/// Frame magic.
pub const MAGIC: [u8; 4] = *b"WFT1";
/// Fixed header length.
pub const HEADER_LEN: usize = 96;
/// Trailer length.
pub const TRAILER_LEN: usize = 4;
/// Dependency table entry length.
pub const DEP_ENTRY_LEN: usize = 48;
/// Chunk table entry length.
pub const CHUNK_ENTRY_LEN: usize = 48;
/// Supported format major.
pub const FORMAT_MAJOR: u8 = 1;
/// Supported format minor.
pub const FORMAT_MINOR: u8 = 0;

/// Payload domain (header `domain = 0`).
pub const DOMAIN_PAYLOAD: u8 = 0;
/// Structured domain (header `domain = 1`).
pub const DOMAIN_STRUCTURED: u8 = 1;

/// Dependency domain: payload bytes.
pub const DEP_DOMAIN_PAYLOAD: u8 = 0;
/// Dependency domain: structured bytes.
pub const DEP_DOMAIN_STRUCTURED: u8 = 1;
/// Dependency domain: raw zstd dictionary bytes.
pub const DEP_DOMAIN_DICTIONARY: u8 = 2;

/// Dependency flag: dependency is required for decoding.
pub const DEP_FLAG_REQUIRED: u8 = 1 << 0;

/// Header `flags` bits.
pub mod flags {
    /// Chunk table present; body is concatenated chunk encodings.
    pub const CHUNKED: u16 = 1 << 0;
    /// Reserved: dependency bytes appended after the trailer.
    pub const SELF_CONTAINED: u16 = 1 << 1;
    /// Dependency 0 is a raw-content zstd dictionary.
    pub const DICT_RAW: u16 = 1 << 2;
    /// Reserved in v1; MUST be zero.
    pub const ORIGINAL_HASH: u16 = 1 << 3;
}

/// Header `features` bits.
pub mod features {
    /// Dependency table may be non-empty.
    pub const DEPENDENCIES: u16 = 1 << 0;
    /// Per-chunk hashes are present (implies CHUNKED).
    pub const CHUNK_HASHES: u16 = 1 << 1;
}

/// Payload modes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u8)]
pub enum Mode {
    Raw = 0,
    Zstd = 1,
    Reference = 2,
    Delta = 3,
    Dictionary = 4,
    Structured = 5,
    /// Canonical tensor payload with WCB descriptor (WEFT-AI-CODEC §2).
    Tensor = 6,
    /// Token-id sequence with WCB descriptor (WEFT-AI-CODEC §3).
    TokenSequence = 7,
    /// Affine quantized tensor (i8/u8) with WCB descriptor (WEFT-AI-CODEC §4).
    QuantizedTensor = 8,
    /// Split float tensor: zstd exponent plane plus raw sign+mantissa plane
    /// with WCB descriptor (WEFT-AI-CODEC §5).
    FloatTensor = 9,
    /// Sharded quantized tensor shard with WCB descriptor (WEFT-AI-CODEC §11).
    TensorSharded = 10,
    /// Exact word-wise tensor delta against a base frame (WEFT-AI-CODEC §12).
    WeightDelta = 11,
    /// One paged attention KV cache page with WCB descriptor
    /// (WEFT-AI-CODEC §13).
    KvCache = 12,
    /// One activation/gradient stream chunk with WCB descriptor
    /// (WEFT-AI-CODEC §14).
    ActivationStream = 13,
}

impl Mode {
    /// Parse the one-byte mode value.
    pub fn from_u8(value: u8) -> Result<Self> {
        match value {
            0 => Ok(Mode::Raw),
            1 => Ok(Mode::Zstd),
            2 => Ok(Mode::Reference),
            3 => Ok(Mode::Delta),
            4 => Ok(Mode::Dictionary),
            5 => Ok(Mode::Structured),
            6 => Ok(Mode::Tensor),
            7 => Ok(Mode::TokenSequence),
            8 => Ok(Mode::QuantizedTensor),
            9 => Ok(Mode::FloatTensor),
            10 => Ok(Mode::TensorSharded),
            11 => Ok(Mode::WeightDelta),
            12 => Ok(Mode::KvCache),
            13 => Ok(Mode::ActivationStream),
            14..=127 => Err(Error::unsupported_mode(format!(
                "reserved mode {value} is not defined in v1"
            ))),
            128..=255 => Err(Error::unsupported_mode(format!(
                "experimental mode {value} must not be persisted"
            ))),
        }
    }

    /// Numeric wire value.
    pub fn as_u8(self) -> u8 {
        self as u8
    }

    /// Lowercase CLI/JSON name.
    pub fn as_str(self) -> &'static str {
        match self {
            Mode::Raw => "raw",
            Mode::Zstd => "zstd",
            Mode::Reference => "reference",
            Mode::Delta => "delta",
            Mode::Dictionary => "dictionary",
            Mode::Structured => "structured",
            Mode::Tensor => "tensor",
            Mode::TokenSequence => "tokens",
            Mode::QuantizedTensor => "quantized_tensor",
            Mode::FloatTensor => "float_tensor",
            Mode::TensorSharded => "tensor_sharded",
            Mode::WeightDelta => "weight_delta",
            Mode::KvCache => "kv_cache",
            Mode::ActivationStream => "activation_stream",
        }
    }

    /// Parse a CLI name (case-insensitive).
    pub fn from_name(name: &str) -> Result<Self> {
        match name.to_ascii_lowercase().as_str() {
            "raw" => Ok(Mode::Raw),
            "zstd" => Ok(Mode::Zstd),
            "reference" => Ok(Mode::Reference),
            "delta" => Ok(Mode::Delta),
            "dictionary" => Ok(Mode::Dictionary),
            "structured" => Ok(Mode::Structured),
            "tensor" => Ok(Mode::Tensor),
            "tokens" | "token_sequence" => Ok(Mode::TokenSequence),
            "quantized" | "quantized_tensor" => Ok(Mode::QuantizedTensor),
            "float_tensor" | "float-tensor" => Ok(Mode::FloatTensor),
            "tensor_sharded" | "tensor-sharded" | "sharded" => Ok(Mode::TensorSharded),
            "weight_delta" | "weight-delta" | "weightdelta" => Ok(Mode::WeightDelta),
            "kv_cache" | "kv-cache" | "kvcache" => Ok(Mode::KvCache),
            "activation_stream" | "activation-stream" | "activation" => Ok(Mode::ActivationStream),
            other => Err(Error::unsupported_mode(format!("unknown mode `{other}`"))),
        }
    }
}

/// Parsed dependency table entry (48 bytes on the wire).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Dependency {
    /// `H("weft/chunk/v1", dependency plaintext)`.
    pub hash: Hash32,
    /// Dependency plaintext length.
    pub decoded_len: u64,
    /// Dependency domain (`0` payload, `1` structured, `2` dictionary bytes).
    pub dep_domain: u8,
    /// Dependency flags; bit 0 (`REQUIRED`) must be set in v1.
    pub dep_flags: u8,
    /// Informational mode of the dependency frame.
    pub codec_hint: u32,
}

impl Dependency {
    /// Build a v1 dependency entry with `REQUIRED` set.
    pub fn new(hash: Hash32, decoded_len: u64, dep_domain: u8, codec_hint: u32) -> Self {
        Dependency {
            hash,
            decoded_len,
            dep_domain,
            dep_flags: DEP_FLAG_REQUIRED,
            codec_hint,
        }
    }
}

/// Parsed chunk table entry (48 bytes on the wire).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChunkEntry {
    /// Offset of the encoded chunk within the body.
    pub encoded_off: u64,
    /// Encoded chunk length.
    pub encoded_len: u32,
    /// Decoded chunk length.
    pub decoded_len: u32,
    /// `H("weft/chunk/v1", chunk plaintext)`.
    pub hash: Hash32,
}

/// Parsed frame header.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FrameHeader {
    pub format_major: u8,
    pub format_minor: u8,
    /// `0` payload, `1` structured.
    pub domain: u8,
    pub mode: Mode,
    pub flags: u16,
    pub level: u16,
    pub decoded_len: u64,
    pub body_len: u64,
    pub dep_count: u32,
    pub chunk_count: u32,
    pub features: u16,
    pub content_hash: Hash32,
    pub table_len: u64,
}

impl FrameHeader {
    /// Names of set flags.
    pub fn flag_names(&self) -> Vec<&'static str> {
        let mut names = Vec::new();
        if self.flags & flags::CHUNKED != 0 {
            names.push("CHUNKED");
        }
        if self.flags & flags::SELF_CONTAINED != 0 {
            names.push("SELF_CONTAINED");
        }
        if self.flags & flags::DICT_RAW != 0 {
            names.push("DICT_RAW");
        }
        if self.flags & flags::ORIGINAL_HASH != 0 {
            names.push("ORIGINAL_HASH");
        }
        names
    }

    /// Names of set features.
    pub fn feature_names(&self) -> Vec<&'static str> {
        let mut names = Vec::new();
        if self.features & features::DEPENDENCIES != 0 {
            names.push("DEPENDENCIES");
        }
        if self.features & features::CHUNK_HASHES != 0 {
            names.push("CHUNK_HASHES");
        }
        names
    }

    /// Whether the CHUNKED flag is set.
    pub fn is_chunked(&self) -> bool {
        self.flags & flags::CHUNKED != 0
    }
}

/// A parsed frame borrowing the input bytes.
#[derive(Clone, Debug)]
pub struct Frame<'a> {
    pub header: FrameHeader,
    pub dependencies: Vec<Dependency>,
    pub chunks: Vec<ChunkEntry>,
    pub body: &'a [u8],
    /// The complete frame bytes (header through trailer).
    pub bytes: &'a [u8],
}

/// Encode a frame from its parts.
///
/// Features are derived from the parts: `DEPENDENCIES` when the dependency
/// table is non-empty, `CHUNK_HASHES` when the chunk table is non-empty. The
/// resulting bytes are re-parsed with `limits` before being returned, so an
/// invalid combination of mode/flags/tables is rejected here as well as on
/// decode.
pub fn encode_frame(parts: &FrameParts<'_>, limits: &Limits) -> Result<Vec<u8>> {
    let dep_count = u32::try_from(parts.dependencies.len())
        .map_err(|_| Error::limit_exceeded("dependency count does not fit u32"))?;
    let chunk_count = u32::try_from(parts.chunks.len())
        .map_err(|_| Error::limit_exceeded("chunk count does not fit u32"))?;
    let table_len = (u64::from(dep_count))
        .checked_mul(DEP_ENTRY_LEN as u64)
        .and_then(|v| {
            (u64::from(chunk_count))
                .checked_mul(CHUNK_ENTRY_LEN as u64)
                .and_then(|c| v.checked_add(c))
        })
        .ok_or_else(|| Error::limit_exceeded("table length overflow"))?;
    let body_len = parts.body.len() as u64;

    let mut features: u16 = 0;
    if !parts.dependencies.is_empty() {
        features |= features::DEPENDENCIES;
    }
    if !parts.chunks.is_empty() {
        features |= features::CHUNK_HASHES;
    }

    let total = HEADER_LEN as u64;
    let total = total
        .checked_add(table_len)
        .and_then(|v| v.checked_add(body_len))
        .and_then(|v| v.checked_add(TRAILER_LEN as u64))
        .ok_or_else(|| Error::limit_exceeded("frame length overflow"))?;
    check_len(total, limits.max_frame_bytes, "frame")?;
    check_len(parts.decoded_len, limits.max_decoded_bytes, "decoded data")?;

    let mut header = [0u8; HEADER_LEN];
    header[0..4].copy_from_slice(&MAGIC);
    header[4] = parts.format_major;
    header[5] = parts.format_minor;
    header[6] = parts.domain;
    header[7] = parts.mode.as_u8();
    header[8..10].copy_from_slice(&parts.flags.to_le_bytes());
    header[10..12].copy_from_slice(&parts.level.to_le_bytes());
    header[12..20].copy_from_slice(&parts.decoded_len.to_le_bytes());
    header[20..28].copy_from_slice(&body_len.to_le_bytes());
    header[28..32].copy_from_slice(&dep_count.to_le_bytes());
    header[32..36].copy_from_slice(&chunk_count.to_le_bytes());
    header[36..38].copy_from_slice(&features.to_le_bytes());
    // reserved0 38..40 zero
    header[40..72].copy_from_slice(parts.content_hash.as_bytes());
    // header CRC 72..76 zero, filled below
    header[76..84].copy_from_slice(&table_len.to_le_bytes());
    // reserved1 84..88 and reserved2 88..96 zero

    let mut crc_input = Vec::with_capacity(92);
    crc_input.extend_from_slice(&header[0..72]);
    crc_input.extend_from_slice(&header[76..96]);
    let header_crc = crc32c::crc32c(&crc_input);
    header[72..76].copy_from_slice(&header_crc.to_le_bytes());

    let capacity = usize::try_from(total)
        .map_err(|_| Error::limit_exceeded("frame length does not fit usize"))?;
    let mut out = Vec::with_capacity(capacity);
    out.extend_from_slice(&header);
    for dep in parts.dependencies {
        out.extend_from_slice(dep.hash.as_bytes());
        out.extend_from_slice(&dep.decoded_len.to_le_bytes());
        out.push(dep.dep_domain);
        out.push(dep.dep_flags);
        out.extend_from_slice(&[0u8, 0u8]);
        out.extend_from_slice(&dep.codec_hint.to_le_bytes());
    }
    for chunk in parts.chunks {
        out.extend_from_slice(&chunk.encoded_off.to_le_bytes());
        out.extend_from_slice(&chunk.encoded_len.to_le_bytes());
        out.extend_from_slice(&chunk.decoded_len.to_le_bytes());
        out.extend_from_slice(chunk.hash.as_bytes());
    }
    out.extend_from_slice(parts.body);
    let trailer = crc32c::crc32c(&out);
    out.extend_from_slice(&trailer.to_le_bytes());

    // Self-validate the layout (but not the content hash, which requires the
    // plaintext).
    Frame::parse(&out, limits)?;
    Ok(out)
}

/// Input to [`encode_frame`].
#[derive(Clone, Debug)]
pub struct FrameParts<'a> {
    pub format_major: u8,
    pub format_minor: u8,
    pub domain: u8,
    pub mode: Mode,
    pub flags: u16,
    pub level: u16,
    pub decoded_len: u64,
    pub content_hash: Hash32,
    pub dependencies: &'a [Dependency],
    pub chunks: &'a [ChunkEntry],
    pub body: &'a [u8],
}

impl<'a> FrameParts<'a> {
    /// Parts for a payload-domain frame with no dependencies or chunks.
    pub fn payload(
        mode: Mode,
        flags: u16,
        level: u16,
        decoded_len: u64,
        content_hash: Hash32,
        body: &'a [u8],
    ) -> Self {
        FrameParts {
            format_major: FORMAT_MAJOR,
            format_minor: FORMAT_MINOR,
            domain: DOMAIN_PAYLOAD,
            mode,
            flags,
            level,
            decoded_len,
            content_hash,
            dependencies: &[],
            chunks: &[],
            body,
        }
    }
}

impl<'a> Frame<'a> {
    /// Parse and fully validate a frame with default limits.
    pub fn parse_default(bytes: &'a [u8]) -> Result<Self> {
        Self::parse(bytes, &Limits::default())
    }

    /// Parse and fully validate a frame.
    pub fn parse(bytes: &'a [u8], limits: &Limits) -> Result<Self> {
        if bytes.len() < HEADER_LEN + TRAILER_LEN {
            return Err(Error::malformed(format!(
                "frame length {} is below the minimum {}",
                bytes.len(),
                HEADER_LEN + TRAILER_LEN
            )));
        }
        check_len(bytes.len() as u64, limits.max_frame_bytes, "frame")?;

        if bytes[0..4] != MAGIC {
            return Err(Error::malformed("bad frame magic (expected WFT1)"));
        }
        let format_major = bytes[4];
        let format_minor = bytes[5];
        if format_major != FORMAT_MAJOR {
            return Err(Error::unsupported_version(format!(
                "unsupported format major {format_major}"
            )));
        }
        if format_minor != FORMAT_MINOR {
            return Err(Error::unsupported_version(format!(
                "unsupported format minor {format_minor}"
            )));
        }

        // Verify the header checksum before trusting any length field.
        let stored_header_crc = le_u32(bytes, 72);
        let mut crc_input = Vec::with_capacity(92);
        crc_input.extend_from_slice(&bytes[0..72]);
        crc_input.extend_from_slice(&bytes[76..96]);
        if crc32c::crc32c(&crc_input) != stored_header_crc {
            return Err(Error::corrupt("header CRC32C mismatch"));
        }

        let domain = bytes[6];
        let mode = Mode::from_u8(bytes[7])?;
        let flags = le_u16(bytes, 8);
        let level = le_u16(bytes, 10);
        let decoded_len = le_u64(bytes, 12);
        let body_len = le_u64(bytes, 20);
        let dep_count = le_u32(bytes, 28);
        let chunk_count = le_u32(bytes, 32);
        let feature_bits = le_u16(bytes, 36);
        let reserved0 = le_u16(bytes, 38);
        let content_hash = Hash32::from_slice(&bytes[40..72])?;
        let table_len = le_u64(bytes, 76);
        let reserved1 = le_u32(bytes, 84);
        let reserved2 = le_u64(bytes, 88);

        if reserved0 != 0 || reserved1 != 0 || reserved2 != 0 {
            return Err(Error::malformed("reserved header fields must be zero"));
        }
        if domain > DOMAIN_STRUCTURED {
            return Err(Error::malformed(format!("invalid domain value {domain}")));
        }
        if flags
            & !(flags::CHUNKED | flags::SELF_CONTAINED | flags::DICT_RAW | flags::ORIGINAL_HASH)
            != 0
        {
            return Err(Error::unsupported_feature(format!(
                "unknown critical flag bits 0x{:04x}",
                flags & !0x000f
            )));
        }
        if flags & flags::ORIGINAL_HASH != 0 {
            return Err(Error::unsupported_feature(
                "ORIGINAL_HASH flag is reserved in v1",
            ));
        }
        if flags & flags::SELF_CONTAINED != 0 {
            return Err(Error::unsupported_feature(
                "SELF_CONTAINED is reserved and not produced in v1",
            ));
        }
        if feature_bits & !(features::DEPENDENCIES | features::CHUNK_HASHES) != 0 {
            return Err(Error::unsupported_feature(format!(
                "unknown critical feature bits 0x{:04x}",
                feature_bits & !0x0003
            )));
        }

        check_len(decoded_len, limits.max_decoded_bytes, "decoded data")?;
        check_len(
            u64::from(dep_count),
            limits.max_dependency_count,
            "dependency count",
        )?;
        check_len(u64::from(chunk_count), limits.max_chunks, "chunk count")?;

        let expected_table_len = (u64::from(dep_count))
            .checked_mul(DEP_ENTRY_LEN as u64)
            .and_then(|v| {
                (u64::from(chunk_count))
                    .checked_mul(CHUNK_ENTRY_LEN as u64)
                    .and_then(|c| v.checked_add(c))
            })
            .ok_or_else(|| Error::malformed("table length overflow"))?;
        if table_len != expected_table_len {
            return Err(Error::malformed(format!(
                "table_len {table_len} does not match {} dependencies and {} chunks",
                dep_count, chunk_count
            )));
        }

        let expected_total = (HEADER_LEN as u64)
            .checked_add(table_len)
            .and_then(|v| v.checked_add(body_len))
            .and_then(|v| v.checked_add(TRAILER_LEN as u64))
            .ok_or_else(|| Error::malformed("frame length overflow"))?;
        if expected_total != bytes.len() as u64 {
            return Err(Error::malformed(format!(
                "frame length {} does not match header-derived length {expected_total}",
                bytes.len()
            )));
        }

        let chunked = flags & flags::CHUNKED != 0;
        if chunked {
            if chunk_count == 0 {
                return Err(Error::malformed("CHUNKED set with an empty chunk table"));
            }
            if feature_bits & features::CHUNK_HASHES == 0 {
                return Err(Error::malformed(
                    "CHUNKED set without the CHUNK_HASHES feature",
                ));
            }
        } else {
            if chunk_count != 0 {
                return Err(Error::malformed(
                    "chunk table present without the CHUNKED flag",
                ));
            }
            if feature_bits & features::CHUNK_HASHES != 0 {
                return Err(Error::malformed(
                    "CHUNK_HASHES set without the CHUNKED flag",
                ));
            }
        }
        if dep_count > 0 && feature_bits & features::DEPENDENCIES == 0 {
            return Err(Error::malformed(
                "dependency table present without the DEPENDENCIES feature",
            ));
        }

        let table_len_usize = usize::try_from(table_len)
            .map_err(|_| Error::malformed("table length does not fit usize"))?;
        let body_start = HEADER_LEN
            .checked_add(table_len_usize)
            .ok_or_else(|| Error::malformed("body offset overflow"))?;
        let body_len_usize = usize::try_from(body_len)
            .map_err(|_| Error::malformed("body length does not fit usize"))?;
        let body_end = body_start
            .checked_add(body_len_usize)
            .ok_or_else(|| Error::malformed("body end overflow"))?;
        // expected_total == bytes.len() and trailer is last, so body_end + 4 == len.
        let body = &bytes[body_start..body_end];

        let trailer_at = bytes.len() - TRAILER_LEN;
        let stored_trailer_crc = le_u32(bytes, trailer_at);
        if crc32c::crc32c(&bytes[..trailer_at]) != stored_trailer_crc {
            return Err(Error::corrupt("frame trailer CRC32C mismatch"));
        }

        // Parse the dependency table.
        let mut dependencies = Vec::with_capacity(dep_count as usize);
        for index in 0..dep_count as usize {
            let off = index
                .checked_mul(DEP_ENTRY_LEN)
                .and_then(|value| HEADER_LEN.checked_add(value))
                .ok_or_else(|| Error::malformed("dependency table offset overflow"))?;
            let hash = Hash32::from_slice(&bytes[off..off + HASH_LEN])?;
            let dep_len = le_u64(bytes, off + 32);
            let dep_domain = bytes[off + 40];
            let dep_flags = bytes[off + 41];
            let reserved = le_u16(bytes, off + 42);
            let codec_hint = le_u32(bytes, off + 44);
            if dep_flags & !DEP_FLAG_REQUIRED != 0 {
                return Err(Error::malformed(format!(
                    "dependency {index} has unknown flag bits"
                )));
            }
            if dep_flags & DEP_FLAG_REQUIRED == 0 {
                return Err(Error::malformed(format!(
                    "dependency {index} does not set REQUIRED"
                )));
            }
            if reserved != 0 {
                return Err(Error::malformed(format!(
                    "dependency {index} reserved field must be zero"
                )));
            }
            if dep_domain > DEP_DOMAIN_DICTIONARY {
                return Err(Error::malformed(format!(
                    "dependency {index} has invalid domain {dep_domain}"
                )));
            }
            check_len(dep_len, limits.max_decoded_bytes, "dependency decoded data")?;
            dependencies.push(Dependency {
                hash,
                decoded_len: dep_len,
                dep_domain,
                dep_flags,
                codec_hint,
            });
        }

        // Parse the chunk table.
        let chunk_table_start = (dep_count as usize)
            .checked_mul(DEP_ENTRY_LEN)
            .and_then(|value| HEADER_LEN.checked_add(value))
            .ok_or_else(|| Error::malformed("chunk table offset overflow"))?;
        let mut chunks = Vec::with_capacity(chunk_count as usize);
        for index in 0..chunk_count as usize {
            let off = index
                .checked_mul(CHUNK_ENTRY_LEN)
                .and_then(|value| chunk_table_start.checked_add(value))
                .ok_or_else(|| Error::malformed("chunk entry offset overflow"))?;
            let encoded_off = le_u64(bytes, off);
            let encoded_len = le_u32(bytes, off + 8);
            let chunk_decoded_len = le_u32(bytes, off + 12);
            let hash = Hash32::from_slice(&bytes[off + 16..off + 48])?;
            chunks.push(ChunkEntry {
                encoded_off,
                encoded_len,
                decoded_len: chunk_decoded_len,
                hash,
            });
        }

        if chunked {
            let mut next_off: u64 = 0;
            let mut decoded_sum: u64 = 0;
            for (index, chunk) in chunks.iter().enumerate() {
                if chunk.encoded_off != next_off {
                    return Err(Error::malformed(format!(
                        "chunk {index} starts at {} but expected {next_off} (gap or overlap)",
                        chunk.encoded_off
                    )));
                }
                if chunk.encoded_len == 0 {
                    return Err(Error::malformed(format!(
                        "chunk {index} has zero encoded length"
                    )));
                }
                next_off = next_off
                    .checked_add(u64::from(chunk.encoded_len))
                    .ok_or_else(|| Error::malformed("chunk extent overflow"))?;
                decoded_sum = decoded_sum
                    .checked_add(u64::from(chunk.decoded_len))
                    .ok_or_else(|| Error::malformed("chunk decoded length sum overflow"))?;
            }
            if next_off != body_len {
                return Err(Error::malformed(format!(
                    "chunk extents end at {next_off} but body_len is {body_len}"
                )));
            }
            if decoded_sum != decoded_len {
                return Err(Error::malformed(format!(
                    "chunk decoded lengths sum to {decoded_sum} but decoded_len is {decoded_len}"
                )));
            }
        }

        let header = FrameHeader {
            format_major,
            format_minor,
            domain,
            mode,
            flags,
            level,
            decoded_len,
            body_len,
            dep_count,
            chunk_count,
            features: feature_bits,
            content_hash,
            table_len,
        };
        validate_mode_rules(&header, &dependencies, limits)?;

        Ok(Frame {
            header,
            dependencies,
            chunks,
            body,
            bytes,
        })
    }
}

fn validate_mode_rules(
    header: &FrameHeader,
    dependencies: &[Dependency],
    limits: &Limits,
) -> Result<()> {
    let dict_flag = header.flags & flags::DICT_RAW != 0;
    match header.mode {
        Mode::Raw => {
            if header.domain != DOMAIN_PAYLOAD {
                return Err(Error::malformed("RAW frame must use the payload domain"));
            }
            if dict_flag {
                return Err(Error::malformed("DICT_RAW is only valid for DICTIONARY"));
            }
            if header.dep_count != 0 {
                return Err(Error::malformed("RAW frame must not have dependencies"));
            }
            // Chunked RAW is explicitly defined by §6 (chunks are literals);
            // the §3 line `chunk_count == 0` is read as applying to the
            // unchunked form. Recorded in NOTES.md.
            if header.body_len != header.decoded_len {
                return Err(Error::malformed("RAW body_len must equal decoded_len"));
            }
        }
        Mode::Zstd => {
            if header.domain != DOMAIN_PAYLOAD {
                return Err(Error::malformed("ZSTD frame must use the payload domain"));
            }
            if dict_flag {
                return Err(Error::malformed("DICT_RAW is only valid for DICTIONARY"));
            }
            if header.dep_count != 0 {
                return Err(Error::malformed(
                    "ZSTD frame must not have dependencies in v1",
                ));
            }
        }
        Mode::Reference => {
            if header.domain != DOMAIN_PAYLOAD {
                return Err(Error::malformed(
                    "REFERENCE frame must use the payload domain",
                ));
            }
            if dict_flag {
                return Err(Error::malformed("DICT_RAW is only valid for DICTIONARY"));
            }
            if header.body_len != 0 {
                return Err(Error::malformed("REFERENCE body must be empty"));
            }
            if header.dep_count != 1 {
                return Err(Error::malformed(
                    "REFERENCE requires exactly one dependency",
                ));
            }
            if header.chunk_count != 0 {
                return Err(Error::malformed("REFERENCE must not be chunked"));
            }
            if header.decoded_len != dependencies[0].decoded_len {
                return Err(Error::malformed(
                    "REFERENCE decoded_len must equal dependency decoded_len",
                ));
            }
        }
        Mode::Delta => {
            if header.domain != DOMAIN_PAYLOAD {
                return Err(Error::malformed("DELTA frame must use the payload domain"));
            }
            if dict_flag {
                return Err(Error::malformed("DICT_RAW is only valid for DICTIONARY"));
            }
            if header.dep_count != 1 {
                return Err(Error::malformed("DELTA requires exactly one dependency"));
            }
            if header.chunk_count != 0 {
                return Err(Error::malformed("DELTA must not be chunked"));
            }
        }
        Mode::Tensor
        | Mode::TokenSequence
        | Mode::QuantizedTensor
        | Mode::FloatTensor
        | Mode::TensorSharded
        | Mode::KvCache
        | Mode::ActivationStream => {
            if header.domain != DOMAIN_PAYLOAD {
                return Err(Error::malformed(
                    "AI-mode frames must use the payload domain",
                ));
            }
            if dict_flag {
                return Err(Error::malformed("DICT_RAW is only valid for DICTIONARY"));
            }
            if header.dep_count != 0 {
                return Err(Error::malformed(
                    "AI-mode frames must not have dependencies in v1",
                ));
            }
            if header.chunk_count != 0 {
                return Err(Error::malformed("AI-mode frames must not be chunked"));
            }
            if header.body_len < 4 {
                return Err(Error::malformed(
                    "AI-mode body is shorter than its 4-byte metadata length",
                ));
            }
        }
        Mode::WeightDelta => {
            if header.domain != DOMAIN_PAYLOAD {
                return Err(Error::malformed(
                    "WEIGHT_DELTA frame must use the payload domain",
                ));
            }
            if dict_flag {
                return Err(Error::malformed("DICT_RAW is only valid for DICTIONARY"));
            }
            if header.dep_count != 1 {
                return Err(Error::malformed(
                    "WEIGHT_DELTA requires exactly one dependency",
                ));
            }
            if header.chunk_count != 0 {
                return Err(Error::malformed("WEIGHT_DELTA must not be chunked"));
            }
            if header.body_len < 4 {
                return Err(Error::malformed(
                    "WEIGHT_DELTA body is shorter than its 4-byte metadata length",
                ));
            }
            if dependencies[0].dep_domain != DEP_DOMAIN_PAYLOAD {
                return Err(Error::malformed(
                    "WEIGHT_DELTA dependency must use dep_domain 0 (payload)",
                ));
            }
            if header.decoded_len != dependencies[0].decoded_len {
                return Err(Error::malformed(
                    "WEIGHT_DELTA decoded_len must equal the base decoded_len",
                ));
            }
        }
        Mode::Dictionary => {
            if header.domain != DOMAIN_PAYLOAD {
                return Err(Error::malformed(
                    "DICTIONARY frame must use the payload domain",
                ));
            }
            if !dict_flag {
                return Err(Error::malformed("DICTIONARY requires the DICT_RAW flag"));
            }
            if header.dep_count != 1 {
                return Err(Error::malformed(
                    "DICTIONARY requires exactly one dependency",
                ));
            }
            if dependencies[0].dep_domain != DEP_DOMAIN_DICTIONARY {
                return Err(Error::malformed(
                    "DICTIONARY dependency must use dep_domain 2 (dictionary bytes)",
                ));
            }
            if header.chunk_count != 0 {
                return Err(Error::malformed("DICTIONARY must not be chunked"));
            }
            check_len(
                dependencies[0].decoded_len,
                limits.max_dictionary_bytes,
                "dictionary",
            )?;
        }
        Mode::Structured => {
            if header.domain != DOMAIN_STRUCTURED {
                return Err(Error::malformed(
                    "STRUCTURED frame must use the structured domain",
                ));
            }
            if dict_flag {
                return Err(Error::malformed("DICT_RAW is only valid for DICTIONARY"));
            }
            if header.dep_count != 0 {
                return Err(Error::malformed(
                    "STRUCTURED must not have dependencies in v1",
                ));
            }
            if header.chunk_count != 0 {
                return Err(Error::malformed("STRUCTURED must not be chunked"));
            }
            if header.body_len != header.decoded_len {
                return Err(Error::malformed(
                    "STRUCTURED body_len must equal decoded_len",
                ));
            }
        }
    }
    Ok(())
}

/// Inspect frame bytes with default limits.
pub fn inspect(bytes: &[u8]) -> Result<Frame<'_>> {
    Frame::parse(bytes, &Limits::default())
}

/// Return an empty-error helper used by callers that need to construct a
/// frame-length error with the exact class.
pub fn malformed_frame(message: impl Into<String>) -> Error {
    Error::new(ErrorClass::Malformed, message)
}

fn le_u16(bytes: &[u8], off: usize) -> u16 {
    u16::from_le_bytes([bytes[off], bytes[off + 1]])
}

fn le_u32(bytes: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]])
}

fn le_u64(bytes: &[u8], off: usize) -> u64 {
    let mut buf = [0u8; 8];
    buf.copy_from_slice(&bytes[off..off + 8]);
    u64::from_le_bytes(buf)
}
