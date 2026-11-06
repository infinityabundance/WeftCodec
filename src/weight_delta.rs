//! Mode 11 `WEIGHT_DELTA` (WEFT-AI-CODEC §12).
//!
//! A frame carries the exact word-wise difference between one tensor
//! generation and a base tensor frame. Every operation is on unsigned words:
//!
//! ```text
//! d_i = (a_i - b_i) mod 2^word_bits
//! a_i = (d_i + b_i) mod 2^word_bits
//! ```
//!
//! No floating-point arithmetic is performed anywhere. `a_i` and `b_i` are raw
//! element words, so NaNs, denormals, and signed zeros reconstruct bit-exactly.
//!
//! The payload is the concatenation of one `huffman-canonical-v1` stream per
//! byte plane (plane 0 = least significant byte). `q4_0` uses a single plane of
//! unpacked nibbles, one byte per element.
//!
//! The base is a dependency frame named by content hash. Decoding resolves it
//! through the caller's resolver (the DAG resolver enforces depth and cycle
//! rules) and, when the resolver can also supply the base frame bytes, verifies
//! that frame's own dtype/shape descriptor.

use std::collections::BTreeMap;

use crate::dependency::{MapResolver, Resolver};
use crate::error::{Error, Result};
use crate::frame::{
    encode_frame, Dependency, Frame, FrameParts, Mode, DEP_DOMAIN_PAYLOAD, DEP_FLAG_REQUIRED,
    DOMAIN_PAYLOAD,
};
use crate::hash::{hash_domain, Hash32, DOMAIN_CHUNK, HASH_LEN};
use crate::huffman;
use crate::limits::{check_len, Limits};
use crate::modes;
use crate::tensor::DOMAIN_TENSOR_VALUES;
use crate::wcb::{self, Value};

/// Canonical descriptor schema.
pub const WEIGHT_DELTA_SCHEMA: &str = "weft.ai.weight_delta.v1";
/// The only delta operation implemented in v1.
pub const WEIGHT_DELTA_OP: &str = "mod-sub";
/// The only plane coder implemented in v1.
pub const WEIGHT_DELTA_PLANE_CODEC: &str = huffman::CODER_ID;

/// Element word dtypes implemented by mode 11.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WeightDtype {
    Bf16,
    Fp16,
    Fp32,
    Q8_0,
    Q4_0,
}

impl WeightDtype {
    /// Parse the canonical dtype name.
    pub fn from_name(name: &str) -> Result<Self> {
        match name {
            "bf16" => Ok(WeightDtype::Bf16),
            "fp16" => Ok(WeightDtype::Fp16),
            "fp32" => Ok(WeightDtype::Fp32),
            "q8_0" => Ok(WeightDtype::Q8_0),
            "q4_0" => Ok(WeightDtype::Q4_0),
            other => Err(Error::malformed(format!(
                "unknown WEIGHT_DELTA dtype `{other}` (v1 allows bf16, fp16, fp32, q8_0, q4_0)"
            ))),
        }
    }

    /// Canonical dtype name.
    pub fn as_str(self) -> &'static str {
        match self {
            WeightDtype::Bf16 => "bf16",
            WeightDtype::Fp16 => "fp16",
            WeightDtype::Fp32 => "fp32",
            WeightDtype::Q8_0 => "q8_0",
            WeightDtype::Q4_0 => "q4_0",
        }
    }

    /// Bits in one unsigned delta word.
    pub fn word_bits(self) -> u64 {
        match self {
            WeightDtype::Bf16 | WeightDtype::Fp16 => 16,
            WeightDtype::Fp32 => 32,
            WeightDtype::Q8_0 => 8,
            WeightDtype::Q4_0 => 4,
        }
    }

    /// Bytes per element in canonical row-major order.
    ///
    /// `q4_0` words are unpacked nibbles: one byte per element.
    pub fn storage_bytes(self) -> u64 {
        match self {
            WeightDtype::Bf16 | WeightDtype::Fp16 => 2,
            WeightDtype::Fp32 => 4,
            WeightDtype::Q8_0 | WeightDtype::Q4_0 => 1,
        }
    }

    /// Number of byte planes in the payload.
    pub fn planes(self) -> u64 {
        match self {
            WeightDtype::Bf16 | WeightDtype::Fp16 => 2,
            WeightDtype::Fp32 => 4,
            WeightDtype::Q8_0 | WeightDtype::Q4_0 => 1,
        }
    }

    pub(crate) fn is_quantized(self) -> bool {
        matches!(self, WeightDtype::Q8_0 | WeightDtype::Q4_0)
    }
}

/// The descriptor of one WEIGHT_DELTA frame.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WeightDeltaMeta {
    pub dtype: WeightDtype,
    pub shape: Vec<u64>,
    pub element_count: u64,
    pub word_bits: u64,
    pub base: Hash32,
    pub base_dtype: WeightDtype,
    pub base_shape: Vec<u64>,
    pub op: String,
    pub planes: u64,
    pub plane_codec: String,
    pub delta_root: Hash32,
    pub payload_len: u64,
}

// ---------------------------------------------------------------------------
// Shape and metadata helpers
// ---------------------------------------------------------------------------

pub(crate) fn validate_shape(shape: &[u64], limits: &Limits) -> Result<u64> {
    if shape.is_empty() {
        return Err(Error::malformed(
            "WEIGHT_DELTA shape rank must be at least 1",
        ));
    }
    if shape.len() as u64 > limits.max_tensor_rank {
        return Err(Error::limit_exceeded(format!(
            "WEIGHT_DELTA rank {} exceeds limit {}",
            shape.len(),
            limits.max_tensor_rank
        )));
    }
    let mut product: u64 = 1;
    for dim in shape {
        if *dim == 0 {
            return Err(Error::malformed(
                "WEIGHT_DELTA shape dimensions must be greater than zero",
            ));
        }
        product = product
            .checked_mul(*dim)
            .ok_or_else(|| Error::limit_exceeded("WEIGHT_DELTA shape product overflows u64"))?;
    }
    if product > limits.max_tensor_elements {
        return Err(Error::limit_exceeded(format!(
            "WEIGHT_DELTA element count {product} exceeds limit {}",
            limits.max_tensor_elements
        )));
    }
    Ok(product)
}

pub(crate) fn payload_len(element_count: u64, dtype: WeightDtype) -> Result<u64> {
    element_count
        .checked_mul(dtype.storage_bytes())
        .ok_or_else(|| Error::limit_exceeded("WEIGHT_DELTA element byte length overflows u64"))
}

fn meta_text(map: &BTreeMap<String, Value>, key: &str) -> Result<String> {
    match map.get(key) {
        Some(Value::Str(text)) => Ok(text.clone()),
        _ => Err(Error::malformed(format!(
            "WEIGHT_DELTA metadata `{key}` must be text"
        ))),
    }
}

fn meta_u64(map: &BTreeMap<String, Value>, key: &str) -> Result<u64> {
    match map.get(key) {
        Some(Value::U64(value)) => Ok(*value),
        _ => Err(Error::malformed(format!(
            "WEIGHT_DELTA metadata `{key}` must be u64"
        ))),
    }
}

fn meta_bytes(map: &BTreeMap<String, Value>, key: &str) -> Result<Vec<u8>> {
    match map.get(key) {
        Some(Value::Bytes(bytes)) => Ok(bytes.clone()),
        _ => Err(Error::malformed(format!(
            "WEIGHT_DELTA metadata `{key}` must be bytes"
        ))),
    }
}

fn meta_shape(map: &BTreeMap<String, Value>, key: &str) -> Result<Vec<u64>> {
    match map.get(key) {
        Some(Value::Array(items)) => {
            let mut shape = Vec::with_capacity(items.len());
            for item in items {
                match item {
                    Value::U64(value) => shape.push(*value),
                    _ => {
                        return Err(Error::malformed(format!(
                            "WEIGHT_DELTA metadata `{key}` elements must be u64"
                        )))
                    }
                }
            }
            Ok(shape)
        }
        _ => Err(Error::malformed(format!(
            "WEIGHT_DELTA metadata `{key}` must be an array"
        ))),
    }
}

fn hash_field(map: &BTreeMap<String, Value>, key: &str) -> Result<Hash32> {
    let bytes = meta_bytes(map, key)?;
    if bytes.len() != HASH_LEN {
        return Err(Error::malformed(format!(
            "WEIGHT_DELTA metadata `{key}` must be {HASH_LEN} bytes, got {}",
            bytes.len()
        )));
    }
    Hash32::from_slice(&bytes)
}

/// Validate every descriptor invariant and return the element count.
pub fn validate_meta(meta: &WeightDeltaMeta, limits: &Limits) -> Result<u64> {
    if meta.op != WEIGHT_DELTA_OP {
        return Err(Error::malformed(format!(
            "WEIGHT_DELTA op `{}` is not implemented (only {WEIGHT_DELTA_OP})",
            meta.op
        )));
    }
    if meta.plane_codec != WEIGHT_DELTA_PLANE_CODEC {
        return Err(Error::malformed(format!(
            "WEIGHT_DELTA plane_codec `{}` is not implemented (only {WEIGHT_DELTA_PLANE_CODEC})",
            meta.plane_codec
        )));
    }
    if meta.dtype.word_bits() != meta.word_bits {
        return Err(Error::malformed(format!(
            "WEIGHT_DELTA word_bits {} disagrees with dtype {} ({} bits)",
            meta.word_bits,
            meta.dtype.as_str(),
            meta.dtype.word_bits()
        )));
    }
    if meta.dtype.planes() != meta.planes {
        return Err(Error::malformed(format!(
            "WEIGHT_DELTA planes {} disagrees with dtype {} ({} planes)",
            meta.planes,
            meta.dtype.as_str(),
            meta.dtype.planes()
        )));
    }
    if meta.base_dtype != meta.dtype {
        return Err(Error::malformed(format!(
            "WEIGHT_DELTA base_dtype `{}` must equal dtype `{}`",
            meta.base_dtype.as_str(),
            meta.dtype.as_str()
        )));
    }
    if meta.base_shape != meta.shape {
        return Err(Error::malformed("WEIGHT_DELTA base_shape must equal shape"));
    }
    let count = validate_shape(&meta.shape, limits)?;
    if meta.element_count != count {
        return Err(Error::malformed(format!(
            "WEIGHT_DELTA element_count {} does not match shape product {count}",
            meta.element_count
        )));
    }
    payload_len(count, meta.dtype)?;
    Ok(count)
}

/// Build the canonical WCB descriptor value.
pub fn descriptor_value(meta: &WeightDeltaMeta) -> Value {
    Value::map([
        ("schema", Value::Str(WEIGHT_DELTA_SCHEMA.to_owned())),
        ("dtype", Value::Str(meta.dtype.as_str().to_owned())),
        (
            "shape",
            Value::Array(meta.shape.iter().map(|dim| Value::U64(*dim)).collect()),
        ),
        ("element_count", Value::U64(meta.element_count)),
        ("word_bits", Value::U64(meta.word_bits)),
        ("base", Value::Bytes(meta.base.as_bytes().to_vec())),
        (
            "base_dtype",
            Value::Str(meta.base_dtype.as_str().to_owned()),
        ),
        (
            "base_shape",
            Value::Array(meta.base_shape.iter().map(|dim| Value::U64(*dim)).collect()),
        ),
        ("op", Value::Str(meta.op.clone())),
        ("planes", Value::U64(meta.planes)),
        ("plane_codec", Value::Str(meta.plane_codec.clone())),
        (
            "delta_root",
            Value::Bytes(meta.delta_root.as_bytes().to_vec()),
        ),
        ("payload_len", Value::U64(meta.payload_len)),
    ])
}

pub(crate) fn split_ai_body(body: &[u8]) -> Result<(&[u8], &[u8])> {
    if body.len() < 4 {
        return Err(Error::malformed(
            "WEIGHT_DELTA body is shorter than its 4-byte metadata length",
        ));
    }
    let meta_len = u64::from(u32::from_le_bytes([body[0], body[1], body[2], body[3]]));
    let end = 4u64
        .checked_add(meta_len)
        .ok_or_else(|| Error::malformed("WEIGHT_DELTA metadata length overflows"))?;
    if end > body.len() as u64 {
        return Err(Error::malformed(
            "WEIGHT_DELTA metadata extends past the frame body",
        ));
    }
    let end = end as usize;
    Ok((&body[4..end], &body[end..]))
}

fn parse_descriptor(bytes: &[u8], limits: &Limits) -> Result<WeightDeltaMeta> {
    let value = wcb::decode(bytes, limits)?;
    let map = match value {
        Value::Map(map) => map,
        _ => return Err(Error::malformed("WEIGHT_DELTA metadata must be a WCB map")),
    };
    const KEYS: [&str; 13] = [
        "schema",
        "dtype",
        "shape",
        "element_count",
        "word_bits",
        "base",
        "base_dtype",
        "base_shape",
        "op",
        "planes",
        "plane_codec",
        "delta_root",
        "payload_len",
    ];
    for key in map.keys() {
        if !KEYS.contains(&key.as_str()) {
            return Err(Error::malformed(format!(
                "WEIGHT_DELTA metadata has unknown key `{key}`"
            )));
        }
    }
    for key in KEYS {
        if !map.contains_key(key) {
            return Err(Error::malformed(format!(
                "WEIGHT_DELTA metadata is missing key `{key}`"
            )));
        }
    }
    let schema = meta_text(&map, "schema")?;
    if schema != WEIGHT_DELTA_SCHEMA {
        return Err(Error::malformed(format!(
            "unsupported WEIGHT_DELTA schema `{schema}`"
        )));
    }
    let dtype = WeightDtype::from_name(&meta_text(&map, "dtype")?)?;
    let shape = meta_shape(&map, "shape")?;
    let element_count = meta_u64(&map, "element_count")?;
    let word_bits = meta_u64(&map, "word_bits")?;
    let base = hash_field(&map, "base")?;
    let base_dtype = WeightDtype::from_name(&meta_text(&map, "base_dtype")?)?;
    let base_shape = meta_shape(&map, "base_shape")?;
    let op = meta_text(&map, "op")?;
    let planes = meta_u64(&map, "planes")?;
    let plane_codec = meta_text(&map, "plane_codec")?;
    let delta_root = hash_field(&map, "delta_root")?;
    let payload_len = meta_u64(&map, "payload_len")?;
    let meta = WeightDeltaMeta {
        dtype,
        shape,
        element_count,
        word_bits,
        base,
        base_dtype,
        base_shape,
        op,
        planes,
        plane_codec,
        delta_root,
        payload_len,
    };
    validate_meta(&meta, limits)?;
    Ok(meta)
}

/// Parse and validate a WEIGHT_DELTA descriptor from a frame (no base
/// resolution).
pub fn weight_delta_metadata(frame: &Frame<'_>, limits: &Limits) -> Result<WeightDeltaMeta> {
    if frame.header.mode != Mode::WeightDelta {
        return Err(Error::malformed(format!(
            "expected a WEIGHT_DELTA frame, got mode {}",
            frame.header.mode.as_str()
        )));
    }
    let (descriptor, _) = split_ai_body(frame.body)?;
    parse_descriptor(descriptor, limits)
}

// ---------------------------------------------------------------------------
// Delta words and planes
// ---------------------------------------------------------------------------

pub(crate) fn word_at(dtype: WeightDtype, bytes: &[u8], index: usize) -> u32 {
    match dtype {
        WeightDtype::Bf16 | WeightDtype::Fp16 => {
            let offset = index * 2;
            u32::from(u16::from_le_bytes([bytes[offset], bytes[offset + 1]]))
        }
        WeightDtype::Fp32 => {
            let offset = index * 4;
            u32::from_le_bytes([
                bytes[offset],
                bytes[offset + 1],
                bytes[offset + 2],
                bytes[offset + 3],
            ])
        }
        WeightDtype::Q8_0 | WeightDtype::Q4_0 => u32::from(bytes[index]),
    }
}

pub(crate) fn push_word(dtype: WeightDtype, word: u32, out: &mut Vec<u8>) {
    match dtype {
        WeightDtype::Bf16 | WeightDtype::Fp16 => {
            out.extend_from_slice(&(word as u16).to_le_bytes());
        }
        WeightDtype::Fp32 => out.extend_from_slice(&word.to_le_bytes()),
        WeightDtype::Q8_0 | WeightDtype::Q4_0 => out.push(word as u8),
    }
}

pub(crate) fn mod_mask(dtype: WeightDtype) -> u32 {
    match dtype.word_bits() {
        32 => u32::MAX,
        16 => 0xFFFF,
        8 => 0xFF,
        4 => 0x0F,
        _ => unreachable!("word_bits is fixed by dtype"),
    }
}

pub(crate) fn delta_word(dtype: WeightDtype, target: u32, base: u32) -> u32 {
    target.wrapping_sub(base) & mod_mask(dtype)
}

pub(crate) fn target_word(dtype: WeightDtype, delta: u32, base: u32) -> u32 {
    delta.wrapping_add(base) & mod_mask(dtype)
}

/// Split `target`/`base` element bytes into the canonical delta word stream and
/// the encoded byte planes.
pub(crate) fn build_planes(
    dtype: WeightDtype,
    target: &[u8],
    base: &[u8],
    element_count: usize,
) -> (Vec<u8>, Vec<Vec<u8>>) {
    let plane_count = dtype.planes() as usize;
    let mut planes: Vec<Vec<u8>> = (0..plane_count)
        .map(|_| Vec::with_capacity(element_count))
        .collect();
    let mut delta_bytes =
        Vec::with_capacity(element_count.saturating_mul(dtype.storage_bytes() as usize));
    for index in 0..element_count {
        let delta = delta_word(
            dtype,
            word_at(dtype, target, index),
            word_at(dtype, base, index),
        );
        push_word(dtype, delta, &mut delta_bytes);
        match dtype {
            WeightDtype::Bf16 | WeightDtype::Fp16 => {
                planes[0].push(delta as u8);
                planes[1].push((delta >> 8) as u8);
            }
            WeightDtype::Fp32 => {
                planes[0].push(delta as u8);
                planes[1].push((delta >> 8) as u8);
                planes[2].push((delta >> 16) as u8);
                planes[3].push((delta >> 24) as u8);
            }
            WeightDtype::Q8_0 | WeightDtype::Q4_0 => planes[0].push(delta as u8),
        }
    }
    (delta_bytes, planes)
}

// ---------------------------------------------------------------------------
// Base frame descriptor
// ---------------------------------------------------------------------------

fn tensor_dtype_to_weight(dtype: modes::TensorDtype) -> Result<WeightDtype> {
    match dtype {
        modes::TensorDtype::Bf16 => Ok(WeightDtype::Bf16),
        modes::TensorDtype::F16 => Ok(WeightDtype::Fp16),
        modes::TensorDtype::F32 => Ok(WeightDtype::Fp32),
        modes::TensorDtype::I8 => Ok(WeightDtype::Q8_0),
        modes::TensorDtype::U8 => Ok(WeightDtype::Q4_0),
        other => Err(Error::malformed(format!(
            "base frame dtype `{}` cannot back a WEIGHT_DELTA word plane",
            other.as_str()
        ))),
    }
}

fn float_dtype_to_weight(dtype: modes::FloatDtype) -> Result<WeightDtype> {
    match dtype {
        modes::FloatDtype::Bf16 => Ok(WeightDtype::Bf16),
        modes::FloatDtype::F16 => Ok(WeightDtype::Fp16),
        modes::FloatDtype::F32 => Ok(WeightDtype::Fp32),
        other => Err(Error::malformed(format!(
            "base frame dtype `{}` cannot back a WEIGHT_DELTA word plane",
            other.as_str()
        ))),
    }
}

/// The logical weight dtype and shape carried by a base frame.
///
/// `q8_0` is backed by a mode-6 `i8` tensor frame and `q4_0` by a mode-6 `u8`
/// frame (one unpacked nibble per byte), because mode 6 has no quantized dtype
/// names. A mode-11 base frame carries the weight dtype directly.
pub fn base_frame_descriptor(
    frame: &Frame<'_>,
    limits: &Limits,
) -> Result<(WeightDtype, Vec<u64>)> {
    match frame.header.mode {
        Mode::Tensor => {
            let meta = modes::tensor_metadata(frame, limits)?;
            Ok((tensor_dtype_to_weight(meta.dtype)?, meta.shape))
        }
        Mode::FloatTensor => {
            let meta = modes::float_tensor_metadata(frame, limits)?;
            Ok((float_dtype_to_weight(meta.dtype)?, meta.shape))
        }
        Mode::WeightDelta | Mode::SemanticDelta | Mode::SparseDelta => {
            let (dtype, shape) = match frame.header.mode {
                Mode::WeightDelta => {
                    let meta = weight_delta_metadata(frame, limits)?;
                    (meta.dtype, meta.shape)
                }
                Mode::SemanticDelta => {
                    let meta = crate::semantic_delta::semantic_delta_metadata(frame, limits)?;
                    (meta.dtype, meta.shape)
                }
                _ => {
                    let meta = crate::sparse_delta::sparse_delta_metadata(frame, limits)?;
                    (meta.dtype, meta.shape)
                }
            };
            Ok((dtype, shape))
        }
        other => Err(Error::malformed(format!(
            "base frame mode `{}` does not carry tensor dtype/shape metadata",
            other.as_str()
        ))),
    }
}

fn dependency_mismatch_base(message: impl Into<String>) -> Error {
    Error::dependency_mismatch(message)
}

pub(crate) fn check_quantized_words(dtype: WeightDtype, bytes: &[u8], what: &str) -> Result<()> {
    if dtype == WeightDtype::Q4_0 {
        for (index, byte) in bytes.iter().enumerate() {
            if *byte > 15 {
                return Err(Error::corrupt(format!(
                    "WEIGHT_DELTA {what} q4_0 word {index} is {byte}, outside 0..=15"
                )));
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Encoding
// ---------------------------------------------------------------------------

/// Build a frame from an explicit descriptor and store payload.
///
/// This is the low-level constructor shared by the encoder, adversarial vector
/// generation, and the cycle vector. It does not re-check descriptor
/// semantics; the resulting frame is re-parsed, so frame-level mode rules
/// still apply.
pub fn encode_descriptor_frame(
    meta: &WeightDeltaMeta,
    payload: &[u8],
    decoded_len: u64,
    content_hash: Hash32,
    base_mode: u32,
    limits: &Limits,
) -> Result<Vec<u8>> {
    let meta_bytes = wcb::encode(&descriptor_value(meta), limits)?;
    let meta_len = u32::try_from(meta_bytes.len())
        .map_err(|_| Error::limit_exceeded("WEIGHT_DELTA metadata does not fit u32"))?;
    let capacity = 4usize
        .checked_add(meta_bytes.len())
        .and_then(|value| value.checked_add(payload.len()))
        .ok_or_else(|| Error::limit_exceeded("WEIGHT_DELTA body length overflow"))?;
    let mut body = Vec::with_capacity(capacity);
    body.extend_from_slice(&meta_len.to_le_bytes());
    body.extend_from_slice(&meta_bytes);
    body.extend_from_slice(payload);

    let dependencies = [Dependency {
        hash: meta.base,
        decoded_len,
        dep_domain: DEP_DOMAIN_PAYLOAD,
        dep_flags: DEP_FLAG_REQUIRED,
        codec_hint: base_mode,
    }];
    let parts = FrameParts {
        format_major: crate::frame::FORMAT_MAJOR,
        format_minor: crate::frame::FORMAT_MINOR,
        domain: DOMAIN_PAYLOAD,
        mode: Mode::WeightDelta,
        flags: 0,
        level: 0,
        decoded_len,
        content_hash,
        dependencies: &dependencies,
        chunks: &[],
        body: &body,
    };
    encode_frame(&parts, limits)
}

/// Encode a target tensor as a WEIGHT_DELTA against `base_frame`.
pub fn encode_weight_delta(
    target: &[u8],
    dtype: WeightDtype,
    shape: &[u64],
    base_frame: &[u8],
    limits: &Limits,
) -> Result<Vec<u8>> {
    encode_weight_delta_with_resolver(
        target,
        dtype,
        shape,
        base_frame,
        &MapResolver::new(),
        limits,
    )
}

/// Encode a target tensor against a base frame whose own dependencies are
/// resolved through `resolver`.
pub fn encode_weight_delta_with_resolver(
    target: &[u8],
    dtype: WeightDtype,
    shape: &[u64],
    base_frame: &[u8],
    resolver: &dyn Resolver,
    limits: &Limits,
) -> Result<Vec<u8>> {
    let base_parsed = Frame::parse(base_frame, limits)?;
    let (base_dtype, base_shape) = base_frame_descriptor(&base_parsed, limits)?;
    if base_dtype != dtype {
        return Err(Error::malformed(format!(
            "base frame dtype `{}` does not match WEIGHT_DELTA dtype `{}`",
            base_dtype.as_str(),
            dtype.as_str()
        )));
    }
    if base_shape != shape {
        return Err(Error::malformed(
            "base frame shape does not match the WEIGHT_DELTA shape",
        ));
    }
    let element_count = validate_shape(shape, limits)?;
    let expected_len = payload_len(element_count, dtype)?;
    check_len(
        expected_len,
        limits.max_decoded_bytes,
        "WEIGHT_DELTA element bytes",
    )?;
    if target.len() as u64 != expected_len {
        return Err(Error::malformed(format!(
            "WEIGHT_DELTA target is {} bytes but shape {} x {} requires {expected_len}",
            target.len(),
            dtype.as_str(),
            element_count
        )));
    }
    let base_bytes = modes::decode_bytes(base_frame, resolver, limits)?;
    if base_bytes.len() as u64 != expected_len {
        return Err(dependency_mismatch_base(format!(
            "base frame decodes to {} bytes but the descriptor requires {expected_len}",
            base_bytes.len()
        )));
    }
    if dtype.is_quantized() {
        check_quantized_words(dtype, target, "target")?;
        check_quantized_words(dtype, &base_bytes, "base")?;
    }

    let count = usize::try_from(element_count)
        .map_err(|_| Error::limit_exceeded("WEIGHT_DELTA element count does not fit usize"))?;
    let (delta_bytes, planes) = build_planes(dtype, target, &base_bytes, count);
    let delta_root = hash_domain(DOMAIN_TENSOR_VALUES, &delta_bytes);
    let mut payload = Vec::new();
    for plane in &planes {
        payload.extend_from_slice(&huffman::encode(plane));
    }

    let meta = WeightDeltaMeta {
        dtype,
        shape: shape.to_vec(),
        element_count,
        word_bits: dtype.word_bits(),
        base: base_parsed.header.content_hash,
        base_dtype: dtype,
        base_shape: shape.to_vec(),
        op: WEIGHT_DELTA_OP.to_owned(),
        planes: dtype.planes(),
        plane_codec: WEIGHT_DELTA_PLANE_CODEC.to_owned(),
        delta_root,
        payload_len: payload.len() as u64,
    };
    encode_descriptor_frame(
        &meta,
        &payload,
        target.len() as u64,
        hash_domain(DOMAIN_CHUNK, target),
        u32::from(base_parsed.header.mode.as_u8()),
        limits,
    )
}

// ---------------------------------------------------------------------------
// Decoding
// ---------------------------------------------------------------------------

pub(crate) fn decode_planes(
    dtype: WeightDtype,
    payload: &[u8],
    element_count: usize,
) -> Result<Vec<Vec<u8>>> {
    let plane_count = usize::try_from(dtype.planes())
        .map_err(|_| Error::limit_exceeded("WEIGHT_DELTA plane count does not fit usize"))?;
    let mut planes = Vec::with_capacity(plane_count);
    let mut offset = 0usize;
    for _ in 0..plane_count {
        if offset >= payload.len() {
            return Err(Error::corrupt(
                "WEIGHT_DELTA payload ends before all declared planes",
            ));
        }
        let (plane, consumed) = huffman::decode_prefix(&payload[offset..], element_count)?;
        offset = offset
            .checked_add(consumed)
            .ok_or_else(|| Error::limit_exceeded("WEIGHT_DELTA payload offset overflows"))?;
        planes.push(plane);
    }
    if offset != payload.len() {
        return Err(Error::corrupt(
            "WEIGHT_DELTA payload has trailing bytes after the declared planes",
        ));
    }
    Ok(planes)
}

pub(crate) fn delta_word_from_planes(
    dtype: WeightDtype,
    planes: &[Vec<u8>],
    index: usize,
) -> Result<u32> {
    let word = match dtype {
        WeightDtype::Bf16 | WeightDtype::Fp16 => {
            u32::from(planes[0][index]) | (u32::from(planes[1][index]) << 8)
        }
        WeightDtype::Fp32 => {
            u32::from(planes[0][index])
                | (u32::from(planes[1][index]) << 8)
                | (u32::from(planes[2][index]) << 16)
                | (u32::from(planes[3][index]) << 24)
        }
        WeightDtype::Q8_0 => u32::from(planes[0][index]),
        WeightDtype::Q4_0 => {
            let nibble = planes[0][index];
            if nibble > 15 {
                return Err(Error::corrupt(format!(
                    "WEIGHT_DELTA q4_0 delta word {index} is {nibble}, outside 0..=15"
                )));
            }
            u32::from(nibble)
        }
    };
    Ok(word)
}

/// Decode a WEIGHT_DELTA frame, resolving the base through `resolver`.
///
/// The resolver must return the base plaintext. A resolver that can also
/// return the base frame bytes (the existing DAG resolver does) additionally
/// gets its dtype/shape descriptor verified.
pub fn decode_frame(
    frame: &Frame<'_>,
    resolver: &dyn Resolver,
    limits: &Limits,
) -> Result<Vec<u8>> {
    decode_with_optional_base(frame, None, resolver, limits)
}

/// Decode a WEIGHT_DELTA frame with the base frame supplied out of band.
pub fn decode_with_base_frame(
    frame: &Frame<'_>,
    base_frame: &[u8],
    resolver: &dyn Resolver,
    limits: &Limits,
) -> Result<Vec<u8>> {
    decode_with_optional_base(frame, Some(base_frame), resolver, limits)
}

fn decode_with_optional_base(
    frame: &Frame<'_>,
    base_frame: Option<&[u8]>,
    resolver: &dyn Resolver,
    limits: &Limits,
) -> Result<Vec<u8>> {
    if frame.header.mode != Mode::WeightDelta {
        return Err(Error::malformed(format!(
            "expected a WEIGHT_DELTA frame, got mode {}",
            frame.header.mode.as_str()
        )));
    }
    let (descriptor_bytes, payload) = split_ai_body(frame.body)?;
    let meta = parse_descriptor(descriptor_bytes, limits)?;
    let element_count = validate_meta(&meta, limits)?;
    let expected_len = payload_len(element_count, meta.dtype)?;
    if frame.header.decoded_len != expected_len {
        return Err(Error::corrupt(format!(
            "WEIGHT_DELTA decoded_len is {} but the descriptor requires {expected_len}",
            frame.header.decoded_len
        )));
    }
    check_len(
        expected_len,
        limits.max_decoded_bytes,
        "WEIGHT_DELTA element bytes",
    )?;
    if meta.payload_len != payload.len() as u64 {
        return Err(Error::malformed(format!(
            "WEIGHT_DELTA payload is {} bytes but payload_len declares {}",
            payload.len(),
            meta.payload_len
        )));
    }
    if frame.header.dep_count != 1 {
        return Err(Error::malformed(
            "WEIGHT_DELTA requires exactly one dependency",
        ));
    }
    let dep = &frame.dependencies[0];
    if dep.hash != meta.base {
        return Err(Error::malformed(
            "WEIGHT_DELTA dependency hash must equal the descriptor base",
        ));
    }
    if dep.decoded_len != frame.header.decoded_len {
        return Err(Error::malformed(
            "WEIGHT_DELTA dependency decoded_len must equal the frame decoded_len",
        ));
    }

    let count = usize::try_from(element_count)
        .map_err(|_| Error::limit_exceeded("WEIGHT_DELTA element count does not fit usize"))?;
    let planes = decode_planes(meta.dtype, payload, count)?;

    // Rebuild the canonical delta word stream and verify delta_root before
    // touching the base.
    let mut delta_bytes =
        Vec::with_capacity(count.saturating_mul(meta.dtype.storage_bytes() as usize));
    for index in 0..count {
        let delta = delta_word_from_planes(meta.dtype, &planes, index)?;
        push_word(meta.dtype, delta, &mut delta_bytes);
    }
    if hash_domain(DOMAIN_TENSOR_VALUES, &delta_bytes) != meta.delta_root {
        return Err(Error::corrupt("WEIGHT_DELTA delta_root mismatch"));
    }

    let (base_bytes, supplied_frame_bytes) = match base_frame {
        Some(bytes) => (
            modes::decode_bytes(bytes, resolver, limits)?,
            Some(bytes.to_vec()),
        ),
        None => {
            let fetched = resolver.resolve_frame(&meta.base)?;
            let bytes =
                resolver.resolve(&meta.base, frame.header.decoded_len, DEP_DOMAIN_PAYLOAD)?;
            (bytes, fetched)
        }
    };
    if let Some(base_frame_bytes) = supplied_frame_bytes.as_deref() {
        let base_parsed = Frame::parse(base_frame_bytes, limits)?;
        if base_parsed.header.content_hash != meta.base {
            return Err(dependency_mismatch_base(
                "supplied base frame content hash does not equal the descriptor base",
            ));
        }
        let (base_dtype, base_shape) = base_frame_descriptor(&base_parsed, limits)
            .map_err(|err| dependency_mismatch_base(err.message()))?;
        if base_dtype != meta.dtype {
            return Err(dependency_mismatch_base(format!(
                "base frame dtype `{}` does not match WEIGHT_DELTA dtype `{}`",
                base_dtype.as_str(),
                meta.dtype.as_str()
            )));
        }
        if base_shape != meta.shape {
            return Err(dependency_mismatch_base(
                "base frame shape does not match the WEIGHT_DELTA shape",
            ));
        }
    }
    if base_bytes.len() as u64 != expected_len {
        return Err(dependency_mismatch_base(format!(
            "base frame decodes to {} bytes but the descriptor requires {expected_len}",
            base_bytes.len()
        )));
    }
    if meta.dtype.is_quantized() {
        check_quantized_words(meta.dtype, &base_bytes, "base")?;
    }

    let mut target = Vec::with_capacity(count.saturating_mul(meta.dtype.storage_bytes() as usize));
    for index in 0..count {
        let delta = delta_word_from_planes(meta.dtype, &planes, index)?;
        let base_word = word_at(meta.dtype, &base_bytes, index);
        let word = target_word(meta.dtype, delta, base_word);
        push_word(meta.dtype, word, &mut target);
    }
    if target.len() as u64 != expected_len {
        return Err(Error::corrupt(
            "WEIGHT_DELTA reconstructed element length is inconsistent",
        ));
    }
    if hash_domain(DOMAIN_CHUNK, &target) != frame.header.content_hash {
        return Err(Error::corrupt(
            "WEIGHT_DELTA content_hash mismatch against the reconstructed tensor",
        ));
    }
    Ok(target)
}
