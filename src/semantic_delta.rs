//! Mode 14 `SEMANTIC_DELTA` (WEFT-AI-CODEC §15).
//!
//! SEMANTIC_DELTA has exactly the WEIGHT_DELTA reconstruction scheme — modular
//! element-wise subtraction on raw words, one `huffman-canonical-v1` stream per
//! byte plane — but its descriptor binds the *semantic root* under which the
//! base was selected:
//!
//! ```text
//! base_semantic_root = H("weft/tensor/semantic/v1", WCB(weft.tensor.semantic.v1))
//! ```
//!
//! Decoding recomputes that root from the resolved base bytes and the
//! descriptor policy. It is a verification input, never a substitute for the
//! base content hash, dtype, shape, delta root, or reconstructed content hash.
//! No lossy reconstruction exists: a descriptor mismatch fails closed.

use std::collections::BTreeMap;

use crate::dependency::Resolver;
use crate::error::{Error, Result};
use crate::frame::{
    encode_frame, Dependency, Frame, FrameParts, Mode, DEP_DOMAIN_PAYLOAD, DEP_FLAG_REQUIRED,
    DOMAIN_PAYLOAD,
};
use crate::hash::{hash_domain, Hash32, DOMAIN_CHUNK, HASH_LEN};
use crate::huffman;
use crate::limits::{check_len, Limits};
use crate::modes;
use crate::semantic::{SemanticPolicy, SemanticRootRecord};
use crate::tensor::DOMAIN_TENSOR_VALUES;
use crate::wcb::{self, Value};
use crate::weight_delta::{
    base_frame_descriptor, build_planes, check_quantized_words, decode_planes,
    delta_word_from_planes, payload_len, push_word, split_ai_body, validate_shape, word_at,
    WeightDtype, WEIGHT_DELTA_PLANE_CODEC,
};

/// Canonical descriptor schema for mode 14.
pub const SEMANTIC_DELTA_SCHEMA: &str = "weft.ai.semantic_delta.v1";

/// The descriptor of one SEMANTIC_DELTA frame.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SemanticDeltaMeta {
    pub dtype: WeightDtype,
    pub shape: Vec<u64>,
    pub element_count: u64,
    pub word_bits: u64,
    pub base: Hash32,
    pub base_dtype: WeightDtype,
    pub base_shape: Vec<u64>,
    pub base_semantic_root: Hash32,
    pub policy: SemanticPolicy,
    pub delta_root: Hash32,
    pub plane_codec: String,
    pub payload_len: u64,
}

fn meta_text(map: &BTreeMap<String, Value>, key: &str) -> Result<String> {
    match map.get(key) {
        Some(Value::Str(text)) => Ok(text.clone()),
        _ => Err(Error::malformed(format!(
            "SEMANTIC_DELTA metadata `{key}` must be text"
        ))),
    }
}

fn meta_u64(map: &BTreeMap<String, Value>, key: &str) -> Result<u64> {
    match map.get(key) {
        Some(Value::U64(value)) => Ok(*value),
        _ => Err(Error::malformed(format!(
            "SEMANTIC_DELTA metadata `{key}` must be u64"
        ))),
    }
}

fn meta_bytes(map: &BTreeMap<String, Value>, key: &str) -> Result<Vec<u8>> {
    match map.get(key) {
        Some(Value::Bytes(bytes)) => Ok(bytes.clone()),
        _ => Err(Error::malformed(format!(
            "SEMANTIC_DELTA metadata `{key}` must be bytes"
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
                            "SEMANTIC_DELTA metadata `{key}` elements must be u64"
                        )))
                    }
                }
            }
            Ok(shape)
        }
        _ => Err(Error::malformed(format!(
            "SEMANTIC_DELTA metadata `{key}` must be an array"
        ))),
    }
}

fn hash_field(map: &BTreeMap<String, Value>, key: &str) -> Result<Hash32> {
    let bytes = meta_bytes(map, key)?;
    if bytes.len() != HASH_LEN {
        return Err(Error::malformed(format!(
            "SEMANTIC_DELTA metadata `{key}` must be {HASH_LEN} bytes, got {}",
            bytes.len()
        )));
    }
    Hash32::from_slice(&bytes)
}

/// Validate every descriptor invariant and return the element count.
pub fn validate_meta(meta: &SemanticDeltaMeta, limits: &Limits) -> Result<u64> {
    if meta.plane_codec != WEIGHT_DELTA_PLANE_CODEC {
        return Err(Error::malformed(format!(
            "SEMANTIC_DELTA plane_codec `{}` is not implemented (only {WEIGHT_DELTA_PLANE_CODEC})",
            meta.plane_codec
        )));
    }
    if meta.dtype.word_bits() != meta.word_bits {
        return Err(Error::malformed(format!(
            "SEMANTIC_DELTA word_bits {} disagrees with dtype {} ({} bits)",
            meta.word_bits,
            meta.dtype.as_str(),
            meta.dtype.word_bits()
        )));
    }
    if meta.base_dtype != meta.dtype {
        return Err(Error::malformed(format!(
            "SEMANTIC_DELTA base_dtype `{}` must equal dtype `{}`",
            meta.base_dtype.as_str(),
            meta.dtype.as_str()
        )));
    }
    if meta.base_shape != meta.shape {
        return Err(Error::malformed(
            "SEMANTIC_DELTA base_shape must equal shape",
        ));
    }
    let count = validate_shape(&meta.shape, limits)?;
    if meta.element_count != count {
        return Err(Error::malformed(format!(
            "SEMANTIC_DELTA element_count {} does not match shape product {count}",
            meta.element_count
        )));
    }
    payload_len(count, meta.dtype)?;
    meta.policy.validate(meta.dtype, count)?;
    Ok(count)
}

/// Build the canonical WCB descriptor value.
pub fn descriptor_value(meta: &SemanticDeltaMeta) -> Value {
    Value::map([
        ("schema", Value::Str(SEMANTIC_DELTA_SCHEMA.to_owned())),
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
        (
            "base_semantic_root",
            Value::Bytes(meta.base_semantic_root.as_bytes().to_vec()),
        ),
        ("policy", meta.policy.to_wcb_value()),
        (
            "delta_root",
            Value::Bytes(meta.delta_root.as_bytes().to_vec()),
        ),
        ("plane_codec", Value::Str(meta.plane_codec.clone())),
        ("payload_len", Value::U64(meta.payload_len)),
    ])
}

fn parse_descriptor(bytes: &[u8], limits: &Limits) -> Result<SemanticDeltaMeta> {
    let value = wcb::decode(bytes, limits)?;
    let map = match value {
        Value::Map(map) => map,
        _ => {
            return Err(Error::malformed(
                "SEMANTIC_DELTA metadata must be a WCB map",
            ))
        }
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
        "base_semantic_root",
        "policy",
        "delta_root",
        "plane_codec",
        "payload_len",
    ];
    for key in map.keys() {
        if !KEYS.contains(&key.as_str()) {
            return Err(Error::malformed(format!(
                "SEMANTIC_DELTA metadata has unknown key `{key}`"
            )));
        }
    }
    for key in KEYS {
        if !map.contains_key(key) {
            return Err(Error::malformed(format!(
                "SEMANTIC_DELTA metadata is missing key `{key}`"
            )));
        }
    }
    let schema = meta_text(&map, "schema")?;
    if schema != SEMANTIC_DELTA_SCHEMA {
        return Err(Error::malformed(format!(
            "unsupported SEMANTIC_DELTA schema `{schema}`"
        )));
    }
    let dtype = WeightDtype::from_name(&meta_text(&map, "dtype")?)?;
    let shape = meta_shape(&map, "shape")?;
    let element_count = meta_u64(&map, "element_count")?;
    let word_bits = meta_u64(&map, "word_bits")?;
    let base = hash_field(&map, "base")?;
    let base_dtype = WeightDtype::from_name(&meta_text(&map, "base_dtype")?)?;
    let base_shape = meta_shape(&map, "base_shape")?;
    let base_semantic_root = hash_field(&map, "base_semantic_root")?;
    let policy_value = map
        .get("policy")
        .ok_or_else(|| Error::malformed("SEMANTIC_DELTA metadata is missing key `policy`"))?;
    let policy = SemanticPolicy::from_wcb_value(policy_value, dtype, element_count)?;
    let delta_root = hash_field(&map, "delta_root")?;
    let plane_codec = meta_text(&map, "plane_codec")?;
    let payload_len = meta_u64(&map, "payload_len")?;
    let meta = SemanticDeltaMeta {
        dtype,
        shape,
        element_count,
        word_bits,
        base,
        base_dtype,
        base_shape,
        base_semantic_root,
        policy,
        delta_root,
        plane_codec,
        payload_len,
    };
    validate_meta(&meta, limits)?;
    Ok(meta)
}

/// Parse and validate a SEMANTIC_DELTA descriptor from a frame (no base
/// resolution).
pub fn semantic_delta_metadata(frame: &Frame<'_>, limits: &Limits) -> Result<SemanticDeltaMeta> {
    if frame.header.mode != Mode::SemanticDelta {
        return Err(Error::malformed(format!(
            "expected a SEMANTIC_DELTA frame, got mode {}",
            frame.header.mode.as_str()
        )));
    }
    let (descriptor, _) = split_ai_body(frame.body)?;
    parse_descriptor(descriptor, limits)
}

// ---------------------------------------------------------------------------
// Encoding
// ---------------------------------------------------------------------------

/// Build a frame from an explicit descriptor and stored payload.
///
/// This low-level constructor is shared with conformance vector generation. It
/// does not re-check descriptor semantics; the resulting frame is re-parsed, so
/// frame-level mode rules still apply.
pub fn encode_descriptor_frame(
    meta: &SemanticDeltaMeta,
    payload: &[u8],
    decoded_len: u64,
    content_hash: Hash32,
    base_mode: u32,
    limits: &Limits,
) -> Result<Vec<u8>> {
    let meta_bytes = wcb::encode(&descriptor_value(meta), limits)?;
    let meta_len = u32::try_from(meta_bytes.len())
        .map_err(|_| Error::limit_exceeded("SEMANTIC_DELTA metadata does not fit u32"))?;
    let capacity = 4usize
        .checked_add(meta_bytes.len())
        .and_then(|value| value.checked_add(payload.len()))
        .ok_or_else(|| Error::limit_exceeded("SEMANTIC_DELTA body length overflow"))?;
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
        mode: Mode::SemanticDelta,
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

/// The semantic root of a base frame's plaintext under `policy`.
pub fn base_semantic_root(
    base_frame: &Frame<'_>,
    base_bytes: &[u8],
    policy: &SemanticPolicy,
    limits: &Limits,
) -> Result<Hash32> {
    let (dtype, shape) = base_frame_descriptor(base_frame, limits)?;
    SemanticRootRecord::from_row_major(dtype, &shape, base_bytes, policy.clone(), limits)?
        .root(limits)
}

/// Encode a target tensor as a SEMANTIC_DELTA against `base_frame`.
pub fn encode_semantic_delta(
    target: &[u8],
    dtype: WeightDtype,
    shape: &[u64],
    base_frame: &[u8],
    policy: SemanticPolicy,
    limits: &Limits,
) -> Result<Vec<u8>> {
    let base_parsed = Frame::parse(base_frame, limits)?;
    let (base_dtype, base_shape) = base_frame_descriptor(&base_parsed, limits)?;
    if base_dtype != dtype {
        return Err(Error::malformed(format!(
            "base frame dtype `{}` does not match SEMANTIC_DELTA dtype `{}`",
            base_dtype.as_str(),
            dtype.as_str()
        )));
    }
    if base_shape != shape {
        return Err(Error::malformed(
            "base frame shape does not match the SEMANTIC_DELTA shape",
        ));
    }
    let element_count = validate_shape(shape, limits)?;
    let expected_len = payload_len(element_count, dtype)?;
    check_len(
        expected_len,
        limits.max_decoded_bytes,
        "SEMANTIC_DELTA element bytes",
    )?;
    if target.len() as u64 != expected_len {
        return Err(Error::malformed(format!(
            "SEMANTIC_DELTA target is {} bytes but shape {} x {} requires {expected_len}",
            target.len(),
            dtype.as_str(),
            element_count
        )));
    }
    policy.validate(dtype, element_count)?;
    let base_bytes =
        modes::decode_bytes(base_frame, &crate::dependency::MapResolver::new(), limits)?;
    if base_bytes.len() as u64 != expected_len {
        return Err(Error::dependency_mismatch(format!(
            "base frame decodes to {} bytes but the descriptor requires {expected_len}",
            base_bytes.len()
        )));
    }
    if dtype.is_quantized() {
        check_quantized_words(dtype, target, "target")?;
        check_quantized_words(dtype, &base_bytes, "base")?;
    }
    let base_root =
        SemanticRootRecord::from_row_major(dtype, shape, &base_bytes, policy.clone(), limits)?
            .root(limits)?;

    let count = usize::try_from(element_count)
        .map_err(|_| Error::limit_exceeded("SEMANTIC_DELTA element count does not fit usize"))?;
    let (delta_bytes, planes) = build_planes(dtype, target, &base_bytes, count);
    let delta_root = hash_domain(DOMAIN_TENSOR_VALUES, &delta_bytes);
    let mut payload = Vec::new();
    for plane in &planes {
        payload.extend_from_slice(&huffman::encode(plane));
    }

    let meta = SemanticDeltaMeta {
        dtype,
        shape: shape.to_vec(),
        element_count,
        word_bits: dtype.word_bits(),
        base: base_parsed.header.content_hash,
        base_dtype: dtype,
        base_shape: shape.to_vec(),
        base_semantic_root: base_root,
        policy,
        delta_root,
        plane_codec: WEIGHT_DELTA_PLANE_CODEC.to_owned(),
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

/// Decode a SEMANTIC_DELTA frame, resolving the base through `resolver`.
pub fn decode_frame(
    frame: &Frame<'_>,
    resolver: &dyn Resolver,
    limits: &Limits,
) -> Result<Vec<u8>> {
    decode_with_optional_base(frame, None, resolver, limits)
}

/// Decode a SEMANTIC_DELTA frame with the base frame supplied out of band.
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
    if frame.header.mode != Mode::SemanticDelta {
        return Err(Error::malformed(format!(
            "expected a SEMANTIC_DELTA frame, got mode {}",
            frame.header.mode.as_str()
        )));
    }
    let (descriptor_bytes, payload) = split_ai_body(frame.body)?;
    let meta = parse_descriptor(descriptor_bytes, limits)?;
    let element_count = validate_meta(&meta, limits)?;
    let expected_len = payload_len(element_count, meta.dtype)?;
    if frame.header.decoded_len != expected_len {
        return Err(Error::corrupt(format!(
            "SEMANTIC_DELTA decoded_len is {} but the descriptor requires {expected_len}",
            frame.header.decoded_len
        )));
    }
    check_len(
        expected_len,
        limits.max_decoded_bytes,
        "SEMANTIC_DELTA element bytes",
    )?;
    if meta.payload_len != payload.len() as u64 {
        return Err(Error::malformed(format!(
            "SEMANTIC_DELTA payload is {} bytes but payload_len declares {}",
            payload.len(),
            meta.payload_len
        )));
    }
    if frame.header.dep_count != 1 {
        return Err(Error::malformed(
            "SEMANTIC_DELTA requires exactly one dependency",
        ));
    }
    let dep = &frame.dependencies[0];
    if dep.hash != meta.base {
        return Err(Error::malformed(
            "SEMANTIC_DELTA dependency hash must equal the descriptor base",
        ));
    }
    if dep.decoded_len != frame.header.decoded_len {
        return Err(Error::malformed(
            "SEMANTIC_DELTA dependency decoded_len must equal the frame decoded_len",
        ));
    }

    let count = usize::try_from(element_count)
        .map_err(|_| Error::limit_exceeded("SEMANTIC_DELTA element count does not fit usize"))?;
    let planes = decode_planes(meta.dtype, payload, count)?;

    let mut delta_bytes =
        Vec::with_capacity(count.saturating_mul(meta.dtype.storage_bytes() as usize));
    for index in 0..count {
        let delta = delta_word_from_planes(meta.dtype, &planes, index)?;
        push_word(meta.dtype, delta, &mut delta_bytes);
    }
    if hash_domain(DOMAIN_TENSOR_VALUES, &delta_bytes) != meta.delta_root {
        return Err(Error::corrupt("SEMANTIC_DELTA delta_root mismatch"));
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
            return Err(Error::dependency_mismatch(
                "supplied base frame content hash does not equal the descriptor base",
            ));
        }
        let (base_dtype, base_shape) = base_frame_descriptor(&base_parsed, limits)
            .map_err(|err| Error::dependency_mismatch(err.message()))?;
        if base_dtype != meta.dtype {
            return Err(Error::dependency_mismatch(format!(
                "base frame dtype `{}` does not match SEMANTIC_DELTA dtype `{}`",
                base_dtype.as_str(),
                meta.dtype.as_str()
            )));
        }
        if base_shape != meta.shape {
            return Err(Error::dependency_mismatch(
                "base frame shape does not match the SEMANTIC_DELTA shape",
            ));
        }
    }
    if base_bytes.len() as u64 != expected_len {
        return Err(Error::dependency_mismatch(format!(
            "base frame decodes to {} bytes but the descriptor requires {expected_len}",
            base_bytes.len()
        )));
    }
    if meta.dtype.is_quantized() {
        check_quantized_words(meta.dtype, &base_bytes, "base")?;
    }

    // The semantic root is a verification input: recompute it from the actual
    // base bytes under the declared policy. A mismatch fails closed.
    let recomputed = SemanticRootRecord::from_row_major(
        meta.dtype,
        &meta.shape,
        &base_bytes,
        meta.policy.clone(),
        limits,
    )
    .map_err(|err| Error::dependency_mismatch(err.message()))?
    .root(limits)?;
    if recomputed != meta.base_semantic_root {
        return Err(Error::dependency_mismatch(format!(
            "base semantic root {recomputed} under policy {} does not equal the descriptor {}{}",
            meta.policy.as_str(),
            meta.base_semantic_root,
            " (no approximate reconstruction exists)"
        )));
    }

    let mut target = Vec::with_capacity(count.saturating_mul(meta.dtype.storage_bytes() as usize));
    for index in 0..count {
        let delta = delta_word_from_planes(meta.dtype, &planes, index)?;
        let base_word = word_at(meta.dtype, &base_bytes, index);
        let word = delta.wrapping_add(base_word) & mask(meta.dtype);
        push_word(meta.dtype, word, &mut target);
    }
    if target.len() as u64 != expected_len {
        return Err(Error::corrupt(
            "SEMANTIC_DELTA reconstructed element length is inconsistent",
        ));
    }
    if hash_domain(DOMAIN_CHUNK, &target) != frame.header.content_hash {
        return Err(Error::corrupt(
            "SEMANTIC_DELTA content_hash mismatch against the reconstructed tensor",
        ));
    }
    Ok(target)
}

fn mask(dtype: WeightDtype) -> u32 {
    match dtype.word_bits() {
        32 => u32::MAX,
        16 => 0xFFFF,
        8 => 0xFF,
        4 => 0x0F,
        _ => unreachable!("word_bits is fixed by dtype"),
    }
}
