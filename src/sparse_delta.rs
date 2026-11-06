//! Mode 16 `SPARSE_DELTA` (WEFT-AI-CODEC §17).
//!
//! Mode 16 reconstructs a tensor from the same exact word-wise difference mode
//! 11 defines:
//!
//! ```text
//! d_i = (a_i - b_i) mod 2^word_bits
//! a_i = (d_i + b_i) mod 2^word_bits
//! ```
//!
//! The difference is the *payload*: where mode 11 spends two to four
//! `huffman-canonical-v1` planes per element regardless of how many delta words
//! are non-zero, mode 16 stores a run-length stream over the delta word array.
//! It is the sparse counterpart of mode 11, not a replacement: a dense delta is
//! larger under mode 16 (one varint plus one word per non-zero element) and the
//! planner therefore keeps mode 11 for dense pairs.
//!
//! Payload:
//!
//! ```text
//! u32le(run_count) || run_count records
//! record = varint(zero_run_len) || storage_bytes(dtype) bytes of the word
//! ```
//!
//! A record means "skip `zero_run_len` zero words, then emit this non-zero
//! word". Trailing zeros are implicit and are never stored, so an all-zero
//! delta is `run_count = 0` and reconstructs the base exactly. Reconstruction
//! from `run_count = 0` is a legal sparse encoding, not a `REFERENCE` alias:
//! the frame still carries its own descriptor, `delta_root`, and content hash.
//!
//! No floating-point arithmetic is performed anywhere. `a_i` and `b_i` are raw
//! element words, so NaNs, denormals, and signed zeros reconstruct bit-exactly.
//! The base is a dependency frame named by content hash; decoding verifies the
//! base descriptor's dtype/shape when the resolver can also supply the base
//! frame bytes.

use std::collections::BTreeMap;

use crate::dependency::{MapResolver, Resolver};
use crate::error::{Error, Result};
use crate::frame::{
    encode_frame, Dependency, Frame, FrameParts, Mode, DEP_DOMAIN_PAYLOAD, DEP_FLAG_REQUIRED,
    DOMAIN_PAYLOAD,
};
use crate::hash::{hash_domain, Hash32, DOMAIN_CHUNK, HASH_LEN};
use crate::limits::{check_len, to_usize, Limits};
use crate::modes;
use crate::tensor::DOMAIN_TENSOR_VALUES;
use crate::varint;
use crate::wcb::{self, Value};
use crate::weight_delta::{
    check_quantized_words, delta_word, payload_len, push_word, target_word, word_at, WeightDtype,
};

/// Canonical descriptor schema.
pub const SPARSE_DELTA_SCHEMA: &str = "weft.ai.sparse_delta.v1";
/// The only delta operation implemented in v1.
pub const SPARSE_DELTA_OP: &str = "mod-sub";
/// The only sparse coder implemented in v1.
pub const SPARSE_DELTA_CODEC: &str = "zero-run-v1";

/// The descriptor of one SPARSE_DELTA frame.
///
/// Exactly the nine keys of `WEFT-AI-CODEC.md` §17.2; `run_count` is read from
/// the payload, not the descriptor, so a stored payload must be consumed
/// exactly (trailing bytes fail closed).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SparseDeltaMeta {
    pub dtype: WeightDtype,
    pub shape: Vec<u64>,
    pub element_count: u64,
    pub storage_bytes: u64,
    pub op: String,
    pub sparse_codec: String,
    /// The base frame's content hash (`H("weft/chunk/v1", base bytes)`).
    pub base: Hash32,
    pub delta_root: Hash32,
}

fn meta_text(map: &BTreeMap<String, Value>, key: &str) -> Result<String> {
    match map.get(key) {
        Some(Value::Str(text)) => Ok(text.clone()),
        _ => Err(Error::malformed(format!(
            "SPARSE_DELTA metadata `{key}` must be text"
        ))),
    }
}

fn meta_u64(map: &BTreeMap<String, Value>, key: &str) -> Result<u64> {
    match map.get(key) {
        Some(Value::U64(value)) => Ok(*value),
        _ => Err(Error::malformed(format!(
            "SPARSE_DELTA metadata `{key}` must be u64"
        ))),
    }
}

fn meta_bytes(map: &BTreeMap<String, Value>, key: &str) -> Result<Vec<u8>> {
    match map.get(key) {
        Some(Value::Bytes(bytes)) => Ok(bytes.clone()),
        _ => Err(Error::malformed(format!(
            "SPARSE_DELTA metadata `{key}` must be bytes"
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
                            "SPARSE_DELTA metadata `{key}` elements must be u64"
                        )))
                    }
                }
            }
            Ok(shape)
        }
        _ => Err(Error::malformed(format!(
            "SPARSE_DELTA metadata `{key}` must be an array"
        ))),
    }
}

fn hash_field(map: &BTreeMap<String, Value>, key: &str) -> Result<Hash32> {
    let bytes = meta_bytes(map, key)?;
    if bytes.len() != HASH_LEN {
        return Err(Error::malformed(format!(
            "SPARSE_DELTA metadata `{key}` must be {HASH_LEN} bytes, got {}",
            bytes.len()
        )));
    }
    Hash32::from_slice(&bytes)
}

/// Validate the shape and return the element count.
pub(crate) fn validate_shape(shape: &[u64], limits: &Limits) -> Result<u64> {
    if shape.is_empty() {
        return Err(Error::malformed(
            "SPARSE_DELTA shape rank must be at least 1",
        ));
    }
    if shape.len() as u64 > limits.max_tensor_rank {
        return Err(Error::limit_exceeded(format!(
            "SPARSE_DELTA rank {} exceeds limit {}",
            shape.len(),
            limits.max_tensor_rank
        )));
    }
    let mut product: u64 = 1;
    for dim in shape {
        if *dim == 0 {
            return Err(Error::malformed(
                "SPARSE_DELTA shape dimensions must be greater than zero",
            ));
        }
        product = product
            .checked_mul(*dim)
            .ok_or_else(|| Error::limit_exceeded("SPARSE_DELTA shape product overflows u64"))?;
    }
    if product > limits.max_tensor_elements {
        return Err(Error::limit_exceeded(format!(
            "SPARSE_DELTA element count {product} exceeds limit {}",
            limits.max_tensor_elements
        )));
    }
    Ok(product)
}

/// Validate every descriptor invariant and return the element count.
pub fn validate_meta(meta: &SparseDeltaMeta, limits: &Limits) -> Result<u64> {
    if meta.op != SPARSE_DELTA_OP {
        return Err(Error::malformed(format!(
            "SPARSE_DELTA op `{}` is not implemented (only {SPARSE_DELTA_OP})",
            meta.op
        )));
    }
    if meta.sparse_codec != SPARSE_DELTA_CODEC {
        return Err(Error::malformed(format!(
            "SPARSE_DELTA sparse_codec `{}` is not implemented (only {SPARSE_DELTA_CODEC})",
            meta.sparse_codec
        )));
    }
    if meta.storage_bytes != meta.dtype.storage_bytes() {
        return Err(Error::malformed(format!(
            "SPARSE_DELTA storage_bytes {} disagrees with dtype {} ({} bytes)",
            meta.storage_bytes,
            meta.dtype.as_str(),
            meta.dtype.storage_bytes()
        )));
    }
    let count = validate_shape(&meta.shape, limits)?;
    if meta.element_count != count {
        return Err(Error::malformed(format!(
            "SPARSE_DELTA element_count {} does not match shape product {count}",
            meta.element_count
        )));
    }
    payload_len(count, meta.dtype)?;
    Ok(count)
}

/// Build the canonical WCB descriptor value.
pub fn descriptor_value(meta: &SparseDeltaMeta) -> Value {
    Value::map([
        ("schema", Value::Str(SPARSE_DELTA_SCHEMA.to_owned())),
        ("dtype", Value::Str(meta.dtype.as_str().to_owned())),
        (
            "shape",
            Value::Array(meta.shape.iter().map(|dim| Value::U64(*dim)).collect()),
        ),
        ("element_count", Value::U64(meta.element_count)),
        ("storage_bytes", Value::U64(meta.storage_bytes)),
        ("op", Value::Str(meta.op.clone())),
        ("sparse_codec", Value::Str(meta.sparse_codec.clone())),
        (
            "base_content_root",
            Value::Bytes(meta.base.as_bytes().to_vec()),
        ),
        (
            "delta_root",
            Value::Bytes(meta.delta_root.as_bytes().to_vec()),
        ),
    ])
}

pub(crate) fn split_ai_body(body: &[u8]) -> Result<(&[u8], &[u8])> {
    if body.len() < 4 {
        return Err(Error::malformed(
            "SPARSE_DELTA body is shorter than its 4-byte metadata length",
        ));
    }
    let meta_len = u64::from(u32::from_le_bytes([body[0], body[1], body[2], body[3]]));
    let end = 4u64
        .checked_add(meta_len)
        .ok_or_else(|| Error::malformed("SPARSE_DELTA metadata length overflows"))?;
    if end > body.len() as u64 {
        return Err(Error::malformed(
            "SPARSE_DELTA metadata extends past the frame body",
        ));
    }
    let end = end as usize;
    Ok((&body[4..end], &body[end..]))
}

fn parse_descriptor(bytes: &[u8], limits: &Limits) -> Result<SparseDeltaMeta> {
    let value = wcb::decode(bytes, limits)?;
    let map = match value {
        Value::Map(map) => map,
        _ => return Err(Error::malformed("SPARSE_DELTA metadata must be a WCB map")),
    };
    const KEYS: [&str; 9] = [
        "schema",
        "dtype",
        "shape",
        "element_count",
        "storage_bytes",
        "op",
        "sparse_codec",
        "base_content_root",
        "delta_root",
    ];
    for key in map.keys() {
        if !KEYS.contains(&key.as_str()) {
            return Err(Error::malformed(format!(
                "SPARSE_DELTA metadata has unknown key `{key}`"
            )));
        }
    }
    for key in KEYS {
        if !map.contains_key(key) {
            return Err(Error::malformed(format!(
                "SPARSE_DELTA metadata is missing key `{key}`"
            )));
        }
    }
    let schema = meta_text(&map, "schema")?;
    if schema != SPARSE_DELTA_SCHEMA {
        return Err(Error::malformed(format!(
            "unsupported SPARSE_DELTA schema `{schema}`"
        )));
    }
    let dtype = WeightDtype::from_name(&meta_text(&map, "dtype")?)?;
    let shape = meta_shape(&map, "shape")?;
    let element_count = meta_u64(&map, "element_count")?;
    let storage_bytes = meta_u64(&map, "storage_bytes")?;
    let op = meta_text(&map, "op")?;
    let sparse_codec = meta_text(&map, "sparse_codec")?;
    let base = hash_field(&map, "base_content_root")?;
    let delta_root = hash_field(&map, "delta_root")?;
    let meta = SparseDeltaMeta {
        dtype,
        shape,
        element_count,
        storage_bytes,
        op,
        sparse_codec,
        base,
        delta_root,
    };
    validate_meta(&meta, limits)?;
    Ok(meta)
}

/// Parse and validate a SPARSE_DELTA descriptor from a frame (no base
/// resolution).
pub fn sparse_delta_metadata(frame: &Frame<'_>, limits: &Limits) -> Result<SparseDeltaMeta> {
    if frame.header.mode != Mode::SparseDelta {
        return Err(Error::malformed(format!(
            "expected a SPARSE_DELTA frame, got mode {}",
            frame.header.mode.as_str()
        )));
    }
    let (descriptor, _) = split_ai_body(frame.body)?;
    parse_descriptor(descriptor, limits)
}

// ---------------------------------------------------------------------------
// Zero-run stream
// ---------------------------------------------------------------------------

/// Encode the delta words as a `zero-run-v1` stream.
///
/// Returns the canonical delta word bytes (the `delta_root` input) and the
/// stored payload. Both implementations emit the same bytes: records are
/// emitted in element order, trailing zeros are never stored, and each varint
/// is minimal.
pub(crate) fn build_runs(
    dtype: WeightDtype,
    target: &[u8],
    base: &[u8],
    element_count: usize,
    limits: &Limits,
) -> Result<(Vec<u8>, Vec<u8>)> {
    let storage = to_usize(dtype.storage_bytes(), "SPARSE_DELTA storage bytes")?;
    let mut delta_bytes = Vec::with_capacity(element_count.saturating_mul(storage));
    let mut run_count: u64 = 0;
    for index in 0..element_count {
        let delta = delta_word(
            dtype,
            word_at(dtype, target, index),
            word_at(dtype, base, index),
        );
        push_word(dtype, delta, &mut delta_bytes);
        if delta != 0 {
            run_count += 1;
        }
    }
    check_len(
        run_count,
        limits.max_sparse_runs,
        "SPARSE_DELTA non-zero run count",
    )?;
    let run_count_u32 = u32::try_from(run_count)
        .map_err(|_| Error::limit_exceeded("SPARSE_DELTA run_count does not fit u32"))?;
    let mut payload =
        Vec::with_capacity(4 + usize::try_from(run_count).unwrap_or(0) * (storage + 1));
    payload.extend_from_slice(&run_count_u32.to_le_bytes());
    let mut zero_run: u64 = 0;
    for index in 0..element_count {
        let start = index * storage;
        let word = &delta_bytes[start..start + storage];
        if word.iter().all(|byte| *byte == 0) {
            zero_run += 1;
            continue;
        }
        varint::encode(zero_run, &mut payload);
        payload.extend_from_slice(word);
        zero_run = 0;
    }
    // Trailing zeros are implicit; they are deliberately not stored.
    Ok((delta_bytes, payload))
}

/// Decode a `zero-run-v1` payload into the canonical delta word bytes.
pub(crate) fn decode_runs(
    dtype: WeightDtype,
    payload: &[u8],
    element_count: usize,
    limits: &Limits,
) -> Result<Vec<u8>> {
    if payload.len() < 4 {
        return Err(Error::malformed(
            "SPARSE_DELTA body is shorter than its 4-byte run_count",
        ));
    }
    let run_count = u64::from(u32::from_le_bytes([
        payload[0], payload[1], payload[2], payload[3],
    ]));
    check_len(run_count, limits.max_sparse_runs, "SPARSE_DELTA run_count")?;
    if run_count > element_count as u64 {
        return Err(Error::malformed(format!(
            "SPARSE_DELTA run_count {run_count} exceeds element_count {element_count}"
        )));
    }
    let storage = to_usize(dtype.storage_bytes(), "SPARSE_DELTA storage bytes")?;
    let delta_len = element_count
        .checked_mul(storage)
        .ok_or_else(|| Error::limit_exceeded("SPARSE_DELTA delta bytes overflow"))?;
    let mut delta_bytes = vec![0u8; delta_len];
    let mut pos = 4usize;
    let mut emitted: u64 = 0;
    for record in 0..run_count {
        let (zero_run, next) = varint::decode_at(payload, pos)?;
        pos = next;
        emitted = emitted
            .checked_add(zero_run)
            .ok_or_else(|| Error::limit_exceeded("SPARSE_DELTA zero-run length overflows u64"))?;
        if emitted >= element_count as u64 {
            return Err(Error::malformed(format!(
                "SPARSE_DELTA run {record} skips {zero_run} words past element_count {element_count}"
            )));
        }
        let end = pos
            .checked_add(storage)
            .ok_or_else(|| Error::limit_exceeded("SPARSE_DELTA run offset overflows"))?;
        if end > payload.len() {
            return Err(Error::malformed(format!(
                "SPARSE_DELTA run {record} record is truncated"
            )));
        }
        let word = &payload[pos..end];
        if word.iter().all(|byte| *byte == 0) {
            return Err(Error::corrupt(format!(
                "SPARSE_DELTA run {record} stores a zero word (non-canonical: zeros are implicit)"
            )));
        }
        if dtype == WeightDtype::Q4_0 && word[0] > 15 {
            return Err(Error::corrupt(format!(
                "SPARSE_DELTA q4_0 run {record} word is {}, outside 0..=15",
                word[0]
            )));
        }
        let at = to_usize(emitted, "SPARSE_DELTA word index")? * storage;
        delta_bytes[at..at + storage].copy_from_slice(word);
        pos = end;
        emitted += 1;
    }
    if pos != payload.len() {
        return Err(Error::corrupt(format!(
            "SPARSE_DELTA payload has {} trailing bytes after the {run_count} declared runs",
            payload.len() - pos
        )));
    }
    Ok(delta_bytes)
}

// ---------------------------------------------------------------------------
// Encoding
// ---------------------------------------------------------------------------

/// Build a frame from an explicit descriptor and stored payload.
///
/// This is the low-level constructor shared by the encoder and by adversarial
/// conformance-vector generation. It does not re-check descriptor semantics;
/// the resulting frame is re-parsed, so frame-level mode rules still apply.
pub fn encode_descriptor_frame(
    meta: &SparseDeltaMeta,
    payload: &[u8],
    decoded_len: u64,
    content_hash: Hash32,
    base_mode: u32,
    limits: &Limits,
) -> Result<Vec<u8>> {
    let meta_bytes = wcb::encode(&descriptor_value(meta), limits)?;
    let meta_len = u32::try_from(meta_bytes.len())
        .map_err(|_| Error::limit_exceeded("SPARSE_DELTA metadata does not fit u32"))?;
    let capacity = 4usize
        .checked_add(meta_bytes.len())
        .and_then(|value| value.checked_add(payload.len()))
        .ok_or_else(|| Error::limit_exceeded("SPARSE_DELTA body length overflow"))?;
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
        mode: Mode::SparseDelta,
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

/// Encode a target tensor as a SPARSE_DELTA against `base_frame`.
pub fn encode_sparse_delta(
    target: &[u8],
    dtype: WeightDtype,
    shape: &[u64],
    base_frame: &[u8],
    limits: &Limits,
) -> Result<Vec<u8>> {
    encode_sparse_delta_with_resolver(
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
pub fn encode_sparse_delta_with_resolver(
    target: &[u8],
    dtype: WeightDtype,
    shape: &[u64],
    base_frame: &[u8],
    resolver: &dyn Resolver,
    limits: &Limits,
) -> Result<Vec<u8>> {
    let base_parsed = Frame::parse(base_frame, limits)?;
    let (base_dtype, base_shape) =
        crate::weight_delta::base_frame_descriptor(&base_parsed, limits)?;
    if base_dtype != dtype {
        return Err(Error::malformed(format!(
            "base frame dtype `{}` does not match SPARSE_DELTA dtype `{}`",
            base_dtype.as_str(),
            dtype.as_str()
        )));
    }
    if base_shape != shape {
        return Err(Error::malformed(
            "base frame shape does not match the SPARSE_DELTA shape",
        ));
    }
    let element_count = validate_shape(shape, limits)?;
    let expected_len = payload_len(element_count, dtype)?;
    check_len(
        expected_len,
        limits.max_decoded_bytes,
        "SPARSE_DELTA element bytes",
    )?;
    if target.len() as u64 != expected_len {
        return Err(Error::malformed(format!(
            "SPARSE_DELTA target is {} bytes but shape {} x {} requires {expected_len}",
            target.len(),
            dtype.as_str(),
            element_count
        )));
    }
    let base_bytes = modes::decode_bytes(base_frame, resolver, limits)?;
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

    let count = to_usize(element_count, "SPARSE_DELTA element count")?;
    let (delta_bytes, payload) = build_runs(dtype, target, &base_bytes, count, limits)?;
    let delta_root = hash_domain(DOMAIN_TENSOR_VALUES, &delta_bytes);

    let meta = SparseDeltaMeta {
        dtype,
        shape: shape.to_vec(),
        element_count,
        storage_bytes: dtype.storage_bytes(),
        op: SPARSE_DELTA_OP.to_owned(),
        sparse_codec: SPARSE_DELTA_CODEC.to_owned(),
        base: base_parsed.header.content_hash,
        delta_root,
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

/// Decode a SPARSE_DELTA frame, resolving the base through `resolver`.
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

/// Decode a SPARSE_DELTA frame with the base frame supplied out of band.
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
    if frame.header.mode != Mode::SparseDelta {
        return Err(Error::malformed(format!(
            "expected a SPARSE_DELTA frame, got mode {}",
            frame.header.mode.as_str()
        )));
    }
    let (descriptor_bytes, payload) = split_ai_body(frame.body)?;
    let meta = parse_descriptor(descriptor_bytes, limits)?;
    let element_count = validate_meta(&meta, limits)?;
    let expected_len = payload_len(element_count, meta.dtype)?;
    if frame.header.decoded_len != expected_len {
        return Err(Error::corrupt(format!(
            "SPARSE_DELTA decoded_len is {} but the descriptor requires {expected_len}",
            frame.header.decoded_len
        )));
    }
    check_len(
        expected_len,
        limits.max_decoded_bytes,
        "SPARSE_DELTA element bytes",
    )?;
    if frame.header.dep_count != 1 {
        return Err(Error::malformed(
            "SPARSE_DELTA requires exactly one dependency",
        ));
    }
    let dep = &frame.dependencies[0];
    if dep.hash != meta.base {
        return Err(Error::malformed(
            "SPARSE_DELTA dependency hash must equal the descriptor base_content_root",
        ));
    }
    if dep.decoded_len != frame.header.decoded_len {
        return Err(Error::malformed(
            "SPARSE_DELTA dependency decoded_len must equal the frame decoded_len",
        ));
    }

    let count = to_usize(element_count, "SPARSE_DELTA element count")?;
    // Rebuild the canonical delta word stream and verify delta_root before
    // touching the base.
    let delta_bytes = decode_runs(meta.dtype, payload, count, limits)?;
    if hash_domain(DOMAIN_TENSOR_VALUES, &delta_bytes) != meta.delta_root {
        return Err(Error::corrupt("SPARSE_DELTA delta_root mismatch"));
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
        let (base_dtype, base_shape) =
            crate::weight_delta::base_frame_descriptor(&base_parsed, limits)
                .map_err(|err| Error::dependency_mismatch(err.message()))?;
        if base_dtype != meta.dtype {
            return Err(Error::dependency_mismatch(format!(
                "base frame dtype `{}` does not match SPARSE_DELTA dtype `{}`",
                base_dtype.as_str(),
                meta.dtype.as_str()
            )));
        }
        if base_shape != meta.shape {
            return Err(Error::dependency_mismatch(
                "base frame shape does not match the SPARSE_DELTA shape",
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

    let mut target = Vec::with_capacity(expected_len as usize);
    for index in 0..count {
        let delta = word_at(meta.dtype, &delta_bytes, index);
        let base_word = word_at(meta.dtype, &base_bytes, index);
        let word = target_word(meta.dtype, delta, base_word);
        push_word(meta.dtype, word, &mut target);
    }
    if target.len() as u64 != expected_len {
        return Err(Error::corrupt(
            "SPARSE_DELTA reconstructed element length is inconsistent",
        ));
    }
    if hash_domain(DOMAIN_CHUNK, &target) != frame.header.content_hash {
        return Err(Error::corrupt(
            "SPARSE_DELTA content_hash mismatch against the reconstructed tensor",
        ));
    }
    Ok(target)
}
