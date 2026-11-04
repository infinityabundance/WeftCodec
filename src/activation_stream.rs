//! Mode 13 `ACTIVATION_STREAM` (WEFT-AI-CODEC §14).
//!
//! A frame carries one chunk of an activation or gradient stream. Chunks are
//! independent: encoding chunk `k+1` never decodes chunk `k`, no frame depends
//! on another, and a decoder may materialize one chunk without its siblings.
//!
//! The descriptor carries the stream kind, step, rank/world coordinates, chunk
//! position, dtype, shape, element count, and the exact digest of the chunk's
//! canonical row-major element bytes. The payload is the same byte-plane
//! `huffman-canonical-v1` layout as mode 12.
//!
//! Chunks of one `(step, rank)` are assembled in `chunk_index` order.
//! [`assemble_stream`] rejects duplicate or missing indices, re-verifies each
//! chunk's `digest_root`, and returns the concatenated element bytes plus the
//! deterministic stream root
//!
//! ```text
//! H("weft/ai/stream/v1",
//!   WCB({stream, step, rank, world, chunk_count, digest_roots}))
//! ```

use std::collections::{BTreeMap, BTreeSet};

use crate::error::{Error, Result};
use crate::frame::{encode_frame, Frame, FrameParts, Mode, DOMAIN_PAYLOAD};
use crate::hash::{hash_domain, Hash32, DOMAIN_CHUNK, HASH_LEN};
use crate::huffman;
use crate::limits::{check_len, Limits};
use crate::tensor::DOMAIN_TENSOR_VALUES;
use crate::wcb::{self, Value};

/// Canonical descriptor schema.
pub const ACTIVATION_STREAM_SCHEMA: &str = "weft.ai.activation_stream.v1";
/// The only plane coder implemented in v1.
pub const ACTIVATION_STREAM_PLANE_CODEC: &str = huffman::CODER_ID;
/// Stream-root domain.
pub const DOMAIN_AI_STREAM: &str = "weft/ai/stream/v1";

/// Stream kinds implemented by mode 13.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StreamKind {
    Activation,
    Gradient,
}

impl StreamKind {
    /// Parse the canonical stream name.
    pub fn from_name(name: &str) -> Result<Self> {
        match name {
            "activation" => Ok(StreamKind::Activation),
            "gradient" => Ok(StreamKind::Gradient),
            other => Err(Error::malformed(format!(
                "unknown ACTIVATION_STREAM stream `{other}` (v1 allows activation, gradient)"
            ))),
        }
    }

    /// Canonical stream name.
    pub fn as_str(self) -> &'static str {
        match self {
            StreamKind::Activation => "activation",
            StreamKind::Gradient => "gradient",
        }
    }
}

/// Element dtypes implemented by mode 13.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StreamDtype {
    Bf16,
    Fp16,
    Fp32,
}

impl StreamDtype {
    /// Parse the canonical dtype name.
    pub fn from_name(name: &str) -> Result<Self> {
        match name {
            "bf16" => Ok(StreamDtype::Bf16),
            "fp16" => Ok(StreamDtype::Fp16),
            "fp32" => Ok(StreamDtype::Fp32),
            other => Err(Error::malformed(format!(
                "unknown ACTIVATION_STREAM dtype `{other}` (v1 allows bf16, fp16, fp32)"
            ))),
        }
    }

    /// Canonical dtype name.
    pub fn as_str(self) -> &'static str {
        match self {
            StreamDtype::Bf16 => "bf16",
            StreamDtype::Fp16 => "fp16",
            StreamDtype::Fp32 => "fp32",
        }
    }

    /// Bytes per element.
    pub fn word_bytes(self) -> u64 {
        match self {
            StreamDtype::Bf16 | StreamDtype::Fp16 => 2,
            StreamDtype::Fp32 => 4,
        }
    }

    /// Number of byte planes.
    pub fn planes(self) -> u64 {
        match self {
            StreamDtype::Bf16 | StreamDtype::Fp16 => 2,
            StreamDtype::Fp32 => 4,
        }
    }
}

/// The descriptor of one activation chunk.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ActivationStreamMeta {
    pub stream: StreamKind,
    pub step: u64,
    pub rank: u32,
    pub world: u32,
    pub chunk_index: u64,
    pub chunk_count: u64,
    pub dtype: StreamDtype,
    pub shape: Vec<u64>,
    pub element_count: u64,
    pub digest_root: Hash32,
    pub plane_codec: String,
    pub payload_len: u64,
}

/// The stream-wide head shared by every chunk of one `(step, rank)`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ActivationStreamHead {
    pub stream: StreamKind,
    pub step: u64,
    pub rank: u32,
    pub world: u32,
    pub chunk_count: u64,
    pub dtype: StreamDtype,
    pub shape: Vec<u64>,
}

/// A decoded activation chunk.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DecodedActivationChunk {
    pub meta: ActivationStreamMeta,
    pub bytes: Vec<u8>,
}

/// The result of assembling one stream.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ActivationStreamAssembly {
    pub stream: StreamKind,
    pub step: u64,
    pub rank: u32,
    pub world: u32,
    pub chunk_count: u64,
    pub dtype: StreamDtype,
    pub bytes: Vec<u8>,
    pub stream_root: Hash32,
    pub chunk_digests: Vec<Hash32>,
    /// Number of chunks decoded to produce this assembly.
    pub chunks_decoded: u64,
}

// ---------------------------------------------------------------------------
// Descriptor helpers
// ---------------------------------------------------------------------------

fn meta_text(map: &BTreeMap<String, Value>, key: &str) -> Result<String> {
    match map.get(key) {
        Some(Value::Str(text)) => Ok(text.clone()),
        _ => Err(Error::malformed(format!(
            "ACTIVATION_STREAM metadata `{key}` must be text"
        ))),
    }
}

fn meta_u64(map: &BTreeMap<String, Value>, key: &str) -> Result<u64> {
    match map.get(key) {
        Some(Value::U64(value)) => Ok(*value),
        _ => Err(Error::malformed(format!(
            "ACTIVATION_STREAM metadata `{key}` must be u64"
        ))),
    }
}

fn meta_u32(map: &BTreeMap<String, Value>, key: &str) -> Result<u32> {
    let value = meta_u64(map, key)?;
    u32::try_from(value).map_err(|_| {
        Error::malformed(format!(
            "ACTIVATION_STREAM metadata `{key}` = {value} exceeds u32"
        ))
    })
}

fn meta_shape(map: &BTreeMap<String, Value>) -> Result<Vec<u64>> {
    match map.get("shape") {
        Some(Value::Array(items)) => {
            let mut shape = Vec::with_capacity(items.len());
            for item in items {
                match item {
                    Value::U64(value) => shape.push(*value),
                    _ => {
                        return Err(Error::malformed(
                            "ACTIVATION_STREAM metadata `shape` elements must be u64",
                        ))
                    }
                }
            }
            Ok(shape)
        }
        _ => Err(Error::malformed(
            "ACTIVATION_STREAM metadata `shape` must be an array",
        )),
    }
}

fn meta_hash(map: &BTreeMap<String, Value>, key: &str) -> Result<Hash32> {
    match map.get(key) {
        Some(Value::Bytes(bytes)) if bytes.len() == HASH_LEN => Hash32::from_slice(bytes),
        Some(Value::Bytes(bytes)) => Err(Error::malformed(format!(
            "ACTIVATION_STREAM metadata `{key}` must be {HASH_LEN} bytes, got {}",
            bytes.len()
        ))),
        _ => Err(Error::malformed(format!(
            "ACTIVATION_STREAM metadata `{key}` must be bytes"
        ))),
    }
}

fn exact_keys(map: &BTreeMap<String, Value>, expected: &[&str], what: &str) -> Result<()> {
    for key in map.keys() {
        if !expected.contains(&key.as_str()) {
            return Err(Error::malformed(format!(
                "{what} metadata has unknown key `{key}`"
            )));
        }
    }
    for key in expected {
        if !map.contains_key(*key) {
            return Err(Error::malformed(format!(
                "{what} metadata is missing key `{key}`"
            )));
        }
    }
    Ok(())
}

fn shape_product(shape: &[u64], limits: &Limits) -> Result<u64> {
    if shape.is_empty() {
        return Err(Error::malformed(
            "ACTIVATION_STREAM shape rank must be at least 1",
        ));
    }
    if shape.len() as u64 > limits.max_tensor_rank {
        return Err(Error::limit_exceeded(format!(
            "ACTIVATION_STREAM rank {} exceeds limit {}",
            shape.len(),
            limits.max_tensor_rank
        )));
    }
    let mut product: u64 = 1;
    for dim in shape {
        if *dim == 0 {
            return Err(Error::malformed(
                "ACTIVATION_STREAM shape dimensions must be greater than zero",
            ));
        }
        product = product.checked_mul(*dim).ok_or_else(|| {
            Error::limit_exceeded("ACTIVATION_STREAM shape product overflows u64")
        })?;
    }
    if product > limits.max_tensor_elements {
        return Err(Error::limit_exceeded(format!(
            "ACTIVATION_STREAM element count {product} exceeds limit {}",
            limits.max_tensor_elements
        )));
    }
    Ok(product)
}

fn validate_head(head: &ActivationStreamHead, limits: &Limits) -> Result<u64> {
    if head.world == 0 {
        return Err(Error::malformed(
            "ACTIVATION_STREAM world must be at least 1",
        ));
    }
    if head.rank >= head.world {
        return Err(Error::malformed(format!(
            "ACTIVATION_STREAM rank {} must be below world {}",
            head.rank, head.world
        )));
    }
    if head.chunk_count == 0 {
        return Err(Error::malformed(
            "ACTIVATION_STREAM chunk_count must be at least 1",
        ));
    }
    let element_count = shape_product(&head.shape, limits)?;
    let bytes = element_count
        .checked_mul(head.dtype.word_bytes())
        .ok_or_else(|| Error::limit_exceeded("ACTIVATION_STREAM chunk byte count overflows u64"))?;
    check_len(bytes, limits.max_decoded_bytes, "ACTIVATION_STREAM chunk")?;
    Ok(bytes)
}

/// Validate the descriptor, returning the exact chunk byte length.
pub fn validate_meta(meta: &ActivationStreamMeta, limits: &Limits) -> Result<u64> {
    if meta.plane_codec != ACTIVATION_STREAM_PLANE_CODEC {
        return Err(Error::malformed(format!(
            "ACTIVATION_STREAM plane_codec `{}` is not implemented (only {ACTIVATION_STREAM_PLANE_CODEC})",
            meta.plane_codec
        )));
    }
    let head = ActivationStreamHead {
        stream: meta.stream,
        step: meta.step,
        rank: meta.rank,
        world: meta.world,
        chunk_count: meta.chunk_count,
        dtype: meta.dtype,
        shape: meta.shape.clone(),
    };
    let bytes = validate_head(&head, limits)?;
    if meta.chunk_index >= meta.chunk_count {
        return Err(Error::malformed(format!(
            "ACTIVATION_STREAM chunk_index {} is not below chunk_count {}",
            meta.chunk_index, meta.chunk_count
        )));
    }
    let product = shape_product(&meta.shape, limits)?;
    if meta.element_count != product {
        return Err(Error::malformed(format!(
            "ACTIVATION_STREAM element_count {} does not match shape product {product}",
            meta.element_count
        )));
    }
    Ok(bytes)
}

/// Build the canonical WCB descriptor value.
pub fn descriptor_value(meta: &ActivationStreamMeta) -> Value {
    Value::map([
        ("schema", Value::Str(ACTIVATION_STREAM_SCHEMA.to_owned())),
        ("stream", Value::Str(meta.stream.as_str().to_owned())),
        ("step", Value::U64(meta.step)),
        ("rank", Value::U64(u64::from(meta.rank))),
        ("world", Value::U64(u64::from(meta.world))),
        ("chunk_index", Value::U64(meta.chunk_index)),
        ("chunk_count", Value::U64(meta.chunk_count)),
        ("dtype", Value::Str(meta.dtype.as_str().to_owned())),
        (
            "shape",
            Value::Array(meta.shape.iter().map(|dim| Value::U64(*dim)).collect()),
        ),
        ("element_count", Value::U64(meta.element_count)),
        (
            "digest_root",
            Value::Bytes(meta.digest_root.as_bytes().to_vec()),
        ),
        ("plane_codec", Value::Str(meta.plane_codec.clone())),
        ("payload_len", Value::U64(meta.payload_len)),
    ])
}

fn split_ai_body(body: &[u8]) -> Result<(&[u8], &[u8])> {
    if body.len() < 4 {
        return Err(Error::malformed(
            "ACTIVATION_STREAM body is shorter than its 4-byte metadata length",
        ));
    }
    let meta_len = u64::from(u32::from_le_bytes([body[0], body[1], body[2], body[3]]));
    let end = 4u64
        .checked_add(meta_len)
        .ok_or_else(|| Error::malformed("ACTIVATION_STREAM metadata length overflows"))?;
    if end > body.len() as u64 {
        return Err(Error::malformed(
            "ACTIVATION_STREAM metadata extends past the frame body",
        ));
    }
    let end = end as usize;
    Ok((&body[4..end], &body[end..]))
}

fn parse_descriptor(bytes: &[u8], limits: &Limits) -> Result<ActivationStreamMeta> {
    let value = wcb::decode(bytes, limits)?;
    let map = match value {
        Value::Map(map) => map,
        _ => {
            return Err(Error::malformed(
                "ACTIVATION_STREAM metadata must be a WCB map",
            ))
        }
    };
    const KEYS: [&str; 13] = [
        "schema",
        "stream",
        "step",
        "rank",
        "world",
        "chunk_index",
        "chunk_count",
        "dtype",
        "shape",
        "element_count",
        "digest_root",
        "plane_codec",
        "payload_len",
    ];
    exact_keys(&map, &KEYS, "ACTIVATION_STREAM")?;
    let schema = meta_text(&map, "schema")?;
    if schema != ACTIVATION_STREAM_SCHEMA {
        return Err(Error::malformed(format!(
            "unsupported ACTIVATION_STREAM schema `{schema}`"
        )));
    }
    let meta = ActivationStreamMeta {
        stream: StreamKind::from_name(&meta_text(&map, "stream")?)?,
        step: meta_u64(&map, "step")?,
        rank: meta_u32(&map, "rank")?,
        world: meta_u32(&map, "world")?,
        chunk_index: meta_u64(&map, "chunk_index")?,
        chunk_count: meta_u64(&map, "chunk_count")?,
        dtype: StreamDtype::from_name(&meta_text(&map, "dtype")?)?,
        shape: meta_shape(&map)?,
        element_count: meta_u64(&map, "element_count")?,
        digest_root: meta_hash(&map, "digest_root")?,
        plane_codec: meta_text(&map, "plane_codec")?,
        payload_len: meta_u64(&map, "payload_len")?,
    };
    validate_meta(&meta, limits)?;
    Ok(meta)
}

/// Parse and validate an ACTIVATION_STREAM descriptor from a frame.
pub fn activation_chunk_metadata(
    frame: &Frame<'_>,
    limits: &Limits,
) -> Result<ActivationStreamMeta> {
    if frame.header.mode != Mode::ActivationStream {
        return Err(Error::malformed(format!(
            "expected an ACTIVATION_STREAM frame, got mode {}",
            frame.header.mode.as_str()
        )));
    }
    let (descriptor, _) = split_ai_body(frame.body)?;
    parse_descriptor(descriptor, limits)
}

// ---------------------------------------------------------------------------
// Byte planes
// ---------------------------------------------------------------------------

fn element_word(dtype: StreamDtype, bytes: &[u8], index: usize) -> u32 {
    match dtype {
        StreamDtype::Bf16 | StreamDtype::Fp16 => {
            let offset = index * 2;
            u32::from(u16::from_le_bytes([bytes[offset], bytes[offset + 1]]))
        }
        StreamDtype::Fp32 => {
            let offset = index * 4;
            u32::from_le_bytes([
                bytes[offset],
                bytes[offset + 1],
                bytes[offset + 2],
                bytes[offset + 3],
            ])
        }
    }
}

fn build_planes(dtype: StreamDtype, bytes: &[u8], element_count: usize) -> Vec<Vec<u8>> {
    let plane_count = dtype.planes() as usize;
    let mut planes: Vec<Vec<u8>> = (0..plane_count)
        .map(|_| Vec::with_capacity(element_count))
        .collect();
    for index in 0..element_count {
        let word = element_word(dtype, bytes, index);
        for (plane_index, plane) in planes.iter_mut().enumerate() {
            plane.push((word >> (8 * plane_index)) as u8);
        }
    }
    planes
}

fn decode_planes(dtype: StreamDtype, payload: &[u8], byte_len: u64) -> Result<Vec<u8>> {
    let word_bytes = dtype.word_bytes();
    if byte_len % word_bytes != 0 {
        return Err(Error::malformed(
            "ACTIVATION_STREAM byte length is not a whole number of elements",
        ));
    }
    let element_count = usize::try_from(byte_len / word_bytes)
        .map_err(|_| Error::limit_exceeded("ACTIVATION_STREAM element count does not fit usize"))?;
    let plane_count = usize::try_from(dtype.planes())
        .map_err(|_| Error::limit_exceeded("ACTIVATION_STREAM plane count does not fit usize"))?;
    let mut planes = Vec::with_capacity(plane_count);
    let mut offset = 0usize;
    for _ in 0..plane_count {
        if offset >= payload.len() {
            return Err(Error::corrupt(
                "ACTIVATION_STREAM payload ends before all declared planes",
            ));
        }
        let (plane, consumed) = huffman::decode_prefix(&payload[offset..], element_count)?;
        offset = offset
            .checked_add(consumed)
            .ok_or_else(|| Error::limit_exceeded("ACTIVATION_STREAM payload offset overflows"))?;
        planes.push(plane);
    }
    if offset != payload.len() {
        return Err(Error::corrupt(
            "ACTIVATION_STREAM payload has trailing bytes after the declared planes",
        ));
    }
    let mut bytes = Vec::with_capacity(element_count.saturating_mul(word_bytes as usize));
    for index in 0..element_count {
        let mut word = 0u32;
        for (plane_index, plane) in planes.iter().enumerate() {
            word |= u32::from(plane[index]) << (8 * plane_index);
        }
        match dtype {
            StreamDtype::Bf16 | StreamDtype::Fp16 => {
                bytes.extend_from_slice(&(word as u16).to_le_bytes());
            }
            StreamDtype::Fp32 => bytes.extend_from_slice(&word.to_le_bytes()),
        }
    }
    Ok(bytes)
}

fn build_ai_body(meta: &ActivationStreamMeta, payload: &[u8], limits: &Limits) -> Result<Vec<u8>> {
    let meta_bytes = wcb::encode(&descriptor_value(meta), limits)?;
    let meta_len = u32::try_from(meta_bytes.len())
        .map_err(|_| Error::limit_exceeded("ACTIVATION_STREAM metadata does not fit u32"))?;
    let capacity = 4usize
        .checked_add(meta_bytes.len())
        .and_then(|value| value.checked_add(payload.len()))
        .ok_or_else(|| Error::limit_exceeded("ACTIVATION_STREAM body length overflow"))?;
    let mut body = Vec::with_capacity(capacity);
    body.extend_from_slice(&meta_len.to_le_bytes());
    body.extend_from_slice(&meta_bytes);
    body.extend_from_slice(payload);
    Ok(body)
}

// ---------------------------------------------------------------------------
// Encoding
// ---------------------------------------------------------------------------

/// Build a frame from an explicit descriptor and stored payload.
pub fn encode_descriptor_frame(
    meta: &ActivationStreamMeta,
    payload: &[u8],
    decoded_len: u64,
    content_hash: Hash32,
    limits: &Limits,
) -> Result<Vec<u8>> {
    let body = build_ai_body(meta, payload, limits)?;
    let parts = FrameParts {
        format_major: crate::frame::FORMAT_MAJOR,
        format_minor: crate::frame::FORMAT_MINOR,
        domain: DOMAIN_PAYLOAD,
        mode: Mode::ActivationStream,
        flags: 0,
        level: 0,
        decoded_len,
        content_hash,
        dependencies: &[],
        chunks: &[],
        body: &body,
    };
    encode_frame(&parts, limits)
}

/// Encode one activation/gradient chunk.
pub fn encode_activation_chunk(
    head: &ActivationStreamHead,
    chunk_index: u64,
    elements: &[u8],
    limits: &Limits,
) -> Result<Vec<u8>> {
    let expected = validate_head(head, limits)?;
    if chunk_index >= head.chunk_count {
        return Err(Error::malformed(format!(
            "ACTIVATION_STREAM chunk_index {chunk_index} is not below chunk_count {}",
            head.chunk_count
        )));
    }
    if elements.len() as u64 != expected {
        return Err(Error::malformed(format!(
            "ACTIVATION_STREAM chunk is {} bytes but the descriptor requires {expected}",
            elements.len()
        )));
    }
    let element_count = shape_product(&head.shape, limits)?;
    let count = usize::try_from(element_count)
        .map_err(|_| Error::limit_exceeded("ACTIVATION_STREAM element count does not fit usize"))?;
    let planes = build_planes(head.dtype, elements, count);
    let mut payload = Vec::new();
    for plane in &planes {
        payload.extend_from_slice(&huffman::encode(plane));
    }
    let meta = ActivationStreamMeta {
        stream: head.stream,
        step: head.step,
        rank: head.rank,
        world: head.world,
        chunk_index,
        chunk_count: head.chunk_count,
        dtype: head.dtype,
        shape: head.shape.clone(),
        element_count,
        digest_root: hash_domain(DOMAIN_TENSOR_VALUES, elements),
        plane_codec: ACTIVATION_STREAM_PLANE_CODEC.to_owned(),
        payload_len: payload.len() as u64,
    };
    validate_meta(&meta, limits)?;
    encode_descriptor_frame(
        &meta,
        &payload,
        elements.len() as u64,
        hash_domain(DOMAIN_CHUNK, elements),
        limits,
    )
}

// ---------------------------------------------------------------------------
// Decoding
// ---------------------------------------------------------------------------

/// Decode one activation chunk, verifying its descriptor, digest, and content
/// hash. No other chunk is read or decoded.
pub fn decode_activation_chunk(
    frame: &Frame<'_>,
    limits: &Limits,
) -> Result<DecodedActivationChunk> {
    if frame.header.mode != Mode::ActivationStream {
        return Err(Error::malformed(format!(
            "expected an ACTIVATION_STREAM frame, got mode {}",
            frame.header.mode.as_str()
        )));
    }
    let (descriptor, payload) = split_ai_body(frame.body)?;
    let meta = parse_descriptor(descriptor, limits)?;
    let byte_len = validate_meta(&meta, limits)?;
    if frame.header.decoded_len != byte_len {
        return Err(Error::corrupt(format!(
            "ACTIVATION_STREAM decoded_len is {} but the descriptor requires {byte_len}",
            frame.header.decoded_len
        )));
    }
    if meta.payload_len != payload.len() as u64 {
        return Err(Error::malformed(format!(
            "ACTIVATION_STREAM payload is {} bytes but payload_len declares {}",
            payload.len(),
            meta.payload_len
        )));
    }
    let bytes = decode_planes(meta.dtype, payload, byte_len)?;
    if bytes.len() as u64 != byte_len {
        return Err(Error::corrupt(
            "ACTIVATION_STREAM reconstructed chunk length is inconsistent",
        ));
    }
    if hash_domain(DOMAIN_TENSOR_VALUES, &bytes) != meta.digest_root {
        return Err(Error::corrupt("ACTIVATION_STREAM digest_root mismatch"));
    }
    if hash_domain(DOMAIN_CHUNK, &bytes) != frame.header.content_hash {
        return Err(Error::corrupt(
            "ACTIVATION_STREAM content_hash mismatch against the reconstructed chunk",
        ));
    }
    Ok(DecodedActivationChunk { meta, bytes })
}

/// Decode a byte sub-range inside one chunk. Only the supplied chunk frame is
/// touched.
pub fn decode_activation_chunk_range(
    frame: &Frame<'_>,
    start: u64,
    end: u64,
    limits: &Limits,
) -> Result<Vec<u8>> {
    let chunk = decode_activation_chunk(frame, limits)?;
    let length = chunk.bytes.len() as u64;
    if start > end || end > length {
        return Err(Error::malformed(format!(
            "ACTIVATION_STREAM range {start}..{end} is outside the {length}-byte chunk"
        )));
    }
    Ok(chunk.bytes[start as usize..end as usize].to_vec())
}

/// Compute the normative stream root from per-chunk digests.
pub fn stream_root(
    stream: StreamKind,
    step: u64,
    rank: u32,
    world: u32,
    chunk_count: u64,
    digest_roots: &[Hash32],
    limits: &Limits,
) -> Result<Hash32> {
    if digest_roots.len() as u64 != chunk_count {
        return Err(Error::malformed(format!(
            "ACTIVATION_STREAM stream root needs {chunk_count} digests, got {}",
            digest_roots.len()
        )));
    }
    let value = Value::map([
        ("stream", Value::Str(stream.as_str().to_owned())),
        ("step", Value::U64(step)),
        ("rank", Value::U64(u64::from(rank))),
        ("world", Value::U64(u64::from(world))),
        ("chunk_count", Value::U64(chunk_count)),
        (
            "digest_roots",
            Value::Array(
                digest_roots
                    .iter()
                    .map(|digest| Value::Bytes(digest.as_bytes().to_vec()))
                    .collect(),
            ),
        ),
    ]);
    let bytes = wcb::encode(&value, limits)?;
    Ok(hash_domain(DOMAIN_AI_STREAM, &bytes))
}

/// Assemble one stream from its chunk frames.
///
/// Frames may arrive in any order. Every chunk is decoded and its
/// `digest_root` re-verified. Duplicate or missing `chunk_index` values and
/// inconsistent stream heads are `NonCanonical`/`Malformed`.
pub fn assemble_stream(frames: &[&[u8]], limits: &Limits) -> Result<ActivationStreamAssembly> {
    if frames.is_empty() {
        return Err(Error::malformed(
            "ACTIVATION_STREAM assembly needs at least one chunk",
        ));
    }
    let mut decoded = Vec::with_capacity(frames.len());
    for bytes in frames {
        let frame = Frame::parse(bytes, limits)?;
        decoded.push(decode_activation_chunk(&frame, limits)?);
    }
    let first = decoded[0].meta.clone();
    if frames.len() as u64 != first.chunk_count {
        return Err(Error::malformed(format!(
            "ACTIVATION_STREAM has {} chunk frames but chunk_count is {}",
            frames.len(),
            first.chunk_count
        )));
    }
    for chunk in &decoded {
        let meta = &chunk.meta;
        if meta.stream != first.stream
            || meta.step != first.step
            || meta.rank != first.rank
            || meta.world != first.world
            || meta.chunk_count != first.chunk_count
            || meta.dtype != first.dtype
        {
            return Err(Error::malformed(
                "ACTIVATION_STREAM chunk heads disagree on stream/step/rank/world/chunk_count/dtype",
            ));
        }
        if meta.shape != first.shape || meta.element_count != first.element_count {
            return Err(Error::malformed(
                "ACTIVATION_STREAM chunk heads disagree on shape/element_count",
            ));
        }
    }
    decoded.sort_by_key(|chunk| chunk.meta.chunk_index);
    let mut seen = BTreeSet::new();
    for chunk in &decoded {
        if !seen.insert(chunk.meta.chunk_index) {
            return Err(Error::non_canonical(format!(
                "ACTIVATION_STREAM has duplicate chunk_index {}",
                chunk.meta.chunk_index
            )));
        }
    }
    for (expected, chunk) in decoded.iter().enumerate() {
        if chunk.meta.chunk_index != expected as u64 {
            return Err(Error::malformed(format!(
                "ACTIVATION_STREAM is missing chunk_index {expected}"
            )));
        }
    }
    let mut bytes = Vec::new();
    let mut digests = Vec::with_capacity(decoded.len());
    for chunk in &decoded {
        bytes.extend_from_slice(&chunk.bytes);
        digests.push(chunk.meta.digest_root);
    }
    let root = stream_root(
        first.stream,
        first.step,
        first.rank,
        first.world,
        first.chunk_count,
        &digests,
        limits,
    )?;
    Ok(ActivationStreamAssembly {
        stream: first.stream,
        step: first.step,
        rank: first.rank,
        world: first.world,
        chunk_count: first.chunk_count,
        dtype: first.dtype,
        bytes,
        stream_root: root,
        chunk_digests: digests,
        chunks_decoded: decoded.len() as u64,
    })
}

// ---------------------------------------------------------------------------
// Append and lazy readers
// ---------------------------------------------------------------------------

/// Appends chunks in `chunk_index` order without ever decoding a previous
/// chunk. `decoded_chunks` is incremented only by [`Self::assemble`], so it
/// stays zero across arbitrary appends.
#[derive(Clone, Debug)]
pub struct ActivationStreamWriter {
    head: ActivationStreamHead,
    next_index: u64,
    encoded_chunks: u64,
    decoded_chunks: u64,
}

impl ActivationStreamWriter {
    /// Create a writer for one stream head.
    pub fn new(head: ActivationStreamHead, limits: &Limits) -> Result<Self> {
        validate_head(&head, limits)?;
        Ok(ActivationStreamWriter {
            head,
            next_index: 0,
            encoded_chunks: 0,
            decoded_chunks: 0,
        })
    }

    /// Encode and append the next chunk. This path never decodes a chunk.
    pub fn append_chunk(&mut self, elements: &[u8], limits: &Limits) -> Result<Vec<u8>> {
        let frame = encode_activation_chunk(&self.head, self.next_index, elements, limits)?;
        self.next_index += 1;
        self.encoded_chunks += 1;
        Ok(frame)
    }

    /// Number of chunks encoded through this writer.
    pub fn encoded_chunks(&self) -> u64 {
        self.encoded_chunks
    }

    /// Number of chunks decoded through this writer.
    pub fn decoded_chunks(&self) -> u64 {
        self.decoded_chunks
    }

    /// Next implicit `chunk_index`.
    pub fn next_index(&self) -> u64 {
        self.next_index
    }

    /// Assemble the supplied frames and count the chunks decoded by the
    /// assembly.
    pub fn assemble(
        &mut self,
        frames: &[&[u8]],
        limits: &Limits,
    ) -> Result<ActivationStreamAssembly> {
        let assembly = assemble_stream(frames, limits)?;
        self.decoded_chunks += assembly.chunks_decoded;
        Ok(assembly)
    }
}

/// An ordered set of chunk frames that decodes one chunk at a time.
#[derive(Clone, Debug)]
pub struct ActivationChunkSet {
    frames: Vec<Vec<u8>>,
    decoded_chunks: u64,
}

impl ActivationChunkSet {
    /// Build a set from chunk frames in `chunk_index` order.
    pub fn new(frames: Vec<Vec<u8>>) -> Self {
        ActivationChunkSet {
            frames,
            decoded_chunks: 0,
        }
    }

    /// Number of chunk frames in the set.
    pub fn len(&self) -> usize {
        self.frames.len()
    }

    /// Whether the set contains no chunks.
    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    /// Number of chunk-level decodes performed through this set.
    pub fn decoded_chunks(&self) -> u64 {
        self.decoded_chunks
    }

    /// Decode exactly one chunk by index.
    pub fn decode_chunk(&mut self, index: u64, limits: &Limits) -> Result<DecodedActivationChunk> {
        let slot = usize::try_from(index)
            .map_err(|_| Error::malformed("ACTIVATION_STREAM chunk index does not fit usize"))?;
        let frame_bytes = self.frames.get(slot).ok_or_else(|| {
            Error::malformed(format!(
                "ACTIVATION_STREAM chunk index {index} is outside the {}-chunk set",
                self.frames.len()
            ))
        })?;
        let frame = Frame::parse(frame_bytes, limits)?;
        let chunk = decode_activation_chunk(&frame, limits)?;
        if chunk.meta.chunk_index != index {
            return Err(Error::corrupt(format!(
                "ACTIVATION_STREAM chunk frame at slot {index} declares chunk_index {}",
                chunk.meta.chunk_index
            )));
        }
        if chunk.meta.chunk_count != self.frames.len() as u64 {
            return Err(Error::corrupt(format!(
                "ACTIVATION_STREAM chunk declares chunk_count {} but the set has {} frames",
                chunk.meta.chunk_count,
                self.frames.len()
            )));
        }
        self.decoded_chunks += 1;
        Ok(chunk)
    }

    /// Decode a byte sub-range inside exactly one chunk.
    pub fn read_chunk_range(
        &mut self,
        index: u64,
        start: u64,
        end: u64,
        limits: &Limits,
    ) -> Result<Vec<u8>> {
        let chunk = self.decode_chunk(index, limits)?;
        let length = chunk.bytes.len() as u64;
        if start > end || end > length {
            return Err(Error::malformed(format!(
                "ACTIVATION_STREAM range {start}..{end} is outside the {length}-byte chunk"
            )));
        }
        Ok(chunk.bytes[start as usize..end as usize].to_vec())
    }
}
