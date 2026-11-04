//! Mode 12 `KV_CACHE` (WEFT-AI-CODEC §13).
//!
//! A frame carries exactly one page of a paged attention KV cache. Pages are
//! independent: no frame depends on another, no decoder state crosses a page,
//! and a page can be fetched and decoded without touching its siblings.
//!
//! The descriptor carries the model/session identity, the layer, the page
//! position, the token range, the head geometry, and the exact page byte
//! length. The payload is the canonical row-major element byte sequence split
//! into byte planes (plane 0 = least significant byte), each plane coded with
//! its own `huffman-canonical-v1` stream and concatenated in plane order.
//! `bf16` and `fp16` use two planes; `fp32` uses four.
//!
//! Decoding reconstructs the raw element bytes exactly. NaNs, denormals,
//! signed zeros, and every other bit pattern pass through untouched.

use std::collections::BTreeMap;

use crate::error::{Error, Result};
use crate::frame::{encode_frame, Frame, FrameParts, Mode, DOMAIN_PAYLOAD};
use crate::hash::{hash_domain, Hash32, DOMAIN_CHUNK};
use crate::huffman;
use crate::limits::{check_len, Limits};
use crate::wcb::{self, Value};

/// Canonical descriptor schema.
pub const KV_CACHE_SCHEMA: &str = "weft.ai.kv_cache.v1";
/// The only plane coder implemented in v1.
pub const KV_CACHE_PLANE_CODEC: &str = huffman::CODER_ID;

/// Element dtypes implemented by mode 12.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KvDtype {
    Bf16,
    Fp16,
    Fp32,
}

impl KvDtype {
    /// Parse the canonical dtype name.
    pub fn from_name(name: &str) -> Result<Self> {
        match name {
            "bf16" => Ok(KvDtype::Bf16),
            "fp16" => Ok(KvDtype::Fp16),
            "fp32" => Ok(KvDtype::Fp32),
            other => Err(Error::malformed(format!(
                "unknown KV_CACHE dtype `{other}` (v1 allows bf16, fp16, fp32)"
            ))),
        }
    }

    /// Canonical dtype name.
    pub fn as_str(self) -> &'static str {
        match self {
            KvDtype::Bf16 => "bf16",
            KvDtype::Fp16 => "fp16",
            KvDtype::Fp32 => "fp32",
        }
    }

    /// Bytes per element in canonical row-major order.
    pub fn word_bytes(self) -> u64 {
        match self {
            KvDtype::Bf16 | KvDtype::Fp16 => 2,
            KvDtype::Fp32 => 4,
        }
    }

    /// Number of byte planes in the payload.
    pub fn planes(self) -> u64 {
        match self {
            KvDtype::Bf16 | KvDtype::Fp16 => 2,
            KvDtype::Fp32 => 4,
        }
    }
}

/// The descriptor of one KV cache page.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KvCacheMeta {
    pub model: String,
    pub layer: u32,
    pub session: String,
    pub page_index: u64,
    pub page_count: u64,
    pub seq_start: u64,
    pub seq_end: u64,
    pub dtype: KvDtype,
    pub head_count: u32,
    pub head_dim: u32,
    pub shape: Vec<u64>,
    pub page_bytes: u64,
    pub plane_codec: String,
    pub payload_len: u64,
}

/// A page to encode. `shape` and `page_bytes` are derived.
#[derive(Clone, Debug)]
pub struct KvCacheInput {
    pub model: String,
    pub layer: u32,
    pub session: String,
    pub page_index: u64,
    pub page_count: u64,
    pub seq_start: u64,
    pub seq_end: u64,
    pub dtype: KvDtype,
    pub head_count: u32,
    pub head_dim: u32,
}

/// A decoded page.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DecodedKvPage {
    pub meta: KvCacheMeta,
    pub bytes: Vec<u8>,
}

// ---------------------------------------------------------------------------
// Descriptor helpers
// ---------------------------------------------------------------------------

fn meta_text(map: &BTreeMap<String, Value>, key: &str) -> Result<String> {
    match map.get(key) {
        Some(Value::Str(text)) => Ok(text.clone()),
        _ => Err(Error::malformed(format!(
            "KV_CACHE metadata `{key}` must be text"
        ))),
    }
}

fn meta_u64(map: &BTreeMap<String, Value>, key: &str) -> Result<u64> {
    match map.get(key) {
        Some(Value::U64(value)) => Ok(*value),
        _ => Err(Error::malformed(format!(
            "KV_CACHE metadata `{key}` must be u64"
        ))),
    }
}

fn meta_u32(map: &BTreeMap<String, Value>, key: &str) -> Result<u32> {
    let value = meta_u64(map, key)?;
    u32::try_from(value)
        .map_err(|_| Error::malformed(format!("KV_CACHE metadata `{key}` = {value} exceeds u32")))
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
                            "KV_CACHE metadata `shape` elements must be u64",
                        ))
                    }
                }
            }
            Ok(shape)
        }
        _ => Err(Error::malformed(
            "KV_CACHE metadata `shape` must be an array",
        )),
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

fn page_rows(meta: &KvCacheMeta) -> Result<u64> {
    if meta.seq_start >= meta.seq_end {
        return Err(Error::malformed(format!(
            "KV_CACHE seq_start {} must be less than seq_end {}",
            meta.seq_start, meta.seq_end
        )));
    }
    Ok(meta.seq_end - meta.seq_start)
}

/// Validate the descriptor, returning the exact page byte length.
pub fn validate_meta(meta: &KvCacheMeta, limits: &Limits) -> Result<u64> {
    if meta.plane_codec != KV_CACHE_PLANE_CODEC {
        return Err(Error::malformed(format!(
            "KV_CACHE plane_codec `{}` is not implemented (only {KV_CACHE_PLANE_CODEC})",
            meta.plane_codec
        )));
    }
    if meta.page_count == 0 {
        return Err(Error::malformed("KV_CACHE page_count must be at least 1"));
    }
    if meta.page_index >= meta.page_count {
        return Err(Error::malformed(format!(
            "KV_CACHE page_index {} is not below page_count {}",
            meta.page_index, meta.page_count
        )));
    }
    let rows = page_rows(meta)?;
    if meta.head_count == 0 {
        return Err(Error::malformed(
            "KV_CACHE head_count must be greater than zero",
        ));
    }
    if meta.head_dim == 0 {
        return Err(Error::malformed(
            "KV_CACHE head_dim must be greater than zero",
        ));
    }
    let expected_shape = [rows, u64::from(meta.head_count), u64::from(meta.head_dim)];
    if meta.shape.len() != 3 || meta.shape.as_slice() != expected_shape {
        return Err(Error::malformed(format!(
            "KV_CACHE shape {:?} must equal [page_rows, head_count, head_dim] = {expected_shape:?}",
            meta.shape
        )));
    }
    if meta.shape.len() as u64 > limits.max_tensor_rank {
        return Err(Error::limit_exceeded(format!(
            "KV_CACHE rank {} exceeds limit {}",
            meta.shape.len(),
            limits.max_tensor_rank
        )));
    }
    let element_count = rows
        .checked_mul(u64::from(meta.head_count))
        .and_then(|value| value.checked_mul(u64::from(meta.head_dim)))
        .ok_or_else(|| Error::limit_exceeded("KV_CACHE element count overflows u64"))?;
    if element_count > limits.max_tensor_elements {
        return Err(Error::limit_exceeded(format!(
            "KV_CACHE element count {element_count} exceeds limit {}",
            limits.max_tensor_elements
        )));
    }
    let computed = element_count
        .checked_mul(meta.dtype.word_bytes())
        .ok_or_else(|| Error::limit_exceeded("KV_CACHE page byte count overflows u64"))?;
    if meta.page_bytes != computed {
        return Err(Error::malformed(format!(
            "KV_CACHE page_bytes {} does not equal page_rows * head_count * head_dim * word_bytes = {computed}",
            meta.page_bytes
        )));
    }
    check_len(meta.page_bytes, limits.max_decoded_bytes, "KV_CACHE page")?;
    Ok(computed)
}

/// Build the canonical WCB descriptor value.
pub fn descriptor_value(meta: &KvCacheMeta) -> Value {
    Value::map([
        ("schema", Value::Str(KV_CACHE_SCHEMA.to_owned())),
        ("model", Value::Str(meta.model.clone())),
        ("layer", Value::U64(u64::from(meta.layer))),
        ("session", Value::Str(meta.session.clone())),
        ("page_index", Value::U64(meta.page_index)),
        ("page_count", Value::U64(meta.page_count)),
        ("seq_start", Value::U64(meta.seq_start)),
        ("seq_end", Value::U64(meta.seq_end)),
        ("dtype", Value::Str(meta.dtype.as_str().to_owned())),
        ("head_count", Value::U64(u64::from(meta.head_count))),
        ("head_dim", Value::U64(u64::from(meta.head_dim))),
        (
            "shape",
            Value::Array(meta.shape.iter().map(|dim| Value::U64(*dim)).collect()),
        ),
        ("page_bytes", Value::U64(meta.page_bytes)),
        ("plane_codec", Value::Str(meta.plane_codec.clone())),
        ("payload_len", Value::U64(meta.payload_len)),
    ])
}

fn split_ai_body(body: &[u8]) -> Result<(&[u8], &[u8])> {
    if body.len() < 4 {
        return Err(Error::malformed(
            "KV_CACHE body is shorter than its 4-byte metadata length",
        ));
    }
    let meta_len = u64::from(u32::from_le_bytes([body[0], body[1], body[2], body[3]]));
    let end = 4u64
        .checked_add(meta_len)
        .ok_or_else(|| Error::malformed("KV_CACHE metadata length overflows"))?;
    if end > body.len() as u64 {
        return Err(Error::malformed(
            "KV_CACHE metadata extends past the frame body",
        ));
    }
    let end = end as usize;
    Ok((&body[4..end], &body[end..]))
}

fn parse_descriptor(bytes: &[u8], limits: &Limits) -> Result<KvCacheMeta> {
    let value = wcb::decode(bytes, limits)?;
    let map = match value {
        Value::Map(map) => map,
        _ => return Err(Error::malformed("KV_CACHE metadata must be a WCB map")),
    };
    const KEYS: [&str; 15] = [
        "schema",
        "model",
        "layer",
        "session",
        "page_index",
        "page_count",
        "seq_start",
        "seq_end",
        "dtype",
        "head_count",
        "head_dim",
        "shape",
        "page_bytes",
        "plane_codec",
        "payload_len",
    ];
    exact_keys(&map, &KEYS, "KV_CACHE")?;
    let schema = meta_text(&map, "schema")?;
    if schema != KV_CACHE_SCHEMA {
        return Err(Error::malformed(format!(
            "unsupported KV_CACHE schema `{schema}`"
        )));
    }
    let meta = KvCacheMeta {
        model: meta_text(&map, "model")?,
        layer: meta_u32(&map, "layer")?,
        session: meta_text(&map, "session")?,
        page_index: meta_u64(&map, "page_index")?,
        page_count: meta_u64(&map, "page_count")?,
        seq_start: meta_u64(&map, "seq_start")?,
        seq_end: meta_u64(&map, "seq_end")?,
        dtype: KvDtype::from_name(&meta_text(&map, "dtype")?)?,
        head_count: meta_u32(&map, "head_count")?,
        head_dim: meta_u32(&map, "head_dim")?,
        shape: meta_shape(&map)?,
        page_bytes: meta_u64(&map, "page_bytes")?,
        plane_codec: meta_text(&map, "plane_codec")?,
        payload_len: meta_u64(&map, "payload_len")?,
    };
    validate_meta(&meta, limits)?;
    Ok(meta)
}

/// Parse and validate a KV_CACHE descriptor from a frame.
pub fn kv_cache_metadata(frame: &Frame<'_>, limits: &Limits) -> Result<KvCacheMeta> {
    if frame.header.mode != Mode::KvCache {
        return Err(Error::malformed(format!(
            "expected a KV_CACHE frame, got mode {}",
            frame.header.mode.as_str()
        )));
    }
    let (descriptor, _) = split_ai_body(frame.body)?;
    parse_descriptor(descriptor, limits)
}

// ---------------------------------------------------------------------------
// Byte planes
// ---------------------------------------------------------------------------

fn element_word(dtype: KvDtype, bytes: &[u8], index: usize) -> u32 {
    match dtype {
        KvDtype::Bf16 | KvDtype::Fp16 => {
            let offset = index * 2;
            u32::from(u16::from_le_bytes([bytes[offset], bytes[offset + 1]]))
        }
        KvDtype::Fp32 => {
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

fn build_planes(dtype: KvDtype, bytes: &[u8], element_count: usize) -> Vec<Vec<u8>> {
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

fn decode_planes(dtype: KvDtype, payload: &[u8], page_bytes: u64) -> Result<Vec<u8>> {
    let word_bytes = dtype.word_bytes();
    if page_bytes % word_bytes != 0 {
        return Err(Error::malformed(
            "KV_CACHE page byte length is not a whole number of elements",
        ));
    }
    let element_count = usize::try_from(page_bytes / word_bytes)
        .map_err(|_| Error::limit_exceeded("KV_CACHE element count does not fit usize"))?;
    let plane_count = usize::try_from(dtype.planes())
        .map_err(|_| Error::limit_exceeded("KV_CACHE plane count does not fit usize"))?;
    let mut planes = Vec::with_capacity(plane_count);
    let mut offset = 0usize;
    for _ in 0..plane_count {
        if offset >= payload.len() {
            return Err(Error::corrupt(
                "KV_CACHE payload ends before all declared planes",
            ));
        }
        let (plane, consumed) = huffman::decode_prefix(&payload[offset..], element_count)?;
        offset = offset
            .checked_add(consumed)
            .ok_or_else(|| Error::limit_exceeded("KV_CACHE payload offset overflows"))?;
        planes.push(plane);
    }
    if offset != payload.len() {
        return Err(Error::corrupt(
            "KV_CACHE payload has trailing bytes after the declared planes",
        ));
    }
    let mut bytes = Vec::with_capacity(element_count.saturating_mul(word_bytes as usize));
    for index in 0..element_count {
        let mut word = 0u32;
        for (plane_index, plane) in planes.iter().enumerate() {
            word |= u32::from(plane[index]) << (8 * plane_index);
        }
        match dtype {
            KvDtype::Bf16 | KvDtype::Fp16 => {
                bytes.extend_from_slice(&(word as u16).to_le_bytes());
            }
            KvDtype::Fp32 => bytes.extend_from_slice(&word.to_le_bytes()),
        }
    }
    Ok(bytes)
}

fn build_ai_body(meta: &KvCacheMeta, payload: &[u8], limits: &Limits) -> Result<Vec<u8>> {
    let meta_bytes = wcb::encode(&descriptor_value(meta), limits)?;
    let meta_len = u32::try_from(meta_bytes.len())
        .map_err(|_| Error::limit_exceeded("KV_CACHE metadata does not fit u32"))?;
    let capacity = 4usize
        .checked_add(meta_bytes.len())
        .and_then(|value| value.checked_add(payload.len()))
        .ok_or_else(|| Error::limit_exceeded("KV_CACHE body length overflow"))?;
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
///
/// Low-level constructor shared by the encoder and adversarial vector
/// generation. It does not re-validate descriptor semantics; the resulting
/// frame is re-parsed and decoded by the caller.
pub fn encode_descriptor_frame(
    meta: &KvCacheMeta,
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
        mode: Mode::KvCache,
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

/// Encode one KV cache page. `elements` is the exact canonical row-major
/// little-endian element bytes for `shape`.
pub fn encode_kv_page(input: &KvCacheInput, elements: &[u8], limits: &Limits) -> Result<Vec<u8>> {
    let rows = if input.seq_start < input.seq_end {
        input.seq_end - input.seq_start
    } else {
        return Err(Error::malformed(format!(
            "KV_CACHE seq_start {} must be less than seq_end {}",
            input.seq_start, input.seq_end
        )));
    };
    let element_count = rows
        .checked_mul(u64::from(input.head_count))
        .and_then(|value| value.checked_mul(u64::from(input.head_dim)))
        .ok_or_else(|| Error::limit_exceeded("KV_CACHE element count overflows u64"))?;
    let page_bytes = element_count
        .checked_mul(input.dtype.word_bytes())
        .ok_or_else(|| Error::limit_exceeded("KV_CACHE page byte count overflows u64"))?;
    if elements.len() as u64 != page_bytes {
        return Err(Error::malformed(format!(
            "KV_CACHE page is {} bytes but the descriptor requires {page_bytes}",
            elements.len()
        )));
    }
    check_len(page_bytes, limits.max_decoded_bytes, "KV_CACHE page")?;
    let meta = KvCacheMeta {
        model: input.model.clone(),
        layer: input.layer,
        session: input.session.clone(),
        page_index: input.page_index,
        page_count: input.page_count,
        seq_start: input.seq_start,
        seq_end: input.seq_end,
        dtype: input.dtype,
        head_count: input.head_count,
        head_dim: input.head_dim,
        shape: vec![rows, u64::from(input.head_count), u64::from(input.head_dim)],
        page_bytes,
        plane_codec: KV_CACHE_PLANE_CODEC.to_owned(),
        payload_len: 0,
    };
    validate_meta(&meta, limits)?;
    let count = usize::try_from(element_count)
        .map_err(|_| Error::limit_exceeded("KV_CACHE element count does not fit usize"))?;
    let planes = build_planes(input.dtype, elements, count);
    let mut payload = Vec::new();
    for plane in &planes {
        payload.extend_from_slice(&huffman::encode(plane));
    }
    let meta = KvCacheMeta {
        payload_len: payload.len() as u64,
        ..meta
    };
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

/// Decode one KV cache page, verifying its descriptor, planes, and content
/// hash. No other page is read or decoded.
pub fn decode_kv_page(frame: &Frame<'_>, limits: &Limits) -> Result<DecodedKvPage> {
    if frame.header.mode != Mode::KvCache {
        return Err(Error::malformed(format!(
            "expected a KV_CACHE frame, got mode {}",
            frame.header.mode.as_str()
        )));
    }
    let (descriptor, payload) = split_ai_body(frame.body)?;
    let meta = parse_descriptor(descriptor, limits)?;
    let page_bytes = validate_meta(&meta, limits)?;
    if frame.header.decoded_len != page_bytes {
        return Err(Error::corrupt(format!(
            "KV_CACHE decoded_len is {} but the descriptor requires {page_bytes}",
            frame.header.decoded_len
        )));
    }
    if meta.payload_len != payload.len() as u64 {
        return Err(Error::malformed(format!(
            "KV_CACHE payload is {} bytes but payload_len declares {}",
            payload.len(),
            meta.payload_len
        )));
    }
    let bytes = decode_planes(meta.dtype, payload, page_bytes)?;
    if bytes.len() as u64 != page_bytes {
        return Err(Error::corrupt(
            "KV_CACHE reconstructed page length is inconsistent",
        ));
    }
    if hash_domain(DOMAIN_CHUNK, &bytes) != frame.header.content_hash {
        return Err(Error::corrupt(
            "KV_CACHE content_hash mismatch against the reconstructed page",
        ));
    }
    Ok(DecodedKvPage { meta, bytes })
}

/// Decode a byte sub-range inside one page.
///
/// Huffman streams are not seekable, so the page's planes are decoded and the
/// requested slice is returned. Only the supplied page frame is touched;
/// sibling pages are never read.
pub fn decode_kv_page_range(
    frame: &Frame<'_>,
    start: u64,
    end: u64,
    limits: &Limits,
) -> Result<Vec<u8>> {
    let page = decode_kv_page(frame, limits)?;
    let length = page.bytes.len() as u64;
    if start > end || end > length {
        return Err(Error::malformed(format!(
            "KV_CACHE range {start}..{end} is outside the {length}-byte page"
        )));
    }
    Ok(page.bytes[start as usize..end as usize].to_vec())
}

/// An ordered set of page frames that decodes one page at a time.
///
/// `decoded_pages` counts page-level decodes performed through this set; it is
/// the evidence counter for page independence and lazy range reads.
#[derive(Clone, Debug)]
pub struct KvPageSet {
    frames: Vec<Vec<u8>>,
    decoded_pages: u64,
}

impl KvPageSet {
    /// Build a set from page frames in page-index order.
    pub fn new(frames: Vec<Vec<u8>>) -> Self {
        KvPageSet {
            frames,
            decoded_pages: 0,
        }
    }

    /// Number of page frames in the set.
    pub fn len(&self) -> usize {
        self.frames.len()
    }

    /// Whether the set contains no pages.
    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    /// Number of page-level decodes performed through this set.
    pub fn decoded_pages(&self) -> u64 {
        self.decoded_pages
    }

    /// Decode exactly one page by index. Other page frames are not parsed,
    /// decoded, or inspected.
    pub fn decode_page(&mut self, index: u64, limits: &Limits) -> Result<DecodedKvPage> {
        let slot = usize::try_from(index)
            .map_err(|_| Error::malformed("KV_CACHE page index does not fit usize"))?;
        let frame_bytes = self.frames.get(slot).ok_or_else(|| {
            Error::malformed(format!(
                "KV_CACHE page index {index} is outside the {}-page set",
                self.frames.len()
            ))
        })?;
        let frame = Frame::parse(frame_bytes, limits)?;
        let page = decode_kv_page(&frame, limits)?;
        if page.meta.page_index != index {
            return Err(Error::corrupt(format!(
                "KV_CACHE page frame at slot {index} declares page_index {}",
                page.meta.page_index
            )));
        }
        if page.meta.page_count != self.frames.len() as u64 {
            return Err(Error::corrupt(format!(
                "KV_CACHE page declares page_count {} but the set has {} frames",
                page.meta.page_count,
                self.frames.len()
            )));
        }
        self.decoded_pages += 1;
        Ok(page)
    }

    /// Decode a byte sub-range inside exactly one page.
    pub fn read_page_range(
        &mut self,
        index: u64,
        start: u64,
        end: u64,
        limits: &Limits,
    ) -> Result<Vec<u8>> {
        let page = self.decode_page(index, limits)?;
        let length = page.bytes.len() as u64;
        if start > end || end > length {
            return Err(Error::malformed(format!(
                "KV_CACHE range {start}..{end} is outside the {length}-byte page"
            )));
        }
        Ok(page.bytes[start as usize..end as usize].to_vec())
    }
}
