//! Mode 10 `TENSOR_SHARDED` (WEFT-AI-CODEC §11).
//!
//! A frame carries one shard of a quantized tensor. The descriptor names the
//! whole logical shape, the shard position, the global quantization blocks, and
//! the WeftTensor `LogicalTensorRoot` of the unsharded tensor. The payload is a
//! single `huffman-canonical-v1` stream over
//!
//! ```text
//! q8_0: [shard i8 words][global block scale bits]
//! q4_0: [shard packed i4 nibbles][global block scale bits]
//! ```
//!
//! Decoding reconstructs those planes exactly; no dequantization happens here.
//! The logical root is computed from the *quantized* words and the exact
//! scale/zero-point block records, so it is independent of sharding.

use std::collections::BTreeMap;

use crate::error::{Error, Result};
use crate::frame::{encode_frame, Frame, FrameParts, Mode, DOMAIN_PAYLOAD};
use crate::hash::{hash_domain, Hash32};
use crate::huffman;
use crate::limits::Limits;
use crate::modes::TensorDtype;
use crate::tensor;
use crate::wcb::{self, Value};

/// Canonical descriptor schema.
pub const TENSOR_SHARDED_SCHEMA: &str = "weft.ai.tensor_sharded.v1";
/// Stored Huffman-stream binding domain.
pub const DOMAIN_TENSOR_SHARDED_PAYLOAD: &str = "weft/tensor-sharded/payload/v1";
/// The only block sizes implemented in v1.
pub const BLOCK_SIZES: [u64; 5] = [16, 32, 64, 128, 256];

/// Quantized dtypes implemented by mode 10.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShardedDtype {
    /// One signed 8-bit quantized value per element.
    Q8_0,
    /// Two signed 4-bit quantized values per byte, low nibble first.
    Q4_0,
}

impl ShardedDtype {
    /// Parse the canonical dtype name.
    pub fn from_name(name: &str) -> Result<Self> {
        match name {
            "q8_0" => Ok(ShardedDtype::Q8_0),
            "q4_0" => Ok(ShardedDtype::Q4_0),
            other => Err(Error::malformed(format!(
                "unknown TENSOR_SHARDED dtype `{other}` (v1 allows q8_0, q4_0)"
            ))),
        }
    }

    /// Canonical dtype name.
    pub fn as_str(self) -> &'static str {
        match self {
            ShardedDtype::Q8_0 => "q8_0",
            ShardedDtype::Q4_0 => "q4_0",
        }
    }

    /// Number of quantized bytes for `shard_elements` elements.
    pub fn quant_plane_len(self, shard_elements: u64) -> Result<u64> {
        match self {
            ShardedDtype::Q8_0 => Ok(shard_elements),
            ShardedDtype::Q4_0 => {
                if shard_elements % 2 != 0 {
                    return Err(Error::malformed(
                        "q4_0 shard element count must be even to pack nibble pairs",
                    ));
                }
                Ok(shard_elements / 2)
            }
        }
    }
}

/// The descriptor of one shard.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorShardedMeta {
    pub dtype: ShardedDtype,
    pub shape: Vec<u64>,
    pub element_count: u64,
    pub shard_axis: u32,
    pub shard_count: u32,
    pub shard_index: u32,
    pub block_size: u64,
    pub scales: Vec<u32>,
    pub zero_points: Vec<i64>,
    pub layout_root: Hash32,
    pub payload_len: u64,
}

impl TensorShardedMeta {
    /// Number of elements in each shard.
    pub fn shard_elements(&self) -> Result<u64> {
        if self.shard_count == 0 {
            return Err(Error::malformed(
                "TENSOR_SHARDED shard_count must be at least 1",
            ));
        }
        let count = u64::from(self.shard_count);
        if self.element_count % count != 0 {
            return Err(Error::malformed(format!(
                "TENSOR_SHARDED element_count {} is not divisible by shard_count {}",
                self.element_count, self.shard_count
            )));
        }
        Ok(self.element_count / count)
    }

    /// Number of global quantization blocks (derived, never on the wire).
    pub fn block_count(&self) -> Result<u64> {
        if self.block_size == 0 {
            return Err(Error::malformed(
                "TENSOR_SHARDED block_size must be greater than zero",
            ));
        }
        if self.element_count % self.block_size != 0 {
            return Err(Error::malformed(format!(
                "TENSOR_SHARDED element_count {} is not divisible by block_size {}",
                self.element_count, self.block_size
            )));
        }
        Ok(self.element_count / self.block_size)
    }

    /// Reconstructed plane length: quantized bytes plus one f32 scale per block.
    pub fn planes_len(&self) -> Result<u64> {
        let quant = self.dtype.quant_plane_len(self.shard_elements()?)?;
        let block_count = self.block_count()?;
        let scales = block_count.checked_mul(4).ok_or_else(|| {
            Error::limit_exceeded("TENSOR_SHARDED scale plane length overflows u64")
        })?;
        quant
            .checked_add(scales)
            .ok_or_else(|| Error::limit_exceeded("TENSOR_SHARDED plane length overflows u64"))
    }
}

/// A whole quantized tensor to be split into shards.
#[derive(Clone, Debug)]
pub struct ShardedTensorInput {
    pub dtype: ShardedDtype,
    pub shape: Vec<u64>,
    pub shard_axis: u32,
    pub shard_count: u32,
    pub block_size: u64,
    /// One f32 raw-bit scale per global block, in block order.
    pub scales: Vec<u32>,
    /// One zero point per global block, in block order.
    pub zero_points: Vec<i64>,
    /// Whole-tensor canonical row-major quantized values.
    pub words: Vec<i8>,
}

/// A decoded shard: descriptor, shard-local quantized words, and planes.
#[derive(Clone, Debug)]
pub struct DecodedTensorShard {
    pub meta: TensorShardedMeta,
    /// Shard-local canonical row-major quantized values.
    pub words: Vec<i8>,
    /// Exact reconstructed planes (quantized bytes followed by scale bits).
    pub planes: Vec<u8>,
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

fn validate_shape(shape: &[u64], limits: &Limits) -> Result<u64> {
    if shape.is_empty() {
        return Err(Error::malformed(
            "TENSOR_SHARDED shape rank must be at least 1",
        ));
    }
    if shape.len() as u64 > limits.max_tensor_rank {
        return Err(Error::limit_exceeded(format!(
            "TENSOR_SHARDED rank {} exceeds limit {}",
            shape.len(),
            limits.max_tensor_rank
        )));
    }
    let mut product: u64 = 1;
    for dim in shape {
        if *dim == 0 {
            return Err(Error::malformed(
                "TENSOR_SHARDED shape dimensions must be greater than zero",
            ));
        }
        product = product
            .checked_mul(*dim)
            .ok_or_else(|| Error::limit_exceeded("TENSOR_SHARDED shape product overflows u64"))?;
    }
    if product > limits.max_tensor_elements {
        return Err(Error::limit_exceeded(format!(
            "TENSOR_SHARDED element count {product} exceeds limit {}",
            limits.max_tensor_elements
        )));
    }
    Ok(product)
}

fn validate_scale_bits(bits: u32) -> Result<()> {
    let scale = f32::from_bits(bits);
    if !scale.is_finite() {
        return Err(Error::malformed(
            "TENSOR_SHARDED scale bits must be a finite f32",
        ));
    }
    if scale <= 0.0 {
        return Err(Error::malformed(
            "TENSOR_SHARDED scale must be greater than zero",
        ));
    }
    Ok(())
}

fn validate_block_size(block_size: u64) -> Result<()> {
    if !BLOCK_SIZES.contains(&block_size) {
        return Err(Error::malformed(format!(
            "TENSOR_SHARDED block_size {block_size} is not one of 16, 32, 64, 128, 256"
        )));
    }
    Ok(())
}

/// Validate every descriptor invariant and return `(element_count,
/// shard_elements, block_count)`.
fn validate_descriptor(meta: &TensorShardedMeta, limits: &Limits) -> Result<(u64, u64, u64)> {
    let element_count = validate_shape(&meta.shape, limits)?;
    if meta.element_count != element_count {
        return Err(Error::malformed(format!(
            "TENSOR_SHARDED element_count {} does not match shape product {element_count}",
            meta.element_count
        )));
    }
    if meta.shard_count == 0 {
        return Err(Error::malformed(
            "TENSOR_SHARDED shard_count must be at least 1",
        ));
    }
    if u64::from(meta.shard_axis) >= meta.shape.len() as u64 {
        return Err(Error::malformed(format!(
            "TENSOR_SHARDED shard_axis {} is not below rank {}",
            meta.shard_axis,
            meta.shape.len()
        )));
    }
    if meta.shard_index >= meta.shard_count {
        return Err(Error::malformed(format!(
            "TENSOR_SHARDED shard_index {} is not below shard_count {}",
            meta.shard_index, meta.shard_count
        )));
    }
    let axis_dim = meta.shape[meta.shard_axis as usize];
    if axis_dim % u64::from(meta.shard_count) != 0 {
        return Err(Error::malformed(format!(
            "TENSOR_SHARDED shape[{}] = {axis_dim} is not divisible by shard_count {}",
            meta.shard_axis, meta.shard_count
        )));
    }
    let shard_elements = meta.shard_elements()?;
    validate_block_size(meta.block_size)?;
    let block_count = meta.block_count()?;
    if block_count > limits.max_structured_items {
        return Err(Error::limit_exceeded(format!(
            "TENSOR_SHARDED block count {} exceeds WCB item limit {}",
            block_count, limits.max_structured_items
        )));
    }
    let count = usize::try_from(block_count)
        .map_err(|_| Error::limit_exceeded("TENSOR_SHARDED block count does not fit usize"))?;
    if meta.scales.len() != count || meta.zero_points.len() != count {
        return Err(Error::malformed(format!(
            "TENSOR_SHARDED descriptor has {} scales and {} zero points but requires {count}",
            meta.scales.len(),
            meta.zero_points.len()
        )));
    }
    for bits in &meta.scales {
        validate_scale_bits(*bits)?;
    }
    let planes = meta.planes_len()?;
    crate::limits::check_len(
        planes,
        limits.max_decoded_bytes,
        "TENSOR_SHARDED reconstructed planes",
    )?;
    Ok((element_count, shard_elements, block_count))
}

fn validate_words(dtype: ShardedDtype, words: &[i8]) -> Result<()> {
    if dtype == ShardedDtype::Q4_0 {
        for &word in words {
            if !(-8..=7).contains(&word) {
                return Err(Error::malformed(format!(
                    "q4_0 quantized value {word} is outside the signed 4-bit range -8..=7"
                )));
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Logical root
// ---------------------------------------------------------------------------

/// `values_hash = H("weft/tensor/values/v1", W || B)` where `W` is the
/// whole-tensor row-major sequence of sign-extended quantized words and `B` is
/// the block records in block order.
fn quantized_values_hash(words: &[i8], scales: &[u32], zero_points: &[i64]) -> Hash32 {
    let domain = tensor::DOMAIN_TENSOR_VALUES;
    let mut hasher = blake3::Hasher::new();
    let len = u32::try_from(domain.len()).unwrap_or(u32::MAX);
    hasher.update(&len.to_le_bytes());
    hasher.update(domain.as_bytes());
    let mut byte = [0u8; 1];
    for &word in words {
        byte[0] = word as u8;
        hasher.update(&byte);
    }
    for (scale, zero) in scales.iter().zip(zero_points.iter()) {
        hasher.update(&scale.to_le_bytes());
        hasher.update(&zero.to_le_bytes());
    }
    Hash32::from_bytes(*hasher.finalize().as_bytes())
}

/// Compute the WeftTensor `LogicalTensorRoot` over a whole quantized tensor.
pub fn logical_root(
    dtype: ShardedDtype,
    shape: &[u64],
    words: &[i8],
    scales: &[u32],
    zero_points: &[i64],
    limits: &Limits,
) -> Result<Hash32> {
    let element_count = validate_shape(shape, limits)?;
    if words.len() as u64 != element_count {
        return Err(Error::malformed(format!(
            "TENSOR_SHARDED tensor has {} words but shape requires {element_count}",
            words.len()
        )));
    }
    if scales.len() != zero_points.len() {
        return Err(Error::malformed(
            "TENSOR_SHARDED scales and zero_points must have the same length",
        ));
    }
    validate_words(dtype, words)?;
    for bits in scales {
        validate_scale_bits(*bits)?;
    }
    let values_hash = quantized_values_hash(words, scales, zero_points);
    let record = Value::map([
        ("schema", Value::Str(tensor::LOGICAL_SCHEMA.to_owned())),
        ("dtype", Value::Str(dtype.as_str().to_owned())),
        (
            "shape",
            Value::Array(shape.iter().map(|dim| Value::U64(*dim)).collect()),
        ),
        ("element_count", Value::U64(element_count)),
        ("values_hash", Value::Bytes(values_hash.as_bytes().to_vec())),
    ]);
    let bytes = wcb::encode(&record, limits)?;
    Ok(hash_domain(tensor::DOMAIN_TENSOR_LOGICAL, &bytes))
}

// ---------------------------------------------------------------------------
// Encoding
// ---------------------------------------------------------------------------

fn pack_q4(words: &[i8]) -> Result<Vec<u8>> {
    if words.len() % 2 != 0 {
        return Err(Error::malformed(
            "q4_0 shard element count must be even to pack nibble pairs",
        ));
    }
    let mut out = Vec::with_capacity(words.len() / 2);
    for pair in words.chunks_exact(2) {
        let low = (pair[0] as u8) & 0x0f;
        let high = (pair[1] as u8) & 0x0f;
        out.push(low | (high << 4));
    }
    Ok(out)
}

fn unpack_q4(bytes: &[u8], count: usize) -> Vec<i8> {
    let mut out = Vec::with_capacity(count);
    for &byte in bytes {
        out.push(nibble_to_i8(byte & 0x0f));
        out.push(nibble_to_i8(byte >> 4));
    }
    out.truncate(count);
    out
}

fn nibble_to_i8(nibble: u8) -> i8 {
    if nibble < 8 {
        nibble as i8
    } else {
        nibble as i8 - 16
    }
}

fn quantize_shard(dtype: ShardedDtype, shard_bytes: &[u8]) -> Result<Vec<u8>> {
    match dtype {
        ShardedDtype::Q8_0 => Ok(shard_bytes.to_vec()),
        ShardedDtype::Q4_0 => {
            let words: Vec<i8> = shard_bytes.iter().map(|byte| *byte as i8).collect();
            pack_q4(&words)
        }
    }
}

fn scale_plane(scales: &[u32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(scales.len().saturating_mul(4));
    for scale in scales {
        out.extend_from_slice(&scale.to_le_bytes());
    }
    out
}

pub(crate) fn descriptor_value(meta: &TensorShardedMeta, payload_hash: &Hash32) -> Value {
    Value::map([
        ("schema", Value::Str(TENSOR_SHARDED_SCHEMA.to_owned())),
        ("dtype", Value::Str(meta.dtype.as_str().to_owned())),
        (
            "shape",
            Value::Array(meta.shape.iter().map(|dim| Value::U64(*dim)).collect()),
        ),
        ("element_count", Value::U64(meta.element_count)),
        ("shard_axis", Value::U64(u64::from(meta.shard_axis))),
        ("shard_count", Value::U64(u64::from(meta.shard_count))),
        ("shard_index", Value::U64(u64::from(meta.shard_index))),
        ("block_size", Value::U64(meta.block_size)),
        (
            "scales",
            Value::Array(
                meta.scales
                    .iter()
                    .map(|bits| Value::U64(u64::from(*bits)))
                    .collect(),
            ),
        ),
        (
            "zero_points",
            Value::Array(
                meta.zero_points
                    .iter()
                    .map(|zero| Value::I64(*zero))
                    .collect(),
            ),
        ),
        (
            "layout_root",
            Value::Bytes(meta.layout_root.as_bytes().to_vec()),
        ),
        ("payload_len", Value::U64(meta.payload_len)),
        (
            "payload_hash",
            Value::Bytes(payload_hash.as_bytes().to_vec()),
        ),
    ])
}

fn split_ai_body(body: &[u8]) -> Result<(&[u8], &[u8])> {
    if body.len() < 4 {
        return Err(Error::malformed(
            "TENSOR_SHARDED body is shorter than its 4-byte metadata length",
        ));
    }
    let meta_len = u64::from(u32::from_le_bytes([body[0], body[1], body[2], body[3]]));
    let end = 4u64
        .checked_add(meta_len)
        .ok_or_else(|| Error::malformed("TENSOR_SHARDED metadata length overflows"))?;
    if end > body.len() as u64 {
        return Err(Error::malformed(
            "TENSOR_SHARDED metadata extends past the frame body",
        ));
    }
    let end = end as usize;
    Ok((&body[4..end], &body[end..]))
}

/// Build a frame from explicit planes and an already-encoded Huffman stream.
///
/// This is the low-level constructor used by the encoder and by adversarial
/// vector generation. It validates only the stored-length binding; callers are
/// responsible for descriptor semantics and for `decoded_len` consistency.
pub fn encode_shard_frame_with_stored(
    meta: &TensorShardedMeta,
    planes: &[u8],
    stored: &[u8],
    limits: &Limits,
) -> Result<Vec<u8>> {
    if meta.payload_len != stored.len() as u64 {
        return Err(Error::malformed(format!(
            "TENSOR_SHARDED payload_len {} does not match the stored stream length {}",
            meta.payload_len,
            stored.len()
        )));
    }
    let payload_hash = hash_domain(DOMAIN_TENSOR_SHARDED_PAYLOAD, stored);
    let meta_bytes = wcb::encode(&descriptor_value(meta, &payload_hash), limits)?;
    let meta_len = u32::try_from(meta_bytes.len())
        .map_err(|_| Error::limit_exceeded("TENSOR_SHARDED metadata does not fit u32"))?;
    let mut body = Vec::with_capacity(4 + meta_bytes.len() + stored.len());
    body.extend_from_slice(&meta_len.to_le_bytes());
    body.extend_from_slice(&meta_bytes);
    body.extend_from_slice(stored);
    let parts = FrameParts {
        format_major: crate::frame::FORMAT_MAJOR,
        format_minor: crate::frame::FORMAT_MINOR,
        domain: DOMAIN_PAYLOAD,
        mode: Mode::TensorSharded,
        flags: 0,
        level: 0,
        decoded_len: planes.len() as u64,
        content_hash: hash_domain(crate::hash::DOMAIN_CHUNK, planes),
        dependencies: &[],
        chunks: &[],
        body: &body,
    };
    encode_frame(&parts, limits)
}

/// Encode planes with `huffman-canonical-v1` and build the shard frame.
pub fn encode_shard_frame(
    meta: &TensorShardedMeta,
    planes: &[u8],
    limits: &Limits,
) -> Result<Vec<u8>> {
    let expected = meta.planes_len()?;
    if planes.len() as u64 != expected {
        return Err(Error::malformed(format!(
            "TENSOR_SHARDED planes are {} bytes but the descriptor requires {expected}",
            planes.len()
        )));
    }
    let stored = huffman::encode(planes);
    let mut meta = meta.clone();
    meta.payload_len = stored.len() as u64;
    encode_shard_frame_with_stored(&meta, planes, &stored, limits)
}

/// Encode a whole quantized tensor into `shard_count` mode-10 frames.
///
/// Frames are returned in `shard_index` order.
pub fn encode_sharded_tensor(input: &ShardedTensorInput, limits: &Limits) -> Result<Vec<Vec<u8>>> {
    let element_count = validate_shape(&input.shape, limits)?;
    let meta = TensorShardedMeta {
        dtype: input.dtype,
        shape: input.shape.clone(),
        element_count,
        shard_axis: input.shard_axis,
        shard_count: input.shard_count,
        shard_index: 0,
        block_size: input.block_size,
        scales: input.scales.clone(),
        zero_points: input.zero_points.clone(),
        layout_root: Hash32::ZERO,
        payload_len: 0,
    };
    let (element_count, _, _) = validate_descriptor(&meta, limits)?;
    if input.words.len() as u64 != element_count {
        return Err(Error::malformed(format!(
            "TENSOR_SHARDED tensor has {} words but shape requires {element_count}",
            input.words.len()
        )));
    }
    validate_words(input.dtype, &input.words)?;
    let root = logical_root(
        input.dtype,
        &input.shape,
        &input.words,
        &input.scales,
        &input.zero_points,
        limits,
    )?;
    let word_bytes: Vec<u8> = input.words.iter().map(|word| *word as u8).collect();
    let shard_bytes = tensor::split_shards(
        TensorDtype::I8,
        &input.shape,
        input.shard_axis,
        input.shard_count,
        &word_bytes,
        limits,
    )?;
    let scales = scale_plane(&input.scales);
    let mut frames = Vec::with_capacity(shard_bytes.len());
    for (index, shard) in shard_bytes.iter().enumerate() {
        let mut planes = quantize_shard(input.dtype, shard)?;
        planes.extend_from_slice(&scales);
        let meta = TensorShardedMeta {
            dtype: input.dtype,
            shape: input.shape.clone(),
            element_count,
            shard_axis: input.shard_axis,
            shard_count: input.shard_count,
            shard_index: index as u32,
            block_size: input.block_size,
            scales: input.scales.clone(),
            zero_points: input.zero_points.clone(),
            layout_root: root,
            payload_len: 0,
        };
        frames.push(encode_shard_frame(&meta, &planes, limits)?);
    }
    Ok(frames)
}

// ---------------------------------------------------------------------------
// Decoding
// ---------------------------------------------------------------------------

fn meta_text(map: &BTreeMap<String, Value>, key: &str) -> Result<String> {
    match map.get(key) {
        Some(Value::Str(text)) => Ok(text.clone()),
        _ => Err(Error::malformed(format!(
            "TENSOR_SHARDED metadata `{key}` must be text"
        ))),
    }
}

fn meta_u64(map: &BTreeMap<String, Value>, key: &str) -> Result<u64> {
    match map.get(key) {
        Some(Value::U64(value)) => Ok(*value),
        _ => Err(Error::malformed(format!(
            "TENSOR_SHARDED metadata `{key}` must be u64"
        ))),
    }
}

fn meta_u32(map: &BTreeMap<String, Value>, key: &str) -> Result<u32> {
    let value = meta_u64(map, key)?;
    u32::try_from(value).map_err(|_| {
        Error::malformed(format!(
            "TENSOR_SHARDED metadata `{key}` = {value} does not fit u32"
        ))
    })
}

fn meta_bytes(map: &BTreeMap<String, Value>, key: &str) -> Result<Vec<u8>> {
    match map.get(key) {
        Some(Value::Bytes(bytes)) => Ok(bytes.clone()),
        _ => Err(Error::malformed(format!(
            "TENSOR_SHARDED metadata `{key}` must be bytes"
        ))),
    }
}

fn meta_shape(map: &BTreeMap<String, Value>) -> Result<Vec<u64>> {
    match map.get("shape") {
        Some(Value::Array(items)) => {
            let mut shape = Vec::with_capacity(items.len());
            for item in items {
                match item {
                    Value::U64(dim) => shape.push(*dim),
                    _ => {
                        return Err(Error::malformed(
                            "TENSOR_SHARDED `shape` elements must be u64",
                        ))
                    }
                }
            }
            Ok(shape)
        }
        _ => Err(Error::malformed(
            "TENSOR_SHARDED metadata `shape` must be an array",
        )),
    }
}

fn meta_scales(map: &BTreeMap<String, Value>) -> Result<Vec<u32>> {
    match map.get("scales") {
        Some(Value::Array(items)) => {
            let mut scales = Vec::with_capacity(items.len());
            for item in items {
                match item {
                    Value::U64(value) => {
                        let bits = u32::try_from(*value).map_err(|_| {
                            Error::malformed(
                                "TENSOR_SHARDED scale bits exceed the 32-bit f32 pattern",
                            )
                        })?;
                        scales.push(bits);
                    }
                    _ => {
                        return Err(Error::malformed(
                            "TENSOR_SHARDED `scales` elements must be u64",
                        ))
                    }
                }
            }
            Ok(scales)
        }
        _ => Err(Error::malformed(
            "TENSOR_SHARDED metadata `scales` must be an array",
        )),
    }
}

fn meta_zero_points(map: &BTreeMap<String, Value>) -> Result<Vec<i64>> {
    match map.get("zero_points") {
        Some(Value::Array(items)) => {
            let mut points = Vec::with_capacity(items.len());
            for item in items {
                match item {
                    Value::I64(value) => points.push(*value),
                    _ => {
                        return Err(Error::malformed(
                            "TENSOR_SHARDED `zero_points` elements must be i64",
                        ))
                    }
                }
            }
            Ok(points)
        }
        _ => Err(Error::malformed(
            "TENSOR_SHARDED metadata `zero_points` must be an array",
        )),
    }
}

fn parse_descriptor(bytes: &[u8], limits: &Limits) -> Result<(TensorShardedMeta, Vec<u8>)> {
    let value = wcb::decode(bytes, limits)?;
    let map = match value {
        Value::Map(map) => map,
        _ => {
            return Err(Error::malformed(
                "TENSOR_SHARDED metadata must be a WCB map",
            ))
        }
    };
    const KEYS: [&str; 13] = [
        "schema",
        "dtype",
        "shape",
        "element_count",
        "shard_axis",
        "shard_count",
        "shard_index",
        "block_size",
        "scales",
        "zero_points",
        "layout_root",
        "payload_len",
        "payload_hash",
    ];
    for key in map.keys() {
        if !KEYS.contains(&key.as_str()) {
            return Err(Error::malformed(format!(
                "TENSOR_SHARDED metadata has unknown key `{key}`"
            )));
        }
    }
    for key in KEYS {
        if !map.contains_key(key) {
            return Err(Error::malformed(format!(
                "TENSOR_SHARDED metadata is missing key `{key}`"
            )));
        }
    }
    let schema = meta_text(&map, "schema")?;
    if schema != TENSOR_SHARDED_SCHEMA {
        return Err(Error::malformed(format!(
            "unsupported TENSOR_SHARDED schema `{schema}`"
        )));
    }
    let dtype = ShardedDtype::from_name(&meta_text(&map, "dtype")?)?;
    let shape = meta_shape(&map)?;
    let element_count = meta_u64(&map, "element_count")?;
    let shard_axis = meta_u32(&map, "shard_axis")?;
    let shard_count = meta_u32(&map, "shard_count")?;
    let shard_index = meta_u32(&map, "shard_index")?;
    let block_size = meta_u64(&map, "block_size")?;
    let scales = meta_scales(&map)?;
    let zero_points = meta_zero_points(&map)?;
    let root_bytes = meta_bytes(&map, "layout_root")?;
    if root_bytes.len() != crate::hash::HASH_LEN {
        return Err(Error::malformed(format!(
            "TENSOR_SHARDED layout_root must be {} bytes, got {}",
            crate::hash::HASH_LEN,
            root_bytes.len()
        )));
    }
    let layout_root = Hash32::from_slice(&root_bytes)?;
    let payload_len = meta_u64(&map, "payload_len")?;
    let payload_hash = meta_bytes(&map, "payload_hash")?;
    if payload_hash.len() != crate::hash::HASH_LEN {
        return Err(Error::malformed(format!(
            "TENSOR_SHARDED payload_hash must be {} bytes, got {}",
            crate::hash::HASH_LEN,
            payload_hash.len()
        )));
    }
    let meta = TensorShardedMeta {
        dtype,
        shape,
        element_count,
        shard_axis,
        shard_count,
        shard_index,
        block_size,
        scales,
        zero_points,
        layout_root,
        payload_len,
    };
    Ok((meta, payload_hash))
}

/// Validate and decode one TENSOR_SHARDED frame.
pub fn decode_shard_frame(frame: &Frame<'_>, limits: &Limits) -> Result<DecodedTensorShard> {
    if frame.header.mode != Mode::TensorSharded {
        return Err(Error::malformed(format!(
            "expected a TENSOR_SHARDED frame, got mode {}",
            frame.header.mode.as_str()
        )));
    }
    let (descriptor_bytes, stored) = split_ai_body(frame.body)?;
    let (meta, declared_payload_hash) = parse_descriptor(descriptor_bytes, limits)?;
    let (_, shard_elements, block_count) = validate_descriptor(&meta, limits)?;
    let planes_len = meta.planes_len()?;
    if meta.payload_len != stored.len() as u64 {
        return Err(Error::malformed(format!(
            "TENSOR_SHARDED payload is {} bytes but payload_len declares {}",
            stored.len(),
            meta.payload_len
        )));
    }
    if frame.header.decoded_len != planes_len {
        return Err(Error::corrupt(format!(
            "TENSOR_SHARDED decoded_len is {} but the descriptor requires {planes_len}",
            frame.header.decoded_len
        )));
    }
    let actual_payload_hash = hash_domain(DOMAIN_TENSOR_SHARDED_PAYLOAD, stored);
    if declared_payload_hash.as_slice() != actual_payload_hash.as_bytes().as_slice() {
        return Err(Error::corrupt("TENSOR_SHARDED payload_hash mismatch"));
    }
    crate::limits::check_len(
        planes_len,
        limits.max_decoded_bytes,
        "TENSOR_SHARDED reconstructed planes",
    )?;
    let planes_usize = usize::try_from(planes_len)
        .map_err(|_| Error::limit_exceeded("TENSOR_SHARDED planes do not fit usize"))?;
    let planes = huffman::decode(stored, planes_usize)?;
    let quant_len = usize::try_from(meta.dtype.quant_plane_len(shard_elements)?)
        .map_err(|_| Error::limit_exceeded("TENSOR_SHARDED quant plane does not fit usize"))?;
    if planes.len() != planes_usize || quant_len > planes.len() {
        return Err(Error::corrupt(
            "TENSOR_SHARDED reconstructed plane length is inconsistent",
        ));
    }
    let (quant_plane, scale_bytes) = planes.split_at(quant_len);
    let scale_count = usize::try_from(block_count)
        .map_err(|_| Error::limit_exceeded("TENSOR_SHARDED block count does not fit usize"))?;
    if scale_bytes.len() != scale_count.saturating_mul(4) {
        return Err(Error::corrupt(
            "TENSOR_SHARDED scale plane length is inconsistent",
        ));
    }
    for (index, chunk) in scale_bytes.chunks_exact(4).enumerate() {
        let actual = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        if actual != meta.scales[index] {
            return Err(Error::corrupt(format!(
                "TENSOR_SHARDED scale plane block {index} does not match the descriptor"
            )));
        }
    }
    if frame.header.content_hash != hash_domain(crate::hash::DOMAIN_CHUNK, &planes) {
        return Err(Error::corrupt(
            "TENSOR_SHARDED content_hash mismatch against the reconstructed planes",
        ));
    }
    let words = match meta.dtype {
        ShardedDtype::Q8_0 => quant_plane.iter().map(|byte| *byte as i8).collect(),
        ShardedDtype::Q4_0 => {
            let count = usize::try_from(shard_elements).map_err(|_| {
                Error::limit_exceeded("TENSOR_SHARDED shard elements do not fit usize")
            })?;
            unpack_q4(quant_plane, count)
        }
    };
    if words.len() as u64 != shard_elements {
        return Err(Error::corrupt(
            "TENSOR_SHARDED reconstructed word count is inconsistent",
        ));
    }
    Ok(DecodedTensorShard {
        meta,
        words,
        planes,
    })
}

/// Decode one shard and return only its reconstructed planes (the value bound
/// by `decoded_len` and `content_hash`).
pub fn decode_planes(frame: &Frame<'_>, limits: &Limits) -> Result<Vec<u8>> {
    Ok(decode_shard_frame(frame, limits)?.planes)
}

// ---------------------------------------------------------------------------
// Shard-set reassembly
// ---------------------------------------------------------------------------

fn same_tensor(left: &TensorShardedMeta, right: &TensorShardedMeta) -> bool {
    left.dtype == right.dtype
        && left.shape == right.shape
        && left.element_count == right.element_count
        && left.shard_axis == right.shard_axis
        && left.shard_count == right.shard_count
        && left.block_size == right.block_size
        && left.scales == right.scales
        && left.zero_points == right.zero_points
        && left.layout_root == right.layout_root
}

fn ordered_shards<'a>(
    shards: &'a [DecodedTensorShard],
    limits: &Limits,
) -> Result<(Vec<&'a DecodedTensorShard>, TensorShardedMeta)> {
    if shards.is_empty() {
        return Err(Error::malformed("TENSOR_SHARDED shard set is empty"));
    }
    let first = shards[0].meta.clone();
    validate_descriptor(&first, limits)?;
    let count = first.shard_count as usize;
    if shards.len() != count {
        return Err(Error::malformed(format!(
            "TENSOR_SHARDED shard set has {} entries but the descriptor declares {}",
            shards.len(),
            first.shard_count
        )));
    }
    let mut ordered: Vec<Option<&DecodedTensorShard>> = vec![None; count];
    for shard in shards {
        if !same_tensor(&first, &shard.meta) {
            return Err(Error::non_canonical(
                "TENSOR_SHARDED shard descriptors disagree on tensor identity",
            ));
        }
        let index = shard.meta.shard_index as usize;
        if index >= count {
            return Err(Error::non_canonical(format!(
                "TENSOR_SHARDED shard index {} is not below shard count {}",
                shard.meta.shard_index, first.shard_count
            )));
        }
        if ordered[index].is_some() {
            return Err(Error::non_canonical(format!(
                "duplicate TENSOR_SHARDED shard index {}",
                shard.meta.shard_index
            )));
        }
        ordered[index] = Some(shard);
    }
    let mut result = Vec::with_capacity(count);
    for (index, entry) in ordered.into_iter().enumerate() {
        match entry {
            Some(shard) => result.push(shard),
            None => {
                return Err(Error::non_canonical(format!(
                    "missing TENSOR_SHARDED shard index {index}"
                )))
            }
        }
    }
    Ok((result, first))
}

/// Reassemble the whole-tensor quantized words from a complete shard set.
pub fn assemble_sharded_words(shards: &[DecodedTensorShard], limits: &Limits) -> Result<Vec<i8>> {
    let (ordered, meta) = ordered_shards(shards, limits)?;
    assemble_ordered(&ordered, &meta, limits)
}

fn assemble_ordered(
    ordered: &[&DecodedTensorShard],
    meta: &TensorShardedMeta,
    limits: &Limits,
) -> Result<Vec<i8>> {
    let payloads: Vec<Vec<u8>> = ordered
        .iter()
        .map(|shard| shard.words.iter().map(|word| *word as u8).collect())
        .collect();
    let physical = tensor::PhysicalTensor::sharded(
        TensorDtype::I8,
        &meta.shape,
        meta.shard_axis,
        payloads,
        limits,
    )?;
    let bytes = physical.assemble_with_limits(limits)?;
    Ok(bytes.iter().map(|byte| *byte as i8).collect())
}

/// Compute the logical root from a complete shard set in any order.
///
/// Duplicate, missing, out-of-range, and inconsistent shards are rejected.
/// The recomputed root must equal every shard's declared `layout_root`.
pub fn shard_set_root(shards: &[DecodedTensorShard], limits: &Limits) -> Result<Hash32> {
    let (ordered, meta) = ordered_shards(shards, limits)?;
    let words = assemble_ordered(&ordered, &meta, limits)?;
    let root = logical_root(
        meta.dtype,
        &meta.shape,
        &words,
        &meta.scales,
        &meta.zero_points,
        limits,
    )?;
    if root != meta.layout_root {
        return Err(Error::corrupt(format!(
            "TENSOR_SHARDED shard set recomputes logical root {root}, but the descriptor declares {}",
            meta.layout_root
        )));
    }
    Ok(root)
}
