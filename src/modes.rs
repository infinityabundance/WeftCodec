//! Payload mode encoders/decoders (WEFTCODEC-FORMAT §3–§6).
//!
//! Every encoder computes the whole-frame bytes including header, tables, and
//! trailer. Every decoder verifies the header/trailer, every chunk hash, the
//! dependency hash/length at the materialization boundary, and the whole
//! content hash.

use std::collections::{BTreeMap, HashMap};

use crate::dependency::Resolver;
use crate::error::{Error, Result};
use crate::frame::{
    encode_frame, ChunkEntry, Dependency, Frame, FrameParts, Mode, DEP_DOMAIN_DICTIONARY,
    DEP_DOMAIN_PAYLOAD, DOMAIN_PAYLOAD, DOMAIN_STRUCTURED, FORMAT_MAJOR, FORMAT_MINOR,
};
use crate::hash::{
    hash_domain, Hash32, DOMAIN_CHUNK, DOMAIN_FLOAT_TENSOR_PAYLOAD,
    DOMAIN_QUANTIZED_TENSOR_PAYLOAD, DOMAIN_TENSOR_PAYLOAD, DOMAIN_TOKENS_PAYLOAD,
};
use crate::huffman;
use crate::limits::{check_len, Limits};
use crate::varint;
use crate::wcb::{self, Value};

/// Encode a RAW frame. `chunk_size` requests chunking when `Some`; an empty
/// input is never chunked (the chunk table cannot be empty when CHUNKED).
pub fn encode_raw(data: &[u8], chunk_size: Option<usize>, limits: &Limits) -> Result<Vec<u8>> {
    check_len(data.len() as u64, limits.max_decoded_bytes, "decoded data")?;
    validate_chunk_size(chunk_size)?;
    let content_hash = hash_domain(DOMAIN_CHUNK, data);
    let chunked = maybe_chunked(data, chunk_size, limits, |chunk| Ok(chunk.to_vec()))?;
    let (body, chunks, flags) = match chunked {
        Some((body, chunks)) => (body, chunks, crate::frame::flags::CHUNKED),
        None => (data.to_vec(), Vec::new(), 0),
    };
    let parts = FrameParts {
        format_major: FORMAT_MAJOR,
        format_minor: FORMAT_MINOR,
        domain: DOMAIN_PAYLOAD,
        mode: Mode::Raw,
        flags,
        level: 0,
        decoded_len: data.len() as u64,
        content_hash,
        dependencies: &[],
        chunks: &chunks,
        body: &body,
    };
    encode_frame(&parts, limits)
}

/// Encode a ZSTD frame with the given zstd level.
pub fn encode_zstd(
    data: &[u8],
    level: i32,
    chunk_size: Option<usize>,
    limits: &Limits,
) -> Result<Vec<u8>> {
    check_len(data.len() as u64, limits.max_decoded_bytes, "decoded data")?;
    validate_chunk_size(chunk_size)?;
    let content_hash = hash_domain(DOMAIN_CHUNK, data);
    let chunked = maybe_chunked(data, chunk_size, limits, |chunk| compress(chunk, level))?;
    let (body, chunks, flags) = match chunked {
        Some((body, chunks)) => (body, chunks, crate::frame::flags::CHUNKED),
        None => (compress(data, level)?, Vec::new(), 0),
    };
    let parts = FrameParts {
        format_major: FORMAT_MAJOR,
        format_minor: FORMAT_MINOR,
        domain: DOMAIN_PAYLOAD,
        mode: Mode::Zstd,
        flags,
        level: level_to_u16(level)?,
        decoded_len: data.len() as u64,
        content_hash,
        dependencies: &[],
        chunks: &chunks,
        body: &body,
    };
    encode_frame(&parts, limits)
}

/// Encode a REFERENCE frame. The input must equal `base` byte for byte.
pub fn encode_reference(data: &[u8], base: &[u8], limits: &Limits) -> Result<Vec<u8>> {
    if data != base {
        return Err(Error::malformed(
            "REFERENCE encoding requires the input to equal the supplied base",
        ));
    }
    check_len(data.len() as u64, limits.max_decoded_bytes, "decoded data")?;
    let dependency = dependency_for(base, DEP_DOMAIN_PAYLOAD, 0);
    let parts = FrameParts {
        format_major: FORMAT_MAJOR,
        format_minor: FORMAT_MINOR,
        domain: DOMAIN_PAYLOAD,
        mode: Mode::Reference,
        flags: 0,
        level: 0,
        decoded_len: data.len() as u64,
        content_hash: hash_domain(DOMAIN_CHUNK, data),
        dependencies: &[dependency],
        chunks: &[],
        body: &[],
    };
    encode_frame(&parts, limits)
}

/// Encode a DELTA frame against `base`.
pub fn encode_delta(data: &[u8], base: &[u8], limits: &Limits) -> Result<Vec<u8>> {
    check_len(data.len() as u64, limits.max_decoded_bytes, "decoded data")?;
    check_len(base.len() as u64, limits.max_decoded_bytes, "delta base")?;
    let body = encode_delta_ops(data, base, limits)?;
    let dependency = dependency_for(base, DEP_DOMAIN_PAYLOAD, 0);
    let parts = FrameParts {
        format_major: FORMAT_MAJOR,
        format_minor: FORMAT_MINOR,
        domain: DOMAIN_PAYLOAD,
        mode: Mode::Delta,
        flags: 0,
        level: 0,
        decoded_len: data.len() as u64,
        content_hash: hash_domain(DOMAIN_CHUNK, data),
        dependencies: &[dependency],
        chunks: &[],
        body: &body,
    };
    encode_frame(&parts, limits)
}

/// Encode a DICTIONARY frame using `dict` as a raw-content zstd dictionary.
pub fn encode_dictionary(data: &[u8], dict: &[u8], level: i32, limits: &Limits) -> Result<Vec<u8>> {
    check_len(data.len() as u64, limits.max_decoded_bytes, "decoded data")?;
    check_len(dict.len() as u64, limits.max_dictionary_bytes, "dictionary")?;
    let body = compress_with_dictionary(data, dict, level)?;
    let dependency = dependency_for(dict, DEP_DOMAIN_DICTIONARY, 0);
    let parts = FrameParts {
        format_major: FORMAT_MAJOR,
        format_minor: FORMAT_MINOR,
        domain: DOMAIN_PAYLOAD,
        mode: Mode::Dictionary,
        flags: crate::frame::flags::DICT_RAW,
        level: level_to_u16(level)?,
        decoded_len: data.len() as u64,
        content_hash: hash_domain(DOMAIN_CHUNK, data),
        dependencies: &[dependency],
        chunks: &[],
        body: &body,
    };
    encode_frame(&parts, limits)
}

/// Encode a STRUCTURED frame containing one canonical WCB value.
pub fn encode_structured(value: &Value, limits: &Limits) -> Result<Vec<u8>> {
    let body = wcb::encode(value, limits)?;
    check_len(
        body.len() as u64,
        limits.max_decoded_bytes,
        "structured body",
    )?;
    let parts = FrameParts {
        format_major: FORMAT_MAJOR,
        format_minor: FORMAT_MINOR,
        domain: DOMAIN_STRUCTURED,
        mode: Mode::Structured,
        flags: 0,
        level: 0,
        decoded_len: body.len() as u64,
        content_hash: hash_domain(DOMAIN_CHUNK, &body),
        dependencies: &[],
        chunks: &[],
        body: &body,
    };
    encode_frame(&parts, limits)
}

/// Decode a complete frame, resolving dependencies through `resolver`.
///
/// The returned bytes always satisfy `decoded_len == bytes.len()` and
/// `H("weft/chunk/v1", bytes) == header.content_hash`.
pub fn decode_bytes(
    frame_bytes: &[u8],
    resolver: &dyn Resolver,
    limits: &Limits,
) -> Result<Vec<u8>> {
    let frame = Frame::parse(frame_bytes, limits)?;
    let plaintext = match frame.header.mode {
        Mode::Raw => {
            if frame.header.is_chunked() {
                decode_chunks(&frame, limits)?
            } else {
                frame.body.to_vec()
            }
        }
        Mode::Zstd => {
            if frame.header.is_chunked() {
                decode_chunks(&frame, limits)?
            } else {
                decompress(frame.body, frame.header.decoded_len)?
            }
        }
        Mode::Reference => {
            let dep = resolve_dependency(&frame.dependencies[0], resolver)?;
            // The frame parser already checked decoded_len == dep.decoded_len.
            dep
        }
        Mode::Delta => {
            let dep = resolve_dependency(&frame.dependencies[0], resolver)?;
            decode_delta_ops(frame.body, &dep, frame.header.decoded_len, limits)?
        }
        Mode::Dictionary => {
            let dict = resolve_dependency(&frame.dependencies[0], resolver)?;
            decompress_with_dictionary(frame.body, frame.header.decoded_len, &dict)?
        }
        Mode::Structured => {
            // Validate canonical WCB before materializing the bytes.
            wcb::decode(frame.body, limits)?;
            frame.body.to_vec()
        }
        Mode::Tensor => decode_tensor(&frame, limits)?,
        Mode::TokenSequence => decode_token_sequence(&frame, limits)?,
        Mode::QuantizedTensor => decode_quantized_tensor(&frame, limits)?,
        Mode::FloatTensor => decode_float_tensor(&frame, limits)?,
        Mode::TensorSharded => crate::tensor_sharded::decode_planes(&frame, limits)?,
        Mode::WeightDelta => crate::weight_delta::decode_frame(&frame, resolver, limits)?,
        Mode::SemanticDelta => crate::semantic_delta::decode_frame(&frame, resolver, limits)?,
        Mode::SparseDelta => crate::sparse_delta::decode_frame(&frame, resolver, limits)?,
        Mode::KvCache => crate::kv_cache::decode_kv_page(&frame, limits)?.bytes,
        Mode::ActivationStream => {
            crate::activation_stream::decode_activation_chunk(&frame, limits)?.bytes
        }
        Mode::TensorTiled => decode_tiled_tensor(&frame, limits)?,
    };

    if plaintext.len() as u64 != frame.header.decoded_len {
        return Err(Error::corrupt(format!(
            "decoded length {} differs from declared {}",
            plaintext.len(),
            frame.header.decoded_len
        )));
    }
    if hash_domain(DOMAIN_CHUNK, &plaintext) != frame.header.content_hash {
        return Err(Error::corrupt("content hash mismatch"));
    }
    Ok(plaintext)
}

fn decode_chunks(frame: &Frame<'_>, limits: &Limits) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    let total = usize::try_from(frame.header.decoded_len)
        .map_err(|_| Error::limit_exceeded("decoded length does not fit usize"))?;
    out.reserve(total.min(1 << 20));
    for index in 0..frame.chunks.len() {
        let decoded = decode_chunk(frame, index)?;
        check_len(out.len() as u64, limits.max_decoded_bytes, "decoded data")?;
        out.extend_from_slice(&decoded);
    }
    Ok(out)
}

/// Decode and hash-verify one chunk independently, as if it were the body of a
/// frame with the same mode. Used by random-range reads to avoid decoding
/// chunks outside the requested range.
pub fn decode_chunk(frame: &Frame<'_>, index: usize) -> Result<Vec<u8>> {
    let chunk = frame
        .chunks
        .get(index)
        .ok_or_else(|| Error::malformed(format!("chunk index {index} out of range")))?;
    let start = usize::try_from(chunk.encoded_off)
        .map_err(|_| Error::malformed("chunk offset does not fit usize"))?;
    let len = chunk.encoded_len as usize;
    let end = start
        .checked_add(len)
        .ok_or_else(|| Error::malformed("chunk extent overflow"))?;
    let encoded = frame
        .body
        .get(start..end)
        .ok_or_else(|| Error::malformed(format!("chunk {index} extends past body")))?;
    let decoded = match frame.header.mode {
        Mode::Raw => {
            if encoded.len() as u64 != u64::from(chunk.decoded_len) {
                return Err(Error::malformed(format!(
                    "RAW chunk {index} encoded_len does not equal decoded_len"
                )));
            }
            encoded.to_vec()
        }
        Mode::Zstd => decompress(encoded, u64::from(chunk.decoded_len))?,
        _ => {
            return Err(Error::malformed(
                "chunk table is only valid for RAW and ZSTD frames",
            ))
        }
    };
    if decoded.len() as u64 != u64::from(chunk.decoded_len) {
        return Err(Error::corrupt(format!(
            "chunk {index} decoded length {} differs from declared {}",
            decoded.len(),
            chunk.decoded_len
        )));
    }
    if hash_domain(DOMAIN_CHUNK, &decoded) != chunk.hash {
        return Err(Error::corrupt(format!("chunk {index} hash mismatch")));
    }
    Ok(decoded)
}

/// Resolve and verify dependency bytes.
pub fn resolve_dependency(dep: &Dependency, resolver: &dyn Resolver) -> Result<Vec<u8>> {
    let bytes = resolver.resolve(&dep.hash, dep.decoded_len, dep.dep_domain)?;
    if bytes.len() as u64 != dep.decoded_len {
        return Err(Error::dependency_mismatch(format!(
            "dependency length {} differs from declared {}",
            bytes.len(),
            dep.decoded_len
        )));
    }
    if hash_domain(DOMAIN_CHUNK, &bytes) != dep.hash {
        return Err(Error::dependency_mismatch(
            "dependency content hash mismatch",
        ));
    }
    Ok(bytes)
}

fn dependency_for(bytes: &[u8], dep_domain: u8, codec_hint: u32) -> Dependency {
    Dependency::new(
        hash_domain(DOMAIN_CHUNK, bytes),
        bytes.len() as u64,
        dep_domain,
        codec_hint,
    )
}

fn validate_chunk_size(chunk_size: Option<usize>) -> Result<()> {
    if let Some(size) = chunk_size {
        if size == 0 {
            return Err(Error::malformed("chunk size must be greater than zero"));
        }
        if size > u32::MAX as usize {
            return Err(Error::limit_exceeded(
                "chunk size does not fit the 32-bit encoded_len field",
            ));
        }
    }
    Ok(())
}

fn maybe_chunked<F>(
    data: &[u8],
    chunk_size: Option<usize>,
    limits: &Limits,
    mut encode_chunk: F,
) -> Result<Option<(Vec<u8>, Vec<ChunkEntry>)>>
where
    F: FnMut(&[u8]) -> Result<Vec<u8>>,
{
    let size = match chunk_size {
        Some(size) if !data.is_empty() => size,
        _ => return Ok(None),
    };
    let count = data.len().div_ceil(size);
    check_len(count as u64, limits.max_chunks, "chunk count")?;
    let mut body = Vec::new();
    let mut chunks = Vec::with_capacity(count);
    let mut offset: u64 = 0;
    for chunk in data.chunks(size) {
        let encoded = encode_chunk(chunk)?;
        let encoded_len = u32::try_from(encoded.len())
            .map_err(|_| Error::limit_exceeded("encoded chunk does not fit u32"))?;
        let decoded_len = u32::try_from(chunk.len())
            .map_err(|_| Error::limit_exceeded("decoded chunk does not fit u32"))?;
        chunks.push(ChunkEntry {
            encoded_off: offset,
            encoded_len,
            decoded_len,
            hash: hash_domain(DOMAIN_CHUNK, chunk),
        });
        offset = offset
            .checked_add(u64::from(encoded_len))
            .ok_or_else(|| Error::limit_exceeded("chunk offset overflow"))?;
        body.extend_from_slice(&encoded);
    }
    Ok(Some((body, chunks)))
}

fn level_to_u16(level: i32) -> Result<u16> {
    u16::try_from(level).map_err(|_| {
        Error::limit_exceeded("zstd level is informational and must fit the u16 header field")
    })
}

fn compress(data: &[u8], level: i32) -> Result<Vec<u8>> {
    let mut compressor = zstd::bulk::Compressor::new(level)
        .map_err(|e| Error::malformed(format!("zstd compressor init failed: {e}")))?;
    compressor
        .compress(data)
        .map_err(|e| Error::malformed(format!("zstd compression failed: {e}")))
}

fn compress_with_dictionary(data: &[u8], dict: &[u8], level: i32) -> Result<Vec<u8>> {
    let mut compressor = zstd::bulk::Compressor::with_dictionary(level, dict)
        .map_err(|e| Error::malformed(format!("zstd dictionary init failed: {e}")))?;
    compressor
        .compress(data)
        .map_err(|e| Error::malformed(format!("zstd compression failed: {e}")))
}

/// Read exactly `declared_len` bytes from a zstd stream, never preallocating
/// the declared length. A tiny frame that *claims* a huge output therefore
/// costs only the bytes it actually produces.
fn read_exact_zstd<R: std::io::Read>(reader: R, declared_len: u64) -> Result<Vec<u8>> {
    use std::io::Read;

    let bound = declared_len.saturating_add(1);
    let mut out = Vec::new();
    reader
        .take(bound)
        .read_to_end(&mut out)
        .map_err(|e| Error::corrupt(format!("zstd decompression failed: {e}")))?;
    if out.len() as u64 != declared_len {
        return Err(Error::corrupt(format!(
            "zstd output length {} differs from declared {declared_len}",
            out.len()
        )));
    }
    Ok(out)
}

/// Validate the zstd frame preamble shared by `ZSTD`, `DICTIONARY`, and
/// chunked `ZSTD` payloads before decompression.
///
/// An empty payload is structural (`Malformed`): a zstd frame is never zero
/// bytes, even for empty plaintext. If the frame header declares a content
/// size, it MUST equal the Weft `decoded_len` (`Corrupt` on disagreement), so a
/// stream cannot declare a different length than the frame embedding it.
fn check_zstd_preamble(body: &[u8], declared_len: u64) -> Result<()> {
    if body.is_empty() {
        return Err(Error::malformed(
            "zstd payload is empty; a zstd frame is never zero bytes",
        ));
    }
    if let Ok(Some(frame_len)) = zstd::zstd_safe::get_frame_content_size(body) {
        if frame_len != declared_len {
            return Err(Error::corrupt(format!(
                "zstd frame declares content size {frame_len}, Weft header declares {declared_len}"
            )));
        }
    }
    Ok(())
}

/// Decompress one zstd frame with an exact output bound.
pub fn decompress(body: &[u8], declared_len: u64) -> Result<Vec<u8>> {
    check_zstd_preamble(body, declared_len)?;
    let decoder = zstd::stream::read::Decoder::new(body)
        .map_err(|e| Error::corrupt(format!("zstd decompressor init failed: {e}")))?;
    read_exact_zstd(decoder, declared_len)
}

/// Decompress one zstd frame using a raw-content dictionary, with an exact
/// output bound.
pub fn decompress_with_dictionary(body: &[u8], declared_len: u64, dict: &[u8]) -> Result<Vec<u8>> {
    check_zstd_preamble(body, declared_len)?;
    let decoder = zstd::stream::read::Decoder::with_dictionary(body, dict)
        .map_err(|e| Error::corrupt(format!("zstd dictionary init failed: {e}")))?;
    read_exact_zstd(decoder, declared_len)
}

const MIN_MATCH: usize = 8;
const MAX_CHAIN: usize = 4;
const MAX_INDEXED_POSITIONS: usize = 1_000_000;

#[derive(Clone, Copy, Default)]
struct MatchChain {
    len: u8,
    positions: [u32; MAX_CHAIN],
}

/// Encode a COPY/INSERT op stream against `base`.
///
/// The matcher indexes sampled 4-byte sequences of the base in a bounded hash
/// table (at most one million indexed positions, four candidates per key) and
/// greedily emits the longest match of at least 8 bytes. Each COPY reads the
/// original base, never the output. Deterministic for identical inputs.
pub fn encode_delta_ops(target: &[u8], base: &[u8], limits: &Limits) -> Result<Vec<u8>> {
    check_len(
        target.len() as u64,
        limits.max_decoded_bytes,
        "decoded data",
    )?;
    let mut out = Vec::new();
    let mut ops: u64 = 0;

    let indexable = base.len() <= u32::MAX as usize && target.len() >= 4;
    let stride = if base.len() > MAX_INDEXED_POSITIONS {
        base.len() / MAX_INDEXED_POSITIONS
    } else {
        1
    }
    .max(1);

    let mut table: HashMap<u32, MatchChain> = HashMap::new();
    if indexable && base.len() >= 4 {
        let mut pos = 0usize;
        while base.len() - pos >= 4 {
            if pos % stride == 0 {
                let key =
                    u32::from_le_bytes([base[pos], base[pos + 1], base[pos + 2], base[pos + 3]]);
                let chain = table.entry(key).or_default();
                let used = chain.len as usize;
                if used < MAX_CHAIN {
                    chain.positions[used] = pos as u32;
                    chain.len += 1;
                }
            }
            pos += 1;
        }
    }

    let mut literal_start = 0usize;
    let mut i = 0usize;
    while i < target.len() {
        let mut best_len = 0usize;
        let mut best_pos = 0usize;
        if indexable && target.len() - i >= 4 {
            let key = u32::from_le_bytes([target[i], target[i + 1], target[i + 2], target[i + 3]]);
            if let Some(chain) = table.get(&key) {
                for index in 0..chain.len as usize {
                    let base_pos = chain.positions[index] as usize;
                    let max_len = (base.len() - base_pos).min(target.len() - i);
                    let mut len = 0usize;
                    while len < max_len && base[base_pos + len] == target[i + len] {
                        len += 1;
                    }
                    if len > best_len {
                        best_len = len;
                        best_pos = base_pos;
                    }
                }
            }
        }
        if best_len >= MIN_MATCH {
            if literal_start < i {
                emit_insert(&target[literal_start..i], &mut out, &mut ops, limits)?;
            }
            out.push(0x00);
            varint::encode(best_pos as u64, &mut out);
            varint::encode(best_len as u64, &mut out);
            bump_op(&mut ops, limits)?;
            i += best_len;
            literal_start = i;
        } else {
            i += 1;
        }
    }
    if literal_start < target.len() {
        emit_insert(&target[literal_start..], &mut out, &mut ops, limits)?;
    }
    Ok(out)
}

fn bump_op(ops: &mut u64, limits: &Limits) -> Result<()> {
    *ops = ops
        .checked_add(1)
        .ok_or_else(|| Error::limit_exceeded("delta op count overflow"))?;
    if *ops > limits.max_delta_ops {
        return Err(Error::limit_exceeded(format!(
            "delta op count {} exceeds limit {}",
            *ops, limits.max_delta_ops
        )));
    }
    Ok(())
}

fn emit_insert(bytes: &[u8], out: &mut Vec<u8>, ops: &mut u64, limits: &Limits) -> Result<()> {
    if bytes.is_empty() {
        return Ok(());
    }
    out.push(0x01);
    varint::encode(bytes.len() as u64, out);
    out.extend_from_slice(bytes);
    bump_op(ops, limits)
}

/// Decode a COPY/INSERT op stream against `base`.
pub fn decode_delta_ops(
    body: &[u8],
    base: &[u8],
    expected_len: u64,
    limits: &Limits,
) -> Result<Vec<u8>> {
    check_len(expected_len, limits.max_decoded_bytes, "decoded data")?;
    let expected = usize::try_from(expected_len)
        .map_err(|_| Error::limit_exceeded("decoded length does not fit usize"))?;
    let mut out = Vec::with_capacity(expected.min(1 << 20));
    let mut pos = 0usize;
    let mut ops: u64 = 0;

    while pos < body.len() {
        bump_op(&mut ops, limits)?;
        let opcode = body[pos];
        pos += 1;
        match opcode {
            0x00 => {
                let offset = varint::decode(body, &mut pos)?;
                let length = varint::decode(body, &mut pos)?;
                if length == 0 {
                    return Err(Error::malformed("delta COPY has zero length"));
                }
                let end = offset
                    .checked_add(length)
                    .ok_or_else(|| Error::malformed("delta COPY offset+length overflow"))?;
                if end > base.len() as u64 {
                    return Err(Error::malformed(format!(
                        "delta COPY range {offset}..{end} exceeds base length {}",
                        base.len()
                    )));
                }
                let new_len = (out.len() as u64)
                    .checked_add(length)
                    .ok_or_else(|| Error::corrupt("delta output length overflow"))?;
                if new_len > expected_len {
                    return Err(Error::corrupt(
                        "delta emits more bytes than declared decoded_len",
                    ));
                }
                let start = usize::try_from(offset)
                    .map_err(|_| Error::malformed("delta COPY offset does not fit usize"))?;
                let len = usize::try_from(length)
                    .map_err(|_| Error::malformed("delta COPY length does not fit usize"))?;
                out.extend_from_slice(&base[start..start + len]);
            }
            0x01 => {
                let length = varint::decode(body, &mut pos)?;
                if length == 0 {
                    return Err(Error::malformed("delta INSERT has zero length"));
                }
                let len = usize::try_from(length)
                    .map_err(|_| Error::malformed("delta INSERT length does not fit usize"))?;
                let end = pos
                    .checked_add(len)
                    .ok_or_else(|| Error::malformed("delta INSERT extent overflow"))?;
                if end > body.len() {
                    return Err(Error::malformed("delta INSERT extends past body"));
                }
                let new_len = (out.len() as u64)
                    .checked_add(length)
                    .ok_or_else(|| Error::corrupt("delta output length overflow"))?;
                if new_len > expected_len {
                    return Err(Error::corrupt(
                        "delta emits more bytes than declared decoded_len",
                    ));
                }
                out.extend_from_slice(&body[pos..end]);
                pos = end;
            }
            other => {
                return Err(Error::malformed(format!(
                    "unknown delta opcode 0x{other:02x}"
                )))
            }
        }
    }
    if out.len() as u64 != expected_len {
        return Err(Error::corrupt(format!(
            "delta produced {} bytes but decoded_len is {expected_len}",
            out.len()
        )));
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// AI-native modes (WEFT-AI-CODEC §2–§3)
// ---------------------------------------------------------------------------
//
// Both modes carry a WCB descriptor followed by an exact payload:
//
//     u32le(metadata_len) || WCB(metadata) || payload
//
// The frame's `decoded_len` is the payload length and `content_hash` is
// `H("weft/chunk/v1", payload)`. The descriptor additionally binds the payload
// with a mode-specific hash domain, so swapping a descriptor cannot silently
// reinterpret bytes.

/// Tensor element dtypes implemented in v1.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TensorDtype {
    F16,
    Bf16,
    F32,
    F64,
    I8,
    U8,
    I16,
    I32,
    I64,
}

impl TensorDtype {
    /// Parse the canonical dtype name.
    pub fn from_name(name: &str) -> Result<Self> {
        match name {
            "f16" => Ok(TensorDtype::F16),
            "bf16" => Ok(TensorDtype::Bf16),
            "f32" => Ok(TensorDtype::F32),
            "f64" => Ok(TensorDtype::F64),
            "i8" => Ok(TensorDtype::I8),
            "u8" => Ok(TensorDtype::U8),
            "i16" => Ok(TensorDtype::I16),
            "i32" => Ok(TensorDtype::I32),
            "i64" => Ok(TensorDtype::I64),
            other => Err(Error::malformed(format!(
                "unknown tensor dtype `{other}` (v1 allows f16, bf16, f32, f64, i8, u8, i16, i32, i64)"
            ))),
        }
    }

    /// Canonical dtype name.
    pub fn as_str(self) -> &'static str {
        match self {
            TensorDtype::F16 => "f16",
            TensorDtype::Bf16 => "bf16",
            TensorDtype::F32 => "f32",
            TensorDtype::F64 => "f64",
            TensorDtype::I8 => "i8",
            TensorDtype::U8 => "u8",
            TensorDtype::I16 => "i16",
            TensorDtype::I32 => "i32",
            TensorDtype::I64 => "i64",
        }
    }

    /// Fixed element size in bytes.
    pub fn element_size(self) -> u64 {
        match self {
            TensorDtype::F16 | TensorDtype::Bf16 | TensorDtype::I16 => 2,
            TensorDtype::F32 | TensorDtype::I32 => 4,
            TensorDtype::F64 | TensorDtype::I64 => 8,
            TensorDtype::I8 | TensorDtype::U8 => 1,
        }
    }
}

/// Quantized tensor descriptor recovering the exact affine parameters.
#[derive(Clone, Debug, PartialEq)]
pub struct QuantizedTensorMeta {
    pub dtype: TensorDtype,
    pub shape: Vec<u64>,
    pub element_count: u64,
    pub payload_bytes: u64,
    pub scale: f64,
    pub zero_point: i64,
}

/// Token payload wire encoding implemented in v1.
pub const TOKEN_ENCODING: &str = "u64-zigzag-delta-leb128";

/// Canonical quantized-tensor schema.
pub const QUANTIZED_TENSOR_SCHEMA: &str = "weft.quantized-tensor.v1";
/// The only affine quantization layout implemented in v1.
pub const QUANTIZED_TENSOR_QUANTIZATION: &str = "affine-asymmetric-per-tensor";

/// Tensor descriptor recovered from a frame.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorMeta {
    pub dtype: TensorDtype,
    pub shape: Vec<u64>,
    pub element_count: u64,
    pub payload_bytes: u64,
}

/// Token-sequence descriptor recovered from a frame.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TokenMeta {
    pub vocab_schema: String,
    pub token_count: u64,
    pub payload_bytes: u64,
}

fn ai_map<'a>(value: &'a Value, what: &str) -> Result<&'a BTreeMap<String, Value>> {
    match value {
        Value::Map(map) => Ok(map),
        _ => Err(Error::malformed(format!(
            "{what} metadata must be a WCB map"
        ))),
    }
}

fn check_exact_keys(map: &BTreeMap<String, Value>, expected: &[&str], what: &str) -> Result<()> {
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

fn meta_text<'a>(map: &'a BTreeMap<String, Value>, key: &str) -> Result<&'a str> {
    match map.get(key) {
        Some(Value::Str(text)) => Ok(text.as_str()),
        _ => Err(Error::malformed(format!("metadata `{key}` must be text"))),
    }
}

fn meta_u64(map: &BTreeMap<String, Value>, key: &str) -> Result<u64> {
    match map.get(key) {
        Some(Value::U64(value)) => Ok(*value),
        _ => Err(Error::malformed(format!("metadata `{key}` must be u64"))),
    }
}

fn meta_i64(map: &BTreeMap<String, Value>, key: &str) -> Result<i64> {
    match map.get(key) {
        Some(Value::I64(value)) => Ok(*value),
        _ => Err(Error::malformed(format!("metadata `{key}` must be i64"))),
    }
}

fn meta_bytes<'a>(map: &'a BTreeMap<String, Value>, key: &str) -> Result<&'a [u8]> {
    match map.get(key) {
        Some(Value::Bytes(bytes)) => Ok(bytes.as_slice()),
        _ => Err(Error::malformed(format!("metadata `{key}` must be bytes"))),
    }
}

fn meta_shape(map: &BTreeMap<String, Value>) -> Result<Vec<u64>> {
    match map.get("shape") {
        Some(Value::Array(items)) => {
            let mut shape = Vec::with_capacity(items.len());
            for item in items {
                match item {
                    Value::U64(value) => shape.push(*value),
                    _ => return Err(Error::malformed("metadata `shape` elements must be u64")),
                }
            }
            Ok(shape)
        }
        _ => Err(Error::malformed("metadata `shape` must be an array")),
    }
}

fn tensor_product(shape: &[u64], limits: &Limits) -> Result<u64> {
    if shape.len() as u64 > limits.max_tensor_rank {
        return Err(Error::limit_exceeded(format!(
            "tensor rank {} exceeds limit {}",
            shape.len(),
            limits.max_tensor_rank
        )));
    }
    let mut product: u64 = 1;
    for dim in shape {
        product = product
            .checked_mul(*dim)
            .ok_or_else(|| Error::limit_exceeded("tensor shape product overflows u64"))?;
    }
    if product > limits.max_tensor_elements {
        return Err(Error::limit_exceeded(format!(
            "tensor element count {product} exceeds limit {}",
            limits.max_tensor_elements
        )));
    }
    Ok(product)
}

fn build_ai_body(meta: &Value, payload: &[u8], limits: &Limits) -> Result<Vec<u8>> {
    let meta_bytes = wcb::encode(meta, limits)?;
    let meta_len = u32::try_from(meta_bytes.len())
        .map_err(|_| Error::limit_exceeded("AI metadata does not fit u32"))?;
    let capacity = 4usize
        .checked_add(meta_bytes.len())
        .and_then(|value| value.checked_add(payload.len()))
        .ok_or_else(|| Error::limit_exceeded("AI body length overflow"))?;
    let mut body = Vec::with_capacity(capacity);
    body.extend_from_slice(&meta_len.to_le_bytes());
    body.extend_from_slice(&meta_bytes);
    body.extend_from_slice(payload);
    Ok(body)
}

fn split_ai_body<'a>(body: &'a [u8], decoded_len: u64) -> Result<(&'a [u8], &'a [u8])> {
    let (descriptor, payload) = split_ai_body_prefix(body)?;
    if payload.len() as u64 != decoded_len {
        return Err(Error::malformed(format!(
            "AI payload length {} differs from decoded_len {decoded_len}",
            payload.len()
        )));
    }
    Ok((descriptor, payload))
}

/// Split the descriptor prefix from the stored payload without assuming the
/// stored payload length equals `decoded_len`. FLOAT_TENSOR needs this because
/// the stored planes are smaller than the reconstructed element bytes.
fn split_ai_body_prefix(body: &[u8]) -> Result<(&[u8], &[u8])> {
    if body.len() < 4 {
        return Err(Error::malformed(
            "AI body is shorter than its 4-byte metadata length",
        ));
    }
    let meta_len = u32::from_le_bytes([body[0], body[1], body[2], body[3]]) as usize;
    let meta_end = 4usize
        .checked_add(meta_len)
        .ok_or_else(|| Error::malformed("AI metadata length overflow"))?;
    if meta_end > body.len() {
        return Err(Error::malformed("AI metadata extends past the frame body"));
    }
    Ok((&body[4..meta_end], &body[meta_end..]))
}

/// Encode a canonical tensor payload (WEFT-AI-CODEC §2).
///
/// The payload is the exact little-endian element bytes in row-major order.
/// No dtype conversion, normalization, or quantization happens here.
pub fn encode_tensor(
    payload: &[u8],
    dtype: TensorDtype,
    shape: &[u64],
    limits: &Limits,
) -> Result<Vec<u8>> {
    check_len(
        payload.len() as u64,
        limits.max_decoded_bytes,
        "tensor payload",
    )?;
    let element_count = tensor_product(shape, limits)?;
    let payload_bytes = element_count
        .checked_mul(dtype.element_size())
        .ok_or_else(|| Error::limit_exceeded("tensor payload byte count overflows u64"))?;
    if payload_bytes != payload.len() as u64 {
        return Err(Error::malformed(format!(
            "tensor payload is {} bytes but shape {} x {} is {payload_bytes} bytes",
            payload.len(),
            dtype.as_str(),
            element_count
        )));
    }

    let meta = Value::map([
        ("schema", Value::Str("weft.tensor.v1".to_owned())),
        ("dtype", Value::Str(dtype.as_str().to_owned())),
        (
            "shape",
            Value::Array(shape.iter().map(|dim| Value::U64(*dim)).collect()),
        ),
        ("layout", Value::Str("row_major".to_owned())),
        ("endianness", Value::Str("little".to_owned())),
        ("element_count", Value::U64(element_count)),
        ("payload_bytes", Value::U64(payload_bytes)),
        (
            "payload_hash",
            Value::Bytes(
                hash_domain(DOMAIN_TENSOR_PAYLOAD, payload)
                    .as_bytes()
                    .to_vec(),
            ),
        ),
    ]);
    let body = build_ai_body(&meta, payload, limits)?;
    let parts = FrameParts {
        format_major: FORMAT_MAJOR,
        format_minor: FORMAT_MINOR,
        domain: DOMAIN_PAYLOAD,
        mode: Mode::Tensor,
        flags: 0,
        level: 0,
        decoded_len: payload.len() as u64,
        content_hash: hash_domain(DOMAIN_CHUNK, payload),
        dependencies: &[],
        chunks: &[],
        body: &body,
    };
    encode_frame(&parts, limits)
}

/// Validate and decode a TENSOR frame, returning the exact element payload.
pub fn decode_tensor(frame: &Frame<'_>, limits: &Limits) -> Result<Vec<u8>> {
    let (meta, payload) = parse_tensor_body(frame, limits)?;
    let _ = meta;
    Ok(payload.to_vec())
}

/// Parse and validate a TENSOR descriptor, returning it with the payload slice.
pub fn tensor_metadata(frame: &Frame<'_>, limits: &Limits) -> Result<TensorMeta> {
    let (meta, _) = parse_tensor_body(frame, limits)?;
    Ok(meta)
}

fn parse_tensor_body<'a>(frame: &'a Frame<'_>, limits: &Limits) -> Result<(TensorMeta, &'a [u8])> {
    let (descriptor_bytes, payload) = split_ai_body(frame.body, frame.header.decoded_len)?;
    let value = wcb::decode(descriptor_bytes, limits)?;
    let map = ai_map(&value, "tensor")?;
    check_exact_keys(
        map,
        &[
            "schema",
            "dtype",
            "shape",
            "layout",
            "endianness",
            "element_count",
            "payload_bytes",
            "payload_hash",
        ],
        "tensor",
    )?;
    let schema = meta_text(map, "schema")?;
    if schema != "weft.tensor.v1" {
        return Err(Error::malformed(format!(
            "unsupported tensor schema `{schema}`"
        )));
    }
    let dtype = TensorDtype::from_name(meta_text(map, "dtype")?)?;
    let layout = meta_text(map, "layout")?;
    if layout != "row_major" {
        return Err(Error::malformed(format!(
            "tensor layout `{layout}` is specified but not implemented in v1 (only row_major)"
        )));
    }
    let endianness = meta_text(map, "endianness")?;
    if endianness != "little" {
        return Err(Error::malformed(format!(
            "tensor endianness `{endianness}` is specified but not implemented in v1 (only little)"
        )));
    }
    let shape = meta_shape(map)?;
    let declared_count = meta_u64(map, "element_count")?;
    let declared_bytes = meta_u64(map, "payload_bytes")?;
    let expected_count = tensor_product(&shape, limits)?;
    if declared_count != expected_count {
        return Err(Error::malformed(format!(
            "tensor element_count {declared_count} does not match shape product {expected_count}"
        )));
    }
    let expected_bytes = declared_count
        .checked_mul(dtype.element_size())
        .ok_or_else(|| Error::limit_exceeded("tensor payload byte count overflows u64"))?;
    if declared_bytes != expected_bytes {
        return Err(Error::malformed(format!(
            "tensor payload_bytes {declared_bytes} does not match element_count x element_size {expected_bytes}"
        )));
    }
    if declared_bytes != payload.len() as u64 {
        return Err(Error::malformed(format!(
            "tensor payload is {} bytes but metadata declares {declared_bytes}",
            payload.len()
        )));
    }
    let declared_hash = meta_bytes(map, "payload_hash")?;
    if declared_hash.len() != 32 {
        return Err(Error::malformed(format!(
            "tensor payload_hash must be 32 bytes, got {}",
            declared_hash.len()
        )));
    }
    let actual_hash = hash_domain(DOMAIN_TENSOR_PAYLOAD, payload);
    if declared_hash != actual_hash.as_bytes().as_slice() {
        return Err(Error::corrupt("tensor payload_hash mismatch"));
    }
    Ok((
        TensorMeta {
            dtype,
            shape,
            element_count: declared_count,
            payload_bytes: declared_bytes,
        },
        payload,
    ))
}

/// Encode an affine-quantized tensor (WEFT-AI-CODEC §4).
///
/// The payload is the packed `i8`/`u8` element bytes in row-major order. The
/// descriptor carries the exact f64 scale bit pattern and the i64 zero point;
/// no dequantization or dtype conversion happens here.
pub fn encode_quantized_tensor(
    payload: &[u8],
    dtype: TensorDtype,
    shape: &[u64],
    scale: f64,
    zero_point: i64,
    limits: &Limits,
) -> Result<Vec<u8>> {
    if !matches!(dtype, TensorDtype::I8 | TensorDtype::U8) {
        return Err(Error::malformed(format!(
            "quantized tensor dtype `{}` is not implemented (v1 allows i8, u8)",
            dtype.as_str()
        )));
    }
    check_len(
        payload.len() as u64,
        limits.max_decoded_bytes,
        "quantized tensor payload",
    )?;
    validate_quantized_scale(scale)?;
    validate_zero_point(dtype, zero_point)?;
    let element_count = tensor_product(shape, limits)?;
    if element_count != payload.len() as u64 {
        return Err(Error::malformed(format!(
            "quantized tensor payload is {} bytes but shape {} has {element_count} elements",
            payload.len(),
            dtype.as_str()
        )));
    }

    let meta = Value::map([
        ("schema", Value::Str(QUANTIZED_TENSOR_SCHEMA.to_owned())),
        ("dtype", Value::Str(dtype.as_str().to_owned())),
        (
            "shape",
            Value::Array(shape.iter().map(|dim| Value::U64(*dim)).collect()),
        ),
        ("layout", Value::Str("row_major".to_owned())),
        (
            "quantization",
            Value::Str(QUANTIZED_TENSOR_QUANTIZATION.to_owned()),
        ),
        (
            "scale_bits_le",
            Value::Bytes(scale.to_bits().to_le_bytes().to_vec()),
        ),
        ("zero_point", Value::I64(zero_point)),
        ("element_count", Value::U64(element_count)),
        ("payload_bytes", Value::U64(element_count)),
        (
            "payload_hash",
            Value::Bytes(
                hash_domain(DOMAIN_QUANTIZED_TENSOR_PAYLOAD, payload)
                    .as_bytes()
                    .to_vec(),
            ),
        ),
    ]);
    let body = build_ai_body(&meta, payload, limits)?;
    let parts = FrameParts {
        format_major: FORMAT_MAJOR,
        format_minor: FORMAT_MINOR,
        domain: DOMAIN_PAYLOAD,
        mode: Mode::QuantizedTensor,
        flags: 0,
        level: 0,
        decoded_len: payload.len() as u64,
        content_hash: hash_domain(DOMAIN_CHUNK, payload),
        dependencies: &[],
        chunks: &[],
        body: &body,
    };
    encode_frame(&parts, limits)
}

/// Validate and decode a QUANTIZED_TENSOR frame, returning the exact packed
/// element payload.
pub fn decode_quantized_tensor(frame: &Frame<'_>, limits: &Limits) -> Result<Vec<u8>> {
    let (_, payload) = parse_quantized_tensor_body(frame, limits)?;
    Ok(payload.to_vec())
}

/// Parse and validate a QUANTIZED_TENSOR descriptor, returning it with the
/// payload slice.
pub fn quantized_tensor_metadata(
    frame: &Frame<'_>,
    limits: &Limits,
) -> Result<QuantizedTensorMeta> {
    let (meta, _) = parse_quantized_tensor_body(frame, limits)?;
    Ok(meta)
}

fn validate_quantized_scale(scale: f64) -> Result<()> {
    if !scale.is_finite() {
        return Err(Error::malformed(
            "quantized tensor scale must be finite (no NaN or infinity)",
        ));
    }
    if scale <= 0.0 {
        return Err(Error::malformed(
            "quantized tensor scale must be greater than zero",
        ));
    }
    Ok(())
}

fn validate_zero_point(dtype: TensorDtype, zero_point: i64) -> Result<()> {
    let (min, max) = match dtype {
        TensorDtype::I8 => (i8::MIN as i64, i8::MAX as i64),
        TensorDtype::U8 => (0, u8::MAX as i64),
        _ => return Err(Error::malformed("quantized tensor dtype must be i8 or u8")),
    };
    if zero_point < min || zero_point > max {
        return Err(Error::malformed(format!(
            "quantized tensor zero_point {zero_point} is outside {} range {min}..={max}",
            dtype.as_str()
        )));
    }
    Ok(())
}

fn parse_quantized_tensor_body<'a>(
    frame: &'a Frame<'_>,
    limits: &Limits,
) -> Result<(QuantizedTensorMeta, &'a [u8])> {
    let (descriptor_bytes, payload) = split_ai_body(frame.body, frame.header.decoded_len)?;
    let value = wcb::decode(descriptor_bytes, limits)?;
    let map = ai_map(&value, "quantized tensor")?;
    check_exact_keys(
        map,
        &[
            "schema",
            "dtype",
            "shape",
            "layout",
            "quantization",
            "scale_bits_le",
            "zero_point",
            "element_count",
            "payload_bytes",
            "payload_hash",
        ],
        "quantized tensor",
    )?;
    let schema = meta_text(map, "schema")?;
    if schema != QUANTIZED_TENSOR_SCHEMA {
        return Err(Error::malformed(format!(
            "unsupported quantized tensor schema `{schema}`"
        )));
    }
    let dtype = TensorDtype::from_name(meta_text(map, "dtype")?)?;
    if !matches!(dtype, TensorDtype::I8 | TensorDtype::U8) {
        return Err(Error::malformed(format!(
            "quantized tensor dtype `{}` is specified but not implemented in v1 (only i8, u8)",
            dtype.as_str()
        )));
    }
    let layout = meta_text(map, "layout")?;
    if layout != "row_major" {
        return Err(Error::malformed(format!(
            "quantized tensor layout `{layout}` is specified but not implemented in v1 (only row_major)"
        )));
    }
    let quantization = meta_text(map, "quantization")?;
    if quantization != QUANTIZED_TENSOR_QUANTIZATION {
        return Err(Error::malformed(format!(
            "quantized tensor quantization `{quantization}` is specified but not implemented in v1"
        )));
    }
    let scale_bits = meta_bytes(map, "scale_bits_le")?;
    if scale_bits.len() != 8 {
        return Err(Error::malformed(format!(
            "quantized tensor scale_bits_le must be 8 bytes, got {}",
            scale_bits.len()
        )));
    }
    let scale =
        f64::from_bits(u64::from_le_bytes(scale_bits.try_into().map_err(|_| {
            Error::malformed("quantized tensor scale_bits_le is not 8 bytes")
        })?));
    validate_quantized_scale(scale)?;
    let zero_point = meta_i64(map, "zero_point")?;
    validate_zero_point(dtype, zero_point)?;
    let shape = meta_shape(map)?;
    let declared_count = meta_u64(map, "element_count")?;
    let declared_bytes = meta_u64(map, "payload_bytes")?;
    let expected_count = tensor_product(&shape, limits)?;
    if declared_count != expected_count {
        return Err(Error::malformed(format!(
            "quantized tensor element_count {declared_count} does not match shape product {expected_count}"
        )));
    }
    if declared_bytes != declared_count {
        return Err(Error::malformed(format!(
            "quantized tensor payload_bytes {declared_bytes} does not match element_count {declared_count} for one-byte elements"
        )));
    }
    if declared_bytes != payload.len() as u64 {
        return Err(Error::malformed(format!(
            "quantized tensor payload is {} bytes but metadata declares {declared_bytes}",
            payload.len()
        )));
    }
    let declared_hash = meta_bytes(map, "payload_hash")?;
    if declared_hash.len() != 32 {
        return Err(Error::malformed(format!(
            "quantized tensor payload_hash must be 32 bytes, got {}",
            declared_hash.len()
        )));
    }
    let actual_hash = hash_domain(DOMAIN_QUANTIZED_TENSOR_PAYLOAD, payload);
    if declared_hash != actual_hash.as_bytes().as_slice() {
        return Err(Error::corrupt("quantized tensor payload_hash mismatch"));
    }
    Ok((
        QuantizedTensorMeta {
            dtype,
            shape,
            element_count: declared_count,
            payload_bytes: declared_bytes,
            scale,
            zero_point,
        },
        payload,
    ))
}

// ---------------------------------------------------------------------------
// FLOAT_TENSOR (mode 9, WEFT-AI-CODEC §5)
// ---------------------------------------------------------------------------

/// Canonical float-tensor schema.
pub const FLOAT_TENSOR_SCHEMA: &str = "weft.ai.float_tensor.v1";
/// The exponent-plane coder (WEFT-AI-CODEC §5).
pub const FLOAT_TENSOR_EXPONENT_CODER: &str = huffman::CODER_ID;
/// Frame `level` for FLOAT_TENSOR. No zstd frame is used, so the field is 0.
pub const FLOAT_TENSOR_LEVEL: u16 = 0;

/// Float element dtypes implemented in v1. All are byte-aligned.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FloatDtype {
    Bf16,
    F16,
    F32,
    Fp8E4m3,
    Fp8E5m2,
}

impl FloatDtype {
    /// Parse the canonical dtype name.
    pub fn from_name(name: &str) -> Result<Self> {
        match name {
            "bf16" => Ok(FloatDtype::Bf16),
            "fp16" => Ok(FloatDtype::F16),
            "fp32" => Ok(FloatDtype::F32),
            "fp8-e4m3" => Ok(FloatDtype::Fp8E4m3),
            "fp8-e5m2" => Ok(FloatDtype::Fp8E5m2),
            other => Err(Error::malformed(format!(
                "unknown float tensor dtype `{other}` (v1 allows bf16, fp16, fp32, fp8-e4m3, fp8-e5m2)"
            ))),
        }
    }

    /// Canonical dtype name.
    pub fn as_str(self) -> &'static str {
        match self {
            FloatDtype::Bf16 => "bf16",
            FloatDtype::F16 => "fp16",
            FloatDtype::F32 => "fp32",
            FloatDtype::Fp8E4m3 => "fp8-e4m3",
            FloatDtype::Fp8E5m2 => "fp8-e5m2",
        }
    }

    /// Field widths `(sign, exponent, mantissa)`.
    pub fn field_bits(self) -> (u64, u64, u64) {
        match self {
            FloatDtype::Bf16 => (1, 8, 7),
            FloatDtype::F16 => (1, 5, 10),
            FloatDtype::F32 => (1, 8, 23),
            FloatDtype::Fp8E4m3 => (1, 4, 3),
            FloatDtype::Fp8E5m2 => (1, 5, 2),
        }
    }

    pub fn sign_bits(self) -> u64 {
        self.field_bits().0
    }

    pub fn exponent_bits(self) -> u64 {
        self.field_bits().1
    }

    pub fn mantissa_bits(self) -> u64 {
        self.field_bits().2
    }

    /// Total element width in bits. Byte-aligned for every v1 dtype.
    pub fn element_bits(self) -> u64 {
        let (sign, exponent, mantissa) = self.field_bits();
        sign + exponent + mantissa
    }

    /// Element width in bytes; only valid for byte-aligned dtypes.
    pub fn element_size(self) -> u64 {
        self.element_bits() / 8
    }
}

/// Float-tensor descriptor recovered from a frame.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FloatTensorMeta {
    pub dtype: FloatDtype,
    pub shape: Vec<u64>,
    pub element_count: u64,
    pub sign_bits: u64,
    pub exponent_bits: u64,
    pub mantissa_bits: u64,
    pub exponent_plane_len: u64,
    pub mantissa_plane_len: u64,
}

fn meta_byte_field(map: &BTreeMap<String, Value>, key: &str) -> Result<u64> {
    let value = meta_u64(map, key)?;
    if value > u64::from(u8::MAX) {
        return Err(Error::malformed(format!(
            "float tensor `{key}` must fit an unsigned byte, got {value}"
        )));
    }
    Ok(value)
}

fn float_tensor_product(shape: &[u64], limits: &Limits) -> Result<u64> {
    if shape.is_empty() {
        return Err(Error::malformed(
            "float tensor shape rank must be at least 1",
        ));
    }
    if shape.len() as u64 > limits.max_tensor_rank {
        return Err(Error::limit_exceeded(format!(
            "float tensor rank {} exceeds limit {}",
            shape.len(),
            limits.max_tensor_rank
        )));
    }
    let mut product: u64 = 1;
    for dim in shape {
        if *dim == 0 {
            return Err(Error::malformed(
                "float tensor shape dimensions must be greater than zero",
            ));
        }
        product = product
            .checked_mul(*dim)
            .ok_or_else(|| Error::limit_exceeded("float tensor shape product overflows u64"))?;
    }
    if product > limits.max_tensor_elements {
        return Err(Error::limit_exceeded(format!(
            "float tensor element count {product} exceeds limit {}",
            limits.max_tensor_elements
        )));
    }
    Ok(product)
}

/// Append `count` low bits of `value`, most significant first, at the current
/// bit position. `out` grows by one byte whenever a new byte is started.
fn push_bits(out: &mut Vec<u8>, bit_len: &mut usize, value: u64, count: usize) {
    for shift in (0..count).rev() {
        let bit = ((value >> shift) & 1) as u8;
        let byte_index = *bit_len / 8;
        if byte_index == out.len() {
            out.push(0);
        }
        out[byte_index] |= bit << (7 - (*bit_len % 8));
        *bit_len += 1;
    }
}

/// Split a little-endian float tensor payload into its two planes.
///
/// Returns `(exponent_plane_huffman, mantissa_plane)`. The exponent plane is
/// encoded with `huffman-canonical-v1`; the sign+mantissa plane is raw.
pub fn split_float_tensor_payload(
    payload: &[u8],
    dtype: FloatDtype,
    element_count: u64,
) -> Result<(Vec<u8>, Vec<u8>)> {
    let expected_bytes = element_count
        .checked_mul(dtype.element_size())
        .ok_or_else(|| Error::limit_exceeded("float tensor decoded byte count overflows u64"))?;
    if payload.len() as u64 != expected_bytes {
        return Err(Error::malformed(format!(
            "float tensor payload is {} bytes but the descriptor requires {expected_bytes}",
            payload.len()
        )));
    }
    let element_bytes = usize::try_from(dtype.element_size())
        .map_err(|_| Error::limit_exceeded("float tensor element size does not fit usize"))?;
    let count = usize::try_from(element_count)
        .map_err(|_| Error::limit_exceeded("float tensor element count does not fit usize"))?;
    let sign_bits = dtype.sign_bits() as usize;
    let exponent_bits = dtype.exponent_bits() as usize;
    let mantissa_bits = dtype.mantissa_bits() as usize;
    let mantissa_mask = (1u64 << mantissa_bits) - 1;
    let exponent_mask = (1u64 << exponent_bits) - 1;

    let mut exponent_plane = Vec::with_capacity(count.min(1 << 20));
    let mut mantissa_plane = Vec::new();
    let mut bit_len = 0usize;
    for chunk in payload.chunks_exact(element_bytes) {
        let mut word = 0u64;
        for (index, byte) in chunk.iter().enumerate() {
            word |= u64::from(*byte) << (8 * index);
        }
        let exponent = (word >> mantissa_bits) & exponent_mask;
        let mantissa = word & mantissa_mask;
        let sign = (word >> (exponent_bits + mantissa_bits)) & 1;
        exponent_plane.push(exponent as u8);
        push_bits(&mut mantissa_plane, &mut bit_len, sign, sign_bits);
        push_bits(&mut mantissa_plane, &mut bit_len, mantissa, mantissa_bits);
    }
    Ok((huffman::encode(&exponent_plane), mantissa_plane))
}

/// Reconstruct the exact little-endian element bytes from the two planes.
pub fn join_float_tensor_payload(
    exponent_plane: &[u8],
    mantissa_plane: &[u8],
    dtype: FloatDtype,
    element_count: u64,
    limits: &Limits,
) -> Result<Vec<u8>> {
    let count = usize::try_from(element_count)
        .map_err(|_| Error::limit_exceeded("float tensor element count does not fit usize"))?;
    if exponent_plane.len() != count {
        return Err(Error::corrupt(format!(
            "float tensor exponent plane has {} bytes but element_count is {element_count}",
            exponent_plane.len()
        )));
    }
    let element_bytes = usize::try_from(dtype.element_size())
        .map_err(|_| Error::limit_exceeded("float tensor element size does not fit usize"))?;
    let decoded_bytes = element_count
        .checked_mul(dtype.element_size())
        .ok_or_else(|| Error::limit_exceeded("float tensor decoded byte count overflows u64"))?;
    check_len(
        decoded_bytes,
        limits.max_decoded_bytes,
        "float tensor decoded bytes",
    )?;
    let sign_bits = dtype.sign_bits() as usize;
    let exponent_bits = dtype.exponent_bits() as usize;
    let mantissa_bits = dtype.mantissa_bits() as usize;
    let expected_mantissa_bits = element_count
        .checked_mul((sign_bits + mantissa_bits) as u64)
        .ok_or_else(|| Error::limit_exceeded("float tensor mantissa bit count overflows u64"))?;
    let expected_mantissa_bytes = expected_mantissa_bits
        .checked_add(7)
        .ok_or_else(|| Error::limit_exceeded("float tensor mantissa length overflows u64"))?
        / 8;
    if mantissa_plane.len() as u64 != expected_mantissa_bytes {
        return Err(Error::corrupt(format!(
            "float tensor mantissa plane has {} bytes but descriptor arithmetic requires {expected_mantissa_bytes}",
            mantissa_plane.len()
        )));
    }

    let capacity = usize::try_from(decoded_bytes)
        .map_err(|_| Error::limit_exceeded("float tensor decoded bytes do not fit usize"))?;

    // The output is exactly the declared decoded byte count, which was bounded
    // by `max_decoded_bytes` above, so the allocation is bounded as well.
    let mut out = vec![0u8; capacity];
    let mantissa_mask = (1u64 << mantissa_bits) - 1;
    let width = sign_bits + mantissa_bits;
    let width_bits = width as u32;

    // MSB-first 64-bit accumulator holding the next `accumulated_bits` bits in
    // its high positions and zeros below: refill eight bits at a time while a
    // whole byte still fits, then read the whole sign+mantissa width in one
    // shift and drop it by shifting left. `accumulated_bits >= width` always
    // holds because the mantissa plane is exactly
    // ceil(element_count * width / 8) bytes and `width <= 24`.
    let mut accumulator = 0u64;
    let mut accumulated_bits = 0u32;
    let mut plane_pos = 0usize;
    for (chunk, &exponent_byte) in out
        .chunks_exact_mut(element_bytes)
        .zip(exponent_plane.iter())
    {
        if u64::from(exponent_byte) >> exponent_bits != 0 {
            return Err(Error::corrupt(
                "float tensor exponent byte exceeds the dtype exponent width",
            ));
        }
        while accumulated_bits <= 56 && plane_pos < mantissa_plane.len() {
            accumulator |= u64::from(mantissa_plane[plane_pos]) << (56 - accumulated_bits);
            plane_pos += 1;
            accumulated_bits += 8;
        }
        let value = accumulator >> (64 - width_bits);
        accumulator <<= width_bits;
        accumulated_bits -= width_bits;

        let sign = value >> mantissa_bits;
        let mantissa = value & mantissa_mask;
        let word = (sign << (exponent_bits + mantissa_bits))
            | (u64::from(exponent_byte) << mantissa_bits)
            | mantissa;
        chunk.copy_from_slice(&word.to_le_bytes()[..element_bytes]);
    }

    // Unused low bits of the final mantissa byte are canonical zero padding.
    let bit_pos = expected_mantissa_bits;
    let used_bytes_bits = (mantissa_plane.len() as u64) * 8;
    let padding = used_bytes_bits - bit_pos;
    if padding > 0 {
        let last = *mantissa_plane.last().expect("non-empty mantissa plane");
        let mask = (1u8 << padding) - 1;
        if last & mask != 0 {
            return Err(Error::corrupt(
                "float tensor mantissa padding bits must be zero",
            ));
        }
    }
    Ok(out)
}

/// Encode a FLOAT_TENSOR frame (WEFT-AI-CODEC §5).
///
/// The exponent plane is encoded with the canonical `huffman-canonical-v1`
/// coder; the sign+mantissa plane is stored raw. Every element bit pattern
/// round-trips, including NaN payloads.
pub fn encode_float_tensor(
    payload: &[u8],
    dtype: FloatDtype,
    shape: &[u64],
    limits: &Limits,
) -> Result<Vec<u8>> {
    check_len(
        payload.len() as u64,
        limits.max_decoded_bytes,
        "float tensor payload",
    )?;
    let element_bits = dtype.element_bits();
    if element_bits % 8 != 0 {
        return Err(Error::malformed(format!(
            "float tensor dtype `{}` is not byte-aligned",
            dtype.as_str()
        )));
    }
    let element_count = float_tensor_product(shape, limits)?;
    let decoded_bytes = element_count
        .checked_mul(dtype.element_size())
        .ok_or_else(|| Error::limit_exceeded("float tensor decoded byte count overflows u64"))?;
    check_len(
        decoded_bytes,
        limits.max_decoded_bytes,
        "float tensor decoded bytes",
    )?;
    if decoded_bytes != payload.len() as u64 {
        return Err(Error::malformed(format!(
            "float tensor payload is {} bytes but shape {:?} x {} is {decoded_bytes} bytes",
            payload.len(),
            shape,
            dtype.as_str()
        )));
    }

    let (exponent_plane, mantissa_plane) =
        split_float_tensor_payload(payload, dtype, element_count)?;
    let mut stored = Vec::with_capacity(exponent_plane.len() + mantissa_plane.len());
    stored.extend_from_slice(&exponent_plane);
    stored.extend_from_slice(&mantissa_plane);

    let meta = Value::map([
        ("schema", Value::Str(FLOAT_TENSOR_SCHEMA.to_owned())),
        ("dtype", Value::Str(dtype.as_str().to_owned())),
        (
            "shape",
            Value::Array(shape.iter().map(|dim| Value::U64(*dim)).collect()),
        ),
        ("element_count", Value::U64(element_count)),
        ("sign_bits", Value::U64(dtype.sign_bits())),
        ("exponent_bits", Value::U64(dtype.exponent_bits())),
        ("mantissa_bits", Value::U64(dtype.mantissa_bits())),
        (
            "exponent_coder",
            Value::Str(FLOAT_TENSOR_EXPONENT_CODER.to_owned()),
        ),
        (
            "exponent_plane_len",
            Value::U64(exponent_plane.len() as u64),
        ),
        (
            "mantissa_plane_len",
            Value::U64(mantissa_plane.len() as u64),
        ),
        (
            "payload_hash",
            Value::Bytes(
                hash_domain(DOMAIN_FLOAT_TENSOR_PAYLOAD, &stored)
                    .as_bytes()
                    .to_vec(),
            ),
        ),
    ]);
    let body = build_ai_body(&meta, &stored, limits)?;
    let parts = FrameParts {
        format_major: FORMAT_MAJOR,
        format_minor: FORMAT_MINOR,
        domain: DOMAIN_PAYLOAD,
        mode: Mode::FloatTensor,
        flags: 0,
        level: FLOAT_TENSOR_LEVEL,
        decoded_len: decoded_bytes,
        content_hash: hash_domain(DOMAIN_CHUNK, payload),
        dependencies: &[],
        chunks: &[],
        body: &body,
    };
    encode_frame(&parts, limits)
}

/// Validate and decode a FLOAT_TENSOR frame, returning the reconstructed
/// little-endian element bytes.
pub fn decode_float_tensor(frame: &Frame<'_>, limits: &Limits) -> Result<Vec<u8>> {
    let (_, bytes) = parse_float_tensor_body(frame, limits)?;
    Ok(bytes)
}

/// Parse and validate a FLOAT_TENSOR descriptor, returning it with the
/// reconstructed element bytes.
pub fn float_tensor_metadata(frame: &Frame<'_>, limits: &Limits) -> Result<FloatTensorMeta> {
    let (meta, _) = parse_float_tensor_body(frame, limits)?;
    Ok(meta)
}

fn parse_float_tensor_body(
    frame: &Frame<'_>,
    limits: &Limits,
) -> Result<(FloatTensorMeta, Vec<u8>)> {
    let (descriptor_bytes, stored) = split_ai_body_prefix(frame.body)?;
    let value = wcb::decode(descriptor_bytes, limits)?;
    let map = ai_map(&value, "float tensor")?;
    check_exact_keys(
        map,
        &[
            "schema",
            "dtype",
            "shape",
            "element_count",
            "sign_bits",
            "exponent_bits",
            "mantissa_bits",
            "exponent_coder",
            "exponent_plane_len",
            "mantissa_plane_len",
            "payload_hash",
        ],
        "float tensor",
    )?;
    let schema = meta_text(map, "schema")?;
    if schema != FLOAT_TENSOR_SCHEMA {
        return Err(Error::malformed(format!(
            "unsupported float tensor schema `{schema}`"
        )));
    }
    let dtype = FloatDtype::from_name(meta_text(map, "dtype")?)?;
    let exponent_coder = meta_text(map, "exponent_coder")?;
    if exponent_coder != FLOAT_TENSOR_EXPONENT_CODER {
        return Err(Error::malformed(format!(
            "unsupported float tensor exponent coder `{exponent_coder}` (v1 allows `{FLOAT_TENSOR_EXPONENT_CODER}`)"
        )));
    }
    let sign_bits = meta_byte_field(map, "sign_bits")?;
    let exponent_bits = meta_byte_field(map, "exponent_bits")?;
    let mantissa_bits = meta_byte_field(map, "mantissa_bits")?;
    if (sign_bits, exponent_bits, mantissa_bits) != dtype.field_bits() {
        return Err(Error::malformed(format!(
            "float tensor sub-field widths {}, {}, {} disagree with dtype `{}` ({}, {}, {})",
            sign_bits,
            exponent_bits,
            mantissa_bits,
            dtype.as_str(),
            dtype.sign_bits(),
            dtype.exponent_bits(),
            dtype.mantissa_bits()
        )));
    }
    let element_bits = sign_bits + exponent_bits + mantissa_bits;
    if element_bits % 8 != 0 {
        return Err(Error::malformed(
            "float tensor element width is not byte-aligned",
        ));
    }
    let element_size = element_bits / 8;
    let shape = meta_shape(map)?;
    let declared_count = meta_u64(map, "element_count")?;
    let expected_count = float_tensor_product(&shape, limits)?;
    if declared_count != expected_count {
        return Err(Error::malformed(format!(
            "float tensor element_count {declared_count} does not match shape product {expected_count}"
        )));
    }
    let decoded_bytes = declared_count
        .checked_mul(element_size)
        .ok_or_else(|| Error::limit_exceeded("float tensor decoded byte count overflows u64"))?;
    check_len(
        decoded_bytes,
        limits.max_decoded_bytes,
        "float tensor decoded bytes",
    )?;
    if decoded_bytes != frame.header.decoded_len {
        return Err(Error::corrupt(format!(
            "float tensor decoded_len is {} but the descriptor requires {decoded_bytes}",
            frame.header.decoded_len
        )));
    }

    let mantissa_bit_total = declared_count
        .checked_mul(sign_bits + mantissa_bits)
        .ok_or_else(|| Error::limit_exceeded("float tensor mantissa bit count overflows u64"))?;
    let expected_mantissa_bytes = mantissa_bit_total
        .checked_add(7)
        .ok_or_else(|| Error::limit_exceeded("float tensor mantissa length overflows u64"))?
        / 8;
    let declared_mantissa_len = meta_u64(map, "mantissa_plane_len")?;
    if declared_mantissa_len != expected_mantissa_bytes {
        return Err(Error::malformed(format!(
            "float tensor mantissa_plane_len {declared_mantissa_len} does not match element_count x (sign_bits + mantissa_bits) / 8 {expected_mantissa_bytes}"
        )));
    }
    let declared_exponent_len = meta_u64(map, "exponent_plane_len")?;
    let stored_len = declared_exponent_len
        .checked_add(declared_mantissa_len)
        .ok_or_else(|| Error::limit_exceeded("float tensor plane lengths overflow u64"))?;
    if stored_len != stored.len() as u64 {
        return Err(Error::malformed(format!(
            "float tensor stored payload is {} bytes but the descriptor declares {stored_len}",
            stored.len()
        )));
    }

    let exponent_len = usize::try_from(declared_exponent_len).map_err(|_| {
        Error::limit_exceeded("float tensor exponent plane length does not fit usize")
    })?;
    let (exponent_compressed, mantissa_plane) = stored.split_at(exponent_len);
    let exponent_len_usize = usize::try_from(declared_count).map_err(|_| {
        Error::limit_exceeded("float tensor exponent plane length does not fit usize")
    })?;
    let exponent_plane = huffman::decode(exponent_compressed, exponent_len_usize)?;
    let bytes = join_float_tensor_payload(
        &exponent_plane,
        mantissa_plane,
        dtype,
        declared_count,
        limits,
    )?;

    let declared_hash = meta_bytes(map, "payload_hash")?;
    if declared_hash.len() != 32 {
        return Err(Error::malformed(format!(
            "float tensor payload_hash must be 32 bytes, got {}",
            declared_hash.len()
        )));
    }
    let actual_hash = hash_domain(DOMAIN_FLOAT_TENSOR_PAYLOAD, stored);
    if declared_hash != actual_hash.as_bytes().as_slice() {
        return Err(Error::corrupt("float tensor payload_hash mismatch"));
    }

    Ok((
        FloatTensorMeta {
            dtype,
            shape,
            element_count: declared_count,
            sign_bits,
            exponent_bits,
            mantissa_bits,
            exponent_plane_len: declared_exponent_len,
            mantissa_plane_len: declared_mantissa_len,
        },
        bytes,
    ))
}

/// Encode a token sequence (WEFT-AI-CODEC §3).
pub fn encode_token_sequence(
    token_ids: &[u64],
    vocab_schema: &str,
    limits: &Limits,
) -> Result<Vec<u8>> {
    if vocab_schema.is_empty() {
        return Err(Error::malformed("vocab_schema must not be empty"));
    }
    if token_ids.len() as u64 > limits.max_tokens {
        return Err(Error::limit_exceeded(format!(
            "token count {} exceeds limit {}",
            token_ids.len(),
            limits.max_tokens
        )));
    }
    let payload = encode_token_payload(token_ids, limits)?;
    check_len(
        payload.len() as u64,
        limits.max_decoded_bytes,
        "token payload",
    )?;
    let token_count = token_ids.len() as u64;
    let meta = Value::map([
        ("schema", Value::Str("weft.tokens.v1".to_owned())),
        ("vocab_schema", Value::Str(vocab_schema.to_owned())),
        ("encoding", Value::Str(TOKEN_ENCODING.to_owned())),
        ("token_count", Value::U64(token_count)),
        ("payload_bytes", Value::U64(payload.len() as u64)),
        (
            "payload_hash",
            Value::Bytes(
                hash_domain(DOMAIN_TOKENS_PAYLOAD, &payload)
                    .as_bytes()
                    .to_vec(),
            ),
        ),
    ]);
    let body = build_ai_body(&meta, &payload, limits)?;
    let parts = FrameParts {
        format_major: FORMAT_MAJOR,
        format_minor: FORMAT_MINOR,
        domain: DOMAIN_PAYLOAD,
        mode: Mode::TokenSequence,
        flags: 0,
        level: 0,
        decoded_len: payload.len() as u64,
        content_hash: hash_domain(DOMAIN_CHUNK, &payload),
        dependencies: &[],
        chunks: &[],
        body: &body,
    };
    encode_frame(&parts, limits)
}

/// Encode token ids to the canonical delta payload.
///
/// The first id is an absolute minimal LEB128 u64. Every later id is
/// `token[i] - token[i-1]` in wrapping 64-bit arithmetic, zigzag-encoded and
/// written as a minimal LEB128 u64. This is a bijection over all u64 ids.
pub fn encode_token_payload(token_ids: &[u64], limits: &Limits) -> Result<Vec<u8>> {
    let mut out: Vec<u8> = Vec::new();
    let mut previous: u64 = 0;
    for (index, &token) in token_ids.iter().enumerate() {
        if index == 0 {
            varint::encode(token, &mut out);
        } else {
            let delta = token.wrapping_sub(previous) as i64;
            varint::encode(varint::zigzag_encode(delta), &mut out);
        }
        previous = token;
        check_len(out.len() as u64, limits.max_decoded_bytes, "token payload")?;
    }
    Ok(out)
}

/// Decode a canonical token payload into ids, enforcing exact count and
/// canonical minimal varints. `token_count` must not exceed `max_tokens`.
pub fn decode_token_payload(payload: &[u8], token_count: u64, limits: &Limits) -> Result<Vec<u64>> {
    if token_count > limits.max_tokens {
        return Err(Error::limit_exceeded(format!(
            "token count {token_count} exceeds limit {}",
            limits.max_tokens
        )));
    }
    // Each token consumes at least one byte, so a larger count is truncated
    // before any allocation proportional to the declared count.
    if token_count > payload.len() as u64 {
        return Err(Error::malformed(format!(
            "token count {token_count} cannot fit in a {}-byte payload",
            payload.len()
        )));
    }
    let capacity = usize::try_from(token_count)
        .map_err(|_| Error::limit_exceeded("token count does not fit usize"))?;
    let mut ids = Vec::with_capacity(capacity);
    let mut pos = 0usize;
    let mut previous: u64 = 0;
    for index in 0..capacity {
        let raw = varint::decode(payload, &mut pos)?;
        let token = if index == 0 {
            raw
        } else {
            previous.wrapping_add(varint::zigzag_decode(raw) as u64)
        };
        ids.push(token);
        previous = token;
    }
    if pos != payload.len() {
        return Err(Error::malformed(format!(
            "token payload has {} trailing bytes",
            payload.len() - pos
        )));
    }
    Ok(ids)
}

/// Validate a TOKEN_SEQUENCE payload without materializing the id vector.
pub fn validate_token_payload(payload: &[u8], token_count: u64, limits: &Limits) -> Result<()> {
    if token_count > limits.max_tokens {
        return Err(Error::limit_exceeded(format!(
            "token count {token_count} exceeds limit {}",
            limits.max_tokens
        )));
    }
    if token_count > payload.len() as u64 {
        return Err(Error::malformed(format!(
            "token count {token_count} cannot fit in a {}-byte payload",
            payload.len()
        )));
    }
    let capacity = usize::try_from(token_count)
        .map_err(|_| Error::limit_exceeded("token count does not fit usize"))?;
    let mut pos = 0usize;
    let mut previous: u64 = 0;
    for index in 0..capacity {
        let raw = varint::decode(payload, &mut pos)?;
        previous = if index == 0 {
            raw
        } else {
            previous.wrapping_add(varint::zigzag_decode(raw) as u64)
        };
    }
    if pos != payload.len() {
        return Err(Error::malformed(format!(
            "token payload has {} trailing bytes",
            payload.len() - pos
        )));
    }
    Ok(())
}

/// Validate and decode a TOKEN_SEQUENCE frame, returning the exact payload.
pub fn decode_token_sequence(frame: &Frame<'_>, limits: &Limits) -> Result<Vec<u8>> {
    let (meta, payload) = parse_token_body(frame, limits)?;
    let _ = meta;
    Ok(payload.to_vec())
}

/// Parse and validate a TOKEN_SEQUENCE descriptor.
pub fn token_metadata(frame: &Frame<'_>, limits: &Limits) -> Result<TokenMeta> {
    let (meta, _) = parse_token_body(frame, limits)?;
    Ok(meta)
}

fn parse_token_body<'a>(frame: &'a Frame<'_>, limits: &Limits) -> Result<(TokenMeta, &'a [u8])> {
    let (descriptor_bytes, payload) = split_ai_body(frame.body, frame.header.decoded_len)?;
    let value = wcb::decode(descriptor_bytes, limits)?;
    let map = ai_map(&value, "token")?;
    check_exact_keys(
        map,
        &[
            "schema",
            "vocab_schema",
            "encoding",
            "token_count",
            "payload_bytes",
            "payload_hash",
        ],
        "token",
    )?;
    let schema = meta_text(map, "schema")?;
    if schema != "weft.tokens.v1" {
        return Err(Error::malformed(format!(
            "unsupported token schema `{schema}`"
        )));
    }
    let vocab_schema = meta_text(map, "vocab_schema")?;
    if vocab_schema.is_empty() {
        return Err(Error::malformed("token vocab_schema must not be empty"));
    }
    let encoding = meta_text(map, "encoding")?;
    if encoding != TOKEN_ENCODING {
        return Err(Error::malformed(format!(
            "token encoding `{encoding}` is specified but not implemented in v1 (only {TOKEN_ENCODING})"
        )));
    }
    let token_count = meta_u64(map, "token_count")?;
    if token_count > limits.max_tokens {
        return Err(Error::limit_exceeded(format!(
            "token count {token_count} exceeds limit {}",
            limits.max_tokens
        )));
    }
    let declared_bytes = meta_u64(map, "payload_bytes")?;
    if declared_bytes != payload.len() as u64 {
        return Err(Error::malformed(format!(
            "token payload is {} bytes but metadata declares {declared_bytes}",
            payload.len()
        )));
    }
    let declared_hash = meta_bytes(map, "payload_hash")?;
    if declared_hash.len() != 32 {
        return Err(Error::malformed(format!(
            "token payload_hash must be 32 bytes, got {}",
            declared_hash.len()
        )));
    }
    let actual_hash = hash_domain(DOMAIN_TOKENS_PAYLOAD, payload);
    if declared_hash != actual_hash.as_bytes().as_slice() {
        return Err(Error::corrupt("token payload_hash mismatch"));
    }
    validate_token_payload(payload, token_count, limits)?;
    Ok((
        TokenMeta {
            vocab_schema: vocab_schema.to_owned(),
            token_count,
            payload_bytes: declared_bytes,
        },
        payload,
    ))
}

// ---------------------------------------------------------------------------
// TENSOR_TILED (mode 15, WEFT-AI-CODEC §16)
// ---------------------------------------------------------------------------

/// Canonical tensor-tiled schema.
pub const TENSOR_TILED_SCHEMA: &str = "weft.ai.tensor_tiled.v1";
/// The only tile plane coder implemented in v1.
pub const TENSOR_TILE_CODEC: &str = huffman::CODER_ID;
/// The only tile layout implemented in v1.
pub const TENSOR_TILE_LAYOUT: &str = "row_major";

/// One independently decodable tile entry in a mode-15 descriptor.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TiledTensorEntry {
    /// Canonical grid position (row-major over `tile_counts`).
    pub index: u64,
    /// Tile extents. Every tile is full-size in v1.
    pub dims: Vec<u64>,
    /// Byte offset of the stored tile inside the frame payload.
    pub offset: u64,
    /// Stored bytes for this tile (all byte planes, Huffman-coded).
    pub compressed_len: u64,
    /// Decoded tile bytes (`product(dims) * element_size`).
    pub element_len: u64,
    /// `H("weft/tensor/tile/v1", WCB({dtype, dims, element_len, values_hash}))`.
    pub tile_root: Hash32,
}

/// Mode-15 descriptor recovered from a frame (without the payload bytes).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TiledTensorMeta {
    pub dtype: TensorDtype,
    pub shape: Vec<u64>,
    pub element_count: u64,
    pub tile_shape: Vec<u64>,
    pub tile_counts: Vec<u64>,
    pub tiles: Vec<TiledTensorEntry>,
    pub payload_len: u64,
    pub logical_root: Hash32,
}

/// A parsed mode-15 frame: descriptor plus the stored payload slice.
#[derive(Clone, Debug)]
pub struct TiledTensorDescriptor<'a> {
    pub meta: TiledTensorMeta,
    pub payload: &'a [u8],
}

fn tiled_product(shape: &[u64], limits: &Limits, what: &str) -> Result<u64> {
    if shape.is_empty() {
        return Err(Error::malformed(format!(
            "{what} shape rank must be at least 1"
        )));
    }
    if shape.len() as u64 > limits.max_tensor_rank {
        return Err(Error::limit_exceeded(format!(
            "{what} rank {} exceeds limit {}",
            shape.len(),
            limits.max_tensor_rank
        )));
    }
    let mut product: u64 = 1;
    for dim in shape {
        if *dim == 0 {
            return Err(Error::malformed(format!(
                "{what} shape dimensions must be greater than zero"
            )));
        }
        product = product
            .checked_mul(*dim)
            .ok_or_else(|| Error::limit_exceeded(format!("{what} shape product overflows u64")))?;
    }
    if product > limits.max_tensor_elements {
        return Err(Error::limit_exceeded(format!(
            "{what} element count {product} exceeds limit {}",
            limits.max_tensor_elements
        )));
    }
    Ok(product)
}

fn tile_counts(shape: &[u64], tile_shape: &[u64]) -> Result<Vec<u64>> {
    if shape.len() != tile_shape.len() {
        return Err(Error::malformed(format!(
            "tensor tile rank {} does not equal shape rank {}",
            tile_shape.len(),
            shape.len()
        )));
    }
    let mut counts = Vec::with_capacity(shape.len());
    for (axis, (extent, tile)) in shape.iter().zip(tile_shape.iter()).enumerate() {
        if *tile == 0 {
            return Err(Error::malformed(format!(
                "tensor tile shape axis {axis} must be greater than zero"
            )));
        }
        if *extent % *tile != 0 {
            return Err(Error::malformed(format!(
                "tensor tile shape axis {axis} extent {tile} does not divide shape extent {extent}"
            )));
        }
        counts.push(*extent / *tile);
    }
    Ok(counts)
}

fn row_major_byte_strides(shape: &[u64], element_size: u64) -> Result<Vec<u64>> {
    let mut strides = vec![0u64; shape.len()];
    let mut stride = element_size;
    for axis in (0..shape.len()).rev() {
        strides[axis] = stride;
        stride = stride
            .checked_mul(shape[axis])
            .ok_or_else(|| Error::limit_exceeded("tensor byte stride overflows u64"))?;
    }
    Ok(strides)
}

fn tile_start(counts: &[u64], index: u64) -> Vec<u64> {
    let mut coords = vec![0u64; counts.len()];
    let mut remainder = index;
    for axis in (0..counts.len()).rev() {
        coords[axis] = remainder % counts[axis];
        remainder /= counts[axis];
    }
    coords
}

/// Copy a row-major box from `src` into the row-major `dst`.
fn copy_box_bytes(
    dst: &mut [u8],
    src: &[u8],
    src_shape: &[u64],
    src_start: &[u64],
    extent: &[u64],
    element_size: u64,
) -> Result<()> {
    let rank = src_shape.len();
    if rank == 0 || extent.len() != rank || src_start.len() != rank {
        return Err(Error::malformed("box copy rank mismatch"));
    }
    let strides = row_major_byte_strides(src_shape, element_size)?;
    let row_len_u64 = extent[rank - 1]
        .checked_mul(element_size)
        .ok_or_else(|| Error::limit_exceeded("box row length overflows u64"))?;
    let row_len = usize::try_from(row_len_u64)
        .map_err(|_| Error::limit_exceeded("box row length does not fit usize"))?;
    let outer = &extent[..rank - 1];
    let mut coords = vec![0u64; outer.len()];
    let mut dst_off = 0usize;
    loop {
        let mut src_off: u64 = 0;
        for axis in 0..rank - 1 {
            let position = src_start[axis]
                .checked_add(coords[axis])
                .ok_or_else(|| Error::limit_exceeded("box source offset overflows u64"))?;
            src_off = src_off
                .checked_add(position * strides[axis])
                .ok_or_else(|| Error::limit_exceeded("box source offset overflows u64"))?;
        }
        src_off = src_off
            .checked_add(src_start[rank - 1] * strides[rank - 1])
            .ok_or_else(|| Error::limit_exceeded("box source offset overflows u64"))?;
        let src_off = usize::try_from(src_off)
            .map_err(|_| Error::limit_exceeded("box source offset does not fit usize"))?;
        let src_end = src_off
            .checked_add(row_len)
            .ok_or_else(|| Error::limit_exceeded("box source range overflows usize"))?;
        let dst_end = dst_off
            .checked_add(row_len)
            .ok_or_else(|| Error::limit_exceeded("box destination range overflows usize"))?;
        if src_end > src.len() || dst_end > dst.len() {
            return Err(Error::malformed("box copy is outside its buffers"));
        }
        dst[dst_off..dst_end].copy_from_slice(&src[src_off..src_end]);
        dst_off = dst_end;

        // Odometer over the outer axes.
        let mut axis = outer.len();
        loop {
            if axis == 0 {
                return Ok(());
            }
            axis -= 1;
            coords[axis] += 1;
            if coords[axis] < outer[axis] {
                break;
            }
            coords[axis] = 0;
        }
    }
}

fn encode_tile_planes(tile: &[u8], element_size: usize) -> Vec<u8> {
    let elements = tile.len() / element_size.max(1);
    let mut out = Vec::with_capacity(tile.len());
    for plane in 0..element_size {
        let mut plane_bytes = Vec::with_capacity(elements);
        let mut index = plane;
        while index < tile.len() {
            plane_bytes.push(tile[index]);
            index += element_size;
        }
        out.extend_from_slice(&huffman::encode(&plane_bytes));
    }
    out
}

fn decode_tile_planes(stored: &[u8], element_size: usize, element_len: usize) -> Result<Vec<u8>> {
    if element_size == 0 || element_len % element_size != 0 {
        return Err(Error::malformed(
            "tensor tile element length is not a whole number of elements",
        ));
    }
    let elements = element_len / element_size;
    let mut planes = Vec::with_capacity(element_size);
    let mut cursor = 0usize;
    for _ in 0..element_size {
        let remaining = stored
            .get(cursor..)
            .ok_or_else(|| Error::corrupt("tensor tile plane is truncated"))?;
        let (plane, consumed) = huffman::decode_prefix(remaining, elements)?;
        cursor = cursor
            .checked_add(consumed)
            .ok_or_else(|| Error::limit_exceeded("tensor tile plane offset overflows usize"))?;
        planes.push(plane);
    }
    if cursor != stored.len() {
        return Err(Error::corrupt(
            "tensor tile planes leave trailing bytes inside the tile range",
        ));
    }
    let mut out = vec![0u8; element_len];
    for (plane_index, plane) in planes.iter().enumerate() {
        for (element, byte) in plane.iter().enumerate() {
            out[element * element_size + plane_index] = *byte;
        }
    }
    Ok(out)
}

/// `H("weft/tensor/tile/v1", WCB({dtype, dims, element_len, values_hash}))`.
pub fn tensor_tile_root(
    dtype: TensorDtype,
    dims: &[u64],
    tile_bytes: &[u8],
    limits: &Limits,
) -> Result<Hash32> {
    let value = Value::map([
        ("dtype", Value::Str(dtype.as_str().to_owned())),
        (
            "dims",
            Value::Array(dims.iter().map(|dim| Value::U64(*dim)).collect()),
        ),
        ("element_len", Value::U64(tile_bytes.len() as u64)),
        (
            "values_hash",
            Value::Bytes(
                hash_domain(crate::tensor::DOMAIN_TENSOR_VALUES, tile_bytes)
                    .as_bytes()
                    .to_vec(),
            ),
        ),
    ]);
    Ok(hash_domain(
        crate::hash::DOMAIN_TENSOR_TILE,
        &wcb::encode(&value, limits)?,
    ))
}

fn tiled_meta_value(meta: &TiledTensorMeta) -> Value {
    let tiles: Vec<Value> = meta
        .tiles
        .iter()
        .map(|entry| {
            Value::map([
                ("index", Value::U64(entry.index)),
                (
                    "dims",
                    Value::Array(entry.dims.iter().map(|dim| Value::U64(*dim)).collect()),
                ),
                ("offset", Value::U64(entry.offset)),
                ("compressed_len", Value::U64(entry.compressed_len)),
                ("element_len", Value::U64(entry.element_len)),
                (
                    "tile_root",
                    Value::Bytes(entry.tile_root.as_bytes().to_vec()),
                ),
            ])
        })
        .collect();
    Value::map([
        ("schema", Value::Str(TENSOR_TILED_SCHEMA.to_owned())),
        ("dtype", Value::Str(meta.dtype.as_str().to_owned())),
        (
            "shape",
            Value::Array(meta.shape.iter().map(|dim| Value::U64(*dim)).collect()),
        ),
        ("element_count", Value::U64(meta.element_count)),
        ("layout", Value::Str(TENSOR_TILE_LAYOUT.to_owned())),
        (
            "tile_shape",
            Value::Array(meta.tile_shape.iter().map(|dim| Value::U64(*dim)).collect()),
        ),
        (
            "tile_counts",
            Value::Array(
                meta.tile_counts
                    .iter()
                    .map(|dim| Value::U64(*dim))
                    .collect(),
            ),
        ),
        ("tile_codec", Value::Str(TENSOR_TILE_CODEC.to_owned())),
        ("tiles", Value::Array(tiles)),
        ("payload_len", Value::U64(meta.payload_len)),
        (
            "logical_root",
            Value::Bytes(meta.logical_root.as_bytes().to_vec()),
        ),
    ])
}

/// Encode a mode-15 `TENSOR_TILED` frame.
///
/// Each tile is split into per-byte planes, each plane is coded independently
/// with `huffman-canonical-v1`, and the stored tiles are concatenated with
/// explicit offsets. A decode of any single tile verifies that tile's root.
pub fn encode_tiled_tensor(
    payload: &[u8],
    dtype: TensorDtype,
    shape: &[u64],
    tile_shape: &[u64],
    limits: &Limits,
) -> Result<Vec<u8>> {
    check_len(
        payload.len() as u64,
        limits.max_decoded_bytes,
        "tiled tensor payload",
    )?;
    let element_count = tiled_product(shape, limits, "tiled tensor")?;
    let element_size = dtype.element_size();
    let decoded_bytes = element_count
        .checked_mul(element_size)
        .ok_or_else(|| Error::limit_exceeded("tiled tensor decoded bytes overflow u64"))?;
    if decoded_bytes != payload.len() as u64 {
        return Err(Error::malformed(format!(
            "tiled tensor payload is {} bytes but shape {:?} x {} requires {decoded_bytes}",
            payload.len(),
            shape,
            dtype.as_str()
        )));
    }
    let counts = tile_counts(shape, tile_shape)?;
    let tile_elements = tiled_product(tile_shape, limits, "tensor tile")?;
    let tile_element_len = tile_elements
        .checked_mul(element_size)
        .ok_or_else(|| Error::limit_exceeded("tensor tile byte length overflows u64"))?;
    let tile_count = counts.iter().try_fold(1u64, |acc, count| {
        acc.checked_mul(*count)
            .ok_or_else(|| Error::limit_exceeded("tensor tile count overflows u64"))
    })?;
    if tile_count > limits.max_structured_items {
        return Err(Error::limit_exceeded(format!(
            "tensor tile count {tile_count} exceeds limit {}",
            limits.max_structured_items
        )));
    }

    let element_size_usize = usize::try_from(element_size)
        .map_err(|_| Error::limit_exceeded("tiled tensor element size does not fit usize"))?;
    let tile_len_usize = usize::try_from(tile_element_len)
        .map_err(|_| Error::limit_exceeded("tensor tile byte length does not fit usize"))?;
    let mut stored = Vec::with_capacity(payload.len());
    let mut tiles = Vec::with_capacity(usize::try_from(tile_count).unwrap_or(0));
    for index in 0..tile_count {
        let start = tile_start(&counts, index);
        let mut start_bytes = start.clone();
        for axis in 0..start_bytes.len() {
            start_bytes[axis] = start_bytes[axis]
                .checked_mul(tile_shape[axis])
                .ok_or_else(|| Error::limit_exceeded("tensor tile start overflows u64"))?;
        }
        let mut tile = vec![0u8; tile_len_usize];
        copy_box_bytes(
            &mut tile,
            payload,
            shape,
            &start_bytes,
            tile_shape,
            element_size,
        )?;
        let encoded = encode_tile_planes(&tile, element_size_usize);
        let root = tensor_tile_root(dtype, tile_shape, &tile, limits)?;
        let compressed_len = encoded.len() as u64;
        tiles.push(TiledTensorEntry {
            index,
            dims: tile_shape.to_vec(),
            offset: stored.len() as u64,
            compressed_len,
            element_len: tile_element_len,
            tile_root: root,
        });
        stored.extend_from_slice(&encoded);
    }

    let logical_root = crate::tensor::logical_root(dtype, shape, payload, limits)?;
    let meta = TiledTensorMeta {
        dtype,
        shape: shape.to_vec(),
        element_count,
        tile_shape: tile_shape.to_vec(),
        tile_counts: counts,
        tiles,
        payload_len: stored.len() as u64,
        logical_root,
    };
    let body = build_ai_body(&tiled_meta_value(&meta), &stored, limits)?;
    let parts = FrameParts {
        format_major: FORMAT_MAJOR,
        format_minor: FORMAT_MINOR,
        domain: DOMAIN_PAYLOAD,
        mode: Mode::TensorTiled,
        flags: 0,
        level: 0,
        decoded_len: decoded_bytes,
        content_hash: hash_domain(DOMAIN_CHUNK, payload),
        dependencies: &[],
        chunks: &[],
        body: &body,
    };
    encode_frame(&parts, limits)
}

/// Parse a mode-15 descriptor and return it with the stored payload slice.
pub fn parse_tiled_tensor<'a>(
    frame: &'a Frame<'_>,
    limits: &Limits,
) -> Result<TiledTensorDescriptor<'a>> {
    let (descriptor_bytes, payload) = split_ai_body_prefix(frame.body)?;
    let meta = parse_tiled_descriptor(
        descriptor_bytes,
        frame.header.decoded_len,
        payload.len() as u64,
        limits,
    )?;
    Ok(TiledTensorDescriptor { meta, payload })
}

/// Parse a mode-15 descriptor from its canonical WCB bytes.
///
/// `decoded_len` is the frame header's declared decoded length and
/// `payload_len` is the stored payload length (`body_len - 4 - descriptor`).
/// This is the prefix form used by a range reader that never materialises the
/// whole frame.
pub fn parse_tiled_descriptor(
    descriptor_bytes: &[u8],
    decoded_len: u64,
    payload_len: u64,
    limits: &Limits,
) -> Result<TiledTensorMeta> {
    let value = wcb::decode(descriptor_bytes, limits)?;
    let map = ai_map(&value, "tiled tensor")?;
    check_exact_keys(
        map,
        &[
            "schema",
            "dtype",
            "shape",
            "element_count",
            "layout",
            "tile_shape",
            "tile_counts",
            "tile_codec",
            "tiles",
            "payload_len",
            "logical_root",
        ],
        "tiled tensor",
    )?;
    let schema = meta_text(map, "schema")?;
    if schema != TENSOR_TILED_SCHEMA {
        return Err(Error::malformed(format!(
            "unsupported tiled tensor schema `{schema}`"
        )));
    }
    let dtype = TensorDtype::from_name(meta_text(map, "dtype")?)?;
    let layout = meta_text(map, "layout")?;
    if layout != TENSOR_TILE_LAYOUT {
        return Err(Error::malformed(format!(
            "tiled tensor layout `{layout}` is specified but not implemented in v1 (only {TENSOR_TILE_LAYOUT})"
        )));
    }
    let tile_codec = meta_text(map, "tile_codec")?;
    if tile_codec != TENSOR_TILE_CODEC {
        return Err(Error::malformed(format!(
            "unsupported tensor tile codec `{tile_codec}` (v1 allows `{TENSOR_TILE_CODEC}`)"
        )));
    }
    let shape = meta_shape(map)?;
    let element_count = tiled_product(&shape, limits, "tiled tensor")?;
    let declared_count = meta_u64(map, "element_count")?;
    if declared_count != element_count {
        return Err(Error::malformed(format!(
            "tiled tensor element_count {declared_count} does not match shape product {element_count}"
        )));
    }
    let element_size = dtype.element_size();
    let decoded_bytes = element_count
        .checked_mul(element_size)
        .ok_or_else(|| Error::limit_exceeded("tiled tensor decoded bytes overflow u64"))?;
    check_len(
        decoded_bytes,
        limits.max_decoded_bytes,
        "tiled tensor decoded bytes",
    )?;
    if decoded_len != decoded_bytes {
        return Err(Error::corrupt(format!(
            "tiled tensor decoded_len is {decoded_len} but the descriptor requires {decoded_bytes}"
        )));
    }
    let tile_shape = match map.get("tile_shape") {
        Some(Value::Array(items)) => {
            let mut out = Vec::with_capacity(items.len());
            for item in items {
                match item {
                    Value::U64(dim) => out.push(*dim),
                    _ => {
                        return Err(Error::malformed(
                            "tiled tensor `tile_shape` elements must be u64",
                        ))
                    }
                }
            }
            out
        }
        _ => {
            return Err(Error::malformed(
                "tiled tensor `tile_shape` must be an array",
            ))
        }
    };
    let counts = tile_counts(&shape, &tile_shape)?;
    let declared_counts = match map.get("tile_counts") {
        Some(Value::Array(items)) => {
            let mut out = Vec::with_capacity(items.len());
            for item in items {
                match item {
                    Value::U64(dim) => out.push(*dim),
                    _ => {
                        return Err(Error::malformed(
                            "tiled tensor `tile_counts` elements must be u64",
                        ))
                    }
                }
            }
            out
        }
        _ => {
            return Err(Error::malformed(
                "tiled tensor `tile_counts` must be an array",
            ))
        }
    };
    if declared_counts != counts {
        return Err(Error::malformed(format!(
            "tiled tensor tile_counts {:?} do not match shape/tile_shape {:?}",
            declared_counts, counts
        )));
    }
    let tile_elements = tiled_product(&tile_shape, limits, "tensor tile")?;
    let tile_element_len = tile_elements
        .checked_mul(element_size)
        .ok_or_else(|| Error::limit_exceeded("tensor tile byte length overflows u64"))?;
    let tile_count = counts.iter().try_fold(1u64, |acc, count| {
        acc.checked_mul(*count)
            .ok_or_else(|| Error::limit_exceeded("tensor tile count overflows u64"))
    })?;
    if tile_count > limits.max_structured_items {
        return Err(Error::limit_exceeded(format!(
            "tensor tile count {tile_count} exceeds limit {}",
            limits.max_structured_items
        )));
    }
    let declared_payload_len = meta_u64(map, "payload_len")?;
    if declared_payload_len != payload_len {
        return Err(Error::malformed(format!(
            "tiled tensor payload is {payload_len} bytes but metadata declares {declared_payload_len}"
        )));
    }
    let entries = match map.get("tiles") {
        Some(Value::Array(items)) => items,
        _ => return Err(Error::malformed("tiled tensor `tiles` must be an array")),
    };
    if entries.len() as u64 != tile_count {
        return Err(Error::malformed(format!(
            "tiled tensor has {} tile entries but the grid requires {tile_count}",
            entries.len()
        )));
    }
    let mut tiles = Vec::with_capacity(entries.len());
    let mut expected_offset: u64 = 0;
    for (position, item) in entries.iter().enumerate() {
        let entry = ai_map(item, "tensor tile")?;
        check_exact_keys(
            entry,
            &[
                "index",
                "dims",
                "offset",
                "compressed_len",
                "element_len",
                "tile_root",
            ],
            "tensor tile",
        )?;
        let index = meta_u64(entry, "index")?;
        if index != position as u64 {
            return Err(Error::malformed(format!(
                "tiled tensor entry {position} declares index {index}; entries must be in canonical order"
            )));
        }
        let dims = match entry.get("dims") {
            Some(Value::Array(items)) => {
                let mut out = Vec::with_capacity(items.len());
                for item in items {
                    match item {
                        Value::U64(dim) => out.push(*dim),
                        _ => {
                            return Err(Error::malformed("tensor tile `dims` elements must be u64"))
                        }
                    }
                }
                out
            }
            _ => return Err(Error::malformed("tensor tile `dims` must be an array")),
        };
        if dims != tile_shape {
            return Err(Error::malformed(format!(
                "tensor tile {position} dims {:?} do not equal tile_shape {:?}; partial tiles are not defined in v1",
                dims, tile_shape
            )));
        }
        let declared_element_len = meta_u64(entry, "element_len")?;
        if declared_element_len != tile_element_len {
            return Err(Error::malformed(format!(
                "tensor tile {position} element_len {declared_element_len} does not match {tile_element_len}"
            )));
        }
        let offset = meta_u64(entry, "offset")?;
        if offset != expected_offset {
            return Err(Error::malformed(format!(
                "tensor tile {position} offset {offset} is not the contiguous offset {expected_offset}"
            )));
        }
        let compressed_len = meta_u64(entry, "compressed_len")?;
        if compressed_len == 0 {
            return Err(Error::malformed(format!(
                "tensor tile {position} has an empty stored range"
            )));
        }
        expected_offset = expected_offset
            .checked_add(compressed_len)
            .ok_or_else(|| Error::limit_exceeded("tensor tile offset overflows u64"))?;
        let tile_root = Hash32::from_slice(meta_bytes(entry, "tile_root")?)
            .map_err(|_| Error::malformed("tensor tile_root must be 32 bytes"))?;
        tiles.push(TiledTensorEntry {
            index,
            dims,
            offset,
            compressed_len,
            element_len: declared_element_len,
            tile_root,
        });
    }
    if expected_offset != declared_payload_len {
        return Err(Error::malformed(format!(
            "tensor tile ranges cover {expected_offset} bytes but payload_len is {declared_payload_len}"
        )));
    }
    let logical_root = Hash32::from_slice(meta_bytes(map, "logical_root")?)
        .map_err(|_| Error::malformed("tiled tensor logical_root must be 32 bytes"))?;
    Ok(TiledTensorMeta {
        dtype,
        shape,
        element_count,
        tile_shape,
        tile_counts: counts,
        tiles,
        payload_len: declared_payload_len,
        logical_root,
    })
}

/// Validate and decode a mode-15 frame into the exact canonical row-major bytes.
///
/// Every tile's root is recomputed and matched, and the whole-tensor logical
/// root is recomputed from the assembled bytes.
pub fn decode_tiled_tensor(frame: &Frame<'_>, limits: &Limits) -> Result<Vec<u8>> {
    let descriptor = parse_tiled_tensor(frame, limits)?;
    let meta = &descriptor.meta;
    let element_size = usize::try_from(meta.dtype.element_size())
        .map_err(|_| Error::limit_exceeded("tiled tensor element size does not fit usize"))?;
    let total = usize::try_from(
        meta.element_count
            .checked_mul(meta.dtype.element_size())
            .ok_or_else(|| Error::limit_exceeded("tiled tensor decoded bytes overflow u64"))?,
    )
    .map_err(|_| Error::limit_exceeded("tiled tensor decoded bytes do not fit usize"))?;
    let mut out = vec![0u8; total];
    let dst_extent: Vec<usize> = meta
        .shape
        .iter()
        .map(|dim| {
            usize::try_from(*dim)
                .map_err(|_| Error::limit_exceeded("tiled tensor extent does not fit usize"))
        })
        .collect::<Result<_>>()?;
    let mut tiles_bytes = 0usize;
    for entry in &meta.tiles {
        let start = usize::try_from(entry.offset)
            .map_err(|_| Error::limit_exceeded("tensor tile offset does not fit usize"))?;
        let end = start
            .checked_add(
                usize::try_from(entry.compressed_len)
                    .map_err(|_| Error::limit_exceeded("tensor tile length does not fit usize"))?,
            )
            .ok_or_else(|| Error::limit_exceeded("tensor tile range overflows usize"))?;
        let tile = decode_tile_planes(
            &descriptor.payload[start..end],
            element_size,
            usize::try_from(entry.element_len)
                .map_err(|_| Error::limit_exceeded("tensor tile length does not fit usize"))?,
        )?;
        let root = tensor_tile_root(meta.dtype, &entry.dims, &tile, limits)?;
        if root != entry.tile_root {
            return Err(Error::corrupt(format!(
                "tensor tile {} root {root} does not match the descriptor's {}",
                entry.index, entry.tile_root
            )));
        }
        tiles_bytes = tiles_bytes
            .checked_add(tile.len())
            .ok_or_else(|| Error::limit_exceeded("tiled tensor decoded length overflows usize"))?;
        // Place the tile at its canonical grid position. Concatenation is only
        // correct for a 1-D tile grid; the general case scatters the tile.
        let coords = tile_start(&meta.tile_counts, entry.index);
        let mut dst_start = Vec::with_capacity(meta.shape.len());
        for axis in 0..meta.shape.len() {
            dst_start.push(
                coords[axis]
                    .checked_mul(meta.tile_shape[axis])
                    .ok_or_else(|| Error::limit_exceeded("tensor tile start overflows u64"))?,
            );
        }
        copy_box_into(
            &mut out,
            &dst_extent,
            &dst_start,
            &tile,
            &entry.dims,
            meta.dtype.element_size(),
        )?;
    }
    if tiles_bytes != total {
        return Err(Error::corrupt(format!(
            "tiled tensor decoded to {tiles_bytes} bytes but the descriptor requires {total}"
        )));
    }
    let logical = crate::tensor::logical_root(meta.dtype, &meta.shape, &out, limits)?;
    if logical != meta.logical_root {
        return Err(Error::corrupt(format!(
            "tiled tensor logical root {logical} does not match the descriptor's {}",
            meta.logical_root
        )));
    }
    Ok(out)
}

/// Parse a mode-15 frame and return only its descriptor.
pub fn tiled_tensor_metadata(frame: &Frame<'_>, limits: &Limits) -> Result<TiledTensorMeta> {
    Ok(parse_tiled_tensor(frame, limits)?.meta)
}

/// Read one logical region, decoding only the tiles it intersects.
///
/// Returns the region's canonical row-major element bytes and the number of
/// tiles that were decoded (and whose roots were verified). Untouched tiles are
/// never decoded.
pub fn read_region(
    descriptor: &TiledTensorDescriptor<'_>,
    region: &crate::tensor::Region,
    limits: &Limits,
) -> Result<(Vec<u8>, u64)> {
    read_region_with(&descriptor.meta, region, limits, |entry| {
        let start = usize::try_from(entry.offset)
            .map_err(|_| Error::limit_exceeded("tensor tile offset does not fit usize"))?;
        let end = start
            .checked_add(
                usize::try_from(entry.compressed_len)
                    .map_err(|_| Error::limit_exceeded("tensor tile length does not fit usize"))?,
            )
            .ok_or_else(|| Error::limit_exceeded("tensor tile range overflows usize"))?;
        descriptor
            .payload
            .get(start..end)
            .map(|bytes| bytes.to_vec())
            .ok_or_else(|| Error::corrupt("tensor tile range is outside the payload"))
    })
}

/// Read one logical region with a caller-supplied stored-tile fetch.
///
/// The fetch closure returns the exact stored bytes recorded for an entry. It
/// lets a range reader read only those byte ranges from storage; the codec
/// still decodes the planes and verifies each tile root.
pub fn read_region_with<F>(
    meta: &TiledTensorMeta,
    region: &crate::tensor::Region,
    limits: &Limits,
    mut fetch: F,
) -> Result<(Vec<u8>, u64)>
where
    F: FnMut(&TiledTensorEntry) -> Result<Vec<u8>>,
{
    region.validate(&meta.shape)?;
    let extent = region.extent();
    if region.is_empty() {
        return Ok((Vec::new(), 0));
    }
    let element_size = meta.dtype.element_size();
    let element_size_usize = usize::try_from(element_size)
        .map_err(|_| Error::limit_exceeded("tiled tensor element size does not fit usize"))?;
    let count = extent.iter().try_fold(1u64, |acc, dim| {
        acc.checked_mul(*dim)
            .ok_or_else(|| Error::limit_exceeded("region element count overflows u64"))
    })?;
    let byte_len = count
        .checked_mul(element_size)
        .ok_or_else(|| Error::limit_exceeded("region payload length overflows u64"))?;
    check_len(byte_len, limits.max_decoded_bytes, "region payload")?;
    let byte_len = usize::try_from(byte_len)
        .map_err(|_| Error::limit_exceeded("region payload does not fit usize"))?;
    let mut out = vec![0u8; byte_len];

    // Intersecting tile range per axis.
    let mut first = Vec::with_capacity(meta.shape.len());
    let mut last = Vec::with_capacity(meta.shape.len());
    for axis in 0..meta.shape.len() {
        let tile = meta.tile_shape[axis];
        first.push(region.start[axis] / tile);
        last.push((region.end[axis] + tile - 1) / tile);
    }
    let mut tiles_touched = 0u64;
    let mut coords = first.clone();
    loop {
        // Canonical tile index from the tile coordinates.
        let mut index: u64 = 0;
        for axis in 0..meta.tile_counts.len() {
            index = index
                .checked_mul(meta.tile_counts[axis])
                .and_then(|value| value.checked_add(coords[axis]))
                .ok_or_else(|| Error::limit_exceeded("tensor tile index overflows u64"))?;
        }
        let entry = meta
            .tiles
            .get(usize::try_from(index).unwrap_or(usize::MAX))
            .ok_or_else(|| Error::corrupt("tensor tile index is outside the descriptor"))?;
        let stored = fetch(entry)?;
        if stored.len() as u64 != entry.compressed_len {
            return Err(Error::corrupt(format!(
                "tensor tile {index} stored range is {} bytes but the descriptor records {}",
                stored.len(),
                entry.compressed_len
            )));
        }
        let tile = decode_tile_planes(
            &stored,
            element_size_usize,
            usize::try_from(entry.element_len)
                .map_err(|_| Error::limit_exceeded("tensor tile length does not fit usize"))?,
        )?;
        let root = tensor_tile_root(meta.dtype, &entry.dims, &tile, limits)?;
        if root != entry.tile_root {
            return Err(Error::corrupt(format!(
                "tensor tile {index} root {root} does not match the descriptor's {}",
                entry.tile_root
            )));
        }
        tiles_touched += 1;

        // Intersection of this tile with the requested region.
        let mut local_start = Vec::with_capacity(meta.shape.len());
        let mut local_extent = Vec::with_capacity(meta.shape.len());
        let mut dst_start = Vec::with_capacity(meta.shape.len());
        for axis in 0..meta.shape.len() {
            let base = coords[axis]
                .checked_mul(meta.tile_shape[axis])
                .ok_or_else(|| Error::limit_exceeded("tensor tile start overflows u64"))?;
            let lo = region.start[axis].max(base);
            let hi = region.end[axis].min(base + meta.tile_shape[axis]);
            local_start.push(lo - base);
            local_extent.push(hi - lo);
            dst_start.push(lo - region.start[axis]);
        }
        let dst_len = local_extent.iter().try_fold(1u64, |acc, dim| {
            acc.checked_mul(*dim)
                .ok_or_else(|| Error::limit_exceeded("region box overflows u64"))
        })?;
        let dst_bytes = dst_len
            .checked_mul(element_size)
            .ok_or_else(|| Error::limit_exceeded("region box byte length overflows u64"))?;
        let mut boxed = vec![
            0u8;
            usize::try_from(dst_bytes).map_err(|_| Error::limit_exceeded(
                "region box does not fit usize"
            ))?
        ];
        copy_box_bytes(
            &mut boxed,
            &tile,
            &meta.tile_shape,
            &local_start,
            &local_extent,
            element_size,
        )?;
        // Scatter the box into the requested region's row-major order.
        let dst_extent_usize: Vec<usize> = extent
            .iter()
            .map(|dim| {
                usize::try_from(*dim)
                    .map_err(|_| Error::limit_exceeded("region extent does not fit usize"))
            })
            .collect::<Result<_>>()?;
        let mut coords_in_extent = dst_start.clone();
        let boxed_shape: Vec<u64> = local_extent.clone();
        copy_box_into(
            &mut out,
            &dst_extent_usize,
            &mut coords_in_extent,
            &boxed,
            &boxed_shape,
            element_size,
        )?;

        // Advance the tile odometer.
        let mut axis = first.len();
        loop {
            if axis == 0 {
                return Ok((out, tiles_touched));
            }
            axis -= 1;
            coords[axis] += 1;
            if coords[axis] < last[axis] {
                break;
            }
            coords[axis] = first[axis];
        }
    }
}

/// Scatter a row-major box into a larger row-major destination at `dst_start`.
fn copy_box_into(
    dst: &mut [u8],
    dst_extent: &[usize],
    dst_start: &[u64],
    src: &[u8],
    src_extent: &[u64],
    element_size: u64,
) -> Result<()> {
    let rank = dst_extent.len();
    if rank == 0 || src_extent.len() != rank || dst_start.len() != rank {
        return Err(Error::malformed("box scatter rank mismatch"));
    }
    let mut dst_strides = vec![0usize; rank];
    let mut stride = 1usize;
    for axis in (0..rank).rev() {
        dst_strides[axis] = stride;
        stride = stride
            .checked_mul(dst_extent[axis])
            .ok_or_else(|| Error::limit_exceeded("destination stride overflows usize"))?;
    }
    let row_len = usize::try_from(
        src_extent[rank - 1]
            .checked_mul(element_size)
            .ok_or_else(|| Error::limit_exceeded("box row length overflows u64"))?,
    )
    .map_err(|_| Error::limit_exceeded("box row length does not fit usize"))?;
    let outer = &src_extent[..rank - 1];
    let mut coords = vec![0u64; outer.len()];
    let mut src_off = 0usize;
    loop {
        let mut dst_elements: usize = 0;
        for axis in 0..rank - 1 {
            dst_elements = dst_elements
                .checked_add(
                    usize::try_from(dst_start[axis] + coords[axis]).map_err(|_| {
                        Error::limit_exceeded("destination offset does not fit usize")
                    })? * dst_strides[axis],
                )
                .ok_or_else(|| Error::limit_exceeded("destination offset overflows usize"))?;
        }
        dst_elements = dst_elements
            .checked_add(
                usize::try_from(dst_start[rank - 1])
                    .map_err(|_| Error::limit_exceeded("destination offset does not fit usize"))?,
            )
            .ok_or_else(|| Error::limit_exceeded("destination offset overflows usize"))?;
        let dst_off = dst_elements
            .checked_mul(element_size as usize)
            .ok_or_else(|| Error::limit_exceeded("destination byte offset overflows usize"))?;
        let dst_end = dst_off
            .checked_add(row_len)
            .ok_or_else(|| Error::limit_exceeded("destination row range overflows usize"))?;
        let src_end = src_off
            .checked_add(row_len)
            .ok_or_else(|| Error::limit_exceeded("source row range overflows usize"))?;
        if dst_end > dst.len() || src_end > src.len() {
            return Err(Error::malformed("box scatter is outside its buffers"));
        }
        dst[dst_off..dst_end].copy_from_slice(&src[src_off..src_end]);
        src_off = src_end;

        let mut axis = outer.len();
        loop {
            if axis == 0 {
                return Ok(());
            }
            axis -= 1;
            coords[axis] += 1;
            if coords[axis] < outer[axis] {
                break;
            }
            coords[axis] = 0;
        }
    }
}
