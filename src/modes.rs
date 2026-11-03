//! Payload mode encoders/decoders (WEFTCODEC-FORMAT §3–§6).
//!
//! Every encoder computes the whole-frame bytes including header, tables, and
//! trailer. Every decoder verifies the header/trailer, every chunk hash, the
//! dependency hash/length at the materialization boundary, and the whole
//! content hash.

use std::collections::HashMap;

use crate::dependency::Resolver;
use crate::error::{Error, Result};
use crate::frame::{
    encode_frame, ChunkEntry, Dependency, Frame, FrameParts, Mode, DEP_DOMAIN_DICTIONARY,
    DEP_DOMAIN_PAYLOAD, DOMAIN_PAYLOAD, DOMAIN_STRUCTURED, FORMAT_MAJOR, FORMAT_MINOR,
};
use crate::hash::{hash_domain, DOMAIN_CHUNK};
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

/// Decompress one zstd frame with an exact output bound.
pub fn decompress(body: &[u8], declared_len: u64) -> Result<Vec<u8>> {
    let decoder = zstd::stream::read::Decoder::new(body)
        .map_err(|e| Error::corrupt(format!("zstd decompressor init failed: {e}")))?;
    read_exact_zstd(decoder, declared_len)
}

/// Decompress one zstd frame using a raw-content dictionary, with an exact
/// output bound.
pub fn decompress_with_dictionary(body: &[u8], declared_len: u64, dict: &[u8]) -> Result<Vec<u8>> {
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
