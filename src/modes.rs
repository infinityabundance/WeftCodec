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
    encode_frame, encode_frame_with_body, ChunkEntry, Dependency, Frame, FrameParts, Mode,
    DEP_DOMAIN_DICTIONARY, DEP_DOMAIN_PAYLOAD, DOMAIN_PAYLOAD, DOMAIN_STRUCTURED, FORMAT_MAJOR,
    FORMAT_MINOR,
};
use crate::hash::{
    hash_domain, DomainHasher, Hash32, DOMAIN_CHUNK, DOMAIN_FLOAT_TENSOR_PAYLOAD,
    DOMAIN_QUANTIZED_TENSOR_PAYLOAD, DOMAIN_TENSOR_PAYLOAD, DOMAIN_TOKENS_PAYLOAD,
};
use crate::huffman::{self, TABLE_BYTES};
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

/// Byte and thread bounds for chunk-parallel ZSTD encoding (R22).
///
/// A chunked ZSTD frame is a **concatenation of independent frames**, one per
/// codec chunk (WEFTCODEC-FORMAT §3, CHUNKED). Each chunk's frame is a pure
/// function of that chunk's bytes and the level, so encoding chunks on several
/// threads and concatenating them in index order is byte-identical to encoding
/// them one at a time. This type only carries the bounds of that schedule; it
/// cannot change an emitted byte, a chunk entry or a hash.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ZstdEncodeOptions {
    /// Upper bound on threads used for one tensor's chunk frames. `1` is fully
    /// sequential.
    pub threads: usize,
    /// Inputs smaller than this stay sequential: below it, thread start-up and
    /// the per-batch join cost more than the overlap.
    pub parallel_min_bytes: usize,
    /// Decoded bytes handed to the pool per **ordered batch**. Only one batch's
    /// compressed chunks are retained at a time, so this is the byte bound on
    /// the pool's transient memory (plus a per-frame header slack). It is
    /// independent of the thread count by construction.
    pub batch_decoded_bytes: usize,
}

impl Default for ZstdEncodeOptions {
    fn default() -> Self {
        ZstdEncodeOptions {
            threads: 1,
            parallel_min_bytes: 8 * 1024 * 1024,
            batch_decoded_bytes: 32 * 1024 * 1024,
        }
    }
}

impl ZstdEncodeOptions {
    /// Sequential encoding: the historical path.
    pub fn sequential() -> Self {
        Self::default()
    }

    /// Chunk-parallel encoding with `threads` workers and the default bounds.
    pub fn with_threads(threads: usize) -> Self {
        ZstdEncodeOptions {
            threads: threads.max(1),
            ..Self::default()
        }
    }
}

/// Encode a ZSTD frame, using a bounded chunk-parallel schedule when
/// `options` allows it.
///
/// This is [`encode_zstd`] plus a schedule: with `chunk_size = None`, a
/// one-thread bound, an input below `parallel_min_bytes` or an empty input it
/// calls the sequential encoder directly, so the historical path is the path.
/// When it does parallelise, the chunk frames are produced by the *same*
/// `compress` call as the sequential encoder and concatenated in index order,
/// so the frame is byte-identical (proved by
/// `tests/zstd_chunk_parallel.rs`).
pub fn encode_zstd_with(
    data: &[u8],
    level: i32,
    chunk_size: Option<usize>,
    limits: &Limits,
    options: &ZstdEncodeOptions,
) -> Result<Vec<u8>> {
    if options.batch_decoded_bytes == 0 {
        return Err(Error::malformed(
            "zstd chunk-parallel batch bound must be greater than zero",
        ));
    }
    let chunked_parallel = match chunk_size {
        Some(size) => {
            options.threads > 1
                && !data.is_empty()
                && size > 0
                && data.len() >= options.parallel_min_bytes
        }
        None => false,
    };
    if !chunked_parallel {
        return encode_zstd(data, level, chunk_size, limits);
    }
    let size = chunk_size.expect("chunk size checked above");
    check_len(data.len() as u64, limits.max_decoded_bytes, "decoded data")?;
    validate_chunk_size(chunk_size)?;
    let content_hash = hash_domain(DOMAIN_CHUNK, data);
    let (body, chunks) = encode_zstd_chunks_ordered(
        data,
        size,
        level,
        options.threads.min(64).max(1),
        options.batch_decoded_bytes,
        limits,
    )?;
    let parts = FrameParts {
        format_major: FORMAT_MAJOR,
        format_minor: FORMAT_MINOR,
        domain: DOMAIN_PAYLOAD,
        mode: Mode::Zstd,
        flags: crate::frame::flags::CHUNKED,
        level: level_to_u16(level)?,
        decoded_len: data.len() as u64,
        content_hash,
        dependencies: &[],
        chunks: &chunks,
        body: &body,
    };
    encode_frame(&parts, limits)
}

/// Compress every codec chunk in ordered batches, `threads` chunks at a time.
///
/// The result is exactly [`maybe_chunked`]'s `(body, chunks)` for the same
/// `size` and level: each element of `chunks` and every byte of `body` is
/// produced by the same `compress` call the sequential encoder makes. Only the
/// *order in which chunks are handed to the compressor* changes, and chunks
/// share no compressor state.
fn encode_zstd_chunks_ordered(
    data: &[u8],
    size: usize,
    level: i32,
    threads: usize,
    batch_decoded_bytes: usize,
    limits: &Limits,
) -> Result<(Vec<u8>, Vec<ChunkEntry>)> {
    let count = data.len().div_ceil(size);
    check_len(count as u64, limits.max_chunks, "chunk count")?;
    let per_batch = (batch_decoded_bytes / size.max(1)).max(1).min(count);
    let mut body = Vec::new();
    let mut chunks = Vec::with_capacity(count);
    let mut offset: u64 = 0;
    let mut start = 0usize;
    while start < count {
        let end = (start + per_batch).min(count);
        let width = end - start;
        let workers = threads.min(width).max(1);
        let per_worker = width.div_ceil(workers);
        let mut slots: Vec<Option<Result<(Vec<u8>, Hash32)>>> = (0..width).map(|_| None).collect();
        std::thread::scope(|scope| {
            for (worker, slot_chunk) in slots.chunks_mut(per_worker).enumerate() {
                let base = start + worker * per_worker;
                scope.spawn(move || {
                    // One zstd context per worker instead of one per chunk: the
                    // 272 MB embedding is 4,155 chunks, so a fresh `CCtx` per
                    // chunk is 4,155 allocations. `ZSTD_compress2` resets the
                    // session on every call, so a reused context emits the same
                    // bytes as a fresh one; `tests/zstd_chunk_parallel.rs`
                    // proves it over every thread count and batch assignment.
                    let mut compressor = match zstd::bulk::Compressor::new(level) {
                        Ok(compressor) => compressor,
                        Err(err) => {
                            let message = format!("zstd compressor init failed: {err}");
                            for slot in slot_chunk.iter_mut() {
                                *slot = Some(Err(Error::malformed(message.clone())));
                            }
                            return;
                        }
                    };
                    for (within, slot) in slot_chunk.iter_mut().enumerate() {
                        let index = base + within;
                        let begin = index * size;
                        let finish = (begin + size).min(data.len());
                        let chunk = &data[begin..finish];
                        *slot = Some(match compressor.compress(chunk) {
                            Ok(encoded) => Ok((encoded, hash_domain(DOMAIN_CHUNK, chunk))),
                            Err(err) => {
                                Err(Error::malformed(format!("zstd compression failed: {err}")))
                            }
                        });
                    }
                });
            }
        });
        for (within, slot) in slots.into_iter().enumerate() {
            let index = start + within;
            let (encoded, hash) = match slot {
                Some(result) => result?,
                None => {
                    return Err(Error::corrupt(
                        "chunk-parallel zstd worker left a chunk unencoded",
                    ))
                }
            };
            let decoded = (data.len() - index * size).min(size);
            let encoded_len = u32::try_from(encoded.len())
                .map_err(|_| Error::limit_exceeded("encoded chunk does not fit u32"))?;
            let decoded_len = u32::try_from(decoded)
                .map_err(|_| Error::limit_exceeded("decoded chunk does not fit u32"))?;
            chunks.push(ChunkEntry {
                encoded_off: offset,
                encoded_len,
                decoded_len,
                hash,
            });
            offset = offset
                .checked_add(u64::from(encoded_len))
                .ok_or_else(|| Error::limit_exceeded("chunk offset overflow"))?;
            body.extend_from_slice(&encoded);
        }
        start = end;
    }
    Ok((body, chunks))
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
        Mode::KvCacheProv => crate::kv_prov::decode_kv_prov_page(&frame, limits)?.bytes,
        Mode::ActivationStream => {
            crate::activation_stream::decode_activation_chunk(&frame, limits)?.bytes
        }
        Mode::TensorTiled => decode_tiled_tensor(&frame, limits)?,
        Mode::FloatTiled => decode_float_tiled(&frame, limits)?,
        Mode::FloatTiledStreams => decode_float_tiled_streams(&frame, limits, 1)?,
        Mode::EngramShard => crate::engram::decode_frame(&frame, resolver, limits)?,
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

/// `ceil(log2(value))`, clamped to a valid `WindowLog`.
fn window_log_for(value: usize) -> u32 {
    let mut log = 10u32;
    while log < 31 && (1usize << log) < value {
        log += 1;
    }
    log
}

/// Compress `data` against the whole of `base` with a window that covers it,
/// mirroring `zstd --patch-from` (a raw-content dictionary plus long-distance
/// matching). Plain [`encode_dictionary`] uses the level's default window, which
/// silently truncates the visible base for bases above that window; this entry
/// point makes the whole base addressable, up to the 2 GiB zstd window bound.
fn compress_patch(data: &[u8], base: &[u8], level: i32) -> Result<Vec<u8>> {
    let mut compressor = zstd::bulk::Compressor::with_dictionary(level, base)
        .map_err(|e| Error::malformed(format!("zstd patch init failed: {e}")))?;
    compressor
        .set_parameter(zstd::zstd_safe::CParameter::WindowLog(window_log_for(
            base.len().max(1024),
        )))
        .map_err(|e| Error::malformed(format!("zstd patch window failed: {e}")))?;
    compressor
        .set_parameter(zstd::zstd_safe::CParameter::EnableLongDistanceMatching(
            true,
        ))
        .map_err(|e| Error::malformed(format!("zstd patch long-distance failed: {e}")))?;
    compressor
        .compress(data)
        .map_err(|e| Error::malformed(format!("zstd patch compression failed: {e}")))
}

/// Encode a patch frame: a DICTIONARY frame whose raw-content dictionary is the
/// whole declared base and whose window covers that base. This is the exact
/// analogue of `zstd --patch-from`; the wire mode is the frozen mode 4.
pub fn encode_patch(data: &[u8], base: &[u8], level: i32, limits: &Limits) -> Result<Vec<u8>> {
    check_len(data.len() as u64, limits.max_decoded_bytes, "decoded data")?;
    check_len(base.len() as u64, limits.max_dictionary_bytes, "patch base")?;
    let body = compress_patch(data, base, level)?;
    let dependency = dependency_for(base, DEP_DOMAIN_DICTIONARY, 0);
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
///
/// The decoder's maximum window is raised to cover the supplied dictionary.
/// A patch frame (see [`encode_patch`]) may declare a window as large as its
/// base, and zstd's default decoder limit is far below that. The bound is
/// derived from the dictionary that is *already in memory*, not from the
/// frame, so a malicious frame cannot force an allocation larger than the
/// dependency it names.
pub fn decompress_with_dictionary(body: &[u8], declared_len: u64, dict: &[u8]) -> Result<Vec<u8>> {
    check_zstd_preamble(body, declared_len)?;
    let mut decoder = zstd::stream::read::Decoder::with_dictionary(body, dict)
        .map_err(|e| Error::corrupt(format!("zstd dictionary init failed: {e}")))?;
    // Only ever *raise* the window bound: zstd's default is 27 (128 MiB), and a
    // small dictionary must not lower it below what a normal dictionary frame
    // already needs (which is set from the compression parameters, not the
    // dictionary length). A patch frame needs a window that covers its base, so
    // the bound is the larger of the default and the loaded dependency size.
    let window = window_log_for(dict.len().max(1024)).max(27);
    decoder
        .window_log_max(window)
        .map_err(|e| Error::corrupt(format!("zstd dictionary window init failed: {e}")))?;
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
    match dtype {
        FloatDtype::Bf16 => {
            let count = check_float_payload(payload, dtype, element_count)?;
            // One pass produces both planes *and* the exponent histogram the
            // coder needs, so the encoder never re-counts the plane it just
            // built. The bytes are identical to the reference splitter plus
            // `huffman::encode`; `tests/mode9_encoder_identity.rs` proves it.
            let mut exponent_plane = vec![0u8; count];
            let mut mantissa_plane = vec![0u8; count];
            let histogram = split_bf16_planes(payload, &mut exponent_plane, &mut mantissa_plane);
            let coded = huffman::encode_with_weights(&exponent_plane, &histogram);
            Ok((coded, mantissa_plane))
        }
        _ => {
            let (exponent_plane, mantissa_plane) =
                split_float_tensor_planes_reference(payload, dtype, element_count)?;
            Ok((huffman::encode(&exponent_plane), mantissa_plane))
        }
    }
}

/// Split float element bytes into the raw exponent byte plane and the packed
/// sign+mantissa plane, without coding either plane.
///
/// BF16 uses the specialised kernel; every other dtype keeps the generic
/// word loop, which is the reference oracle.
pub fn split_float_tensor_planes(
    payload: &[u8],
    dtype: FloatDtype,
    element_count: u64,
) -> Result<(Vec<u8>, Vec<u8>)> {
    match dtype {
        FloatDtype::Bf16 => {
            let count = check_float_payload(payload, dtype, element_count)?;
            let mut exponent_plane = vec![0u8; count];
            let mut mantissa_plane = vec![0u8; count];
            split_bf16_planes(payload, &mut exponent_plane, &mut mantissa_plane);
            Ok((exponent_plane, mantissa_plane))
        }
        _ => split_float_tensor_planes_reference(payload, dtype, element_count),
    }
}

/// Validate the payload length against the descriptor before any allocation.
fn check_float_payload(payload: &[u8], dtype: FloatDtype, element_count: u64) -> Result<usize> {
    let expected_bytes = element_count
        .checked_mul(dtype.element_size())
        .ok_or_else(|| Error::limit_exceeded("float tensor decoded byte count overflows u64"))?;
    if payload.len() as u64 != expected_bytes {
        return Err(Error::malformed(format!(
            "float tensor payload is {} bytes but the descriptor requires {expected_bytes}",
            payload.len()
        )));
    }
    usize::try_from(element_count)
        .map_err(|_| Error::limit_exceeded("float tensor element count does not fit usize"))
}

/// BF16 plane split: `exponent = bits 7..15`, `sign|mantissa = sign bit then the
/// low seven bits`, one byte per element on both planes.
///
/// Returns the exponent histogram (four interleaved banks, folded at the end)
/// so the caller can build the canonical table without a second traversal.
pub(crate) fn split_bf16_planes(
    payload: &[u8],
    exponent_plane: &mut [u8],
    mantissa_plane: &mut [u8],
) -> [u64; 256] {
    debug_assert_eq!(mantissa_plane.len() * 2, payload.len());
    debug_assert_eq!(exponent_plane.len(), mantissa_plane.len());
    let mut lanes = [[0u64; 256]; 4];
    let mut chunks = payload.chunks_exact(8);
    let mut out = 0usize;
    for chunk in &mut chunks {
        for (lane, word) in chunk.chunks_exact(2).enumerate() {
            let word = u16::from_le_bytes([word[0], word[1]]);
            let exponent = (word >> 7) as u8;
            exponent_plane[out + lane] = exponent;
            mantissa_plane[out + lane] = ((word >> 8) as u8 & 0x80) | (word as u8 & 0x7f);
            lanes[lane][exponent as usize] += 1;
        }
        out += 4;
    }
    for (lane, word) in chunks.remainder().chunks_exact(2).enumerate() {
        let word = u16::from_le_bytes([word[0], word[1]]);
        let exponent = (word >> 7) as u8;
        exponent_plane[out + lane] = exponent;
        mantissa_plane[out + lane] = ((word >> 8) as u8 & 0x80) | (word as u8 & 0x7f);
        lanes[lane][exponent as usize] += 1;
    }
    let mut histogram = [0u64; 256];
    for symbol in 0..256usize {
        histogram[symbol] =
            lanes[0][symbol] + lanes[1][symbol] + lanes[2][symbol] + lanes[3][symbol];
    }
    histogram
}

/// Split float element bytes into the raw exponent byte plane and the packed
/// sign+mantissa plane, without coding either plane.
///
/// The generic, dtype-parametric word loop. It is the reference oracle for the
/// specialised BF16 kernel and the only splitter for FP16, FP32 and FP8.
pub fn split_float_tensor_planes_reference(
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
    Ok((exponent_plane, mantissa_plane))
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

    // BF16 is the one width where the sign+mantissa plane is exactly one byte
    // per element, so the join is a pure byte interleave and has a
    // runtime-dispatched AVX2 kernel. Other dtypes keep the exact word loop
    // below (their sign+mantissa widths are not byte-aligned).
    let joined_by_kernel = if dtype == FloatDtype::Bf16 {
        crate::simd::join_bf16(exponent_plane, mantissa_plane, &mut out);
        true
    } else {
        false
    };

    let mantissa_mask = (1u64 << mantissa_bits) - 1;
    let width = sign_bits + mantissa_bits;
    let width_bits = width as u32;

    // MSB-first 64-bit accumulator holding the next `accumulated_bits` bits in
    // its high positions and zeros below: refill eight bits at a time while a
    // whole byte still fits, then read the whole sign+mantissa width in one
    // shift and drop it by shifting left. `accumulated_bits >= width` always
    // holds because the mantissa plane is exactly
    // ceil(element_count * width / 8) bytes and `width <= 24`.
    if !joined_by_kernel {
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

/// Count the exponent field of every element in a float-tensor payload.
///
/// One pass, no allocation beyond the 256-slot histogram. The payload must be
/// a whole number of elements; the element count itself is carried by the
/// caller (shape is not needed to build a histogram). BF16 uses a specialised
/// four-bank kernel: the exponent is bits 7..15 of the 16-bit word, and the
/// four banks break the store-to-load chain that a single counter bank has on
/// the heavily skewed exponent distributions real weights produce. The result is
/// an exact count either way.
pub fn float_tensor_exponent_histogram(payload: &[u8], dtype: FloatDtype) -> Result<[u64; 256]> {
    let mut histogram = [0u64; 256];
    accumulate_float_tensor_histogram(&mut histogram, payload, dtype)?;
    Ok(histogram)
}

/// The exponent histogram, accumulated on `threads` workers (R22.2).
///
/// The histogram is a vector of integer counts, so splitting the payload into
/// whole-element windows, accumulating each window's counts independently and
/// adding them is **exact**: addition is associative and commutative, so the
/// result equals [`float_tensor_exponent_histogram`]'s bit for bit. That is
/// what lets the analyse pass parallelise without touching the R18 exact mode-9
/// size prediction: the same counts give the same predicted length.
///
/// Below `min_parallel_bytes`, or with one thread, this is the sequential
/// function.
pub fn float_tensor_exponent_histogram_with(
    payload: &[u8],
    dtype: FloatDtype,
    threads: usize,
    min_parallel_bytes: usize,
) -> Result<[u64; 256]> {
    let element_bytes = usize::try_from(dtype.element_size())
        .map_err(|_| Error::limit_exceeded("float tensor element size does not fit usize"))?;
    let workers = if threads <= 1 || payload.len() < min_parallel_bytes || element_bytes == 0 {
        1
    } else {
        threads.min(64).max(1)
    };
    if element_bytes == 0 || payload.len() % element_bytes != 0 {
        return Err(Error::malformed(format!(
            "float tensor payload of {} bytes is not a whole number of {} elements",
            payload.len(),
            dtype.as_str()
        )));
    }
    let elements = payload.len() / element_bytes;
    let workers = workers.min(elements).max(1);
    if workers == 1 {
        return float_tensor_exponent_histogram(payload, dtype);
    }
    let per_worker = elements.div_ceil(workers);
    let mut partials: Vec<[u64; 256]> = (0..workers).map(|_| [0u64; 256]).collect();
    std::thread::scope(|scope| {
        for (worker, counts) in partials.iter_mut().enumerate() {
            let first = worker * per_worker;
            let last = ((worker + 1) * per_worker).min(elements);
            if first >= last {
                continue;
            }
            let window = &payload[first * element_bytes..last * element_bytes];
            scope.spawn(move || {
                // Every window is a whole number of elements, checked above, so
                // this cannot fail; it is the same accumulation the sequential
                // function performs. It must not be wrapped in a debug assertion:
                // that compiles the call away in a release build.
                let result = accumulate_float_tensor_histogram(counts, window, dtype);
                debug_assert!(result.is_ok());
            });
        }
    });
    let mut histogram = [0u64; 256];
    for counts in &partials {
        for symbol in 0..256usize {
            histogram[symbol] = histogram[symbol].saturating_add(counts[symbol]);
        }
    }
    Ok(histogram)
}

/// Add the exponent multiplicities of `payload` into `counts`.
///
/// `counts` must be the running total of everything already accumulated into
/// it; callers either start from zero or fold consecutive windows of one
/// payload. Every window must be a whole number of elements.
pub fn accumulate_float_tensor_histogram(
    counts: &mut [u64; 256],
    payload: &[u8],
    dtype: FloatDtype,
) -> Result<()> {
    let element_bytes = usize::try_from(dtype.element_size())
        .map_err(|_| Error::limit_exceeded("float tensor element size does not fit usize"))?;
    if element_bytes == 0 || payload.len() % element_bytes != 0 {
        return Err(Error::malformed(format!(
            "float tensor payload of {} bytes is not a whole number of {} elements",
            payload.len(),
            dtype.as_str()
        )));
    }
    match dtype {
        FloatDtype::Bf16 => accumulate_bf16_histogram(counts, payload),
        _ => {
            let mantissa_bits = dtype.mantissa_bits();
            let exponent_mask = (1u64 << dtype.exponent_bits()) - 1;
            for chunk in payload.chunks_exact(element_bytes) {
                let mut word = 0u64;
                for (index, byte) in chunk.iter().enumerate() {
                    word |= u64::from(*byte) << (8 * index);
                }
                let exponent = ((word >> mantissa_bits) & exponent_mask) as usize;
                counts[exponent] = counts[exponent].saturating_add(1);
            }
        }
    }
    Ok(())
}

/// Four-bank BF16 exponent histogram; the sum is exact.
fn accumulate_bf16_histogram(counts: &mut [u64; 256], payload: &[u8]) {
    let mut lanes = [[0u64; 256]; 4];
    let mut chunks = payload.chunks_exact(8);
    for chunk in &mut chunks {
        for (lane, word) in chunk.chunks_exact(2).enumerate() {
            let word = u16::from_le_bytes([word[0], word[1]]);
            lanes[lane][((word >> 7) & 0xff) as usize] += 1;
        }
    }
    for (lane, word) in chunks.remainder().chunks_exact(2).enumerate() {
        let word = u16::from_le_bytes([word[0], word[1]]);
        lanes[lane][((word >> 7) & 0xff) as usize] += 1;
    }
    for symbol in 0..256usize {
        counts[symbol] += lanes[0][symbol] + lanes[1][symbol] + lanes[2][symbol] + lanes[3][symbol];
    }
}

/// The exact length of the FLOAT_TENSOR frame [`encode_float_tensor`] would
/// produce, computed from the exponent histogram instead of coding it.
///
/// Exact, not estimated: the descriptor is rebuilt with the same canonical WCB
/// map, the exponent plane length comes from
/// [`huffman::encoded_len_from_weights`] (the same tree and bit count the
/// encoder emits), and the sign+mantissa plane is a fixed
/// `ceil(element_count * (sign_bits + mantissa_bits) / 8)`. The only field that
/// is not recomputed is `payload_hash`, whose WCB encoding is a 32-byte byte
/// string regardless of content, so its length contribution is exact.
///
/// Validation mirrors [`encode_float_tensor`], so a prediction error implies
/// the encode would have failed too.
pub fn float_tensor_encoded_len_from_histogram(
    dtype: FloatDtype,
    shape: &[u64],
    histogram: &[u64; 256],
    limits: &Limits,
) -> Result<u64> {
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
    let counted = histogram.iter().try_fold(0u64, |total, &count| {
        total
            .checked_add(count)
            .ok_or_else(|| Error::limit_exceeded("float tensor histogram count overflows u64"))
    })?;
    if counted != element_count {
        return Err(Error::malformed(format!(
            "float tensor exponent histogram counts {counted} elements but shape {shape:?} requires {element_count}"
        )));
    }

    let mantissa_bit_total = element_count
        .checked_mul(dtype.sign_bits() + dtype.mantissa_bits())
        .ok_or_else(|| Error::limit_exceeded("float tensor mantissa bit count overflows u64"))?;
    let mantissa_plane_len = mantissa_bit_total
        .checked_add(7)
        .ok_or_else(|| Error::limit_exceeded("float tensor mantissa length overflows u64"))?
        / 8;
    let exponent_plane_len = huffman::encoded_len_from_weights(histogram) as u64;

    // A 32-byte placeholder: WCB encodes `Bytes` as a tag, a length varint and
    // the bytes, so the descriptor length does not depend on the digest value.
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
        ("exponent_plane_len", Value::U64(exponent_plane_len)),
        ("mantissa_plane_len", Value::U64(mantissa_plane_len)),
        ("payload_hash", Value::Bytes(vec![0u8; 32])),
    ]);
    let descriptor_len = wcb::encode(&meta, limits)?.len() as u64;
    let body_len = 4u64
        .checked_add(descriptor_len)
        .and_then(|value| value.checked_add(exponent_plane_len))
        .and_then(|value| value.checked_add(mantissa_plane_len))
        .ok_or_else(|| Error::limit_exceeded("float tensor body length overflows u64"))?;
    let total = (crate::frame::HEADER_LEN as u64)
        .checked_add(body_len)
        .and_then(|value| value.checked_add(crate::frame::TRAILER_LEN as u64))
        .ok_or_else(|| Error::limit_exceeded("float tensor frame length overflows u64"))?;
    check_len(total, limits.max_frame_bytes, "frame")?;
    Ok(total)
}

/// The exact length of the FLOAT_TENSOR frame [`encode_float_tensor`] would
/// produce for `payload`, without encoding it.
pub fn float_tensor_encoded_len(
    payload: &[u8],
    dtype: FloatDtype,
    shape: &[u64],
    limits: &Limits,
) -> Result<u64> {
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
    if decoded_bytes != payload.len() as u64 {
        return Err(Error::malformed(format!(
            "float tensor payload is {} bytes but shape {:?} x {} is {decoded_bytes} bytes",
            payload.len(),
            shape,
            dtype.as_str()
        )));
    }
    let histogram = float_tensor_exponent_histogram(payload, dtype)?;
    float_tensor_encoded_len_from_histogram(dtype, shape, &histogram, limits)
}

/// Encode a FLOAT_TENSOR frame (WEFT-AI-CODEC §5).
///
/// The exponent plane is encoded with the canonical `huffman-canonical-v1`
/// coder; the sign+mantissa plane is stored raw. Every element bit pattern
/// round-trips, including NaN payloads.
///
/// This is the shipped encoder: one fused histogram+content-hash pass over the
/// payload, then one split+emit pass that writes both planes straight into the
/// frame buffer, then the stored-plane hash and the trailer CRC. The emitted
/// bytes are identical to [`encode_float_tensor_reference`], which is the
/// pre-R19 pipeline kept as the byte-identity oracle.
pub fn encode_float_tensor(
    payload: &[u8],
    dtype: FloatDtype,
    shape: &[u64],
    limits: &Limits,
) -> Result<Vec<u8>> {
    encode_float_tensor_with(
        payload,
        dtype,
        shape,
        limits,
        FloatTensorEncodeOptions::default(),
    )
}

/// The pre-R19 mode-9 encoder, kept as the byte-identity oracle.
///
/// This is the plain word-at-a-time implementation: it materialises both
/// planes, concatenates them, builds the body and lets [`encode_frame`] copy it
/// into the frame. Its output is the normative one; the shipped encoder must
/// reproduce it byte for byte, which `tests/mode9_encoder_identity.rs` and the
/// `mode9_encoder_bench` harness check on randomised inputs and on every
/// measured tensor.
#[doc(hidden)]
pub fn encode_float_tensor_reference(
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

    // The reference encoder deliberately uses the generic splitter and the
    // byte-at-a-time `huffman::encode`, i.e. the pre-R19 pipeline, so the bench
    // can measure the old and the new path in one session. The bytes are the
    // same either way; `tests/mode9_encoder_identity.rs` proves it.
    let (raw_exponent_plane, mantissa_plane) =
        split_float_tensor_planes_reference(payload, dtype, element_count)?;
    let exponent_plane = huffman::encode(&raw_exponent_plane);
    drop(raw_exponent_plane);
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

/// Tuning for [`encode_float_tensor_with`].
///
/// Nothing here can change an emitted byte: the exponent table, the symbol
/// order, the canonical codes and the padding are fixed by the histogram alone.
/// The options only choose how much of the same work runs at once, so a
/// one-threaded and a many-threaded encode produce identical frames.
#[derive(Clone, Copy, Debug)]
pub struct FloatTensorEncodeOptions {
    /// Upper bound on threads used for the split/emission pass. `1` is fully
    /// sequential.
    pub threads: usize,
    /// Parallel emission is only used when the exponent bitstream is at least
    /// this large; below it, thread start-up costs more than it saves.
    pub parallel_min_bitstream_bytes: usize,
    /// Hard cap on the bitstream size that may be parallelised. Above it the
    /// encode stays sequential, so the encoder never introduces a large
    /// transient allocation on a memory-constrained court.
    pub parallel_max_bitstream_bytes: usize,
    /// Smallest element range a parallel partition may cover.
    pub min_partition_elements: usize,
}

impl Default for FloatTensorEncodeOptions {
    fn default() -> Self {
        Self {
            threads: 1,
            parallel_min_bitstream_bytes: 1 << 20,
            parallel_max_bitstream_bytes: 512 << 20,
            min_partition_elements: 1 << 16,
        }
    }
}

impl FloatTensorEncodeOptions {
    /// Sequential emission.
    pub fn sequential() -> Self {
        Self::default()
    }

    /// Bounded parallel emission with `threads` workers.
    pub fn with_threads(threads: usize) -> Self {
        Self {
            threads: threads.max(1),
            ..Self::default()
        }
    }
}

/// Shape and payload facts shared by every mode-9 entry point.
struct FloatTensorForm {
    element_count: u64,
    decoded_bytes: u64,
    count: usize,
    element_bytes: usize,
}

fn float_tensor_form(
    payload: &[u8],
    dtype: FloatDtype,
    shape: &[u64],
    limits: &Limits,
) -> Result<FloatTensorForm> {
    check_len(
        payload.len() as u64,
        limits.max_decoded_bytes,
        "float tensor payload",
    )?;
    if dtype.element_bits() % 8 != 0 {
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
    Ok(FloatTensorForm {
        element_count,
        decoded_bytes,
        count: usize::try_from(element_count)
            .map_err(|_| Error::limit_exceeded("float tensor element count does not fit usize"))?,
        element_bytes: usize::try_from(dtype.element_size())
            .map_err(|_| Error::limit_exceeded("float tensor element size does not fit usize"))?,
    })
}

/// The FLOAT_TENSOR descriptor map. One builder for the encoder, the length
/// predictor and the fast encoder keeps the three from drifting.
fn float_tensor_meta(
    dtype: FloatDtype,
    shape: &[u64],
    element_count: u64,
    exponent_plane_len: u64,
    mantissa_plane_len: u64,
    payload_hash: &[u8; 32],
) -> Value {
    Value::map([
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
        ("exponent_plane_len", Value::U64(exponent_plane_len)),
        ("mantissa_plane_len", Value::U64(mantissa_plane_len)),
        ("payload_hash", Value::Bytes(payload_hash.to_vec())),
    ])
}

/// Encode a FLOAT_TENSOR frame with a caller-supplied histogram.
///
/// The histogram must describe **this** payload: it is what the canonical table
/// (and therefore every emitted bit) is derived from. The store obtains it from
/// the analyse stage, which computes it over the same byte slice in the same
/// call, so the plan's exact length prediction and the encoded frame are
/// cross-checked by construction. A wrong histogram yields a frame that still
/// decodes to the payload but whose bytes differ from [`encode_float_tensor`],
/// which is why this entry point is documented rather than guessed at.
pub fn encode_float_tensor_with_histogram(
    payload: &[u8],
    dtype: FloatDtype,
    shape: &[u64],
    limits: &Limits,
    histogram: &[u64; 256],
    options: FloatTensorEncodeOptions,
) -> Result<Vec<u8>> {
    let form = float_tensor_form(payload, dtype, shape, limits)?;
    let counted = histogram.iter().try_fold(0u64, |total, &count| {
        total
            .checked_add(count)
            .ok_or_else(|| Error::limit_exceeded("float tensor histogram count overflows u64"))
    })?;
    if counted != form.element_count {
        return Err(Error::malformed(format!(
            "float tensor histogram counts {counted} elements but shape {shape:?} requires {}",
            form.element_count
        )));
    }
    let content_hash = hash_domain(DOMAIN_CHUNK, payload);
    encode_float_tensor_planned(
        payload,
        dtype,
        shape,
        limits,
        histogram,
        content_hash,
        options,
    )
}

/// Encode a FLOAT_TENSOR frame with explicit encoder tuning.
///
/// Byte-identical to [`encode_float_tensor`] for every option value.
pub fn encode_float_tensor_with(
    payload: &[u8],
    dtype: FloatDtype,
    shape: &[u64],
    limits: &Limits,
    options: FloatTensorEncodeOptions,
) -> Result<Vec<u8>> {
    float_tensor_form(payload, dtype, shape, limits)?;
    // One fused pass: the exponent histogram the coder needs and the plaintext
    // content hash the header needs. Windows keep both hot in cache, so this
    // costs one read of the payload rather than two.
    let mut histogram = [0u64; 256];
    let mut hasher = DomainHasher::new(DOMAIN_CHUNK);
    for window in payload.chunks(1 << 16) {
        accumulate_float_tensor_histogram(&mut histogram, window, dtype)?;
        hasher.update(window);
    }
    let content_hash = hasher.finalize();
    encode_float_tensor_planned(
        payload,
        dtype,
        shape,
        limits,
        &histogram,
        content_hash,
        options,
    )
}

/// The shared second half: plan the exact frame, then write the descriptor and
/// both planes into one buffer with no intermediate copies.
fn encode_float_tensor_planned(
    payload: &[u8],
    dtype: FloatDtype,
    shape: &[u64],
    limits: &Limits,
    histogram: &[u64; 256],
    content_hash: Hash32,
    options: FloatTensorEncodeOptions,
) -> Result<Vec<u8>> {
    let form = float_tensor_form(payload, dtype, shape, limits)?;
    let table = huffman::EncoderTable::from_weights(histogram);
    let exponent_bits = table.bits_from_counts(histogram);
    let exponent_plane_len = TABLE_BYTES as u64 + huffman::bytes_for_bits(exponent_bits);
    let mantissa_bit_total = form
        .element_count
        .checked_mul(dtype.sign_bits() + dtype.mantissa_bits())
        .ok_or_else(|| Error::limit_exceeded("float tensor mantissa bit count overflows u64"))?;
    let mantissa_plane_len = huffman::bytes_for_bits(mantissa_bit_total);
    let stored_len = exponent_plane_len
        .checked_add(mantissa_plane_len)
        .ok_or_else(|| Error::limit_exceeded("float tensor stored plane length overflows u64"))?;
    // The descriptor length does not depend on the digest value, so a placeholder
    // gives the exact final length and therefore the exact final offsets.
    let placeholder = float_tensor_meta(
        dtype,
        shape,
        form.element_count,
        exponent_plane_len,
        mantissa_plane_len,
        &[0u8; 32],
    );
    let descriptor_len = wcb::encode(&placeholder, limits)?.len() as u64;
    let body_len = 4u64
        .checked_add(descriptor_len)
        .and_then(|value| value.checked_add(stored_len))
        .ok_or_else(|| Error::limit_exceeded("float tensor body length overflows u64"))?;

    let parts = FrameParts {
        format_major: FORMAT_MAJOR,
        format_minor: FORMAT_MINOR,
        domain: DOMAIN_PAYLOAD,
        mode: Mode::FloatTensor,
        flags: 0,
        level: FLOAT_TENSOR_LEVEL,
        decoded_len: form.decoded_bytes,
        content_hash,
        dependencies: &[],
        chunks: &[],
        body: &[],
    };
    let descriptor_len_usize = usize::try_from(descriptor_len)
        .map_err(|_| Error::limit_exceeded("float tensor descriptor does not fit usize"))?;
    let stored_len_usize = usize::try_from(stored_len)
        .map_err(|_| Error::limit_exceeded("float tensor stored plane does not fit usize"))?;
    let exponent_plane_len_usize = usize::try_from(exponent_plane_len)
        .map_err(|_| Error::limit_exceeded("float tensor exponent plane does not fit usize"))?;
    encode_frame_with_body(&parts, body_len, limits, |body| {
        body[..4].copy_from_slice(&(descriptor_len as u32).to_le_bytes());
        let stored_at = 4 + descriptor_len_usize;
        let (exponent_region, mantissa_region) =
            body[stored_at..stored_at + stored_len_usize].split_at_mut(exponent_plane_len_usize);
        exponent_region[..TABLE_BYTES].copy_from_slice(table.lengths());
        emit_float_tensor_planes(
            payload,
            dtype,
            &form,
            &table,
            &mut exponent_region[TABLE_BYTES..],
            mantissa_region,
            &options,
        )?;
        let payload_hash = hash_domain(DOMAIN_FLOAT_TENSOR_PAYLOAD, &body[stored_at..]);
        let meta = float_tensor_meta(
            dtype,
            shape,
            form.element_count,
            exponent_plane_len,
            mantissa_plane_len,
            payload_hash.as_bytes(),
        );
        let meta_bytes = wcb::encode(&meta, limits)?;
        if meta_bytes.len() != descriptor_len_usize {
            return Err(Error::corrupt(format!(
                "float tensor descriptor length {} differs from the planned {descriptor_len_usize}",
                meta_bytes.len()
            )));
        }
        body[4..stored_at].copy_from_slice(&meta_bytes);
        Ok(())
    })
}

/// Write the exponent bitstream and the sign+mantissa plane into the frame.
fn emit_float_tensor_planes(
    payload: &[u8],
    dtype: FloatDtype,
    form: &FloatTensorForm,
    table: &huffman::EncoderTable,
    stream: &mut [u8],
    mantissa: &mut [u8],
    options: &FloatTensorEncodeOptions,
) -> Result<()> {
    if dtype == FloatDtype::Bf16 {
        return emit_bf16_planes(payload, form.count, table, stream, mantissa, options);
    }
    let sign_bits = dtype.sign_bits();
    let exponent_bits = dtype.exponent_bits();
    let mantissa_bits = dtype.mantissa_bits();
    let mantissa_mask = (1u64 << mantissa_bits) - 1;
    let exponent_mask = (1u64 << exponent_bits) - 1;
    let width = (sign_bits + mantissa_bits) as u32;
    let mut stream_writer = huffman::BitStream::new(stream);
    let mut mantissa_writer = huffman::BitStream::new(mantissa);
    for chunk in payload.chunks_exact(form.element_bytes) {
        let mut word = 0u64;
        for (index, byte) in chunk.iter().enumerate() {
            word |= u64::from(*byte) << (8 * index);
        }
        let exponent = ((word >> mantissa_bits) & exponent_mask) as u8;
        let sign = (word >> (exponent_bits + mantissa_bits)) & 1;
        let sign_mantissa = (sign << mantissa_bits) | (word & mantissa_mask);
        if table.has_bitstream() {
            table.push(&mut stream_writer, exponent);
        }
        mantissa_writer.push_packed((u64::from(width) << 32) | sign_mantissa);
    }
    if stream_writer.finish() != stream.len() {
        return Err(Error::corrupt(
            "float tensor exponent bitstream length differs from the predicted length",
        ));
    }
    if mantissa_writer.finish() != mantissa.len() {
        return Err(Error::corrupt(
            "float tensor mantissa plane length differs from the predicted length",
        ));
    }
    Ok(())
}

/// BF16 split plus emission. One pass over the payload writes the mantissa plane
/// and the exponent bitstream; nothing else is materialised.
fn emit_bf16_planes(
    payload: &[u8],
    count: usize,
    table: &huffman::EncoderTable,
    stream: &mut [u8],
    mantissa: &mut [u8],
    options: &FloatTensorEncodeOptions,
) -> Result<()> {
    debug_assert_eq!(mantissa.len(), count);
    let partitions = bf16_partitions(count, stream.len(), options);
    if partitions <= 1 || !table.has_bitstream() {
        let mut writer = huffman::BitStream::new(stream);
        for (index, chunk) in payload.chunks_exact(2).enumerate() {
            let word = u16::from_le_bytes([chunk[0], chunk[1]]);
            mantissa[index] = ((word >> 8) as u8 & 0x80) | (word as u8 & 0x7f);
            if table.has_bitstream() {
                table.push(&mut writer, (word >> 7) as u8);
            }
        }
        let written = writer.finish();
        if written != stream.len() {
            return Err(Error::corrupt(
                "float tensor exponent bitstream length differs from the predicted length",
            ));
        }
        return Ok(());
    }
    emit_bf16_parallel(payload, count, table, stream, mantissa, partitions)
}

/// How many partitions the bounded parallel emitter may use.
fn bf16_partitions(count: usize, stream_len: usize, options: &FloatTensorEncodeOptions) -> usize {
    if options.threads <= 1
        || stream_len < options.parallel_min_bitstream_bytes
        || stream_len > options.parallel_max_bitstream_bytes
    {
        return 1;
    }
    let by_elements = count / options.min_partition_elements.max(1);
    options.threads.min(by_elements).max(1).min(64)
}

/// Deterministic parallel BF16 emission.
///
/// The element range is split into `partitions` contiguous ranges. Each worker
/// emits its own range into its own byte range of the final bitstream, starting
/// at the bit offset its range begins at, and spills at most one byte (the byte
/// it shares with the next partition) into a one-byte scratch slot. A short
/// serial pass then ORs each saved scratch byte into the byte the next worker
/// already wrote there. Both writers only ever own disjoint bits of that byte,
/// so the result is exactly the sequential bitstream, byte for byte, and the
/// only memory beyond the frame is one byte per partition.
fn emit_bf16_parallel(
    payload: &[u8],
    count: usize,
    table: &huffman::EncoderTable,
    stream: &mut [u8],
    mantissa: &mut [u8],
    partitions: usize,
) -> Result<()> {
    let ranges = split_range(count, partitions);
    let parts = ranges.len();

    // Pass 1: exact bit length of every partition, in parallel. This is the only
    // extra traversal; it allocates one `u64` per partition and writes nothing.
    let mut bits = vec![0u64; parts];
    {
        let payload_slices: Vec<&[u8]> = ranges
            .iter()
            .map(|(start, end)| &payload[start * 2..end * 2])
            .collect();
        std::thread::scope(|scope| {
            let mut handles = Vec::with_capacity(parts);
            for (slice, slot) in payload_slices.into_iter().zip(bits.iter_mut()) {
                let table = &*table;
                handles.push(scope.spawn(move || {
                    let mut total = 0u64;
                    for chunk in slice.chunks_exact(2) {
                        let word = u16::from_le_bytes([chunk[0], chunk[1]]);
                        total += table.bits_of_byte((word >> 7) as u8);
                    }
                    *slot = total;
                }));
            }
            for handle in handles {
                let _ = handle.join();
            }
        });
    }

    // Prefix sums: `starts[j]` is the global bit offset partition `j` begins at.
    let mut starts = vec![0u64; parts];
    let mut accumulated = 0u64;
    for (index, value) in bits.iter().enumerate() {
        starts[index] = accumulated;
        accumulated += *value;
    }
    if huffman::bytes_for_bits(accumulated) != stream.len() as u64 {
        return Err(Error::corrupt(
            "float tensor exponent bitstream length differs from the predicted length",
        ));
    }

    // Disjoint byte slices: partition `j` owns the bytes of `stream` whose bits
    // lie wholly inside its bit range, and its own `mantissa` element range.
    let mut bounds: Vec<Bf16Partition> = Vec::with_capacity(parts);
    for (index, (start, end)) in ranges.iter().enumerate() {
        let first = (starts[index] / 8) as usize;
        let end_byte = ((starts[index] + bits[index]) / 8) as usize;
        bounds.push(Bf16Partition {
            first,
            complete: end_byte - first,
            offset: (starts[index] % 8) as u32,
            partial: (starts[index] + bits[index]) % 8 != 0,
            start: *start,
            end: *end,
        });
    }
    let mut stream_slices: Vec<&mut [u8]> = Vec::with_capacity(parts);
    {
        let mut rest = &mut stream[..];
        let mut pos = 0usize;
        for partition in &bounds {
            let (_skipped, after) = rest.split_at_mut(partition.first - pos);
            let (mine, after) = after.split_at_mut(partition.complete);
            pos = partition.first + partition.complete;
            rest = after;
            stream_slices.push(mine);
        }
    }
    let mut mantissa_slices: Vec<&mut [u8]> = Vec::with_capacity(parts);
    {
        let mut rest = &mut mantissa[..];
        let mut pos = 0usize;
        for partition in &bounds {
            let (_skipped, after) = rest.split_at_mut(partition.start - pos);
            let (mine, after) = after.split_at_mut(partition.end - partition.start);
            pos = partition.end;
            rest = after;
            mantissa_slices.push(mine);
        }
    }
    let mut scratch = vec![0u8; parts];

    // Pass 2: emit. Each worker starts at its own bit offset and spills the one
    // byte it shares with the next partition into its scratch slot.
    std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(parts);
        for (((stream_slice, mantissa_slice), slot), partition) in stream_slices
            .into_iter()
            .zip(mantissa_slices)
            .zip(scratch.iter_mut())
            .zip(bounds.iter())
        {
            let table = &*table;
            let payload_slice = &payload[partition.start * 2..partition.end * 2];
            let offset = partition.offset;
            handles.push(scope.spawn(move || {
                let mut writer = huffman::BitStream::with_scratch(stream_slice, slot, offset);
                for (index, chunk) in payload_slice.chunks_exact(2).enumerate() {
                    let word = u16::from_le_bytes([chunk[0], chunk[1]]);
                    mantissa_slice[index] = ((word >> 8) as u8 & 0x80) | (word as u8 & 0x7f);
                    table.push(&mut writer, (word >> 7) as u8);
                }
                writer.finish()
            }));
        }
        for (handle, partition) in handles.into_iter().zip(bounds.iter()) {
            let written = handle
                .join()
                .map_err(|_| Error::corrupt("parallel float tensor emission worker panicked"))?;
            let expected = partition.complete + usize::from(partition.partial);
            if written != expected {
                return Err(Error::corrupt(format!(
                    "parallel float tensor emission wrote {written} bytes, expected {expected}"
                )));
            }
        }
        Ok(())
    })?;

    // Stitch: each shared byte was written by exactly one worker with only its
    // own bits set, so OR-ing the other worker's saved byte completes it.
    for (index, partition) in bounds.iter().enumerate() {
        if partition.offset != 0 {
            stream[partition.first] |= scratch[index - 1];
        }
    }
    if bounds[parts - 1].partial {
        let last = parts - 1;
        let tail_at = ((starts[last] + bits[last]) / 8) as usize;
        stream[tail_at] |= scratch[last];
    }
    Ok(())
}

/// One worker's byte range in the parallel BF16 emitter.
struct Bf16Partition {
    /// First byte of `stream` the partition owns.
    first: usize,
    /// Number of whole bytes it writes into `stream`.
    complete: usize,
    /// Bit offset its stream starts at inside `stream[first]`.
    offset: u32,
    /// Whether the bit range ends mid-byte (so its last byte spills to scratch).
    partial: bool,
    /// First and one-past-last element index in the payload.
    start: usize,
    end: usize,
}

fn split_range(count: usize, partitions: usize) -> Vec<(usize, usize)> {
    let mut out = Vec::with_capacity(partitions);
    let base = count / partitions;
    let extra = count % partitions;
    let mut at = 0usize;
    for index in 0..partitions {
        let take = base + usize::from(index < extra);
        out.push((at, at + take));
        at += take;
    }
    out
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

/// Decode one stored tile and verify its `tile_root`.
///
/// `stored` is exactly the tile's recorded stored range (`compressed_len`
/// bytes). The returned bytes are the tile's canonical row-major element bytes.
pub fn decode_tiled_tile(
    dtype: TensorDtype,
    entry: &TiledTensorEntry,
    stored: &[u8],
    limits: &Limits,
) -> Result<Vec<u8>> {
    if stored.len() as u64 != entry.compressed_len {
        return Err(Error::corrupt(format!(
            "tensor tile {} stored range is {} bytes but the descriptor records {}",
            entry.index,
            stored.len(),
            entry.compressed_len
        )));
    }
    let element_size = usize::try_from(dtype.element_size())
        .map_err(|_| Error::limit_exceeded("tensor tile element size does not fit usize"))?;
    let element_len = usize::try_from(entry.element_len)
        .map_err(|_| Error::limit_exceeded("tensor tile length does not fit usize"))?;
    let tile = decode_tile_planes(stored, element_size, element_len)?;
    let root = tensor_tile_root(dtype, &entry.dims, &tile, limits)?;
    if root != entry.tile_root {
        return Err(Error::corrupt(format!(
            "tensor tile {} root {root} does not match the descriptor's {}",
            entry.index, entry.tile_root
        )));
    }
    Ok(tile)
}

fn tiled_dst_extent(shape: &[u64]) -> Result<Vec<usize>> {
    shape
        .iter()
        .map(|dim| {
            usize::try_from(*dim)
                .map_err(|_| Error::limit_exceeded("tiled tensor extent does not fit usize"))
        })
        .collect()
}

/// Place a decoded full tile at its canonical grid position in `dst`, which
/// holds the whole tensor's row-major bytes.
pub fn place_tiled_tile(
    dst: &mut [u8],
    meta: &TiledTensorMeta,
    entry: &TiledTensorEntry,
    tile: &[u8],
) -> Result<()> {
    let dst_extent = tiled_dst_extent(&meta.shape)?;
    let coords = tile_start(&meta.tile_counts, entry.index);
    let mut dst_start = Vec::with_capacity(meta.shape.len());
    for (axis, coord) in coords.iter().enumerate() {
        dst_start.push(
            coord
                .checked_mul(meta.tile_shape[axis])
                .ok_or_else(|| Error::limit_exceeded("tensor tile start overflows u64"))?,
        );
    }
    copy_box_into(
        dst,
        &dst_extent,
        &dst_start,
        tile,
        &entry.dims,
        meta.dtype.element_size(),
    )
}

/// Place the intersection of `region` and a decoded tile into `out`, which
/// holds the region's canonical row-major bytes.
pub fn place_tiled_tile_region(
    out: &mut [u8],
    region: &crate::tensor::Region,
    meta: &TiledTensorMeta,
    entry: &TiledTensorEntry,
    tile: &[u8],
) -> Result<()> {
    let coords = tile_start(&meta.tile_counts, entry.index);
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
        .checked_mul(meta.dtype.element_size())
        .ok_or_else(|| Error::limit_exceeded("region box byte length overflows u64"))?;
    let mut boxed = vec![
        0u8;
        usize::try_from(dst_bytes)
            .map_err(|_| Error::limit_exceeded("region box does not fit usize"))?
    ];
    copy_box_bytes(
        &mut boxed,
        tile,
        &meta.tile_shape,
        &local_start,
        &local_extent,
        meta.dtype.element_size(),
    )?;
    let extent = region.extent();
    let dst_extent: Vec<usize> = extent
        .iter()
        .map(|dim| {
            usize::try_from(*dim)
                .map_err(|_| Error::limit_exceeded("region extent does not fit usize"))
        })
        .collect::<Result<_>>()?;
    let boxed_shape: Vec<u64> = local_extent;
    copy_box_into(
        out,
        &dst_extent,
        &dst_start,
        &boxed,
        &boxed_shape,
        meta.dtype.element_size(),
    )
}

/// Indices of the tiles whose cells intersect `region`, in canonical row-major
/// tile order. Empty when the region is empty.
pub fn tiled_region_indices(
    meta: &TiledTensorMeta,
    region: &crate::tensor::Region,
) -> Result<Vec<usize>> {
    region.validate(&meta.shape)?;
    if region.is_empty() {
        return Ok(Vec::new());
    }
    let mut first = Vec::with_capacity(meta.shape.len());
    let mut last = Vec::with_capacity(meta.shape.len());
    for axis in 0..meta.shape.len() {
        let tile = meta.tile_shape[axis];
        first.push(region.start[axis] / tile);
        last.push((region.end[axis] + tile - 1) / tile);
    }
    let mut indices = Vec::new();
    let mut coords = first.clone();
    loop {
        let mut index: u64 = 0;
        for axis in 0..meta.tile_counts.len() {
            index = index
                .checked_mul(meta.tile_counts[axis])
                .and_then(|value| value.checked_add(coords[axis]))
                .ok_or_else(|| Error::limit_exceeded("tensor tile index overflows u64"))?;
        }
        let index = usize::try_from(index)
            .map_err(|_| Error::limit_exceeded("tensor tile index does not fit usize"))?;
        if index >= meta.tiles.len() {
            return Err(Error::corrupt(
                "tensor tile index is outside the descriptor",
            ));
        }
        indices.push(index);
        let mut axis = first.len();
        loop {
            if axis == 0 {
                return Ok(indices);
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
    let total = usize::try_from(
        meta.element_count
            .checked_mul(meta.dtype.element_size())
            .ok_or_else(|| Error::limit_exceeded("tiled tensor decoded bytes overflow u64"))?,
    )
    .map_err(|_| Error::limit_exceeded("tiled tensor decoded bytes do not fit usize"))?;
    let mut out = vec![0u8; total];
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
        let stored = descriptor
            .payload
            .get(start..end)
            .ok_or_else(|| Error::corrupt("tensor tile range is outside the payload"))?;
        let tile = decode_tiled_tile(meta.dtype, entry, stored, limits)?;
        tiles_bytes = tiles_bytes
            .checked_add(tile.len())
            .ok_or_else(|| Error::limit_exceeded("tiled tensor decoded length overflows usize"))?;
        place_tiled_tile(&mut out, meta, entry, &tile)?;
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

    let indices = tiled_region_indices(meta, region)?;
    let mut tiles_touched = 0u64;
    for index in indices {
        let entry = &meta.tiles[index];
        let stored = fetch(entry)?;
        let tile = decode_tiled_tile(meta.dtype, entry, &stored, limits)?;
        place_tiled_tile_region(&mut out, region, meta, entry, &tile)?;
        tiles_touched += 1;
    }
    Ok((out, tiles_touched))
}

// ---------------------------------------------------------------------------
// FLOAT_TILED (mode 17, WEFT-AI-CODEC §18)
// ---------------------------------------------------------------------------

/// Canonical float-tiled schema.
pub const FLOAT_TILED_SCHEMA: &str = "weft.ai.float_tiled.v1";
/// The only float-tile layout implemented in v1.
pub const FLOAT_TILED_LAYOUT: &str = "row_major";

/// The tensor-identity dtype for a float tile dtype, or `None` when the dtype
/// has no logical-tensor layout in v1 (fp8 has none).
fn float_tiled_tensor_dtype(dtype: FloatDtype) -> Option<TensorDtype> {
    match dtype {
        FloatDtype::Bf16 => Some(TensorDtype::Bf16),
        FloatDtype::F16 => Some(TensorDtype::F16),
        FloatDtype::F32 => Some(TensorDtype::F32),
        FloatDtype::Fp8E4m3 | FloatDtype::Fp8E5m2 => None,
    }
}

/// One independently decodable float tile.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FloatTiledEntry {
    /// Canonical grid position (row-major over `tile_counts`).
    pub index: u64,
    /// Tile extents; every tile is full-size in v1.
    pub dims: Vec<u64>,
    /// Byte offset of the stored tile inside the frame payload.
    pub offset: u64,
    /// `exponent_plane_len + mantissa_plane_len`.
    pub compressed_len: u64,
    /// Huffman-coded exponent plane length for this tile.
    pub exponent_plane_len: u64,
    /// Raw sign+mantissa plane length for this tile.
    pub mantissa_plane_len: u64,
    /// Decoded tile bytes (`product(dims) * element_size`).
    pub element_len: u64,
    /// `H("weft/tensor/float_tile/v1", WCB({dtype, dims, element_len,
    /// exponent_plane_len, mantissa_plane_len, values_hash}))`.
    pub tile_root: Hash32,
}

/// A parsed mode-17 descriptor (without the payload bytes).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FloatTiledMeta {
    pub dtype: FloatDtype,
    pub shape: Vec<u64>,
    pub element_count: u64,
    pub tile_shape: Vec<u64>,
    pub tile_counts: Vec<u64>,
    pub tiles: Vec<FloatTiledEntry>,
    pub payload_len: u64,
    pub logical_root: Hash32,
}

/// A parsed mode-17 frame: descriptor plus the stored payload slice.
#[derive(Clone, Debug)]
pub struct FloatTiledDescriptor<'a> {
    pub meta: FloatTiledMeta,
    pub payload: &'a [u8],
}

/// `H("weft/tensor/float_tile/v1", WCB({...}))` for one decoded tile.
pub fn float_tile_root(
    dtype: FloatDtype,
    dims: &[u64],
    tile_bytes: &[u8],
    exponent_plane_len: u64,
    mantissa_plane_len: u64,
    limits: &Limits,
) -> Result<Hash32> {
    let value = Value::map([
        ("dtype", Value::Str(dtype.as_str().to_owned())),
        (
            "dims",
            Value::Array(dims.iter().map(|dim| Value::U64(*dim)).collect()),
        ),
        ("element_len", Value::U64(tile_bytes.len() as u64)),
        ("exponent_plane_len", Value::U64(exponent_plane_len)),
        ("mantissa_plane_len", Value::U64(mantissa_plane_len)),
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
        crate::hash::DOMAIN_FLOAT_TILE,
        &wcb::encode(&value, limits)?,
    ))
}

/// Decode one stored float tile and verify its `tile_root`.
pub fn decode_float_tile(
    dtype: FloatDtype,
    entry: &FloatTiledEntry,
    stored: &[u8],
    limits: &Limits,
) -> Result<Vec<u8>> {
    if stored.len() as u64 != entry.compressed_len {
        return Err(Error::corrupt(format!(
            "float tile {} stored range is {} bytes but the descriptor records {}",
            entry.index,
            stored.len(),
            entry.compressed_len
        )));
    }
    let exponent_len = usize::try_from(entry.exponent_plane_len).map_err(|_| {
        Error::limit_exceeded("float tile exponent plane length does not fit usize")
    })?;
    let mantissa_len = usize::try_from(entry.mantissa_plane_len).map_err(|_| {
        Error::limit_exceeded("float tile mantissa plane length does not fit usize")
    })?;
    let element_len = usize::try_from(entry.element_len)
        .map_err(|_| Error::limit_exceeded("float tile length does not fit usize"))?;
    let element_size = usize::try_from(dtype.element_size())
        .map_err(|_| Error::limit_exceeded("float tile element size does not fit usize"))?;
    if element_size == 0 || element_len % element_size != 0 {
        return Err(Error::malformed(
            "float tile element length is not a whole number of elements",
        ));
    }
    let elements = element_len / element_size;
    let (exponent_stored, mantissa_plane) = stored.split_at(exponent_len);
    if mantissa_plane.len() != mantissa_len {
        return Err(Error::corrupt(format!(
            "float tile {} mantissa range is {} bytes but the descriptor records {}",
            entry.index,
            mantissa_plane.len(),
            mantissa_len
        )));
    }
    let exponent_plane = huffman::decode(exponent_stored, elements)?;
    let tile = join_float_tensor_payload(
        &exponent_plane,
        mantissa_plane,
        dtype,
        elements as u64,
        limits,
    )?;
    let root = float_tile_root(
        dtype,
        &entry.dims,
        &tile,
        entry.exponent_plane_len,
        entry.mantissa_plane_len,
        limits,
    )?;
    if root != entry.tile_root {
        return Err(Error::corrupt(format!(
            "float tile {} root {root} does not match the descriptor's {}",
            entry.index, entry.tile_root
        )));
    }
    Ok(tile)
}

fn float_tiled_meta_value(meta: &FloatTiledMeta) -> Value {
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
                ("exponent_plane_len", Value::U64(entry.exponent_plane_len)),
                ("mantissa_plane_len", Value::U64(entry.mantissa_plane_len)),
                ("element_len", Value::U64(entry.element_len)),
                (
                    "tile_root",
                    Value::Bytes(entry.tile_root.as_bytes().to_vec()),
                ),
            ])
        })
        .collect();
    Value::map([
        ("schema", Value::Str(FLOAT_TILED_SCHEMA.to_owned())),
        ("dtype", Value::Str(meta.dtype.as_str().to_owned())),
        (
            "shape",
            Value::Array(meta.shape.iter().map(|dim| Value::U64(*dim)).collect()),
        ),
        ("element_count", Value::U64(meta.element_count)),
        ("layout", Value::Str(FLOAT_TILED_LAYOUT.to_owned())),
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
        (
            "exponent_coder",
            Value::Str(FLOAT_TENSOR_EXPONENT_CODER.to_owned()),
        ),
        ("tiles", Value::Array(tiles)),
        ("payload_len", Value::U64(meta.payload_len)),
        (
            "logical_root",
            Value::Bytes(meta.logical_root.as_bytes().to_vec()),
        ),
    ])
}

/// Encode a mode-17 `FLOAT_TILED` frame.
///
/// Each tile is independently split with the exact mode-9 plane split (a
/// `huffman-canonical-v1` exponent plane plus a raw sign+mantissa plane), and
/// the stored tiles are concatenated with explicit offsets. A decode of any
/// single tile verifies that tile's root; a full decode additionally verifies
/// the whole-tensor logical root.
pub fn encode_float_tiled(
    payload: &[u8],
    dtype: FloatDtype,
    shape: &[u64],
    tile_shape: &[u64],
    limits: &Limits,
) -> Result<Vec<u8>> {
    check_len(
        payload.len() as u64,
        limits.max_decoded_bytes,
        "float tiled payload",
    )?;
    let tensor_dtype = float_tiled_tensor_dtype(dtype).ok_or_else(|| {
        Error::unsupported_feature(format!(
            "float-tiled v1 requires a tensor identity layout; dtype `{}` has none",
            dtype.as_str()
        ))
    })?;
    let element_count = float_tensor_product(shape, limits)?;
    let element_size = dtype.element_size();
    let decoded_bytes = element_count
        .checked_mul(element_size)
        .ok_or_else(|| Error::limit_exceeded("float tiled decoded bytes overflow u64"))?;
    if decoded_bytes != payload.len() as u64 {
        return Err(Error::malformed(format!(
            "float tiled payload is {} bytes but shape {:?} x {} requires {decoded_bytes}",
            payload.len(),
            shape,
            dtype.as_str()
        )));
    }
    let counts = tile_counts(shape, tile_shape)?;
    let tile_elements = tiled_product(tile_shape, limits, "float tile")?;
    let tile_element_len = tile_elements
        .checked_mul(element_size)
        .ok_or_else(|| Error::limit_exceeded("float tile byte length overflows u64"))?;
    let tile_count = counts.iter().try_fold(1u64, |acc, count| {
        acc.checked_mul(*count)
            .ok_or_else(|| Error::limit_exceeded("float tile count overflows u64"))
    })?;
    if tile_count > limits.max_structured_items {
        return Err(Error::limit_exceeded(format!(
            "float tile count {tile_count} exceeds limit {}",
            limits.max_structured_items
        )));
    }
    let tile_len_usize = usize::try_from(tile_element_len)
        .map_err(|_| Error::limit_exceeded("float tile byte length does not fit usize"))?;
    let mut stored = Vec::with_capacity(payload.len());
    let mut tiles = Vec::with_capacity(usize::try_from(tile_count).unwrap_or(0));
    for index in 0..tile_count {
        let start = tile_start(&counts, index);
        let mut start_bytes = start.clone();
        for axis in 0..start_bytes.len() {
            start_bytes[axis] = start_bytes[axis]
                .checked_mul(tile_shape[axis])
                .ok_or_else(|| Error::limit_exceeded("float tile start overflows u64"))?;
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
        let (exponent_plane, mantissa_plane) =
            split_float_tensor_payload(&tile, dtype, tile_elements)?;
        let root = float_tile_root(
            dtype,
            tile_shape,
            &tile,
            exponent_plane.len() as u64,
            mantissa_plane.len() as u64,
            limits,
        )?;
        let compressed_len = (exponent_plane.len() + mantissa_plane.len()) as u64;
        tiles.push(FloatTiledEntry {
            index,
            dims: tile_shape.to_vec(),
            offset: stored.len() as u64,
            compressed_len,
            exponent_plane_len: exponent_plane.len() as u64,
            mantissa_plane_len: mantissa_plane.len() as u64,
            element_len: tile_element_len,
            tile_root: root,
        });
        stored.extend_from_slice(&exponent_plane);
        stored.extend_from_slice(&mantissa_plane);
    }

    let logical_root = crate::tensor::logical_root(tensor_dtype, shape, payload, limits)?;
    let meta = FloatTiledMeta {
        dtype,
        shape: shape.to_vec(),
        element_count,
        tile_shape: tile_shape.to_vec(),
        tile_counts: counts,
        tiles,
        payload_len: stored.len() as u64,
        logical_root,
    };
    let body = build_ai_body(&float_tiled_meta_value(&meta), &stored, limits)?;
    let parts = FrameParts {
        format_major: FORMAT_MAJOR,
        format_minor: FORMAT_MINOR,
        domain: DOMAIN_PAYLOAD,
        mode: Mode::FloatTiled,
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

fn meta_u64_array(map: &BTreeMap<String, Value>, key: &str) -> Result<Vec<u64>> {
    match map.get(key) {
        Some(Value::Array(items)) => {
            let mut out = Vec::with_capacity(items.len());
            for item in items {
                match item {
                    Value::U64(dim) => out.push(*dim),
                    _ => {
                        return Err(Error::malformed(format!(
                            "float tiled `{key}` elements must be u64"
                        )))
                    }
                }
            }
            Ok(out)
        }
        _ => Err(Error::malformed(format!(
            "float tiled `{key}` must be an array"
        ))),
    }
}

/// Parse a mode-17 descriptor from its canonical WCB bytes.
pub fn parse_float_tiled_descriptor(
    descriptor_bytes: &[u8],
    decoded_len: u64,
    payload_len: u64,
    limits: &Limits,
) -> Result<FloatTiledMeta> {
    let value = wcb::decode(descriptor_bytes, limits)?;
    let map = ai_map(&value, "float tiled")?;
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
            "exponent_coder",
            "tiles",
            "payload_len",
            "logical_root",
        ],
        "float tiled",
    )?;
    let schema = meta_text(map, "schema")?;
    if schema != FLOAT_TILED_SCHEMA {
        return Err(Error::malformed(format!(
            "unsupported float tiled schema `{schema}`"
        )));
    }
    let dtype = FloatDtype::from_name(meta_text(map, "dtype")?)?;
    let layout = meta_text(map, "layout")?;
    if layout != FLOAT_TILED_LAYOUT {
        return Err(Error::malformed(format!(
            "float tiled layout `{layout}` is specified but not implemented in v1 (only {FLOAT_TILED_LAYOUT})"
        )));
    }
    let coder = meta_text(map, "exponent_coder")?;
    if coder != FLOAT_TENSOR_EXPONENT_CODER {
        return Err(Error::malformed(format!(
            "unsupported float tile exponent coder `{coder}` (v1 allows `{FLOAT_TENSOR_EXPONENT_CODER}`)"
        )));
    }
    let tensor_dtype = float_tiled_tensor_dtype(dtype);
    if tensor_dtype.is_none() {
        return Err(Error::unsupported_feature(format!(
            "float-tiled v1 requires a tensor identity layout; dtype `{}` has none",
            dtype.as_str()
        )));
    }
    let shape = meta_shape(map)?;
    let element_count = float_tensor_product(&shape, limits)?;
    let declared_count = meta_u64(map, "element_count")?;
    if declared_count != element_count {
        return Err(Error::malformed(format!(
            "float tiled element_count {declared_count} does not match shape product {element_count}"
        )));
    }
    let element_size = dtype.element_size();
    let decoded_bytes = element_count
        .checked_mul(element_size)
        .ok_or_else(|| Error::limit_exceeded("float tiled decoded bytes overflow u64"))?;
    check_len(
        decoded_bytes,
        limits.max_decoded_bytes,
        "float tiled decoded bytes",
    )?;
    if decoded_len != decoded_bytes {
        return Err(Error::corrupt(format!(
            "float tiled decoded_len is {decoded_len} but the descriptor requires {decoded_bytes}"
        )));
    }
    let tile_shape = meta_u64_array(map, "tile_shape")?;
    let counts = tile_counts(&shape, &tile_shape)?;
    if meta_u64_array(map, "tile_counts")? != counts {
        return Err(Error::malformed(format!(
            "float tiled tile_counts {:?} do not match shape/tile_shape {:?}",
            meta_u64_array(map, "tile_counts")?,
            counts
        )));
    }
    let tile_elements = tiled_product(&tile_shape, limits, "float tile")?;
    let tile_element_len = tile_elements
        .checked_mul(element_size)
        .ok_or_else(|| Error::limit_exceeded("float tile byte length overflows u64"))?;
    let tile_count = counts.iter().try_fold(1u64, |acc, count| {
        acc.checked_mul(*count)
            .ok_or_else(|| Error::limit_exceeded("float tile count overflows u64"))
    })?;
    if tile_count > limits.max_structured_items {
        return Err(Error::limit_exceeded(format!(
            "float tile count {tile_count} exceeds limit {}",
            limits.max_structured_items
        )));
    }
    let expected_mantissa = tile_elements
        .checked_mul(dtype.sign_bits() + dtype.mantissa_bits())
        .ok_or_else(|| Error::limit_exceeded("float tile mantissa bits overflow u64"))?;
    let expected_mantissa = expected_mantissa
        .checked_add(7)
        .ok_or_else(|| Error::limit_exceeded("float tile mantissa length overflows u64"))?
        / 8;
    let declared_payload_len = meta_u64(map, "payload_len")?;
    if declared_payload_len != payload_len {
        return Err(Error::malformed(format!(
            "float tiled payload is {payload_len} bytes but metadata declares {declared_payload_len}"
        )));
    }
    let entries = match map.get("tiles") {
        Some(Value::Array(items)) => items,
        _ => return Err(Error::malformed("float tiled `tiles` must be an array")),
    };
    if entries.len() as u64 != tile_count {
        return Err(Error::malformed(format!(
            "float tiled has {} tile entries but the grid requires {tile_count}",
            entries.len()
        )));
    }
    let mut tiles = Vec::with_capacity(entries.len());
    let mut expected_offset: u64 = 0;
    for (position, item) in entries.iter().enumerate() {
        let entry = ai_map(item, "float tile")?;
        check_exact_keys(
            entry,
            &[
                "index",
                "dims",
                "offset",
                "compressed_len",
                "exponent_plane_len",
                "mantissa_plane_len",
                "element_len",
                "tile_root",
            ],
            "float tile",
        )?;
        let index = meta_u64(entry, "index")?;
        if index != position as u64 {
            return Err(Error::malformed(format!(
                "float tiled entry {position} declares index {index}; entries must be in canonical order"
            )));
        }
        let dims = meta_u64_array(entry, "dims")?;
        if dims != tile_shape {
            return Err(Error::malformed(format!(
                "float tile {position} dims {:?} do not equal tile_shape {:?}; partial tiles are not defined in v1",
                dims, tile_shape
            )));
        }
        let declared_element_len = meta_u64(entry, "element_len")?;
        if declared_element_len != tile_element_len {
            return Err(Error::malformed(format!(
                "float tile {position} element_len {declared_element_len} does not match {tile_element_len}"
            )));
        }
        let offset = meta_u64(entry, "offset")?;
        if offset != expected_offset {
            return Err(Error::malformed(format!(
                "float tile {position} offset {offset} is not the contiguous offset {expected_offset}"
            )));
        }
        let compressed_len = meta_u64(entry, "compressed_len")?;
        let exponent_plane_len = meta_u64(entry, "exponent_plane_len")?;
        let mantissa_plane_len = meta_u64(entry, "mantissa_plane_len")?;
        if exponent_plane_len < huffman::TABLE_BYTES as u64 {
            return Err(Error::malformed(format!(
                "float tile {position} exponent plane is shorter than its code-length table"
            )));
        }
        if mantissa_plane_len != expected_mantissa {
            return Err(Error::malformed(format!(
                "float tile {position} mantissa_plane_len {mantissa_plane_len} does not match {expected_mantissa}"
            )));
        }
        if compressed_len
            != exponent_plane_len
                .checked_add(mantissa_plane_len)
                .ok_or_else(|| Error::limit_exceeded("float tile length overflows u64"))?
        {
            return Err(Error::malformed(format!(
                "float tile {position} compressed_len {compressed_len} does not equal its plane lengths"
            )));
        }
        expected_offset = expected_offset
            .checked_add(compressed_len)
            .ok_or_else(|| Error::limit_exceeded("float tile offset overflows u64"))?;
        let tile_root = Hash32::from_slice(meta_bytes(entry, "tile_root")?)
            .map_err(|_| Error::malformed("float tile_root must be 32 bytes"))?;
        tiles.push(FloatTiledEntry {
            index,
            dims,
            offset,
            compressed_len,
            exponent_plane_len,
            mantissa_plane_len,
            element_len: declared_element_len,
            tile_root,
        });
    }
    if expected_offset != declared_payload_len {
        return Err(Error::malformed(format!(
            "float tile ranges cover {expected_offset} bytes but payload_len is {declared_payload_len}"
        )));
    }
    let logical_root = Hash32::from_slice(meta_bytes(map, "logical_root")?)
        .map_err(|_| Error::malformed("float tiled logical_root must be 32 bytes"))?;
    Ok(FloatTiledMeta {
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

/// Parse a mode-17 frame and return its descriptor with the stored slice.
pub fn parse_float_tiled<'a>(
    frame: &'a Frame<'_>,
    limits: &Limits,
) -> Result<FloatTiledDescriptor<'a>> {
    let (descriptor_bytes, payload) = split_ai_body_prefix(frame.body)?;
    let meta = parse_float_tiled_descriptor(
        descriptor_bytes,
        frame.header.decoded_len,
        payload.len() as u64,
        limits,
    )?;
    Ok(FloatTiledDescriptor { meta, payload })
}

/// Validate and decode a mode-17 frame into exact row-major element bytes.
pub fn decode_float_tiled(frame: &Frame<'_>, limits: &Limits) -> Result<Vec<u8>> {
    let descriptor = parse_float_tiled(frame, limits)?;
    let meta = &descriptor.meta;
    let total = usize::try_from(
        meta.element_count
            .checked_mul(meta.dtype.element_size())
            .ok_or_else(|| Error::limit_exceeded("float tiled decoded bytes overflow u64"))?,
    )
    .map_err(|_| Error::limit_exceeded("float tiled decoded bytes do not fit usize"))?;
    let mut out = vec![0u8; total];
    let mut tiles_bytes = 0usize;
    for entry in &meta.tiles {
        let start = usize::try_from(entry.offset)
            .map_err(|_| Error::limit_exceeded("float tile offset does not fit usize"))?;
        let end = start
            .checked_add(
                usize::try_from(entry.compressed_len)
                    .map_err(|_| Error::limit_exceeded("float tile length does not fit usize"))?,
            )
            .ok_or_else(|| Error::limit_exceeded("float tile range overflows usize"))?;
        let stored = descriptor
            .payload
            .get(start..end)
            .ok_or_else(|| Error::corrupt("float tile range is outside the payload"))?;
        let tile = decode_float_tile(meta.dtype, entry, stored, limits)?;
        tiles_bytes = tiles_bytes
            .checked_add(tile.len())
            .ok_or_else(|| Error::limit_exceeded("float tiled decoded length overflows usize"))?;
        place_float_tile(&mut out, meta, entry, &tile)?;
    }
    if tiles_bytes != total {
        return Err(Error::corrupt(format!(
            "float tiled decoded to {tiles_bytes} bytes but the descriptor requires {total}"
        )));
    }
    let tensor_dtype = float_tiled_tensor_dtype(meta.dtype)
        .ok_or_else(|| Error::unsupported_feature("float tiled dtype has no tensor identity"))?;
    let logical = crate::tensor::logical_root(tensor_dtype, &meta.shape, &out, limits)?;
    if logical != meta.logical_root {
        return Err(Error::corrupt(format!(
            "float tiled logical root {logical} does not match the descriptor's {}",
            meta.logical_root
        )));
    }
    Ok(out)
}

/// Parse a mode-17 frame and return only its descriptor.
pub fn float_tiled_metadata(frame: &Frame<'_>, limits: &Limits) -> Result<FloatTiledMeta> {
    Ok(parse_float_tiled(frame, limits)?.meta)
}

/// Decode a mode-17 frame with `threads` tile workers.
///
/// `threads <= 1` is exactly [`decode_float_tiled`]. Larger values decode
/// independent tiles on a bounded pool; every tile root, the logical root and
/// the returned bytes are identical to the sequential call.
pub fn decode_float_tiled_with_threads(
    frame: &Frame<'_>,
    limits: &Limits,
    threads: usize,
) -> Result<Vec<u8>> {
    if threads <= 1 {
        return decode_float_tiled(frame, limits);
    }
    let descriptor = parse_float_tiled(frame, limits)?;
    let meta = &descriptor.meta;
    let total = usize::try_from(
        meta.element_count
            .checked_mul(meta.dtype.element_size())
            .ok_or_else(|| Error::limit_exceeded("float tiled decoded bytes overflow u64"))?,
    )
    .map_err(|_| Error::limit_exceeded("float tiled decoded bytes do not fit usize"))?;
    let mut out = vec![0u8; total];
    let decode_tile = |entry: &FloatTiledEntry| -> Result<Vec<u8>> {
        let start = usize::try_from(entry.offset)
            .map_err(|_| Error::limit_exceeded("float tile offset does not fit usize"))?;
        let end = start
            .checked_add(
                usize::try_from(entry.compressed_len)
                    .map_err(|_| Error::limit_exceeded("float tile length does not fit usize"))?,
            )
            .ok_or_else(|| Error::limit_exceeded("float tile range overflows usize"))?;
        let stored = descriptor
            .payload
            .get(start..end)
            .ok_or_else(|| Error::corrupt("float tile range is outside the payload"))?;
        decode_float_tile(meta.dtype, entry, stored, limits)
    };
    let mut tiles: Vec<Vec<u8>> = Vec::with_capacity(meta.tiles.len());
    let workers = threads.min(meta.tiles.len()).min(64).max(1);
    let chunk = meta.tiles.len().div_ceil(workers);
    let mut chunk_results: Vec<Result<Vec<Vec<u8>>>> = Vec::new();
    std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for start in (0..meta.tiles.len()).step_by(chunk) {
            let end = (start + chunk).min(meta.tiles.len());
            let entries = &meta.tiles[start..end];
            let decode_tile = &decode_tile;
            handles.push(scope.spawn(move || {
                let mut decoded = Vec::with_capacity(entries.len());
                for entry in entries {
                    decoded.push(decode_tile(entry)?);
                }
                Ok(decoded)
            }));
        }
        for handle in handles {
            chunk_results.push(
                handle
                    .join()
                    .unwrap_or_else(|_| Err(Error::corrupt("float tile worker panicked"))),
            );
        }
    });
    for result in chunk_results {
        tiles.extend(result?);
    }
    let mut tiles_bytes = 0usize;
    for (entry, tile) in meta.tiles.iter().zip(tiles.iter()) {
        tiles_bytes = tiles_bytes
            .checked_add(tile.len())
            .ok_or_else(|| Error::limit_exceeded("float tiled decoded length overflows usize"))?;
        place_float_tile(&mut out, meta, entry, tile)?;
    }
    if tiles_bytes != total {
        return Err(Error::corrupt(format!(
            "float tiled decoded to {tiles_bytes} bytes but the descriptor requires {total}"
        )));
    }
    let tensor_dtype = float_tiled_tensor_dtype(meta.dtype)
        .ok_or_else(|| Error::unsupported_feature("float tiled dtype has no tensor identity"))?;
    let logical = crate::tensor::logical_root(tensor_dtype, &meta.shape, &out, limits)?;
    if logical != meta.logical_root {
        return Err(Error::corrupt(format!(
            "float tiled logical root {logical} does not match the descriptor's {}",
            meta.logical_root
        )));
    }
    Ok(out)
}

fn float_tile_start_bytes(meta: &FloatTiledMeta, entry: &FloatTiledEntry) -> Result<Vec<u64>> {
    let coords = tile_start(&meta.tile_counts, entry.index);
    let mut start = Vec::with_capacity(coords.len());
    for (axis, coord) in coords.iter().enumerate() {
        start.push(
            coord
                .checked_mul(meta.tile_shape[axis])
                .ok_or_else(|| Error::limit_exceeded("float tile start overflows u64"))?,
        );
    }
    Ok(start)
}

/// Place one decoded full tile at its canonical grid position.
pub fn place_float_tile(
    dst: &mut [u8],
    meta: &FloatTiledMeta,
    entry: &FloatTiledEntry,
    tile: &[u8],
) -> Result<()> {
    let dst_extent = tiled_dst_extent(&meta.shape)?;
    let dst_start = float_tile_start_bytes(meta, entry)?;
    copy_box_into(
        dst,
        &dst_extent,
        &dst_start,
        tile,
        &entry.dims,
        meta.dtype.element_size(),
    )
}

/// Place the intersection of `region` and a decoded tile into `out`.
pub fn place_float_tile_region(
    out: &mut [u8],
    region: &crate::tensor::Region,
    meta: &FloatTiledMeta,
    entry: &FloatTiledEntry,
    tile: &[u8],
) -> Result<()> {
    let base = float_tile_start_bytes(meta, entry)?;
    let mut local_start = Vec::with_capacity(meta.shape.len());
    let mut local_extent = Vec::with_capacity(meta.shape.len());
    let mut dst_start = Vec::with_capacity(meta.shape.len());
    for axis in 0..meta.shape.len() {
        let lo = region.start[axis].max(base[axis]);
        let hi = region.end[axis].min(base[axis] + meta.tile_shape[axis]);
        local_start.push(lo - base[axis]);
        local_extent.push(hi - lo);
        dst_start.push(lo - region.start[axis]);
    }
    let dst_len = local_extent.iter().try_fold(1u64, |acc, dim| {
        acc.checked_mul(*dim)
            .ok_or_else(|| Error::limit_exceeded("region box overflows u64"))
    })?;
    let dst_bytes = dst_len
        .checked_mul(meta.dtype.element_size())
        .ok_or_else(|| Error::limit_exceeded("region box byte length overflows u64"))?;
    let mut boxed = vec![
        0u8;
        usize::try_from(dst_bytes)
            .map_err(|_| Error::limit_exceeded("region box does not fit usize"))?
    ];
    copy_box_bytes(
        &mut boxed,
        tile,
        &meta.tile_shape,
        &local_start,
        &local_extent,
        meta.dtype.element_size(),
    )?;
    let extent = region.extent();
    let dst_extent: Vec<usize> = extent
        .iter()
        .map(|dim| {
            usize::try_from(*dim)
                .map_err(|_| Error::limit_exceeded("region extent does not fit usize"))
        })
        .collect::<Result<_>>()?;
    copy_box_into(
        out,
        &dst_extent,
        &dst_start,
        &boxed,
        &local_extent,
        meta.dtype.element_size(),
    )
}

/// Indices of the tiles whose cells intersect `region`.
pub fn float_tiled_region_indices(
    meta: &FloatTiledMeta,
    region: &crate::tensor::Region,
) -> Result<Vec<usize>> {
    region.validate(&meta.shape)?;
    if region.is_empty() {
        return Ok(Vec::new());
    }
    let mut first = Vec::with_capacity(meta.shape.len());
    let mut last = Vec::with_capacity(meta.shape.len());
    for axis in 0..meta.shape.len() {
        let tile = meta.tile_shape[axis];
        first.push(region.start[axis] / tile);
        last.push((region.end[axis] + tile - 1) / tile);
    }
    let mut indices = Vec::new();
    let mut coords = first.clone();
    loop {
        let mut index: u64 = 0;
        for axis in 0..meta.tile_counts.len() {
            index = index
                .checked_mul(meta.tile_counts[axis])
                .and_then(|value| value.checked_add(coords[axis]))
                .ok_or_else(|| Error::limit_exceeded("float tile index overflows u64"))?;
        }
        let index = usize::try_from(index)
            .map_err(|_| Error::limit_exceeded("float tile index does not fit usize"))?;
        if index >= meta.tiles.len() {
            return Err(Error::corrupt("float tile index is outside the descriptor"));
        }
        indices.push(index);
        let mut axis = first.len();
        loop {
            if axis == 0 {
                return Ok(indices);
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

/// Read one logical region, decoding only the float tiles it intersects.
pub fn read_float_region(
    descriptor: &FloatTiledDescriptor<'_>,
    region: &crate::tensor::Region,
    limits: &Limits,
) -> Result<(Vec<u8>, u64)> {
    read_float_region_with(&descriptor.meta, region, limits, |entry| {
        let start = usize::try_from(entry.offset)
            .map_err(|_| Error::limit_exceeded("float tile offset does not fit usize"))?;
        let end = start
            .checked_add(
                usize::try_from(entry.compressed_len)
                    .map_err(|_| Error::limit_exceeded("float tile length does not fit usize"))?,
            )
            .ok_or_else(|| Error::limit_exceeded("float tile range overflows usize"))?;
        descriptor
            .payload
            .get(start..end)
            .map(|bytes| bytes.to_vec())
            .ok_or_else(|| Error::corrupt("float tile range is outside the payload"))
    })
}

/// Read one logical region with a caller-supplied stored-tile fetch.
pub fn read_float_region_with<F>(
    meta: &FloatTiledMeta,
    region: &crate::tensor::Region,
    limits: &Limits,
    mut fetch: F,
) -> Result<(Vec<u8>, u64)>
where
    F: FnMut(&FloatTiledEntry) -> Result<Vec<u8>>,
{
    region.validate(&meta.shape)?;
    let extent = region.extent();
    if region.is_empty() {
        return Ok((Vec::new(), 0));
    }
    let count = extent.iter().try_fold(1u64, |acc, dim| {
        acc.checked_mul(*dim)
            .ok_or_else(|| Error::limit_exceeded("region element count overflows u64"))
    })?;
    let byte_len = count
        .checked_mul(meta.dtype.element_size())
        .ok_or_else(|| Error::limit_exceeded("region payload length overflows u64"))?;
    check_len(byte_len, limits.max_decoded_bytes, "region payload")?;
    let byte_len = usize::try_from(byte_len)
        .map_err(|_| Error::limit_exceeded("region payload does not fit usize"))?;
    let mut out = vec![0u8; byte_len];
    let indices = float_tiled_region_indices(meta, region)?;
    let mut tiles_touched = 0u64;
    for index in indices {
        let entry = &meta.tiles[index];
        let stored = fetch(entry)?;
        let tile = decode_float_tile(meta.dtype, entry, &stored, limits)?;
        place_float_tile_region(&mut out, region, meta, entry, &tile)?;
        tiles_touched += 1;
    }
    Ok((out, tiles_touched))
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

// ---------------------------------------------------------------------------
// FLOAT_TILED_STREAMS (mode 18, WEFT-AI-CODEC §19)
// ---------------------------------------------------------------------------

/// Canonical multi-stream float-tiled schema.
pub const FLOAT_TILED_STREAMS_SCHEMA: &str = "weft.ai.tensor_tiled_streams.v1";
/// Smallest declared stream count.
pub const FLOAT_TILED_STREAMS_MIN: u64 = 1;
/// Largest declared stream count.
pub const FLOAT_TILED_STREAMS_MAX: u64 = 16;
/// Default stream count offered by the store planner.
pub const FLOAT_TILED_STREAMS_DEFAULT: u64 = 4;

/// One independently decodable multi-stream float tile.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FloatTiledStreamsEntry {
    pub index: u64,
    pub dims: Vec<u64>,
    pub offset: u64,
    pub compressed_len: u64,
    /// Total encoded exponent bytes (all `n_streams` Huffman streams).
    pub exponent_plane_len: u64,
    /// Raw sign+mantissa plane length (all `n_streams` raw streams).
    pub mantissa_plane_len: u64,
    pub element_len: u64,
    /// Starts of the `2 * n_streams` streams, relative to the tile start.
    /// Every exponent stream is one `huffman-canonical-v1` stream with its own
    /// 256-byte table; every mantissa stream is raw. The first `n_streams`
    /// entries are the exponent streams and the next `n_streams` the mantissa
    /// streams.
    pub stream_offsets: Vec<u64>,
    pub tile_root: Hash32,
}

/// A parsed mode-18 descriptor (without the payload bytes).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FloatTiledStreamsMeta {
    pub dtype: FloatDtype,
    pub shape: Vec<u64>,
    pub element_count: u64,
    pub tile_shape: Vec<u64>,
    pub tile_counts: Vec<u64>,
    pub n_streams: u64,
    pub tiles: Vec<FloatTiledStreamsEntry>,
    pub payload_len: u64,
    pub logical_root: Hash32,
}

/// A parsed mode-18 frame: descriptor plus the stored payload slice.
#[derive(Clone, Debug)]
pub struct FloatTiledStreamsDescriptor<'a> {
    pub meta: FloatTiledStreamsMeta,
    pub payload: &'a [u8],
}

/// `H("weft/tensor/float_tile_streams/v1", WCB({...}))` for one decoded tile.
///
/// The record covers `n_streams` and the complete offset table, so changing
/// either the stream count or one offset changes the tile root.
#[allow(clippy::too_many_arguments)]
pub fn float_tile_streams_root(
    dtype: FloatDtype,
    dims: &[u64],
    tile_bytes: &[u8],
    exponent_plane_len: u64,
    mantissa_plane_len: u64,
    n_streams: u64,
    stream_offsets: &[u64],
    limits: &Limits,
) -> Result<Hash32> {
    let value = Value::map([
        ("dtype", Value::Str(dtype.as_str().to_owned())),
        (
            "dims",
            Value::Array(dims.iter().map(|dim| Value::U64(*dim)).collect()),
        ),
        ("element_len", Value::U64(tile_bytes.len() as u64)),
        ("exponent_plane_len", Value::U64(exponent_plane_len)),
        ("mantissa_plane_len", Value::U64(mantissa_plane_len)),
        ("n_streams", Value::U64(n_streams)),
        (
            "stream_offsets",
            Value::Array(
                stream_offsets
                    .iter()
                    .map(|offset| Value::U64(*offset))
                    .collect(),
            ),
        ),
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
        crate::hash::DOMAIN_FLOAT_TILE_STREAMS,
        &wcb::encode(&value, limits)?,
    ))
}

/// Number of bytes the interleaved stream `stream` holds for a plane of
/// `total` bytes split round-robin over `n_streams`.
pub fn interleaved_stream_len(total: u64, stream: u64, n_streams: u64) -> u64 {
    if n_streams == 0 || total <= stream {
        0
    } else {
        (total - stream).div_ceil(n_streams)
    }
}

fn interleave_plane(plane: &[u8], n_streams: usize) -> Vec<Vec<u8>> {
    let mut streams: Vec<Vec<u8>> = (0..n_streams)
        .map(|_| Vec::with_capacity(plane.len() / n_streams + 1))
        .collect();
    for (index, &byte) in plane.iter().enumerate() {
        streams[index % n_streams].push(byte);
    }
    streams
}

/// Scatter each stream back into its round-robin positions.
///
/// The loop is per stream (stride `n_streams`), which is cache-friendlier and
/// avoids the per-element division the naive form performs. A one-stream plane
/// is a plain copy.
fn deinterleave_slices(streams: &[&[u8]], total: usize, what: &str) -> Result<Vec<u8>> {
    let n_streams = streams.len();
    if n_streams == 0 {
        return Err(Error::malformed(format!("{what} declares zero streams")));
    }
    if n_streams == 1 {
        if streams[0].len() != total {
            return Err(Error::corrupt(format!(
                "{what} single stream is {} bytes but the plane requires {total}",
                streams[0].len()
            )));
        }
        return Ok(streams[0].to_vec());
    }
    let mut out = vec![0u8; total];
    for (stream_index, stream) in streams.iter().enumerate() {
        for (position, &byte) in stream.iter().enumerate() {
            let index = position
                .checked_mul(n_streams)
                .and_then(|value| value.checked_add(stream_index))
                .ok_or_else(|| Error::limit_exceeded("interleaved index overflows usize"))?;
            let slot = out.get_mut(index).ok_or_else(|| {
                Error::corrupt(format!(
                    "{what} stream {stream_index} is longer than its interleaved plane"
                ))
            })?;
            *slot = byte;
        }
    }
    Ok(out)
}

/// Decode `n_streams` independent byte planes with the multi-stream kernel (
/// `threads <= 1`) or a bounded thread pool (`threads > 1`).
///
/// Stream errors are returned in global stream order, so the reported class is
/// the first failing stream's class in both paths.
fn decode_stream_planes(
    encoded: &[&[u8]],
    expected: &[usize],
    threads: usize,
) -> Result<Vec<Vec<u8>>> {
    if threads <= 1 || encoded.len() <= 1 {
        return huffman::decode_streams(encoded, expected);
    }
    let workers = threads.min(encoded.len()).min(64).max(1);
    let chunk = encoded.len().div_ceil(workers);
    let mut chunk_results: Vec<Result<Vec<Vec<u8>>>> = Vec::new();
    std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for start in (0..encoded.len()).step_by(chunk) {
            let end = (start + chunk).min(encoded.len());
            let enc = &encoded[start..end];
            let exp = &expected[start..end];
            handles.push(scope.spawn(move || huffman::decode_streams(enc, exp)));
        }
        for handle in handles {
            chunk_results.push(
                handle
                    .join()
                    .unwrap_or_else(|_| Err(Error::corrupt("multi-stream decode worker panicked"))),
            );
        }
    });
    let mut out = Vec::with_capacity(encoded.len());
    for result in chunk_results {
        out.extend(result?);
    }
    Ok(out)
}

/// Decode one stored multi-stream float tile and verify its `tile_root`.
#[allow(clippy::too_many_arguments)]
pub fn decode_float_tile_streams(
    dtype: FloatDtype,
    entry: &FloatTiledStreamsEntry,
    n_streams: u64,
    stored: &[u8],
    limits: &Limits,
    threads: usize,
) -> Result<Vec<u8>> {
    if !(FLOAT_TILED_STREAMS_MIN..=FLOAT_TILED_STREAMS_MAX).contains(&n_streams) {
        return Err(Error::malformed(format!(
            "float tile {} declares n_streams {n_streams} outside {}..={}",
            entry.index, FLOAT_TILED_STREAMS_MIN, FLOAT_TILED_STREAMS_MAX
        )));
    }
    if entry.stream_offsets.len() as u64 != 2 * n_streams {
        return Err(Error::malformed(format!(
            "float tile {} has {} stream offsets but n_streams {n_streams} requires {}",
            entry.index,
            entry.stream_offsets.len(),
            2 * n_streams
        )));
    }
    if stored.len() as u64 != entry.compressed_len {
        return Err(Error::corrupt(format!(
            "float tile {} stored range is {} bytes but the descriptor records {}",
            entry.index,
            stored.len(),
            entry.compressed_len
        )));
    }
    let exponent_len = usize::try_from(entry.exponent_plane_len).map_err(|_| {
        Error::limit_exceeded("float tile exponent plane length does not fit usize")
    })?;
    let element_len = usize::try_from(entry.element_len)
        .map_err(|_| Error::limit_exceeded("float tile length does not fit usize"))?;
    let element_size = usize::try_from(dtype.element_size())
        .map_err(|_| Error::limit_exceeded("float tile element size does not fit usize"))?;
    if element_size == 0 || element_len % element_size != 0 {
        return Err(Error::malformed(
            "float tile element length is not a whole number of elements",
        ));
    }
    let elements = element_len / element_size;
    let (exponent_stored, mantissa_stored) = stored.split_at(exponent_len);
    let n =
        usize::try_from(n_streams).map_err(|_| Error::malformed("n_streams does not fit usize"))?;

    // Exponent streams: each one a `huffman-canonical-v1` stream with its own
    // 256-byte table.
    let mut exponent_streams: Vec<&[u8]> = Vec::with_capacity(n);
    let mut exponent_expected: Vec<usize> = Vec::with_capacity(n);
    let exponent_elements = u64::try_from(elements)
        .map_err(|_| Error::limit_exceeded("float tile element count does not fit u64"))?;
    for stream in 0..n {
        let start = usize::try_from(entry.stream_offsets[stream])
            .map_err(|_| Error::limit_exceeded("stream offset does not fit usize"))?;
        let end = if stream + 1 < n {
            usize::try_from(entry.stream_offsets[stream + 1])
                .map_err(|_| Error::limit_exceeded("stream offset does not fit usize"))?
        } else {
            exponent_len
        };
        let bytes = exponent_stored
            .get(start..end)
            .ok_or_else(|| Error::corrupt("float tile exponent stream is outside the tile"))?;
        exponent_streams.push(bytes);
        exponent_expected.push(
            usize::try_from(interleaved_stream_len(
                exponent_elements,
                stream as u64,
                n_streams,
            ))
            .map_err(|_| Error::limit_exceeded("stream output length does not fit usize"))?,
        );
    }
    let exponent_parts = decode_stream_planes(&exponent_streams, &exponent_expected, threads)?;
    let exponent_plane = if n == 1 {
        exponent_parts
            .into_iter()
            .next()
            .ok_or_else(|| Error::corrupt("float tile has no exponent stream"))?
    } else {
        let refs: Vec<&[u8]> = exponent_parts.iter().map(Vec::as_slice).collect();
        deinterleave_slices(&refs, elements, "exponent plane")?
    };

    // Mantissa streams: raw interleaved segments.
    let mantissa_total = mantissa_stored.len();
    let mantissa_total_u64 = mantissa_total as u64;
    let mut mantissa_parts: Vec<&[u8]> = Vec::with_capacity(n);
    for stream in 0..n {
        let start = usize::try_from(entry.stream_offsets[n + stream] - entry.exponent_plane_len)
            .map_err(|_| Error::corrupt("mantissa stream offset precedes the exponent plane"))?;
        let end = if stream + 1 < n {
            usize::try_from(entry.stream_offsets[n + stream + 1] - entry.exponent_plane_len)
                .map_err(|_| Error::corrupt("mantissa stream offset precedes the exponent plane"))?
        } else {
            mantissa_total
        };
        let bytes = mantissa_stored
            .get(start..end)
            .ok_or_else(|| Error::corrupt("float tile mantissa stream is outside the tile"))?;
        let expected = interleaved_stream_len(mantissa_total_u64, stream as u64, n_streams);
        if bytes.len() as u64 != expected {
            return Err(Error::corrupt(format!(
                "float tile {} mantissa stream {stream} is {} bytes but the interleave requires {expected}",
                entry.index,
                bytes.len()
            )));
        }
        mantissa_parts.push(bytes);
    }

    let tile = if n == 1 {
        // Byte-identical to mode 17: one Huffman exponent stream and the raw
        // mantissa plane, with no deinterleave copy.
        join_float_tensor_payload(
            &exponent_plane,
            mantissa_stored,
            dtype,
            exponent_elements,
            limits,
        )?
    } else {
        let mantissa_plane =
            deinterleave_slices(&mantissa_parts, mantissa_total, "mantissa plane")?;
        join_float_tensor_payload(
            &exponent_plane,
            &mantissa_plane,
            dtype,
            exponent_elements,
            limits,
        )?
    };
    let root = float_tile_streams_root(
        dtype,
        &entry.dims,
        &tile,
        entry.exponent_plane_len,
        entry.mantissa_plane_len,
        n_streams,
        &entry.stream_offsets,
        limits,
    )?;
    if root != entry.tile_root {
        return Err(Error::corrupt(format!(
            "float tile {} stream root {root} does not match the descriptor's {}",
            entry.index, entry.tile_root
        )));
    }
    Ok(tile)
}

fn float_tiled_streams_meta_value(meta: &FloatTiledStreamsMeta) -> Value {
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
                ("exponent_plane_len", Value::U64(entry.exponent_plane_len)),
                ("mantissa_plane_len", Value::U64(entry.mantissa_plane_len)),
                ("element_len", Value::U64(entry.element_len)),
                (
                    "stream_offsets",
                    Value::Array(
                        entry
                            .stream_offsets
                            .iter()
                            .map(|offset| Value::U64(*offset))
                            .collect(),
                    ),
                ),
                (
                    "tile_root",
                    Value::Bytes(entry.tile_root.as_bytes().to_vec()),
                ),
            ])
        })
        .collect();
    Value::map([
        ("schema", Value::Str(FLOAT_TILED_STREAMS_SCHEMA.to_owned())),
        ("dtype", Value::Str(meta.dtype.as_str().to_owned())),
        (
            "shape",
            Value::Array(meta.shape.iter().map(|dim| Value::U64(*dim)).collect()),
        ),
        ("element_count", Value::U64(meta.element_count)),
        ("layout", Value::Str(FLOAT_TILED_LAYOUT.to_owned())),
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
        (
            "exponent_coder",
            Value::Str(FLOAT_TENSOR_EXPONENT_CODER.to_owned()),
        ),
        ("n_streams", Value::U64(meta.n_streams)),
        ("tiles", Value::Array(tiles)),
        ("payload_len", Value::U64(meta.payload_len)),
        (
            "logical_root",
            Value::Bytes(meta.logical_root.as_bytes().to_vec()),
        ),
    ])
}

/// Encode a mode-18 `FLOAT_TILED_STREAMS` frame.
///
/// The tile grid, plane split, offsets and roots are exactly mode 17's
/// discipline. Within one tile the exponent plane is split round-robin into
/// `n_streams` independently coded `huffman-canonical-v1` streams, and the
/// raw mantissa plane is split round-robin into `n_streams` raw streams.
/// With `n_streams == 1` the stored tile bytes are byte-identical to mode 17.
pub fn encode_float_tiled_streams(
    payload: &[u8],
    dtype: FloatDtype,
    shape: &[u64],
    tile_shape: &[u64],
    n_streams: u64,
    limits: &Limits,
) -> Result<Vec<u8>> {
    if !(FLOAT_TILED_STREAMS_MIN..=FLOAT_TILED_STREAMS_MAX).contains(&n_streams) {
        return Err(Error::malformed(format!(
            "n_streams {n_streams} is outside {}..={}",
            FLOAT_TILED_STREAMS_MIN, FLOAT_TILED_STREAMS_MAX
        )));
    }
    check_len(
        payload.len() as u64,
        limits.max_decoded_bytes,
        "float tiled payload",
    )?;
    let tensor_dtype = float_tiled_tensor_dtype(dtype).ok_or_else(|| {
        Error::unsupported_feature(format!(
            "float-tiled v1 requires a tensor identity layout; dtype `{}` has none",
            dtype.as_str()
        ))
    })?;
    let element_count = float_tensor_product(shape, limits)?;
    let element_size = dtype.element_size();
    let decoded_bytes = element_count
        .checked_mul(element_size)
        .ok_or_else(|| Error::limit_exceeded("float tiled decoded bytes overflow u64"))?;
    if decoded_bytes != payload.len() as u64 {
        return Err(Error::malformed(format!(
            "float tiled payload is {} bytes but shape {:?} x {} requires {decoded_bytes}",
            payload.len(),
            shape,
            dtype.as_str()
        )));
    }
    let counts = tile_counts(shape, tile_shape)?;
    let tile_elements = tiled_product(tile_shape, limits, "float tile")?;
    let tile_element_len = tile_elements
        .checked_mul(element_size)
        .ok_or_else(|| Error::limit_exceeded("float tile byte length overflows u64"))?;
    let tile_count = counts.iter().try_fold(1u64, |acc, count| {
        acc.checked_mul(*count)
            .ok_or_else(|| Error::limit_exceeded("float tile count overflows u64"))
    })?;
    if tile_count > limits.max_structured_items {
        return Err(Error::limit_exceeded(format!(
            "float tile count {tile_count} exceeds limit {}",
            limits.max_structured_items
        )));
    }
    let n =
        usize::try_from(n_streams).map_err(|_| Error::malformed("n_streams does not fit usize"))?;
    let tile_len_usize = usize::try_from(tile_element_len)
        .map_err(|_| Error::limit_exceeded("float tile byte length does not fit usize"))?;
    let mut stored = Vec::with_capacity(payload.len());
    let mut tiles = Vec::with_capacity(usize::try_from(tile_count).unwrap_or(0));
    for index in 0..tile_count {
        let start = tile_start(&counts, index);
        let mut start_bytes = start.clone();
        for axis in 0..start_bytes.len() {
            start_bytes[axis] = start_bytes[axis]
                .checked_mul(tile_shape[axis])
                .ok_or_else(|| Error::limit_exceeded("float tile start overflows u64"))?;
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
        let (exponent_plane, mantissa_plane) =
            split_float_tensor_planes(&tile, dtype, tile_elements)?;

        let mut tile_stored = Vec::with_capacity(tile.len());
        let mut stream_offsets = Vec::with_capacity(2 * n);
        let exponent_streams = interleave_plane(&exponent_plane, n);
        for stream in &exponent_streams {
            stream_offsets.push(tile_stored.len() as u64);
            tile_stored.extend_from_slice(&huffman::encode(stream));
        }
        let exponent_plane_len = tile_stored.len() as u64;
        let mantissa_streams = interleave_plane(&mantissa_plane, n);
        for stream in &mantissa_streams {
            stream_offsets.push(tile_stored.len() as u64);
            tile_stored.extend_from_slice(stream);
        }
        let compressed_len = tile_stored.len() as u64;

        let root = float_tile_streams_root(
            dtype,
            tile_shape,
            &tile,
            exponent_plane_len,
            mantissa_plane.len() as u64,
            n_streams,
            &stream_offsets,
            limits,
        )?;
        tiles.push(FloatTiledStreamsEntry {
            index,
            dims: tile_shape.to_vec(),
            offset: stored.len() as u64,
            compressed_len,
            exponent_plane_len,
            mantissa_plane_len: mantissa_plane.len() as u64,
            element_len: tile_element_len,
            stream_offsets,
            tile_root: root,
        });
        stored.extend_from_slice(&tile_stored);
    }

    let logical_root = crate::tensor::logical_root(tensor_dtype, shape, payload, limits)?;
    let meta = FloatTiledStreamsMeta {
        dtype,
        shape: shape.to_vec(),
        element_count,
        tile_shape: tile_shape.to_vec(),
        tile_counts: counts,
        n_streams,
        tiles,
        payload_len: stored.len() as u64,
        logical_root,
    };
    let body = build_ai_body(&float_tiled_streams_meta_value(&meta), &stored, limits)?;
    let parts = FrameParts {
        format_major: FORMAT_MAJOR,
        format_minor: FORMAT_MINOR,
        domain: DOMAIN_PAYLOAD,
        mode: Mode::FloatTiledStreams,
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

fn meta_u64_array_strict(map: &BTreeMap<String, Value>, key: &str, what: &str) -> Result<Vec<u64>> {
    match map.get(key) {
        Some(Value::Array(items)) => {
            let mut out = Vec::with_capacity(items.len());
            for item in items {
                match item {
                    Value::U64(dim) => out.push(*dim),
                    _ => {
                        return Err(Error::malformed(format!(
                            "{what} `{key}` elements must be u64"
                        )))
                    }
                }
            }
            Ok(out)
        }
        _ => Err(Error::malformed(format!("{what} `{key}` must be an array"))),
    }
}

/// Parse a mode-18 descriptor from its canonical WCB bytes.
pub fn parse_float_tiled_streams_descriptor(
    descriptor_bytes: &[u8],
    decoded_len: u64,
    payload_len: u64,
    limits: &Limits,
) -> Result<FloatTiledStreamsMeta> {
    let value = wcb::decode(descriptor_bytes, limits)?;
    let map = ai_map(&value, "float tiled streams")?;
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
            "exponent_coder",
            "n_streams",
            "tiles",
            "payload_len",
            "logical_root",
        ],
        "float tiled streams",
    )?;
    let schema = meta_text(map, "schema")?;
    if schema != FLOAT_TILED_STREAMS_SCHEMA {
        return Err(Error::malformed(format!(
            "unsupported float tiled streams schema `{schema}`"
        )));
    }
    let dtype = FloatDtype::from_name(meta_text(map, "dtype")?)?;
    let layout = meta_text(map, "layout")?;
    if layout != FLOAT_TILED_LAYOUT {
        return Err(Error::malformed(format!(
            "float tiled streams layout `{layout}` is specified but not implemented in v1 (only {FLOAT_TILED_LAYOUT})"
        )));
    }
    let coder = meta_text(map, "exponent_coder")?;
    if coder != FLOAT_TENSOR_EXPONENT_CODER {
        return Err(Error::malformed(format!(
            "unsupported float tile exponent coder `{coder}` (v1 allows `{FLOAT_TENSOR_EXPONENT_CODER}`)"
        )));
    }
    if float_tiled_tensor_dtype(dtype).is_none() {
        return Err(Error::unsupported_feature(format!(
            "float-tiled v1 requires a tensor identity layout; dtype `{}` has none",
            dtype.as_str()
        )));
    }
    let n_streams = meta_u64(map, "n_streams")?;
    if !(FLOAT_TILED_STREAMS_MIN..=FLOAT_TILED_STREAMS_MAX).contains(&n_streams) {
        return Err(Error::malformed(format!(
            "n_streams {n_streams} is outside {}..={}",
            FLOAT_TILED_STREAMS_MIN, FLOAT_TILED_STREAMS_MAX
        )));
    }
    let n =
        usize::try_from(n_streams).map_err(|_| Error::malformed("n_streams does not fit usize"))?;
    let shape = meta_shape(map)?;
    let element_count = float_tensor_product(&shape, limits)?;
    let declared_count = meta_u64(map, "element_count")?;
    if declared_count != element_count {
        return Err(Error::malformed(format!(
            "float tiled element_count {declared_count} does not match shape product {element_count}"
        )));
    }
    let element_size = dtype.element_size();
    let decoded_bytes = element_count
        .checked_mul(element_size)
        .ok_or_else(|| Error::limit_exceeded("float tiled decoded bytes overflow u64"))?;
    check_len(
        decoded_bytes,
        limits.max_decoded_bytes,
        "float tiled decoded bytes",
    )?;
    if decoded_len != decoded_bytes {
        return Err(Error::corrupt(format!(
            "float tiled decoded_len is {decoded_len} but the descriptor requires {decoded_bytes}"
        )));
    }
    let tile_shape = meta_u64_array(map, "tile_shape")?;
    let counts = tile_counts(&shape, &tile_shape)?;
    if meta_u64_array(map, "tile_counts")? != counts {
        return Err(Error::malformed(format!(
            "float tiled tile_counts {:?} do not match shape/tile_shape {:?}",
            meta_u64_array(map, "tile_counts")?,
            counts
        )));
    }
    let tile_elements = tiled_product(&tile_shape, limits, "float tile")?;
    let tile_element_len = tile_elements
        .checked_mul(element_size)
        .ok_or_else(|| Error::limit_exceeded("float tile byte length overflows u64"))?;
    let tile_count = counts.iter().try_fold(1u64, |acc, count| {
        acc.checked_mul(*count)
            .ok_or_else(|| Error::limit_exceeded("float tile count overflows u64"))
    })?;
    if tile_count > limits.max_structured_items {
        return Err(Error::limit_exceeded(format!(
            "float tile count {tile_count} exceeds limit {}",
            limits.max_structured_items
        )));
    }
    let expected_mantissa = tile_elements
        .checked_mul(dtype.sign_bits() + dtype.mantissa_bits())
        .ok_or_else(|| Error::limit_exceeded("float tile mantissa bits overflow u64"))?;
    let expected_mantissa = expected_mantissa
        .checked_add(7)
        .ok_or_else(|| Error::limit_exceeded("float tile mantissa length overflows u64"))?
        / 8;
    let declared_payload_len = meta_u64(map, "payload_len")?;
    if declared_payload_len != payload_len {
        return Err(Error::malformed(format!(
            "float tiled payload is {payload_len} bytes but metadata declares {declared_payload_len}"
        )));
    }
    let entries = match map.get("tiles") {
        Some(Value::Array(items)) => items,
        _ => {
            return Err(Error::malformed(
                "float tiled streams `tiles` must be an array",
            ))
        }
    };
    if entries.len() as u64 != tile_count {
        return Err(Error::malformed(format!(
            "float tiled streams has {} tile entries but the grid requires {tile_count}",
            entries.len()
        )));
    }
    let mut tiles = Vec::with_capacity(entries.len());
    let mut expected_offset: u64 = 0;
    for (position, item) in entries.iter().enumerate() {
        let entry = ai_map(item, "float tile streams tile")?;
        check_exact_keys(
            entry,
            &[
                "index",
                "dims",
                "offset",
                "compressed_len",
                "exponent_plane_len",
                "mantissa_plane_len",
                "element_len",
                "stream_offsets",
                "tile_root",
            ],
            "float tile streams tile",
        )?;
        let index = meta_u64(entry, "index")?;
        if index != position as u64 {
            return Err(Error::malformed(format!(
                "float tiled streams entry {position} declares index {index}; entries must be in canonical order"
            )));
        }
        let dims = meta_u64_array(entry, "dims")?;
        if dims != tile_shape {
            return Err(Error::malformed(format!(
                "float tile {position} dims {:?} do not equal tile_shape {:?}; partial tiles are not defined in v1",
                dims, tile_shape
            )));
        }
        let declared_element_len = meta_u64(entry, "element_len")?;
        if declared_element_len != tile_element_len {
            return Err(Error::malformed(format!(
                "float tile {position} element_len {declared_element_len} does not match {tile_element_len}"
            )));
        }
        let offset = meta_u64(entry, "offset")?;
        if offset != expected_offset {
            return Err(Error::malformed(format!(
                "float tile {position} offset {offset} is not the contiguous offset {expected_offset}"
            )));
        }
        let compressed_len = meta_u64(entry, "compressed_len")?;
        let exponent_plane_len = meta_u64(entry, "exponent_plane_len")?;
        let mantissa_plane_len = meta_u64(entry, "mantissa_plane_len")?;
        if mantissa_plane_len != expected_mantissa {
            return Err(Error::malformed(format!(
                "float tile {position} mantissa_plane_len {mantissa_plane_len} does not match {expected_mantissa}"
            )));
        }
        if exponent_plane_len
            != compressed_len
                .checked_sub(mantissa_plane_len)
                .ok_or_else(|| {
                    Error::malformed("float tile length is shorter than its mantissa plane")
                })?
        {
            return Err(Error::malformed(format!(
                "float tile {position} exponent_plane_len {exponent_plane_len} does not equal compressed_len - mantissa_plane_len"
            )));
        }
        if compressed_len
            != exponent_plane_len
                .checked_add(mantissa_plane_len)
                .ok_or_else(|| Error::limit_exceeded("float tile length overflows u64"))?
        {
            return Err(Error::malformed(format!(
                "float tile {position} compressed_len {compressed_len} does not equal its plane lengths"
            )));
        }
        expected_offset = expected_offset
            .checked_add(compressed_len)
            .ok_or_else(|| Error::limit_exceeded("float tile offset overflows u64"))?;

        let stream_offsets = meta_u64_array_strict(entry, "stream_offsets", "float tile")?;
        if stream_offsets.len() != 2 * n {
            return Err(Error::malformed(format!(
                "float tile {position} has {} stream offsets but n_streams {n_streams} requires {}",
                stream_offsets.len(),
                2 * n_streams
            )));
        }
        if stream_offsets[0] != 0 {
            return Err(Error::malformed(format!(
                "float tile {position} stream_offsets[0] is {} not 0",
                stream_offsets[0]
            )));
        }
        for window in stream_offsets.windows(2) {
            if window[1] < window[0] {
                return Err(Error::malformed(format!(
                    "float tile {position} stream_offsets are not monotonic: {} then {}",
                    window[0], window[1]
                )));
            }
        }
        if stream_offsets[2 * n - 1] > compressed_len {
            return Err(Error::malformed(format!(
                "float tile {position} stream offset {} is past the {compressed_len}-byte tile payload",
                stream_offsets[2 * n - 1]
            )));
        }
        if stream_offsets[n] != exponent_plane_len {
            return Err(Error::malformed(format!(
                "float tile {position} stream_offsets[{n}] is {} but exponent_plane_len is {exponent_plane_len}",
                stream_offsets[n]
            )));
        }
        for stream in 0..n {
            let end = if stream + 1 < n {
                stream_offsets[stream + 1]
            } else {
                exponent_plane_len
            };
            let len = end - stream_offsets[stream];
            if len < huffman::TABLE_BYTES as u64 {
                return Err(Error::malformed(format!(
                    "float tile {position} exponent stream {stream} is {len} bytes, shorter than its {}-byte code-length table",
                    huffman::TABLE_BYTES
                )));
            }
        }
        let mantissa_total = mantissa_plane_len;
        for stream in 0..n {
            let start = stream_offsets[n + stream] - exponent_plane_len;
            let end = if stream + 1 < n {
                stream_offsets[n + stream + 1] - exponent_plane_len
            } else {
                mantissa_total
            };
            if end < start {
                return Err(Error::malformed(format!(
                    "float tile {position} mantissa stream {stream} offsets are inverted"
                )));
            }
            let expected = interleaved_stream_len(mantissa_total, stream as u64, n_streams);
            if end - start != expected {
                return Err(Error::malformed(format!(
                    "float tile {position} mantissa stream {stream} is {} bytes but the interleave requires {expected}",
                    end - start
                )));
            }
        }

        let tile_root = Hash32::from_slice(meta_bytes(entry, "tile_root")?)
            .map_err(|_| Error::malformed("float tile_root must be 32 bytes"))?;
        tiles.push(FloatTiledStreamsEntry {
            index,
            dims,
            offset,
            compressed_len,
            exponent_plane_len,
            mantissa_plane_len,
            element_len: declared_element_len,
            stream_offsets,
            tile_root,
        });
    }
    if expected_offset != declared_payload_len {
        return Err(Error::malformed(format!(
            "float tile ranges cover {expected_offset} bytes but payload_len is {declared_payload_len}"
        )));
    }
    let logical_root = Hash32::from_slice(meta_bytes(map, "logical_root")?)
        .map_err(|_| Error::malformed("float tiled logical_root must be 32 bytes"))?;
    Ok(FloatTiledStreamsMeta {
        dtype,
        shape,
        element_count,
        tile_shape,
        tile_counts: counts,
        n_streams,
        tiles,
        payload_len: declared_payload_len,
        logical_root,
    })
}

/// Parse a mode-18 frame and return its descriptor with the stored slice.
pub fn parse_float_tiled_streams<'a>(
    frame: &'a Frame<'_>,
    limits: &Limits,
) -> Result<FloatTiledStreamsDescriptor<'a>> {
    let (descriptor_bytes, payload) = split_ai_body_prefix(frame.body)?;
    let meta = parse_float_tiled_streams_descriptor(
        descriptor_bytes,
        frame.header.decoded_len,
        payload.len() as u64,
        limits,
    )?;
    Ok(FloatTiledStreamsDescriptor { meta, payload })
}

/// Parse a mode-18 frame and return only its descriptor.
pub fn float_tiled_streams_metadata(
    frame: &Frame<'_>,
    limits: &Limits,
) -> Result<FloatTiledStreamsMeta> {
    Ok(parse_float_tiled_streams(frame, limits)?.meta)
}

/// Validate and decode a mode-18 frame into exact row-major element bytes.
///
/// `threads > 1` decodes tiles on a bounded thread pool; each tile's exponent
/// streams use the wide-word multi-stream kernel. A single-tile region read can
/// instead use [`read_float_tiled_streams_region`], which runs the tile's
/// streams in parallel.
pub fn decode_float_tiled_streams(
    frame: &Frame<'_>,
    limits: &Limits,
    threads: usize,
) -> Result<Vec<u8>> {
    let descriptor = parse_float_tiled_streams(frame, limits)?;
    let meta = &descriptor.meta;
    let total = usize::try_from(
        meta.element_count
            .checked_mul(meta.dtype.element_size())
            .ok_or_else(|| Error::limit_exceeded("float tiled decoded bytes overflow u64"))?,
    )
    .map_err(|_| Error::limit_exceeded("float tiled decoded bytes do not fit usize"))?;
    let mut out = vec![0u8; total];
    let compat = float_tiled_compat_meta(meta);
    let decode_tile = |entry: &FloatTiledStreamsEntry| -> Result<Vec<u8>> {
        let start = usize::try_from(entry.offset)
            .map_err(|_| Error::limit_exceeded("float tile offset does not fit usize"))?;
        let end = start
            .checked_add(
                usize::try_from(entry.compressed_len)
                    .map_err(|_| Error::limit_exceeded("float tile length does not fit usize"))?,
            )
            .ok_or_else(|| Error::limit_exceeded("float tile range overflows usize"))?;
        let stored = descriptor
            .payload
            .get(start..end)
            .ok_or_else(|| Error::corrupt("float tile range is outside the payload"))?;
        decode_float_tile_streams(meta.dtype, entry, meta.n_streams, stored, limits, 1)
    };
    let mut tiles: Vec<Vec<u8>> = Vec::with_capacity(meta.tiles.len());
    if threads > 1 && meta.tiles.len() > 1 {
        let workers = threads.min(meta.tiles.len()).min(64).max(1);
        let chunk = meta.tiles.len().div_ceil(workers);
        let mut chunk_results: Vec<Result<Vec<Vec<u8>>>> = Vec::new();
        std::thread::scope(|scope| {
            let mut handles = Vec::new();
            for start in (0..meta.tiles.len()).step_by(chunk) {
                let end = (start + chunk).min(meta.tiles.len());
                let entries = &meta.tiles[start..end];
                let decode_tile = &decode_tile;
                handles.push(scope.spawn(move || {
                    let mut decoded = Vec::with_capacity(entries.len());
                    for entry in entries {
                        decoded.push(decode_tile(entry)?);
                    }
                    Ok(decoded)
                }));
            }
            for handle in handles {
                chunk_results.push(
                    handle.join().unwrap_or_else(|_| {
                        Err(Error::corrupt("multi-stream tile worker panicked"))
                    }),
                );
            }
        });
        for result in chunk_results {
            tiles.extend(result?);
        }
    } else {
        for entry in &meta.tiles {
            tiles.push(decode_tile(entry)?);
        }
    }
    let mut tiles_bytes = 0usize;
    for (position, tile) in tiles.iter().enumerate() {
        tiles_bytes = tiles_bytes
            .checked_add(tile.len())
            .ok_or_else(|| Error::limit_exceeded("float tiled decoded length overflows usize"))?;
        place_float_tile(&mut out, &compat, &compat.tiles[position], tile)?;
    }
    if tiles_bytes != total {
        return Err(Error::corrupt(format!(
            "float tiled decoded to {tiles_bytes} bytes but the descriptor requires {total}"
        )));
    }
    let tensor_dtype = float_tiled_tensor_dtype(meta.dtype)
        .ok_or_else(|| Error::unsupported_feature("float tiled dtype has no tensor identity"))?;
    let logical = crate::tensor::logical_root(tensor_dtype, &meta.shape, &out, limits)?;
    if logical != meta.logical_root {
        return Err(Error::corrupt(format!(
            "float tiled logical root {logical} does not match the descriptor's {}",
            meta.logical_root
        )));
    }
    Ok(out)
}

/// A mode-17-shaped view for the shared placement helpers.
fn float_tiled_compat_meta(meta: &FloatTiledStreamsMeta) -> FloatTiledMeta {
    FloatTiledMeta {
        dtype: meta.dtype,
        shape: meta.shape.clone(),
        element_count: meta.element_count,
        tile_shape: meta.tile_shape.clone(),
        tile_counts: meta.tile_counts.clone(),
        tiles: meta
            .tiles
            .iter()
            .map(|entry| FloatTiledEntry {
                index: entry.index,
                dims: entry.dims.clone(),
                offset: entry.offset,
                compressed_len: entry.compressed_len,
                exponent_plane_len: entry.exponent_plane_len,
                mantissa_plane_len: entry.mantissa_plane_len,
                element_len: entry.element_len,
                tile_root: entry.tile_root,
            })
            .collect(),
        payload_len: meta.payload_len,
        logical_root: meta.logical_root,
    }
}

/// Indices of the tiles whose cells intersect `region`.
pub fn float_tiled_streams_region_indices(
    meta: &FloatTiledStreamsMeta,
    region: &crate::tensor::Region,
) -> Result<Vec<usize>> {
    let compat = float_tiled_compat_meta(meta);
    float_tiled_region_indices(&compat, region)
}

/// Read one logical region, decoding only the multi-stream tiles it
/// intersects. Returns `(bytes, tiles_touched, streams_touched)`.
///
/// Every touched tile fetches all `2 * n_streams` of its streams, so
/// `streams_touched == tiles_touched * 2 * n_streams`.
pub fn read_float_tiled_streams_region(
    descriptor: &FloatTiledStreamsDescriptor<'_>,
    region: &crate::tensor::Region,
    limits: &Limits,
    threads: usize,
) -> Result<(Vec<u8>, u64, u64)> {
    let meta = &descriptor.meta;
    region.validate(&meta.shape)?;
    let extent = region.extent();
    if region.is_empty() {
        return Ok((Vec::new(), 0, 0));
    }
    let count = extent.iter().try_fold(1u64, |acc, dim| {
        acc.checked_mul(*dim)
            .ok_or_else(|| Error::limit_exceeded("region element count overflows u64"))
    })?;
    let byte_len = count
        .checked_mul(meta.dtype.element_size())
        .ok_or_else(|| Error::limit_exceeded("region payload length overflows u64"))?;
    check_len(byte_len, limits.max_decoded_bytes, "region payload")?;
    let byte_len = usize::try_from(byte_len)
        .map_err(|_| Error::limit_exceeded("region payload does not fit usize"))?;
    let mut out = vec![0u8; byte_len];
    let indices = float_tiled_streams_region_indices(meta, region)?;
    let compat = float_tiled_compat_meta(meta);
    let mut tiles_touched = 0u64;
    for index in indices {
        let entry = &meta.tiles[index];
        let start = usize::try_from(entry.offset)
            .map_err(|_| Error::limit_exceeded("float tile offset does not fit usize"))?;
        let end = start
            .checked_add(
                usize::try_from(entry.compressed_len)
                    .map_err(|_| Error::limit_exceeded("float tile length does not fit usize"))?,
            )
            .ok_or_else(|| Error::limit_exceeded("float tile range overflows usize"))?;
        let stored = descriptor
            .payload
            .get(start..end)
            .ok_or_else(|| Error::corrupt("float tile range is outside the payload"))?;
        let tile =
            decode_float_tile_streams(meta.dtype, entry, meta.n_streams, stored, limits, threads)?;
        place_float_tile_region(&mut out, region, &compat, &compat.tiles[index], &tile)?;
        tiles_touched += 1;
    }
    let streams_touched = tiles_touched
        .checked_mul(2)
        .and_then(|value| value.checked_mul(meta.n_streams))
        .ok_or_else(|| Error::limit_exceeded("streams_touched overflows u64"))?;
    Ok((out, tiles_touched, streams_touched))
}

/// Place one decoded mode-18 tile at its canonical grid position.
pub fn place_float_tiled_streams_tile(
    dst: &mut [u8],
    meta: &FloatTiledStreamsMeta,
    entry: &FloatTiledStreamsEntry,
    tile: &[u8],
) -> Result<()> {
    let compat = float_tiled_compat_meta(meta);
    place_float_tile(dst, &compat, &compat.tiles[entry.index as usize], tile)
}

/// Place the intersection of `region` and a decoded mode-18 tile into `out`.
pub fn place_float_tiled_streams_tile_region(
    out: &mut [u8],
    region: &crate::tensor::Region,
    meta: &FloatTiledStreamsMeta,
    entry: &FloatTiledStreamsEntry,
    tile: &[u8],
) -> Result<()> {
    let compat = float_tiled_compat_meta(meta);
    place_float_tile_region(
        out,
        region,
        &compat,
        &compat.tiles[entry.index as usize],
        tile,
    )
}
