//! Chunked encoding and random-range reads.
//!
//! Chunking is only meaningful for RAW and ZSTD in v1. `read_range` decodes
//! only the chunks intersecting the requested range and verifies the hash of
//! each such chunk. Unchunked frames are decoded fully and then sliced (their
//! whole-content hash is verified in the process).

use crate::dependency::Resolver;
use crate::error::{Error, Result};
use crate::frame::{Frame, Mode};
use crate::limits::Limits;
use crate::modes;

/// Encode with independent chunks.
///
/// `mode` must be RAW or ZSTD. Empty input cannot be chunked (a CHUNKED frame
/// requires a non-empty chunk table), so it is returned as an unchunked frame.
pub fn encode_chunked(
    data: &[u8],
    mode: Mode,
    chunk_size: usize,
    level: i32,
    limits: &Limits,
) -> Result<Vec<u8>> {
    if chunk_size == 0 {
        return Err(Error::malformed("chunk size must be greater than zero"));
    }
    match mode {
        Mode::Raw => modes::encode_raw(data, Some(chunk_size), limits),
        Mode::Zstd => modes::encode_zstd(data, level, Some(chunk_size), limits),
        _ => Err(Error::unsupported_mode(format!(
            "mode {} cannot be chunked in v1",
            mode.as_str()
        ))),
    }
}

/// Decode a chunked (or unchunked) frame.
pub fn decode_chunked(
    frame_bytes: &[u8],
    resolver: &dyn Resolver,
    limits: &Limits,
) -> Result<Vec<u8>> {
    modes::decode_bytes(frame_bytes, resolver, limits)
}

/// Read `len` bytes starting at plaintext `offset`.
///
/// The range is clamped to the end of the plaintext, so an offset at or beyond
/// EOF yields an empty vector. For chunked frames only intersecting chunks are
/// decoded and hash-verified; the whole-content hash cannot be verified without
/// decoding every chunk and therefore is not checked here.
pub fn read_range(
    frame_bytes: &[u8],
    resolver: &dyn Resolver,
    limits: &Limits,
    offset: u64,
    len: u64,
) -> Result<Vec<u8>> {
    let frame = Frame::parse(frame_bytes, limits)?;
    let decoded_len = frame.header.decoded_len;
    if offset >= decoded_len {
        return Ok(Vec::new());
    }
    let available = decoded_len - offset;
    let wanted = available.min(len);
    let end = offset
        .checked_add(wanted)
        .ok_or_else(|| Error::malformed("range end overflow"))?;
    let start_usize = usize::try_from(offset)
        .map_err(|_| Error::limit_exceeded("range offset does not fit usize"))?;
    let end_usize =
        usize::try_from(end).map_err(|_| Error::limit_exceeded("range end does not fit usize"))?;
    if wanted == 0 {
        return Ok(Vec::new());
    }

    if !frame.header.is_chunked() {
        let plaintext = modes::decode_bytes(frame_bytes, resolver, limits)?;
        return Ok(plaintext[start_usize..end_usize].to_vec());
    }

    let mut out = Vec::with_capacity(usize::try_from(wanted).unwrap_or(0));
    let mut chunk_start: u64 = 0;
    for index in 0..frame.chunks.len() {
        let chunk_len = u64::from(frame.chunks[index].decoded_len);
        let chunk_end = chunk_start
            .checked_add(chunk_len)
            .ok_or_else(|| Error::malformed("chunk decoded extent overflow"))?;
        if chunk_end <= offset {
            chunk_start = chunk_end;
            continue;
        }
        if chunk_start >= end {
            break;
        }
        let decoded = modes::decode_chunk(&frame, index)?;
        let from = offset.max(chunk_start) - chunk_start;
        let to = end.min(chunk_end) - chunk_start;
        let from = usize::try_from(from)
            .map_err(|_| Error::limit_exceeded("range slice does not fit usize"))?;
        let to = usize::try_from(to)
            .map_err(|_| Error::limit_exceeded("range slice does not fit usize"))?;
        if to > decoded.len() || from > to {
            return Err(Error::corrupt("chunk decoded extent is inconsistent"));
        }
        out.extend_from_slice(&decoded[from..to]);
        chunk_start = chunk_end;
    }
    if out.len() as u64 != wanted {
        return Err(Error::corrupt(
            "random-range read produced an unexpected length",
        ));
    }
    Ok(out)
}
