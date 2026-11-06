//! Extract the exact plane bytes shipped by modes 9, 11, 12, and 15.
//!
//! Every function here calls the shipped encoder and then reads the plane
//! streams back out of the produced frame. Nothing reimplements the split:
//! the exponent plane is decoded from the shipped `huffman-canonical-v1`
//! stream, and every other plane is obtained from the shipped payload layout.
//!
//! The results are *inputs to the experimental coders*, not a second
//! implementation of any mode.

use crate::error::{Error, Result};
use crate::frame::Frame;
use crate::huffman;
use crate::kv_cache::{encode_kv_page, kv_cache_metadata, KvCacheInput, KvDtype};
use crate::limits::Limits;
use crate::modes::{self, FloatDtype, TensorDtype};
use crate::weight_delta::{self, WeightDtype};

/// Decode one shipped Huffman plane and return its raw bytes.
fn decode_plane(stream: &[u8], expected: usize) -> Result<(Vec<u8>, usize)> {
    huffman::decode_prefix(stream, expected)
}

/// Mode-9 `FLOAT_TENSOR` planes for a whole tensor or a row region.
///
/// Returns `[exponent_plane, sign_and_mantissa_plane]`. The exponent plane is
/// exactly the byte sequence that the shipped encoder Huffman-coded; the
/// sign+mantissa plane is exactly the raw bit-packed plane the shipped encoder
/// stores.
pub fn float_planes(payload: &[u8], dtype: FloatDtype, element_count: u64) -> Result<Vec<Vec<u8>>> {
    let (exponent_stream, mantissa_plane) =
        modes::split_float_tensor_payload(payload, dtype, element_count)?;
    let count = usize::try_from(element_count)
        .map_err(|_| Error::limit_exceeded("float element count does not fit usize"))?;
    let exponent_plane = huffman::decode(&exponent_stream, count)?;
    if exponent_plane.len() != count {
        return Err(Error::corrupt(
            "mode-9 exponent plane does not have one byte per element",
        ));
    }
    Ok(vec![exponent_plane, mantissa_plane])
}

/// Mode-11 `WEIGHT_DELTA` residual byte planes for a target tensor and a base
/// mode-6 tensor frame.
///
/// The target and base slices must be exactly the element bytes for `shape`.
/// The shipped encoder is called; its payload is then split into the
/// per-plane Huffman streams and each stream is decoded with the shipped
/// `huffman-canonical-v1` decoder.
pub fn weight_delta_planes(
    target: &[u8],
    base_frame: &[u8],
    dtype: WeightDtype,
    shape: &[u64],
    limits: &Limits,
) -> Result<Vec<Vec<u8>>> {
    let frame = weight_delta::encode_weight_delta(target, dtype, shape, base_frame, limits)?;
    let parsed = Frame::parse(&frame, limits)?;
    let meta = weight_delta::weight_delta_metadata(&parsed, limits)?;
    let (_, payload) = weight_delta::split_ai_body(parsed.body)?;
    let count = usize::try_from(meta.element_count)
        .map_err(|_| Error::limit_exceeded("weight delta element count does not fit usize"))?;
    let plane_count = usize::try_from(meta.planes)
        .map_err(|_| Error::limit_exceeded("weight delta plane count does not fit usize"))?;
    let mut planes = Vec::with_capacity(plane_count);
    let mut rest = payload;
    for _ in 0..plane_count {
        let (plane, consumed) = decode_plane(rest, count)?;
        planes.push(plane);
        rest = &rest[consumed..];
    }
    if !rest.is_empty() {
        return Err(Error::corrupt(
            "WEIGHT_DELTA payload has bytes after the declared planes",
        ));
    }
    Ok(planes)
}

/// Mode-15 `TENSOR_TILED` planes for a whole tensor or a row region.
///
/// Every tile's planes are extracted from the shipped payload and concatenated
/// per plane slot, giving one logical plane set for the region. This is the
/// plane content an entropy coder would see if it coded the tiled payload.
pub fn tiled_planes(
    payload: &[u8],
    dtype: TensorDtype,
    shape: &[u64],
    tile_shape: &[u64],
    limits: &Limits,
) -> Result<Vec<Vec<u8>>> {
    let frame = modes::encode_tiled_tensor(payload, dtype, shape, tile_shape, limits)?;
    let parsed = Frame::parse(&frame, limits)?;
    let descriptor = modes::parse_tiled_tensor(&parsed, limits)?;
    let element_size = usize::try_from(dtype.element_size())
        .map_err(|_| Error::limit_exceeded("tile element size does not fit usize"))?;
    let tile_elements = descriptor
        .meta
        .tile_shape
        .iter()
        .try_fold(1u64, |acc, dim| {
            acc.checked_mul(*dim)
                .ok_or_else(|| Error::limit_exceeded("tile element count overflows u64"))
        })
        .and_then(|value| {
            usize::try_from(value)
                .map_err(|_| Error::limit_exceeded("tile element count does not fit usize"))
        })?;
    let mut planes: Vec<Vec<u8>> = (0..element_size)
        .map(|_| Vec::with_capacity(payload.len() / element_size))
        .collect();
    for tile in &descriptor.meta.tiles {
        let start = usize::try_from(tile.offset)
            .map_err(|_| Error::limit_exceeded("tile offset does not fit usize"))?;
        let length = usize::try_from(tile.compressed_len)
            .map_err(|_| Error::limit_exceeded("tile length does not fit usize"))?;
        let end = start
            .checked_add(length)
            .ok_or_else(|| Error::limit_exceeded("tile end offset overflows usize"))?;
        if end > descriptor.payload.len() {
            return Err(Error::corrupt("tile payload extends past the frame body"));
        }
        let mut rest = &descriptor.payload[start..end];
        for plane in planes.iter_mut() {
            let (bytes, consumed) = decode_plane(rest, tile_elements)?;
            plane.extend_from_slice(&bytes);
            rest = &rest[consumed..];
        }
        if !rest.is_empty() {
            return Err(Error::corrupt(
                "tile payload has bytes after the declared plane count",
            ));
        }
    }
    Ok(planes)
}

/// Mode-12 `KV_CACHE` planes for one page (or a row region encoded as a page).
///
/// `elements` is the exact canonical row-major little-endian page content.
pub fn kv_page_planes(
    elements: &[u8],
    rows: u64,
    head_count: u32,
    head_dim: u32,
    dtype: KvDtype,
    limits: &Limits,
) -> Result<Vec<Vec<u8>>> {
    let input = KvCacheInput {
        model: "entropy-court".to_owned(),
        layer: 0,
        session: "entropy-court".to_owned(),
        page_index: 0,
        page_count: 1,
        seq_start: 0,
        seq_end: rows,
        dtype,
        head_count,
        head_dim,
    };
    let frame = encode_kv_page(&input, elements, limits)?;
    let parsed = Frame::parse(&frame, limits)?;
    let meta = kv_cache_metadata(&parsed, limits)?;
    let body = parsed.body;
    if body.len() < 4 {
        return Err(Error::malformed("KV_CACHE body lacks its metadata length"));
    }
    let meta_len = u32::from_le_bytes([body[0], body[1], body[2], body[3]]) as usize;
    let payload_start = 4usize
        .checked_add(meta_len)
        .ok_or_else(|| Error::malformed("KV_CACHE metadata length overflows usize"))?;
    if payload_start > body.len() {
        return Err(Error::malformed(
            "KV_CACHE metadata extends past the frame body",
        ));
    }
    let payload = &body[payload_start..];
    let count = usize::try_from(meta.page_bytes / u64::from(dtype.word_bytes()))
        .map_err(|_| Error::limit_exceeded("KV page element count does not fit usize"))?;
    let plane_count = usize::try_from(meta.dtype.planes())
        .map_err(|_| Error::limit_exceeded("KV page plane count does not fit usize"))?;
    let mut planes = Vec::with_capacity(plane_count);
    let mut rest = payload;
    for _ in 0..plane_count {
        let (plane, consumed) = decode_plane(rest, count)?;
        planes.push(plane);
        rest = &rest[consumed..];
    }
    if !rest.is_empty() {
        return Err(Error::corrupt(
            "KV_CACHE payload has bytes after the declared planes",
        ));
    }
    Ok(planes)
}

/// Sum a plane set into one 256-symbol histogram.
pub fn histogram(planes: &[Vec<u8>]) -> [u64; 256] {
    let mut counts = [0u64; 256];
    for plane in planes {
        for &byte in plane {
            counts[byte as usize] += 1;
        }
    }
    counts
}
