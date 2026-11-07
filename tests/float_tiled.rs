//! Mode 17 `FLOAT_TILED`: per-tile mode-9 plane split, independent decode.
//!
//! These tests exercise the shipped API. The frozen byte vectors and every
//! reject case live in `conformance/v1/float-tiled` and are replayed by
//! `tests/conformance.rs`.

use weftcodec::limits::Limits;
use weftcodec::modes::{
    decode_float_tiled, encode_float_tiled, float_tile_root, float_tiled_metadata,
    parse_float_tiled, read_float_region, FloatDtype,
};
use weftcodec::tensor::{logical_root, Region};
use weftcodec::Frame;

fn payload(len: usize, seed: u64) -> Vec<u8> {
    let mut state = seed | 1;
    let mut out = Vec::with_capacity(len);
    for _ in 0..len {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        out.push((state & 0xff) as u8);
    }
    out
}

#[test]
fn bf16_tiles_round_trip_and_verify_every_root() {
    let limits = Limits::default();
    let bytes = payload(8 * 8 * 2, 0x51);
    let frame =
        encode_float_tiled(&bytes, FloatDtype::Bf16, &[8, 8], &[2, 4], &limits).expect("encode");
    let parsed = Frame::parse(&frame, &limits).expect("parse");
    let meta = float_tiled_metadata(&parsed, &limits).expect("metadata");
    assert_eq!(meta.tile_counts, vec![4, 2]);
    assert_eq!(meta.tiles.len(), 8);
    assert_eq!(meta.element_count, 64);
    let descriptor = parse_float_tiled(&parsed, &limits).expect("descriptor");
    assert_eq!(descriptor.meta.payload_len, descriptor.payload.len() as u64);
    let decoded = decode_float_tiled(&parsed, &limits).expect("decode");
    assert_eq!(decoded, bytes);
    assert_eq!(
        logical_root(weftcodec::TensorDtype::Bf16, &[8, 8], &bytes, &limits).expect("root"),
        meta.logical_root
    );

    // Every tile root matches the bytes it decodes to.
    for entry in &descriptor.meta.tiles {
        let start = entry.offset as usize;
        let end = start + entry.compressed_len as usize;
        let tile = weftcodec::decode_float_tile(
            FloatDtype::Bf16,
            entry,
            &descriptor.payload[start..end],
            &limits,
        )
        .expect("tile decode");
        let root = float_tile_root(
            FloatDtype::Bf16,
            &entry.dims,
            &tile,
            entry.exponent_plane_len,
            entry.mantissa_plane_len,
            &limits,
        )
        .expect("tile root");
        assert_eq!(root, entry.tile_root);
    }
}

#[test]
fn region_reads_touch_only_intersecting_tiles() {
    let limits = Limits::default();
    let shape = [8u64, 8];
    let bytes = payload(8 * 8 * 2, 0x77);
    let frame =
        encode_float_tiled(&bytes, FloatDtype::Bf16, &shape, &[2, 4], &limits).expect("encode");
    let parsed = Frame::parse(&frame, &limits).expect("parse");
    let descriptor = parse_float_tiled(&parsed, &limits).expect("descriptor");

    let whole = Region::new(vec![0, 0], vec![8, 8]);
    let (all, touched) = read_float_region(&descriptor, &whole, &limits).expect("whole");
    assert_eq!(all, bytes);
    assert_eq!(touched, 8);

    // One row (8 elements, 16 bytes) spans the two tile columns.
    let row = Region::new(vec![1, 0], vec![2, 8]);
    let (region, touched) = read_float_region(&descriptor, &row, &limits).expect("row");
    assert_eq!(touched, 2);
    assert_eq!(region, &bytes[16..32]);

    // One element.
    let one = Region::new(vec![5, 7], vec![6, 8]);
    let (element, touched) = read_float_region(&descriptor, &one, &limits).expect("one");
    assert_eq!(touched, 1);
    assert_eq!(element, bytes[5 * 8 * 2 + 7 * 2..5 * 8 * 2 + 7 * 2 + 2]);
}

#[test]
fn encoding_is_deterministic_and_rank_one_works() {
    let limits = Limits::default();
    let bytes = payload(32, 0x33);
    let first = encode_float_tiled(&bytes, FloatDtype::F16, &[16], &[4], &limits).expect("a");
    let second = encode_float_tiled(&bytes, FloatDtype::F16, &[16], &[4], &limits).expect("b");
    assert_eq!(first, second);
    let parsed = Frame::parse(&first, &limits).expect("parse");
    assert_eq!(parsed.header.mode, weftcodec::Mode::FloatTiled);
    assert_eq!(decode_float_tiled(&parsed, &limits).expect("decode"), bytes);
}

#[test]
fn non_dividing_tile_shape_is_rejected() {
    let limits = Limits::default();
    let bytes = payload(8 * 8 * 2, 0x11);
    let error = encode_float_tiled(&bytes, FloatDtype::Bf16, &[8, 8], &[3, 4], &limits)
        .expect_err("3 does not divide 8");
    assert_eq!(error.class(), weftcodec::ErrorClass::Malformed);
    assert!(format!("{error}").contains("does not divide"));
}

#[test]
fn fp8_has_no_tensor_identity_layout() {
    let limits = Limits::default();
    // 4 elements x 1 byte.
    let bytes = payload(4, 0x22);
    let error = encode_float_tiled(&bytes, FloatDtype::Fp8E4m3, &[4], &[2], &limits)
        .expect_err("fp8 has no logical-tensor layout");
    assert_eq!(error.class(), weftcodec::ErrorClass::UnsupportedFeature);
}
