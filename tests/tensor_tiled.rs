//! Mode 15 `TENSOR_TILED` exactness, region addressing, and reject vectors
//! (`WEFT-AI-CODEC` §16).

use std::fs;
use std::path::PathBuf;

use serde_json::Value as Json;
use weftcodec::dependency::MapResolver;
use weftcodec::error::ErrorClass;
use weftcodec::frame::Frame;
use weftcodec::limits::Limits;
use weftcodec::modes::{self, TensorDtype};
use weftcodec::tensor::Region;

fn pattern(len: usize, mut state: u64) -> Vec<u8> {
    let mut out = Vec::with_capacity(len);
    for _ in 0..len {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        out.push((state & 0xff) as u8);
    }
    out
}

fn roundtrip(dtype: TensorDtype, shape: &[u64], tile_shape: &[u64], seed: u64) {
    let limits = Limits::default();
    let elements: u64 = shape.iter().product();
    let bytes = pattern((elements * dtype.element_size()) as usize, seed);
    let frame =
        modes::encode_tiled_tensor(&bytes, dtype, shape, tile_shape, &limits).expect("encode");
    let parsed = Frame::parse(&frame, &limits).expect("parse");
    assert_eq!(parsed.header.mode, weftcodec::Mode::TensorTiled);
    let descriptor = modes::parse_tiled_tensor(&parsed, &limits).expect("descriptor");

    let expected_tiles: u64 = descriptor.meta.tile_counts.iter().product();
    assert_eq!(descriptor.meta.tiles.len() as u64, expected_tiles);

    let decoded = modes::decode_tiled_tensor(&parsed, &limits).expect("decode");
    assert_eq!(decoded, bytes, "mode 15 must be exact");

    let full = modes::read_region(
        &descriptor,
        &Region::new(vec![0; shape.len()], shape.to_vec()),
        &limits,
    )
    .expect("region");
    assert_eq!(full.0, bytes);
    assert_eq!(full.1, expected_tiles);
}

#[test]
fn bf16_square_tiles_roundtrip() {
    roundtrip(TensorDtype::Bf16, &[4, 4], &[2, 2], 0x51);
}

#[test]
fn fp32_rank_three_tiles_roundtrip() {
    roundtrip(TensorDtype::F32, &[3, 4, 2], &[1, 2, 2], 0x77);
}

#[test]
fn single_tile_roundtrip() {
    roundtrip(TensorDtype::I8, &[3, 5], &[3, 5], 0x33);
}

#[test]
fn region_reads_decode_only_intersecting_tiles() {
    let limits = Limits::default();
    let shape = [8u64, 8];
    let bytes = pattern(64 * 2, 0x91);
    let frame = modes::encode_tiled_tensor(&bytes, TensorDtype::Bf16, &shape, &[2, 2], &limits)
        .expect("encode");
    let parsed = Frame::parse(&frame, &limits).expect("parse");
    let descriptor = modes::parse_tiled_tensor(&parsed, &limits).expect("descriptor");

    // One row through the tensor intersects four tiles (2 rows x 4 columns).
    let region = Region::new(vec![3, 0], vec![4, 8]);
    let (got, tiles) = modes::read_region(&descriptor, &region, &limits).expect("region");
    let mut want = Vec::new();
    for col in 0..8 {
        let at = (3 * 8 + col) * 2;
        want.extend_from_slice(&bytes[at..at + 2]);
    }
    assert_eq!(got, want);
    assert_eq!(tiles, 4);

    // A single element touches exactly one tile.
    let region = Region::new(vec![5, 5], vec![6, 6]);
    let (got, tiles) = modes::read_region(&descriptor, &region, &limits).expect("region");
    assert_eq!(got, bytes[(5 * 8 + 5) * 2..(5 * 8 + 6) * 2]);
    assert_eq!(tiles, 1);

    // A 3x3 window straddling four tiles touches four.
    let region = Region::new(vec![3, 3], vec![6, 6]);
    let (got, tiles) = modes::read_region(&descriptor, &region, &limits).expect("region");
    let mut want = Vec::new();
    for row in 3..6 {
        for col in 3..6 {
            let at = (row * 8 + col) * 2;
            want.extend_from_slice(&bytes[at..at + 2]);
        }
    }
    assert_eq!(got, want);
    assert_eq!(tiles, 4);
}

#[test]
fn non_dividing_tile_shape_is_rejected() {
    let limits = Limits::default();
    let bytes = pattern(16 * 2, 0x11);
    let err = modes::encode_tiled_tensor(&bytes, TensorDtype::Bf16, &[4, 4], &[3, 2], &limits)
        .expect_err("must reject");
    assert_eq!(err.class(), ErrorClass::Malformed);
}

#[test]
fn rank_mismatched_tile_shape_is_rejected() {
    let limits = Limits::default();
    let bytes = pattern(16 * 2, 0x11);
    let err = modes::encode_tiled_tensor(&bytes, TensorDtype::Bf16, &[4, 4], &[2], &limits)
        .expect_err("must reject");
    assert_eq!(err.class(), ErrorClass::Malformed);
}

fn v1_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("conformance/v1")
}

fn hex(s: &str) -> Vec<u8> {
    hex::decode(s).expect("hex")
}

#[test]
fn conformance_vectors_replay() {
    let dir = v1_dir().join("tensor-tiled");
    let limits = Limits::default();
    let resolver = MapResolver::new();
    let mut ok = 0usize;
    let mut rejects = 0usize;
    let mut entries: Vec<PathBuf> = fs::read_dir(&dir)
        .expect("read tensor-tiled")
        .map(|entry| entry.expect("entry").path())
        .filter(|path| path.extension().map(|ext| ext == "json").unwrap_or(false))
        .collect();
    entries.sort();
    assert!(entries.len() >= 11, "expected 3 valid + 8 reject vectors");
    for path in entries {
        let text = fs::read_to_string(&path).expect("read");
        let vector: Json = serde_json::from_str(&text).expect("json");
        let frame = hex(vector["frame_hex"].as_str().expect("frame_hex"));
        let expected_ok = vector["expect"]["ok"].as_bool().unwrap_or(false);
        if expected_ok {
            let plaintext = hex(vector["plaintext_hex"].as_str().expect("plaintext"));
            let parsed = Frame::parse(&frame, &limits).expect("parse");
            let decoded = modes::decode_bytes(&frame, &resolver, &limits).expect("decode");
            assert_eq!(decoded, plaintext, "{}", path.display());
            let descriptor = modes::parse_tiled_tensor(&parsed, &limits).expect("descriptor");
            assert_eq!(
                descriptor.meta.logical_root.to_hex(),
                vector["logical_root"].as_str().expect("logical_root"),
                "{}",
                path.display()
            );
            if let Some(region) = vector.get("region") {
                let start = region["start"]
                    .as_array()
                    .expect("start")
                    .iter()
                    .map(|value| value.as_u64().expect("u64"))
                    .collect::<Vec<_>>();
                let end = region["end"]
                    .as_array()
                    .expect("end")
                    .iter()
                    .map(|value| value.as_u64().expect("u64"))
                    .collect::<Vec<_>>();
                let (bytes, tiles) =
                    modes::read_region(&descriptor, &Region::new(start, end), &limits)
                        .expect("region");
                assert_eq!(
                    hex::encode(&bytes),
                    region["bytes_hex"].as_str().expect("bytes_hex"),
                    "{}",
                    path.display()
                );
                assert_eq!(
                    tiles,
                    region["tiles_touched"].as_u64().expect("tiles_touched"),
                    "{}",
                    path.display()
                );
            }
            ok += 1;
        } else {
            let expected = vector["expect"]["error"].as_str().expect("error");
            let err = modes::decode_bytes(&frame, &resolver, &limits)
                .expect_err("vector must be rejected");
            assert_eq!(err.class().as_str(), expected, "{}", path.display());
            rejects += 1;
        }
    }
    assert_eq!(ok, 3);
    assert_eq!(rejects, 8);
}
