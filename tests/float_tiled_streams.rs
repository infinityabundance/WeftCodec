//! Mode 18 `FLOAT_TILED_STREAMS`: multi-stream interleaved float tiles.
//!
//! The frozen conformance vectors for this mode live in
//! `conformance/v1/float-tiled-streams`; the replay test at the end of this
//! file decodes every one of them through the shipped API.

use std::fs;
use std::path::{Path, PathBuf};

use serde_json::Value as Json;
use weftcodec::limits::Limits;
use weftcodec::modes::{
    decode_float_tiled_streams, encode_float_tiled, encode_float_tiled_streams,
    float_tile_streams_root, float_tiled_streams_metadata, parse_float_tiled,
    parse_float_tiled_streams, read_float_tiled_streams_region, FloatDtype,
};
use weftcodec::tensor::Region;
use weftcodec::{ErrorClass, Frame, Mode};

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

fn biased_payload(len: usize, seed: u64) -> Vec<u8> {
    // Weight-like exponents: a handful of common high bytes.
    let mut state = seed | 1;
    let mut out = Vec::with_capacity(len);
    for _ in 0..len {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let byte = (state & 0xff) as u8;
        out.push(if byte & 0x80 != 0 { 0x3e } else { byte & 0x1f });
    }
    out
}

/// The strongest correctness anchor: a one-stream mode-18 tile payload is
/// byte-identical to the frozen mode-17 tile payload.
#[test]
fn n_streams_one_payload_is_byte_identical_to_mode17() {
    let limits = Limits::default();
    for (dtype, shape, tile, bytes) in [
        (
            FloatDtype::Bf16,
            vec![8u64, 8],
            vec![2u64, 4],
            payload(8 * 8 * 2, 0x51),
        ),
        (
            FloatDtype::F32,
            vec![4u64, 2],
            vec![2u64, 2],
            payload(4 * 2 * 4, 0x77),
        ),
        (
            FloatDtype::F16,
            vec![16u64],
            vec![4u64],
            biased_payload(16 * 2, 0x33),
        ),
    ] {
        let m17 = encode_float_tiled(&bytes, dtype, &shape, &tile, &limits).expect("mode 17");
        let m18 =
            encode_float_tiled_streams(&bytes, dtype, &shape, &tile, 1, &limits).expect("mode 18");
        let f17 = Frame::parse(&m17, &limits).expect("parse 17");
        let f18 = Frame::parse(&m18, &limits).expect("parse 18");
        assert_eq!(f17.header.mode, Mode::FloatTiled);
        assert_eq!(f18.header.mode, Mode::FloatTiledStreams);
        let d17 = parse_float_tiled(&f17, &limits).expect("descriptor 17");
        let d18 = parse_float_tiled_streams(&f18, &limits).expect("descriptor 18");
        assert_eq!(
            d17.payload, d18.payload,
            "n_streams=1 stored payload must equal mode 17"
        );
        assert_eq!(d17.meta.payload_len, d18.meta.payload_len);
        assert_eq!(d17.meta.tiles.len(), d18.meta.tiles.len());
        for (a, b) in d17.meta.tiles.iter().zip(d18.meta.tiles.iter()) {
            assert_eq!(a.offset, b.offset);
            assert_eq!(a.compressed_len, b.compressed_len);
            assert_eq!(a.exponent_plane_len, b.exponent_plane_len);
            assert_eq!(a.mantissa_plane_len, b.mantissa_plane_len);
            assert_eq!(b.stream_offsets, vec![0, a.exponent_plane_len]);
        }
    }
}

#[test]
fn every_stream_count_round_trips_exactly() {
    let limits = Limits::default();
    let bytes = biased_payload(16 * 16 * 2, 0x2b);
    for n_streams in [1u64, 2, 3, 4, 8, 16] {
        let frame = encode_float_tiled_streams(
            &bytes,
            FloatDtype::Bf16,
            &[16, 16],
            &[8, 8],
            n_streams,
            &limits,
        )
        .expect("encode");
        let parsed = Frame::parse(&frame, &limits).expect("parse");
        let meta = float_tiled_streams_metadata(&parsed, &limits).expect("metadata");
        assert_eq!(meta.n_streams, n_streams);
        assert_eq!(meta.tiles.len(), 4);
        let single = decode_float_tiled_streams(&parsed, &limits, 1).expect("single-thread");
        assert_eq!(single, bytes, "n_streams={n_streams}");
        let threaded = decode_float_tiled_streams(&parsed, &limits, 4).expect("threads");
        assert_eq!(
            threaded, single,
            "n_streams={n_streams} thread pool differs"
        );
    }
}

#[test]
fn more_streams_than_plane_bytes_is_legal() {
    let limits = Limits::default();
    let bytes = payload(2 * 3 * 2, 0x19);
    let frame = encode_float_tiled_streams(&bytes, FloatDtype::F16, &[2, 3], &[2, 3], 16, &limits)
        .expect("encode");
    let parsed = Frame::parse(&frame, &limits).expect("parse");
    let meta = float_tiled_streams_metadata(&parsed, &limits).expect("metadata");
    assert_eq!(meta.n_streams, 16);
    for entry in &meta.tiles {
        assert_eq!(entry.stream_offsets.len(), 32);
        // Empty mantissa streams collapse to equal offsets; the table is
        // non-decreasing by construction.
        assert!(entry.stream_offsets.windows(2).all(|w| w[0] <= w[1]));
    }
    assert_eq!(
        decode_float_tiled_streams(&parsed, &limits, 4).expect("decode"),
        bytes
    );
}

#[test]
fn region_reads_report_tiles_and_streams_touched() {
    let limits = Limits::default();
    let shape = [16u64, 16];
    let bytes = biased_payload(16 * 16 * 2, 0x41);
    let frame = encode_float_tiled_streams(&bytes, FloatDtype::Bf16, &shape, &[8, 8], 4, &limits)
        .expect("encode");
    let parsed = Frame::parse(&frame, &limits).expect("parse");
    let descriptor = parse_float_tiled_streams(&parsed, &limits).expect("descriptor");

    let whole = Region::new(vec![0, 0], vec![16, 16]);
    let (all, tiles, streams) =
        read_float_tiled_streams_region(&descriptor, &whole, &limits, 1).expect("whole");
    assert_eq!(all, bytes);
    assert_eq!(tiles, 4);
    assert_eq!(streams, 4 * 2 * 4);

    // Two rows cross both tile rows and both tile columns: four tiles.
    let rows = Region::new(vec![7, 0], vec![9, 16]);
    let (part, tiles, streams) =
        read_float_tiled_streams_region(&descriptor, &rows, &limits, 4).expect("rows");
    assert_eq!(tiles, 4);
    assert_eq!(streams, 4 * 2 * 4);
    assert_eq!(part, &bytes[7 * 16 * 2..9 * 16 * 2]);

    // One element touches one tile and all of its streams.
    let one = Region::new(vec![15, 15], vec![16, 16]);
    let (element, tiles, streams) =
        read_float_tiled_streams_region(&descriptor, &one, &limits, 1).expect("one");
    assert_eq!(tiles, 1);
    assert_eq!(streams, 2 * 4);
    assert_eq!(
        element,
        &bytes[15 * 16 * 2 + 15 * 2..15 * 16 * 2 + 15 * 2 + 2]
    );
}

#[test]
fn tile_root_covers_stream_count_and_offsets() {
    let limits = Limits::default();
    let bytes = payload(16 * 2, 0x02);
    let dims = [4u64, 4];
    let offsets = vec![0u64, 300, 400, 700, 720];
    let n2 = float_tile_streams_root(
        FloatDtype::Bf16,
        &dims,
        &bytes,
        400,
        320,
        2,
        &offsets[..4],
        &limits,
    )
    .expect("root n2");
    let n2_same = float_tile_streams_root(
        FloatDtype::Bf16,
        &dims,
        &bytes,
        400,
        320,
        2,
        &offsets[..4],
        &limits,
    )
    .expect("root n2");
    assert_eq!(n2, n2_same);
    let mut other_offsets = offsets[..4].to_vec();
    other_offsets[1] += 1;
    let moved = float_tile_streams_root(
        FloatDtype::Bf16,
        &dims,
        &bytes,
        400,
        320,
        2,
        &other_offsets,
        &limits,
    )
    .expect("root moved");
    assert_ne!(n2, moved, "an offset change must change the root");
    // A stream-count change with otherwise identical inputs changes the root.
    let n1 = float_tile_streams_root(
        FloatDtype::Bf16,
        &dims,
        &bytes,
        400,
        320,
        1,
        &offsets[..2],
        &limits,
    )
    .expect("root n1");
    let n1_as_n2 = float_tile_streams_root(
        FloatDtype::Bf16,
        &dims,
        &bytes,
        400,
        320,
        2,
        &offsets[..2],
        &limits,
    )
    .expect("root n1 labelled n2");
    assert_ne!(n1, n1_as_n2, "n_streams must be covered by the tile root");
}

#[test]
fn invalid_stream_counts_are_rejected() {
    let limits = Limits::default();
    let bytes = payload(8 * 2, 0x05);
    for n_streams in [0u64, 17, u64::MAX] {
        let error = encode_float_tiled_streams(
            &bytes,
            FloatDtype::Bf16,
            &[4, 2],
            &[2, 2],
            n_streams,
            &limits,
        )
        .expect_err("out of range");
        assert_eq!(error.class(), ErrorClass::Malformed);
    }
}

// ---------------------------------------------------------------------------
// Committed conformance vector replay
// ---------------------------------------------------------------------------

fn vector_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("conformance/v1/float-tiled-streams")
}

fn load(path: &Path) -> Json {
    let text = fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

#[test]
fn replay_committed_vectors() {
    let limits = Limits::default();
    let dir = vector_dir();
    let mut files: Vec<PathBuf> = fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", dir.display()))
        .map(|entry| entry.expect("entry").path())
        .filter(|path| path.extension().map(|ext| ext == "json").unwrap_or(false))
        .collect();
    files.sort();
    assert!(files.len() >= 14, "expected >= 14 mode-18 vectors");
    let mut valid = 0;
    let mut rejects = 0;
    for path in files {
        let vector = load(&path);
        let frame = hex::decode(
            vector["frame_hex"]
                .as_str()
                .unwrap_or_else(|| panic!("{}: no frame_hex", path.display())),
        )
        .expect("hex");
        let name = vector["name"].as_str().unwrap_or("?");
        if let Some(class) = vector["expect"].get("error").and_then(Json::as_str) {
            let parsed = Frame::parse(&frame, &limits).expect("parse reject");
            let error = decode_float_tiled_streams(&parsed, &limits, 1)
                .err()
                .unwrap_or_else(|| panic!("{name}: reject vector decoded"));
            assert_eq!(error.class().as_str(), class, "{name}");
            rejects += 1;
        } else {
            let parsed = Frame::parse(&frame, &limits).expect("parse valid");
            let decoded = decode_float_tiled_streams(&parsed, &limits, 1).expect(name);
            let plaintext =
                hex::decode(vector["plaintext_hex"].as_str().expect("plaintext")).expect("hex");
            assert_eq!(decoded, plaintext, "{name}");
            let threaded = decode_float_tiled_streams(&parsed, &limits, 4).expect(name);
            assert_eq!(threaded, decoded, "{name} thread pool");
            valid += 1;
        }
    }
    assert!(valid >= 4, "expected >= 4 valid mode-18 vectors");
    assert!(rejects >= 8, "expected >= 8 mode-18 rejects");
}
