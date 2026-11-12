//! R19 byte-identity court for the mode-9 encoder rewrite.
//!
//! Three independent proofs, all of which must hold for the shipped encoder:
//!
//! 1. **Committed vectors** — every valid `float_tensor` vector under
//!    `conformance/v1/` is decoded to its plaintext and re-encoded; the frame
//!    bytes must equal the committed `frame_hex` exactly, both through the
//!    shipped encoder and through the pre-R19 reference encoder.
//! 2. **Specialised BF16 split** — the new one-pass splitter is compared with
//!    the generic word loop over every 16-bit pattern (all 65,536 words, so the
//!    comparison is total, not sampled) and over randomised inputs.
//! 3. **Whole-frame differential** — the shipped encoder is compared with the
//!    reference encoder over randomised payloads for every dtype, every shape
//!    class, flush-word boundary sizes, and every emission thread count,
//!    including a forced-parallel configuration on inputs far below the
//!    production threshold.
//!
//! A single mismatch fails the test; nothing here is a tolerance.

use std::fs;
use std::path::{Path, PathBuf};

use serde_json::Value as Json;
use weftcodec::dependency::MapResolver;
use weftcodec::limits::Limits;
use weftcodec::modes::{
    self, encode_float_tensor, encode_float_tensor_reference, encode_float_tensor_with,
    encode_float_tensor_with_histogram, float_tensor_exponent_histogram, split_float_tensor_planes,
    split_float_tensor_planes_reference, FloatDtype, FloatTensorEncodeOptions,
};
use weftcodec::Frame;

fn v1_codec_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("conformance/v1/codec")
}

fn json_files(dir: &Path) -> Vec<PathBuf> {
    let mut files: Vec<PathBuf> = fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", dir.display()))
        .map(|entry| entry.expect("dir entry").path())
        .filter(|path| path.extension().map(|ext| ext == "json").unwrap_or(false))
        .collect();
    files.sort();
    files
}

fn load(path: &Path) -> Json {
    let text = fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

fn hex_field(value: &Json, field: &str) -> Vec<u8> {
    let text = value[field]
        .as_str()
        .unwrap_or_else(|| panic!("missing `{field}`"));
    hex::decode(text).unwrap_or_else(|e| panic!("bad `{field}`: {e}"))
}

/// Deterministic xorshift64* stream so a failure is reproducible.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed | 1)
    }

    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
}

fn random_bytes(len: usize, seed: u64) -> Vec<u8> {
    let mut rng = Rng::new(seed);
    let mut out = Vec::with_capacity(len);
    while out.len() < len {
        out.extend_from_slice(&rng.next().to_le_bytes());
    }
    out.truncate(len);
    out
}

/// BF16 payloads with a realistic exponent distribution: mostly one exponent
/// (so the canonical code is short and skewed), plus tails of extremes.
fn realistic_bf16(len: usize, seed: u64) -> Vec<u8> {
    let mut rng = Rng::new(seed);
    let mut out = Vec::with_capacity(len);
    while out.len() < len {
        let pick = rng.next();
        let word: u16 = match pick % 8 {
            0 => 0x0000,                        // +0
            1 => 0x8000,                        // -0
            2 => 0x7f80,                        // +inf
            3 => 0xff80,                        // -inf
            4 => 0x7fc1 | (pick as u16 & 0x3f), // NaN with a payload
            5 => 0x0001 | (pick as u16 & 0x7f), // subnormal
            6 => 0x7f7f,                        // max finite
            _ => (((pick >> 16) & 0x807f) as u16) | ((((pick >> 32) % 60 + 120) as u16) << 7),
        };
        out.extend_from_slice(&word.to_le_bytes());
    }
    out.truncate(len);
    out
}

fn dtype_name(dtype: FloatDtype) -> &'static str {
    dtype.as_str()
}

#[test]
fn committed_mode9_vectors_reencode_byte_identically() {
    let limits = Limits::default();
    let mut checked = 0usize;
    for path in json_files(&v1_codec_dir()) {
        let vector = load(&path);
        if vector.get("superseded_by").is_some() {
            continue;
        }
        if vector["mode"].as_str() != Some("float_tensor") {
            continue;
        }
        if vector["expect"].get("error").is_some() {
            continue;
        }
        let name = vector["name"].as_str().unwrap_or("?");
        let frame = hex_field(&vector, "frame_hex");
        let plaintext = hex_field(&vector, "plaintext_hex");
        let parsed = Frame::parse(&frame, &limits).unwrap_or_else(|e| panic!("{name}: {e}"));
        let meta = modes::float_tensor_metadata(&parsed, &limits)
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        let decoded = modes::decode_bytes(&frame, &MapResolver::new(), &limits)
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(decoded, plaintext, "{name}: decoded plaintext");

        let shipped = encode_float_tensor(&plaintext, meta.dtype, &meta.shape, &limits)
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(
            shipped,
            frame,
            "{name}: shipped encoder does not reproduce the committed {} bytes",
            frame.len()
        );
        let reference = encode_float_tensor_reference(&plaintext, meta.dtype, &meta.shape, &limits)
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(
            reference, frame,
            "{name}: reference encoder does not reproduce the committed bytes"
        );
        let histogram = float_tensor_exponent_histogram(&plaintext, meta.dtype)
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        let handoff = encode_float_tensor_with_histogram(
            &plaintext,
            meta.dtype,
            &meta.shape,
            &limits,
            &histogram,
        )
        .unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(
            handoff, frame,
            "{name}: the histogram hand-off path does not reproduce the committed bytes"
        );
        for threads in [2usize, 3, 4, 8] {
            let options = FloatTensorEncodeOptions::with_threads(threads);
            let parallel =
                encode_float_tensor_with(&plaintext, meta.dtype, &meta.shape, &limits, options)
                    .unwrap_or_else(|e| panic!("{name}: {e}"));
            assert_eq!(
                parallel, frame,
                "{name}: {threads} emission threads do not reproduce the committed bytes"
            );
        }
        checked += 1;
    }
    assert!(
        checked >= 3,
        "expected at least three valid mode-9 vectors, found {checked}"
    );
}

#[test]
fn specialised_bf16_split_matches_the_generic_reference() {
    let limits = Limits::default();
    // Every 16-bit word exactly once, so the comparison is total over the
    // dtype's value space (NaN, infinities, subnormals, signed zeros and the
    // maximum finite value are all included by construction).
    let mut all_words = Vec::with_capacity(65536 * 2);
    for word in 0u32..65536 {
        all_words.extend_from_slice(&(word as u16).to_le_bytes());
    }
    let count = (all_words.len() / 2) as u64;
    let (fast_exponent, fast_mantissa) =
        split_float_tensor_planes(&all_words, FloatDtype::Bf16, count).expect("fast split");
    let (reference_exponent, reference_mantissa) =
        split_float_tensor_planes_reference(&all_words, FloatDtype::Bf16, count)
            .expect("reference split");
    assert_eq!(fast_exponent, reference_exponent, "exponent plane");
    assert_eq!(fast_mantissa, reference_mantissa, "sign+mantissa plane");
    assert_eq!(fast_exponent.len(), 65536);
    assert_eq!(fast_mantissa.len(), 65536);
    // Independent statement of the documented bit layout for a few landmarks.
    for (index, word) in [
        (0x0000u16, 0usize),
        (0x8000, 1),
        (0x0001, 2),
        (0x7f80, 3),
        (0xff80, 4),
        (0x7fc1, 5),
        (0x7f7f, 6),
        (0x80ff, 7),
    ] {
        let _ = index;
        let at = word as usize;
        assert_eq!(fast_exponent[at], (word >> 7) as u8);
        assert_eq!(
            fast_mantissa[at],
            ((word >> 8) as u8 & 0x80) | (word as u8 & 0x7f)
        );
    }

    // Random payloads, including lengths that are not a multiple of four
    // elements and not a multiple of the eight-byte window.
    for (len, seed) in [
        (2usize, 1u64),
        (4, 2),
        (6, 3),
        (8, 4),
        (10, 5),
        (14, 6),
        (4094, 7),
        (4096, 8),
        (4098, 9),
        (1_000_000, 10),
    ] {
        let payload = realistic_bf16(len, seed);
        let count = (payload.len() / 2) as u64;
        let (fast_e, fast_m) =
            split_float_tensor_planes(&payload, FloatDtype::Bf16, count).expect("fast split");
        let (ref_e, ref_m) = split_float_tensor_planes_reference(&payload, FloatDtype::Bf16, count)
            .expect("reference split");
        assert_eq!(fast_e, ref_e, "exponent plane at {len} bytes");
        assert_eq!(fast_m, ref_m, "mantissa plane at {len} bytes");
    }
    // The generic oracle is unchanged for the other dtypes, and the shipped
    // splitter delegates to it there.
    for dtype in [
        FloatDtype::F16,
        FloatDtype::F32,
        FloatDtype::Fp8E4m3,
        FloatDtype::Fp8E5m2,
    ] {
        let payload = random_bytes(dtype.element_size() as usize * 777, 0x9e37);
        let count = (payload.len() as u64) / dtype.element_size();
        let (fast_e, fast_m) =
            split_float_tensor_planes(&payload, dtype, count).expect("fast split");
        let (ref_e, ref_m) =
            split_float_tensor_planes_reference(&payload, dtype, count).expect("reference split");
        assert_eq!(fast_e, ref_e, "{} exponent plane", dtype_name(dtype));
        assert_eq!(fast_m, ref_m, "{} mantissa plane", dtype_name(dtype));
    }
    let _ = limits;
}

#[test]
fn fused_histogram_equals_the_standalone_count() {
    for (len, seed) in [(0usize, 11u64), (2, 12), (4096, 13), (1_000_002, 14)] {
        let payload = realistic_bf16(len, seed);
        let direct = float_tensor_exponent_histogram(&payload, FloatDtype::Bf16).expect("count");
        let mut plain = [0u64; 256];
        for chunk in payload.chunks_exact(2) {
            let word = u16::from_le_bytes([chunk[0], chunk[1]]);
            plain[((word >> 7) & 0xff) as usize] += 1;
        }
        assert_eq!(direct, plain, "histogram at {len} bytes");
        // The windowed accumulation the fused pass uses must give the same sum.
        let mut windowed = [0u64; 256];
        for window in payload.chunks(1 << 16) {
            modes::accumulate_float_tensor_histogram(&mut windowed, window, FloatDtype::Bf16)
                .expect("window");
        }
        assert_eq!(windowed, plain, "windowed histogram at {len} bytes");
    }
}

#[test]
fn shipped_encoder_matches_the_reference_across_dtypes_and_sizes() {
    let limits = Limits::default();
    let dtypes = [
        FloatDtype::Bf16,
        FloatDtype::F16,
        FloatDtype::F32,
        FloatDtype::Fp8E4m3,
        FloatDtype::Fp8E5m2,
    ];
    // Sizes around the 32-bit flush word, the 8-byte BF16 window, a plausible
    // partition boundary and a large plane.
    let counts = [
        1u64, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256,
        257, 511, 512, 513, 1023, 4095, 4096, 4097,
    ];
    let mut seed = 0x5eedu64;
    for dtype in dtypes {
        for count in counts {
            seed += 1;
            let len = count as usize * dtype.element_size() as usize;
            let payload = match dtype {
                FloatDtype::Bf16 => realistic_bf16(len, seed),
                _ => random_bytes(len, seed),
            };
            let shape = [count];
            let shipped = encode_float_tensor(&payload, dtype, &shape, &limits)
                .unwrap_or_else(|e| panic!("shipped {} x{count}: {e}", dtype_name(dtype)));
            let reference = encode_float_tensor_reference(&payload, dtype, &shape, &limits)
                .unwrap_or_else(|e| panic!("reference {} x{count}: {e}", dtype_name(dtype)));
            assert_eq!(
                shipped,
                reference,
                "{} x{count}: shipped frame differs from the reference",
                dtype_name(dtype)
            );
            let decoded = modes::decode_bytes(&shipped, &MapResolver::new(), &limits)
                .unwrap_or_else(|e| panic!("decode {} x{count}: {e}", dtype_name(dtype)));
            assert_eq!(
                decoded,
                payload,
                "{} x{count}: round trip",
                dtype_name(dtype)
            );

            let histogram = float_tensor_exponent_histogram(&payload, dtype).expect("histogram");
            let handoff =
                encode_float_tensor_with_histogram(&payload, dtype, &shape, &limits, &histogram)
                    .unwrap_or_else(|e| panic!("hand-off {} x{count}: {e}", dtype_name(dtype)));
            assert_eq!(
                handoff,
                shipped,
                "{} x{count}: histogram hand-off path differs",
                dtype_name(dtype)
            );

            for threads in [2usize, 3, 4, 8] {
                let sequential_equivalent = encode_float_tensor_with(
                    &payload,
                    dtype,
                    &shape,
                    &limits,
                    FloatTensorEncodeOptions::with_threads(threads),
                )
                .unwrap_or_else(|e| {
                    panic!("{threads} threads {} x{count}: {e}", dtype_name(dtype))
                });
                assert_eq!(
                    sequential_equivalent,
                    shipped,
                    "{threads} threads {} x{count}: frame differs",
                    dtype_name(dtype)
                );
                // Force the parallel path on even the smallest inputs so the
                // boundary stitching is exercised at every alignment.
                let forced = FloatTensorEncodeOptions {
                    threads,
                    parallel_min_bitstream_bytes: 0,
                    parallel_max_bitstream_bytes: usize::MAX,
                    min_partition_elements: 1,
                };
                let forced_frame =
                    encode_float_tensor_with(&payload, dtype, &shape, &limits, forced)
                        .unwrap_or_else(|e| panic!("forced {threads} threads: {e}"));
                assert_eq!(
                    forced_frame,
                    shipped,
                    "forced-parallel {threads} threads {} x{count}: frame differs",
                    dtype_name(dtype)
                );
            }
        }
    }
}

#[test]
fn parallel_emission_is_byte_identical_on_a_large_skewed_plane() {
    // A plane big enough to cross many flush words with a strongly skewed
    // exponent distribution, at every thread count from 1 to 8, including the
    // forced-parallel configuration with one element per partition.
    let limits = Limits::default();
    let count = 300_000u64;
    let payload = realistic_bf16(count as usize * 2, 0xc0ffee);
    let shape = [count];
    let reference = encode_float_tensor_reference(&payload, FloatDtype::Bf16, &shape, &limits)
        .expect("reference");
    for threads in 1..=8usize {
        let shipped = encode_float_tensor_with(
            &payload,
            FloatDtype::Bf16,
            &shape,
            &limits,
            FloatTensorEncodeOptions::with_threads(threads),
        )
        .expect("shipped");
        assert_eq!(shipped, reference, "{threads} threads");
        let forced = FloatTensorEncodeOptions {
            threads,
            parallel_min_bitstream_bytes: 0,
            parallel_max_bitstream_bytes: usize::MAX,
            min_partition_elements: if threads % 2 == 0 { 1 } else { 7 },
        };
        let forced_frame =
            encode_float_tensor_with(&payload, FloatDtype::Bf16, &shape, &limits, forced)
                .expect("forced");
        assert_eq!(forced_frame, reference, "forced {threads} threads");
    }
}

#[test]
fn multi_dimensional_shape_indices_do_not_change_the_bytes() {
    let limits = Limits::default();
    let payload = realistic_bf16(2 * (2 * 3 * 5 * 7 * 2), 0xabcdef);
    let flat = [2u64 * 3 * 5 * 7 * 2];
    let nested = [2u64, 3, 5, 7, 2];
    let a = encode_float_tensor(&payload, FloatDtype::Bf16, &flat, &limits).expect("flat");
    let b = encode_float_tensor(&payload, FloatDtype::Bf16, &nested, &limits).expect("nested");
    // The descriptor names the shape, so the frames differ only in that field;
    // both must round-trip and both must match their reference encoder.
    assert_eq!(
        b,
        encode_float_tensor_reference(&payload, FloatDtype::Bf16, &nested, &limits)
            .expect("reference")
    );
    assert_eq!(
        a,
        encode_float_tensor_reference(&payload, FloatDtype::Bf16, &flat, &limits)
            .expect("reference")
    );
    assert_ne!(a, b);
    for frame in [&a, &b] {
        assert_eq!(
            modes::decode_bytes(frame, &MapResolver::new(), &limits).expect("decode"),
            payload
        );
    }
}
