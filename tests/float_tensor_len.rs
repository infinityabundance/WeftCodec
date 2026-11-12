//! **Exactness of the FLOAT_TENSOR size predictor** (R18).
//!
//! `modes::float_tensor_encoded_len` claims to return the exact number of bytes
//! `modes::encode_float_tensor` will produce, without coding the frame. A
//! planner is only allowed to skip an encode when such a prediction is exact,
//! so this file treats any mismatch as a failure of the predictor, not as
//! estimator noise.
//!
//! Cases:
//!
//! 1. Every byte-aligned float dtype the mode implements (bf16, fp16, fp32,
//!    fp8-e4m3, fp8-e5m2) over a size sweep that crosses the same chunking the
//!    rest of the codec uses.
//! 2. Histogram shapes that exercise the coder's own edge cases: a single
//!    symbol (empty bitstream), two symbols, a full 256-symbol alphabet, and
//!    long-tail frequencies.
//! 3. Values that are bit-pattern extremes (NaN, +-inf, subnormals, signed
//!    zero), because the mode stores bit patterns verbatim.
//! 4. `huffman::encoded_len` against `huffman::encode` directly, including the
//!    empty plane.
//! 5. Every committed mode-9 golden frame under `conformance/v1`: the predictor
//!    is re-run from the decoded payload and must reproduce the vector's own
//!    frame length.
//! 6. A mismatch counter that is asserted to be zero; the table is printed when
//!    `WEFT_FLOAT_TENSOR_LEN_TRACE` is set so a failure is diagnosable.

use std::path::{Path, PathBuf};

use weftcodec::dependency::MapResolver;
use weftcodec::frame::{Frame, Mode};
use weftcodec::huffman;
use weftcodec::limits::Limits;
use weftcodec::modes::{self, FloatDtype};

fn deterministic(len: usize, mut state: u64) -> Vec<u8> {
    let mut out = Vec::with_capacity(len);
    for _ in 0..len {
        // xorshift64*: deterministic, no dependency.
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        let value = state.wrapping_mul(0x2545_f491_4f6c_dd1d);
        out.push((value >> 33) as u8);
    }
    out
}

fn float_dtypes() -> [FloatDtype; 5] {
    [
        FloatDtype::Bf16,
        FloatDtype::F16,
        FloatDtype::F32,
        FloatDtype::Fp8E4m3,
        FloatDtype::Fp8E5m2,
    ]
}

/// The predictor must equal the encoder for every dtype at this element count.
fn check_case(dtype: FloatDtype, shape: &[u64], payload: &[u8], label: &str) -> Result<(), String> {
    let limits = Limits::default();
    let frame = modes::encode_float_tensor(payload, dtype, shape, &limits)
        .map_err(|err| format!("{label}: encode failed: {err}"))?;
    let predicted = modes::float_tensor_encoded_len(payload, dtype, shape, &limits)
        .map_err(|err| format!("{label}: predict failed: {err}"))?;
    if predicted != frame.len() as u64 {
        return Err(format!(
            "{label}: predicted {predicted} bytes, encoder produced {} bytes",
            frame.len()
        ));
    }
    // The histogram-based entry point must agree with the payload-based one.
    let histogram = modes::float_tensor_exponent_histogram(payload, dtype)
        .map_err(|err| format!("{label}: histogram failed: {err}"))?;
    let from_histogram =
        modes::float_tensor_encoded_len_from_histogram(dtype, shape, &histogram, &limits)
            .map_err(|err| format!("{label}: histogram predict failed: {err}"))?;
    if from_histogram != predicted {
        return Err(format!(
            "{label}: payload prediction {predicted} != histogram prediction {from_histogram}"
        ));
    }
    if std::env::var_os("WEFT_FLOAT_TENSOR_LEN_TRACE").is_some() {
        println!("{label}: {predicted} bytes (exact)");
    }
    Ok(())
}

/// Build a payload whose elements are all `word` (little-endian, truncated to
/// the dtype width).
fn uniform_payload(dtype: FloatDtype, count: u64, word: u64) -> Vec<u8> {
    let bytes = dtype.element_size() as usize;
    let mut out = Vec::with_capacity(count as usize * bytes);
    for _ in 0..count {
        out.extend_from_slice(&word.to_le_bytes()[..bytes]);
    }
    out
}

#[test]
fn predictor_matches_encoder_across_dtypes_and_sizes() {
    let limits = Limits::default();
    let mut mismatches = 0usize;
    let mut cases = 0usize;
    let sizes: [u64; 9] = [1, 2, 3, 4, 7, 16, 257, 4096, 65_537];
    for dtype in float_dtypes() {
        for count in sizes {
            let shape = [count];
            let bytes = (count * dtype.element_size()) as usize;
            let payload = deterministic(bytes, 0x1234_5678 + count);
            cases += 1;
            if let Err(err) = check_case(
                dtype,
                &shape,
                &payload,
                &format!("random {dtype:?}/{count}"),
            ) {
                eprintln!("MISMATCH {err}");
                mismatches += 1;
            }
        }
    }
    println!("random payloads: {cases} cases, {mismatches} mismatches");
    assert_eq!(mismatches, 0, "size prediction must be exact");
    // Keep `limits` used even if the loop above changes shape.
    let _ = &limits;
}

#[test]
fn predictor_matches_encoder_for_degenerate_histograms() {
    let dtype = FloatDtype::Bf16;
    let limits = Limits::default();
    let mut mismatches = 0usize;

    // One symbol: every element has exponent 0x7f. The coder emits the 256-byte
    // table and no bitstream.
    let count = 1024u64;
    let payload = uniform_payload(dtype, count, (0x7fu64) << 7);
    if let Err(err) = check_case(dtype, &[count], &payload, "single-symbol") {
        eprintln!("MISMATCH {err}");
        mismatches += 1;
    }
    let histogram = modes::float_tensor_exponent_histogram(&payload, dtype).unwrap();
    assert_eq!(histogram[0x7f], count);
    assert_eq!(
        modes::float_tensor_encoded_len_from_histogram(dtype, &[count], &histogram, &limits)
            .unwrap(),
        // table + zero-length bitstream + one byte of sign+mantissa per element
        // + descriptor + frame header/trailer
        modes::encode_float_tensor(&payload, dtype, &[count], &limits)
            .unwrap()
            .len() as u64
    );

    // Two symbols, alternating exponents 0x00 and 0xff.
    let mut two = Vec::with_capacity(count as usize * 2);
    for index in 0..count {
        let exponent = if index % 2 == 0 { 0x00u64 } else { 0xffu64 };
        two.extend_from_slice(&((exponent << 7) as u16).to_le_bytes());
    }
    if let Err(err) = check_case(dtype, &[count], &two, "two-symbol") {
        eprintln!("MISMATCH {err}");
        mismatches += 1;
    }

    // Full alphabet: take one element per exponent byte.
    let mut full = Vec::with_capacity(256 * 2);
    for exponent in 0u64..256 {
        full.extend_from_slice(&((exponent << 7) as u16).to_le_bytes());
    }
    if let Err(err) = check_case(dtype, &[256], &full, "full-alphabet") {
        eprintln!("MISMATCH {err}");
        mismatches += 1;
    }

    // Non-byte-aligned element widths exercise the mantissa rounding.
    for dtype in [FloatDtype::F16, FloatDtype::Fp8E4m3, FloatDtype::Fp8E5m2] {
        for count in [1u64, 3, 9, 257] {
            let bytes = (count * dtype.element_size()) as usize;
            let payload = deterministic(bytes, 0xfeed_0000 + count);
            if let Err(err) = check_case(
                dtype,
                &[count],
                &payload,
                &format!("rounding {dtype:?}/{count}"),
            ) {
                eprintln!("MISMATCH {err}");
                mismatches += 1;
            }
        }
    }

    println!("degenerate histograms: {mismatches} mismatches");
    assert_eq!(mismatches, 0, "size prediction must be exact");
}

#[test]
fn predictor_matches_encoder_for_bit_pattern_extremes() {
    let mut mismatches = 0usize;
    // NaN, +inf, -inf, subnormal, signed zero, and the largest finite value.
    let bf16_words: [u16; 8] = [
        0x7fc0, 0x7f80, 0xff80, 0x0001, 0x8000, 0x0000, 0x7f7f, 0xff7f,
    ];
    let count = bf16_words.len() as u64;
    let mut payload = Vec::new();
    for word in bf16_words {
        payload.extend_from_slice(&word.to_le_bytes());
    }
    if let Err(err) = check_case(FloatDtype::Bf16, &[count], &payload, "bf16 extremes") {
        eprintln!("MISMATCH {err}");
        mismatches += 1;
    }

    // The same bit patterns widened to fp32 (not numerically equal, but the
    // mode stores bit patterns, and the exponent histogram is what matters).
    let f32_words: [u32; 8] = [
        0x7fc0_0000,
        0x7f80_0000,
        0xff80_0000,
        0x0000_0001,
        0x8000_0000,
        0x0000_0000,
        0x7f7f_ffff,
        0xff7f_ffff,
    ];
    let mut payload = Vec::new();
    for word in f32_words {
        payload.extend_from_slice(&word.to_le_bytes());
    }
    if let Err(err) = check_case(FloatDtype::F32, &[8], &payload, "fp32 extremes") {
        eprintln!("MISMATCH {err}");
        mismatches += 1;
    }

    // Multi-dimensional shapes: the descriptor's shape array changes length.
    for shape in [
        vec![2u64, 2],
        vec![1, 1, 1, 1, 1],
        vec![3, 5, 7],
        vec![256, 2],
    ] {
        let count: u64 = shape.iter().product();
        let bytes = (count * 2) as usize;
        let payload = deterministic(bytes, 0xabcd_0000 + count);
        if let Err(err) = check_case(
            FloatDtype::Bf16,
            &shape,
            &payload,
            &format!("shape {shape:?}"),
        ) {
            eprintln!("MISMATCH {err}");
            mismatches += 1;
        }
    }

    println!("bit-pattern extremes and shapes: {mismatches} mismatches");
    assert_eq!(mismatches, 0, "size prediction must be exact");
}

#[test]
fn huffman_encoded_len_matches_encoder() {
    let mut mismatches = 0usize;
    // Empty plane: table only.
    if huffman::encoded_len(&[]) != 256 {
        mismatches += 1;
    }
    if huffman::encoded_len(&[]) != huffman::encode(&[]).len() {
        mismatches += 1;
    }
    // Single symbol, two symbols, and long random planes.
    let mut one = vec![7u8; 4096];
    one[0] = 7;
    for (label, input) in [
        ("single", one),
        (
            "two",
            (0..4096)
                .map(|i| if i % 2 == 0 { 3u8 } else { 9u8 })
                .collect::<Vec<_>>(),
        ),
        ("random-4k", deterministic(4096, 99)),
        ("random-1m", deterministic(1 << 20, 7)),
    ] {
        let predicted = huffman::encoded_len(&input);
        let actual = huffman::encode(&input).len();
        if predicted != actual {
            eprintln!("MISMATCH huffman {label}: predicted {predicted}, actual {actual}");
            mismatches += 1;
        }
    }
    assert_eq!(mismatches, 0, "huffman::encoded_len must be exact");
}

fn conformance_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("conformance/v1")
}

/// Every committed mode-9 golden frame: decode the payload, re-predict the
/// frame length, and compare with the vector's own bytes.
#[test]
fn predictor_reproduces_committed_mode9_vectors() {
    let limits = Limits::default();
    let root = conformance_dir();
    let mut checked = 0usize;
    let mut mismatches = 0usize;
    let mut stack = vec![root];
    while let Some(dir) = stack.pop() {
        let entries = match std::fs::read_dir(&dir) {
            Ok(entries) => entries,
            Err(_) => continue,
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
                continue;
            }
            if path.extension().and_then(|ext| ext.to_str()) != Some("json") {
                continue;
            }
            let bytes = match std::fs::read(&path) {
                Ok(bytes) => bytes,
                Err(_) => continue,
            };
            let value: serde_json::Value = match serde_json::from_slice(&bytes) {
                Ok(value) => value,
                Err(_) => continue,
            };
            // Only valid frames that carry a mode-9 body.
            if value
                .get("expect")
                .and_then(|e| e.get("ok"))
                .and_then(|ok| ok.as_bool())
                != Some(true)
            {
                continue;
            }
            let Some(hex) = value.get("frame_hex").and_then(|h| h.as_str()) else {
                continue;
            };
            let frame_bytes = match hex::decode(hex) {
                Ok(bytes) => bytes,
                Err(_) => continue,
            };
            let frame = match Frame::parse(&frame_bytes, &limits) {
                Ok(frame) => frame,
                Err(_) => continue,
            };
            if frame.header.mode != Mode::FloatTensor {
                continue;
            }
            let metadata = match modes::float_tensor_metadata(&frame, &limits) {
                Ok(metadata) => metadata,
                Err(err) => {
                    eprintln!("MISMATCH {}: metadata: {err}", path.display());
                    mismatches += 1;
                    continue;
                }
            };
            let payload = match modes::decode_float_tensor(&frame, &limits) {
                Ok(payload) => payload,
                Err(err) => {
                    eprintln!("MISMATCH {}: decode: {err}", path.display());
                    mismatches += 1;
                    continue;
                }
            };
            let predicted =
                modes::float_tensor_encoded_len(&payload, metadata.dtype, &metadata.shape, &limits);
            match predicted {
                Ok(predicted) if predicted == frame_bytes.len() as u64 => checked += 1,
                Ok(predicted) => {
                    eprintln!(
                        "MISMATCH {}: predicted {predicted}, vector frame is {} bytes",
                        path.display(),
                        frame_bytes.len()
                    );
                    mismatches += 1;
                }
                Err(err) => {
                    eprintln!("MISMATCH {}: predict: {err}", path.display());
                    mismatches += 1;
                }
            }
        }
    }
    println!("committed mode-9 vectors: {checked} matched, {mismatches} mismatched");
    assert!(
        checked >= 3,
        "expected at least the committed bf16/fp16/fp32 mode-9 vectors, found {checked}"
    );
    assert_eq!(
        mismatches, 0,
        "committed mode-9 vectors must predict exactly"
    );
    // The resolver-free decode path above is the same one the CLI uses; keep
    // `MapResolver` imported so a future delta-bearing mode-9 vector compiles.
    let _ = MapResolver::new();
}
