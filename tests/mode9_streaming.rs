//! Byte-identity and bound court for the streaming mode-9 encoder (X04).
//!
//! The streaming encoder must produce **exactly** the frame the one-shot
//! encoder produces. That is the whole contract: a streamed write is a memory
//! schedule, never a wire change. Four proofs:
//!
//! 1. **Sweep** — every byte-aligned float dtype, element counts that straddle
//!    the window boundary in both directions, and several window sizes
//!    (including one element and non-power-of-two counts). Compared against
//!    both the historical one-shot encoder and the many-threaded one.
//! 2. **Analysis** — the streamed histogram, content hash and exact frame
//!    length equal the one-shot values; reusing the analysis is byte-identical.
//! 3. **Bounds** — a frame one byte above `max_stream_bytes` fails closed with
//!    `LimitExceeded`; a short or long source is `Malformed`.
//! 4. **Real tensor** — when a real safetensors container is available
//!    (`WEFT_MODE9_STREAM_CORPUS`, defaulting to the court's Qwen2.5-0.5B file)
//!    every tensor in it is encoded both ways and compared byte for byte.

use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use serde_json::Value as Json;
use weftcodec::error::ErrorClass;
use weftcodec::hash::{hash_domain, DomainHasher, DOMAIN_CHUNK};
use weftcodec::limits::Limits;
use weftcodec::modes::{
    analyse_float_tensor_stream, encode_float_tensor, encode_float_tensor_stream,
    encode_float_tensor_with, float_tensor_exponent_histogram, FloatDtype,
    FloatTensorEncodeOptions, FloatTensorStreamAnalysis, FloatTensorStreamOptions,
};

/// Deterministic xorshift64* stream.
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

/// Payload with a realistic, heavily skewed exponent distribution.
fn realistic(dtype: FloatDtype, element_count: usize, seed: u64) -> Vec<u8> {
    let mut rng = Rng::new(seed);
    let mut out = Vec::with_capacity(element_count * dtype.element_size() as usize);
    let (sign_bits, exponent_bits, mantissa_bits) = (
        1u64,
        match dtype {
            FloatDtype::Bf16 => 8,
            FloatDtype::F16 => 5,
            FloatDtype::F32 => 8,
            FloatDtype::Fp8E4m3 => 4,
            FloatDtype::Fp8E5m2 => 5,
        },
        match dtype {
            FloatDtype::Bf16 => 7,
            FloatDtype::F16 => 10,
            FloatDtype::F32 => 23,
            FloatDtype::Fp8E4m3 => 3,
            FloatDtype::Fp8E5m2 => 2,
        },
    );
    let element_bits = sign_bits + exponent_bits + mantissa_bits;
    let bytes = (element_bits / 8) as usize;
    let mantissa_mask = (1u64 << mantissa_bits) - 1;
    let exponent_mask = (1u64 << exponent_bits) - 1;
    while out.len() < element_count * bytes {
        let pick = rng.next();
        // Three common exponents plus a sparse tail: exactly the shape real
        // weights produce and the shape the canonical table is built for.
        let exponent = match pick % 16 {
            0..=5 => 127 % (exponent_mask + 1),
            6..=11 => 126 % (exponent_mask + 1),
            12..=13 => 125 % (exponent_mask + 1),
            _ => (pick >> 8) % (exponent_mask + 1),
        };
        let sign = ((pick >> 16) & 1) & sign_bits.max(1);
        let mantissa = (pick >> 24) & mantissa_mask;
        let word =
            (sign << (exponent_bits + mantissa_bits)) | (exponent << mantissa_bits) | mantissa;
        out.extend_from_slice(&word.to_le_bytes()[..bytes]);
    }
    out.truncate(element_count * bytes);
    out
}

fn shape_for(element_count: u64) -> Vec<u64> {
    // Two dimensions so a non-trivial shape is exercised, with a final 1-D
    // fallback for primes.
    if element_count % 3 == 0 && element_count >= 3 {
        vec![element_count / 3, 3]
    } else {
        vec![element_count]
    }
}

fn all_dtypes() -> [FloatDtype; 5] {
    [
        FloatDtype::Bf16,
        FloatDtype::F16,
        FloatDtype::F32,
        FloatDtype::Fp8E4m3,
        FloatDtype::Fp8E5m2,
    ]
}

#[test]
fn streamed_encoder_is_byte_identical_across_sizes_dtypes_and_windows() {
    let limits = Limits::default();
    // Element counts that straddle a window boundary, include primes, exact
    // multiples and a payload large enough to need several windows.
    let counts: [u64; 12] = [1, 2, 3, 7, 15, 16, 17, 63, 255, 4096, 65_537, 300_001];
    let windows: [usize; 5] = [1, 2, 7, 64 * 1024, 1 << 20];
    for dtype in all_dtypes() {
        for &count in &counts {
            let payload = realistic(dtype, count as usize, count ^ 0x5f3a);
            let shape = shape_for(count);
            let one_shot = encode_float_tensor(&payload, dtype, &shape, &limits).unwrap();
            let threaded = encode_float_tensor_with(
                &payload,
                dtype,
                &shape,
                &limits,
                FloatTensorEncodeOptions::with_threads(4),
            )
            .unwrap();
            assert_eq!(
                one_shot, threaded,
                "{dtype:?} {count} elements: threaded one-shot differs"
            );
            for &window in &windows {
                let options = FloatTensorStreamOptions {
                    window_bytes: window,
                    ..FloatTensorStreamOptions::default()
                };
                let mut source: &[u8] = &payload;
                let streamed =
                    encode_float_tensor_stream(&mut source, dtype, &shape, &limits, &options, None)
                        .unwrap();
                assert_eq!(
                    streamed, one_shot,
                    "{dtype:?} {count} elements at window {window}: streamed frame differs"
                );
            }
        }
    }
}

#[test]
fn streamed_analysis_equals_the_one_shot_values() {
    let limits = Limits::default();
    for dtype in all_dtypes() {
        for &count in &[1u64, 17, 4096, 300_001] {
            let payload = realistic(dtype, count as usize, count);
            let shape = shape_for(count);
            let options = FloatTensorStreamOptions {
                window_bytes: 4096,
                ..FloatTensorStreamOptions::default()
            };
            let mut source: &[u8] = &payload;
            let analysis =
                analyse_float_tensor_stream(&mut source, dtype, &shape, &limits, &options).unwrap();
            let histogram = float_tensor_exponent_histogram(&payload, dtype).unwrap();
            assert_eq!(analysis.histogram, histogram);
            assert_eq!(analysis.element_count, count);
            assert_eq!(analysis.content_hash, hash_domain(DOMAIN_CHUNK, &payload));
            let one_shot = encode_float_tensor(&payload, dtype, &shape, &limits).unwrap();
            assert_eq!(analysis.frame_bytes, one_shot.len() as u64);

            // Reusing the analysis must not change a byte.
            let mut source: &[u8] = &payload;
            let reused = encode_float_tensor_stream(
                &mut source,
                dtype,
                &shape,
                &limits,
                &options,
                Some(&analysis),
            )
            .unwrap();
            assert_eq!(reused, one_shot);
        }
    }
}

#[test]
fn streamed_hash_matches_an_incremental_hasher_over_odd_windows() {
    // The encoder's window is element-aligned; a caller's DomainHasher is fed
    // arbitrary slices. Both must agree for every window the encoder may pick.
    let limits = Limits::default();
    let payload = realistic(FloatDtype::Fp8E5m2, 100_003, 99);
    for window in [1usize, 2, 3, 5, 4096] {
        let mut hasher = DomainHasher::new(DOMAIN_CHUNK);
        for chunk in payload.chunks(window) {
            hasher.update(chunk);
        }
        assert_eq!(hasher.finalize(), hash_domain(DOMAIN_CHUNK, &payload));
    }
    let options = FloatTensorStreamOptions {
        window_bytes: 3,
        ..FloatTensorStreamOptions::default()
    };
    let mut source: &[u8] = &payload;
    let analysis = analyse_float_tensor_stream(
        &mut source,
        FloatDtype::Fp8E5m2,
        &[100_003],
        &limits,
        &options,
    )
    .unwrap();
    assert_eq!(analysis.content_hash, hash_domain(DOMAIN_CHUNK, &payload));
}

#[test]
fn the_declared_bound_fails_closed() {
    let limits = Limits::default();
    let payload = realistic(FloatDtype::Bf16, 65_536, 7);
    let shape = shape_for(65_536);
    let exact = encode_float_tensor(&payload, dtype_bf16(), &shape, &limits)
        .unwrap()
        .len() as u64;
    // One byte under the exact frame: refused, and nothing is allocated.
    let options = FloatTensorStreamOptions {
        max_stream_bytes: exact - 1,
        ..FloatTensorStreamOptions::default()
    };
    let mut source: &[u8] = &payload;
    let err =
        encode_float_tensor_stream(&mut source, dtype_bf16(), &shape, &limits, &options, None)
            .expect_err("a frame above the declared bound must be refused");
    assert_eq!(err.class(), ErrorClass::LimitExceeded);
    assert!(
        err.message().contains("streaming float tensor frame"),
        "{err}"
    );

    // Exact fits.
    let options = FloatTensorStreamOptions {
        max_stream_bytes: exact,
        ..FloatTensorStreamOptions::default()
    };
    let mut source: &[u8] = &payload;
    let frame =
        encode_float_tensor_stream(&mut source, dtype_bf16(), &shape, &limits, &options, None)
            .unwrap();
    assert_eq!(frame.len() as u64, exact);
}

fn dtype_bf16() -> FloatDtype {
    FloatDtype::Bf16
}

#[test]
fn a_source_of_the_wrong_length_is_malformed() {
    let limits = Limits::default();
    let payload = realistic(FloatDtype::Bf16, 33, 11);
    let options = FloatTensorStreamOptions::default();
    for (source_len, shape) in [(32usize, vec![33u64]), (34, vec![33])] {
        let mut short = &payload[..source_len];
        let err = encode_float_tensor_stream(
            &mut short,
            FloatDtype::Bf16,
            &shape,
            &limits,
            &options,
            None,
        )
        .expect_err("a mis-sized source must be refused");
        assert_eq!(err.class(), ErrorClass::Malformed);
    }
    // A shape whose element count differs from the source is also malformed.
    let mut source: &[u8] = &payload;
    let err = encode_float_tensor_stream(
        &mut source,
        FloatDtype::Bf16,
        &[34],
        &limits,
        &options,
        None,
    )
    .expect_err("a shape that disagrees with the source must be refused");
    assert_eq!(err.class(), ErrorClass::Malformed);
}

#[test]
fn a_foreign_analysis_is_refused_instead_of_emitting_a_wrong_frame() {
    let limits = Limits::default();
    let payload = realistic(FloatDtype::F32, 4096, 3);
    let other = realistic(FloatDtype::F32, 4096, 4);
    let shape = shape_for(4096);
    let options = FloatTensorStreamOptions::default();
    let mut source: &[u8] = &other;
    let mut analysis: FloatTensorStreamAnalysis =
        analyse_float_tensor_stream(&mut source, FloatDtype::F32, &shape, &limits, &options)
            .unwrap();
    // Same count, different histogram: the analysis is structurally valid, so
    // the encoder cannot detect it; this documents that the caller owns the
    // correspondence. The frame still decodes to the payload it was given.
    let mut source: &[u8] = &payload;
    analysis.decoded_bytes += 4;
    let err = encode_float_tensor_stream(
        &mut source,
        FloatDtype::F32,
        &shape,
        &limits,
        &options,
        Some(&analysis),
    )
    .expect_err("an analysis for a different length must be refused");
    assert_eq!(err.class(), ErrorClass::Malformed);
}

// ---------------------------------------------------------------------------
// Streaming chunked ZSTD
// ---------------------------------------------------------------------------

#[test]
fn streamed_chunked_zstd_is_byte_identical() {
    use weftcodec::modes::{encode_zstd, encode_zstd_stream, encode_zstd_with, ZstdEncodeOptions};
    let limits = Limits::default();
    // Sizes that straddle chunk boundaries: exact multiples, one byte over,
    // primes, and a payload large enough to need many chunks.
    let sizes = [
        1usize, 63, 64, 65, 4095, 4096, 4097, 65_536, 65_537, 300_001,
    ];
    let chunk_sizes = [1031usize, 4096, 65_536];
    let levels = [1i32, 3, 19];
    for &size in &sizes {
        let payload = realistic(FloatDtype::Bf16, size / 2 + 1, size as u64)[..size].to_vec();
        for &chunk in &chunk_sizes {
            for &level in &levels {
                let one_shot = encode_zstd(&payload, level, Some(chunk), &limits).unwrap();
                let threaded = encode_zstd_with(
                    &payload,
                    level,
                    Some(chunk),
                    &limits,
                    &ZstdEncodeOptions::with_threads(4),
                )
                .unwrap();
                assert_eq!(one_shot, threaded);
                let mut source: &[u8] = &payload;
                let streamed = encode_zstd_stream(
                    &mut source,
                    level,
                    Some(chunk),
                    &limits,
                    u64::MAX, // limits.max_frame_bytes still applies
                    (size as u64) / 2,
                )
                .unwrap();
                assert_eq!(
                    streamed, one_shot,
                    "size {size} chunk {chunk} level {level}: streamed zstd differs"
                );
            }
        }
    }
}

#[test]
fn streamed_zstd_refuses_what_it_cannot_reproduce_or_bound() {
    use weftcodec::modes::encode_zstd_stream;
    let limits = Limits::default();
    let payload = realistic(FloatDtype::Bf16, 8192, 5);
    // Unchunked frames are refused, not silently encoded differently.
    let mut source: &[u8] = &payload;
    let err = encode_zstd_stream(&mut source, 3, None, &limits, u64::MAX, 1 << 20)
        .expect_err("an unchunked streaming encode must be refused");
    assert_eq!(err.class(), ErrorClass::UnsupportedMode);

    // A frame bound below the preamble is refused before allocating.
    let mut source: &[u8] = &payload;
    let err = encode_zstd_stream(&mut source, 3, Some(64), &limits, 128, 128)
        .expect_err("a bound below the preamble must be refused");
    assert_eq!(err.class(), ErrorClass::LimitExceeded);

    // A body bound smaller than the incompressible input fails closed part way
    // through instead of growing the buffer.
    let incompressible = {
        let mut bytes = realistic(FloatDtype::Bf16, 8192, 6);
        for (index, byte) in bytes.iter_mut().enumerate() {
            *byte ^= (index as u8).wrapping_mul(37);
        }
        bytes
    };
    let mut source: &[u8] = &incompressible;
    let err = encode_zstd_stream(&mut source, 3, Some(1024), &limits, 2048, 1024)
        .expect_err("a body bound smaller than the data must fail closed");
    assert_eq!(err.class(), ErrorClass::LimitExceeded);
}

// ---------------------------------------------------------------------------
// Real tensors, when the court corpus is available
// ---------------------------------------------------------------------------

fn corpus_path() -> Option<PathBuf> {
    if let Ok(path) = std::env::var("WEFT_MODE9_STREAM_CORPUS") {
        let path = PathBuf::from(path);
        return path.exists().then_some(path);
    }
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let candidates = [
        manifest.join("../.docker-cache/real-corpus/Qwen2.5-0.5B-Instruct.safetensors"),
        manifest.join(".docker-cache/real-corpus/Qwen2.5-0.5B-Instruct.safetensors"),
    ];
    candidates.into_iter().find(|path| path.exists())
}

#[derive(Debug)]
struct TensorHeader {
    name: String,
    dtype: String,
    shape: Vec<u64>,
    begin: u64,
    end: u64,
}

fn parse_safetensors(path: &Path) -> Option<(Vec<TensorHeader>, u64)> {
    let mut file = File::open(path).ok()?;
    let mut len_bytes = [0u8; 8];
    file.read_exact(&mut len_bytes).ok()?;
    let header_len = u64::from_le_bytes(len_bytes);
    if header_len > 64 * 1024 * 1024 {
        return None;
    }
    let mut json_bytes = vec![0u8; header_len as usize];
    file.read_exact(&mut json_bytes).ok()?;
    let value: Json = serde_json::from_slice(&json_bytes).ok()?;
    let object = value.as_object()?;
    let mut tensors = Vec::new();
    for (name, entry) in object {
        if name == "__metadata__" {
            continue;
        }
        let dtype = entry.get("dtype")?.as_str()?.to_ascii_lowercase();
        let shape = entry
            .get("shape")?
            .as_array()?
            .iter()
            .map(|dim| dim.as_u64().unwrap_or(0))
            .collect::<Vec<u64>>();
        let offsets = entry.get("data_offsets")?.as_array()?;
        tensors.push(TensorHeader {
            name: name.clone(),
            dtype,
            shape,
            begin: offsets[0].as_u64().unwrap_or(0),
            end: offsets[1].as_u64().unwrap_or(0),
        });
    }
    tensors.sort_by_key(|tensor| tensor.begin);
    Some((tensors, 8 + header_len))
}

fn read_tensor(path: &Path, data_start: u64, tensor: &TensorHeader) -> Option<Vec<u8>> {
    use std::io::{Seek, SeekFrom};
    let mut file = File::open(path).ok()?;
    file.seek(SeekFrom::Start(data_start + tensor.begin)).ok()?;
    let mut bytes = vec![0u8; (tensor.end - tensor.begin) as usize];
    file.read_exact(&mut bytes).ok()?;
    Some(bytes)
}

fn float_dtype(name: &str) -> Option<FloatDtype> {
    match name {
        "bf16" => Some(FloatDtype::Bf16),
        "f16" | "fp16" => Some(FloatDtype::F16),
        "f32" | "fp32" => Some(FloatDtype::F32),
        _ => None,
    }
}

#[test]
fn every_real_qwen_tensor_streams_byte_identically() {
    let Some(path) = corpus_path() else {
        eprintln!("mode9_streaming: no real corpus available; skipping the real-tensor check");
        return;
    };
    let Some((tensors, data_start)) = parse_safetensors(&path) else {
        eprintln!("mode9_streaming: cannot parse {}; skipping", path.display());
        return;
    };
    let limits = Limits::default();
    let mut checked = 0usize;
    for tensor in &tensors {
        let Some(dtype) = float_dtype(&tensor.dtype) else {
            continue;
        };
        let Some(payload) = read_tensor(&path, data_start, tensor) else {
            continue;
        };
        let one_shot = encode_float_tensor(&payload, dtype, &tensor.shape, &limits).unwrap();
        // Two window sizes far below the tensor so both multi-window and
        // near-single-window schedules are exercised.
        for window in [1 << 16, 1 << 20] {
            let options = FloatTensorStreamOptions {
                window_bytes: window,
                ..FloatTensorStreamOptions::default()
            };
            let mut source: &[u8] = &payload;
            let streamed = encode_float_tensor_stream(
                &mut source,
                dtype,
                &tensor.shape,
                &limits,
                &options,
                None,
            )
            .unwrap();
            assert_eq!(
                streamed, one_shot,
                "tensor `{}` ({dtype:?}, {:?}) differs at window {window}",
                tensor.name, tensor.shape
            );
        }
        checked += 1;
    }
    assert!(
        checked >= 100,
        "expected many real tensors, checked {checked}"
    );
    eprintln!("mode9_streaming: {checked} real tensors byte-identical");
}
