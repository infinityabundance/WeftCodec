//! Decode-throughput floor for the shipped hot loops, single-threaded.
//!
//! Emits one JSON row per `(component, size)` plus a summary. Measured
//! components:
//!
//! * `huffman_decode` — `huffman::decode` of a BF16-style exponent plane
//!   (256-byte canonical table + bit unpack), i.e. exactly what mode 9, 11,
//!   15 and 17 run for one plane.
//! * `plane_join` — `join_float_tensor_payload` on the two planes, with
//!   decoding excluded.
//! * `mode9_decode` — `decode_float_tensor` on a whole frame: descriptor
//!   parse, Huffman decode, join, payload hash, content hash.
//! * `mode9_encode` — `encode_float_tensor`, for reference only.
//!
//! Methodology:
//! * dtypes bf16 and fp32; decoded sizes 128 KiB, 1 MiB, 16 MiB.
//! * Payload words are weight-like: uniformly random mantissa/sign bits and a
//!   skewed exponent distribution (halving over the low half of the exponent
//!   range), so the coder sees realistic symbol statistics.
//! * One warm-up call, then five measured calls; the median is reported.
//! * Every measured decode output is checked byte-identical to the source
//!   payload, so a row can never report a faster wrong answer.
//! * Throughput is decoded MiB/s on one thread with a warm in-process input.
//!
//! This is a single-host, warm-cache microbenchmark of the coder. It makes no
//! claim about end-to-end store throughput, cold cache, or other machines.

use std::time::Instant;

use serde_json::json;

use weftcodec::huffman;
use weftcodec::limits::Limits;
use weftcodec::modes::{
    decode_float_tensor, encode_float_tensor, join_float_tensor_payload,
    split_float_tensor_payload, FloatDtype,
};

fn xorshift(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(|a, b| a.partial_cmp(b).expect("finite timings"));
    let middle = values.len() / 2;
    if values.len() % 2 == 1 {
        values[middle]
    } else {
        (values[middle - 1] + values[middle]) / 2.0
    }
}

/// Weight-like element bytes for `dtype`.
fn payload(dtype: FloatDtype, bytes: usize, seed: u64) -> Vec<u8> {
    let element_size = dtype.element_size() as usize;
    let count = bytes / element_size;
    let (sign_bits, exponent_bits, mantissa_bits) = dtype.field_bits();
    let exponent_values = 1u64 << exponent_bits;
    // Low half of the exponent range, halving frequency.
    let half = exponent_values / 2;
    let mantissa_mask = if mantissa_bits == 64 {
        u64::MAX
    } else {
        (1u64 << mantissa_bits) - 1
    };
    let mut state = seed | 1;
    let mut out = Vec::with_capacity(count * element_size);
    for _ in 0..count {
        let r = xorshift(&mut state);
        // Halving distribution: pick bucket from leading zero count.
        let bucket = (r.leading_zeros() as u64) % half.max(1);
        let exponent = half.saturating_sub(1).saturating_sub(bucket) + 1;
        let mantissa = xorshift(&mut state) & mantissa_mask;
        let sign = xorshift(&mut state) & ((1 << sign_bits) - 1);
        let word = (sign << (exponent_bits + mantissa_bits))
            | ((exponent & (exponent_values - 1)) << mantissa_bits)
            | mantissa;
        out.extend_from_slice(&word.to_le_bytes()[..element_size]);
    }
    out
}

fn cpu_features() -> Vec<&'static str> {
    let mut features = Vec::new();
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("sse2") {
            features.push("sse2");
        }
        if std::is_x86_feature_detected!("avx2") {
            features.push("avx2");
        }
        if std::is_x86_feature_detected!("avx512f") {
            features.push("avx512f");
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            features.push("neon");
        }
    }
    if features.is_empty() {
        features.push("scalar_only");
    }
    features
}

fn main() {
    let reps: usize = std::env::var("WEFT_FLOOR_REPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(5);
    let sizes: Vec<(&str, usize)> = vec![
        ("128KiB", 128 * 1024),
        ("1MiB", 1024 * 1024),
        ("16MiB", 16 * 1024 * 1024),
    ];
    let dtype_names = std::env::var("WEFT_FLOOR_DTYPES").unwrap_or_else(|_| "bf16,fp32".to_owned());
    let dtypes: Vec<FloatDtype> = dtype_names
        .split(',')
        .filter(|s| !s.is_empty())
        .map(|name| FloatDtype::from_name(name.trim()).expect("known dtype"))
        .collect();
    let limits = Limits::default();
    let features = cpu_features();
    println!(
        "{}",
        json!({
            "schema": "weft.decode-floor-bench.v1",
            "event": "header",
            "reps": reps,
            "cpu_features": features,
            "note": "single-threaded, warm in-process input; coder-only microbenchmark",
        })
    );

    for dtype in dtypes {
        for (size_name, size) in &sizes {
            let bytes = *size;
            let source = payload(dtype, bytes, 0x9e37_79b9_7f4a_7c15);
            let element_count = (bytes / dtype.element_size() as usize) as u64;
            let shape = [element_count];
            let frame = encode_float_tensor(&source, dtype, &shape, &limits).expect("encode");
            let parsed = weftcodec::Frame::parse(&frame, &limits).expect("parse");

            let (exponent_plane, mantissa_plane) =
                split_float_tensor_payload(&source, dtype, element_count).expect("split");

            // Warm up.
            let decoded = decode_float_tensor(&parsed, &limits).expect("decode");
            assert_eq!(decoded, source, "warm-up mode-9 decode mismatch");
            let joined = join_float_tensor_payload(
                &huffman::decode(&exponent_plane, element_count as usize).expect("huffman"),
                &mantissa_plane,
                dtype,
                element_count,
                &limits,
            )
            .expect("join");
            assert_eq!(joined, source, "warm-up plane join mismatch");

            let decoded_exponent =
                huffman::decode(&exponent_plane, element_count as usize).expect("huffman");

            let mut huffman_ms = Vec::new();
            let mut join_ms = Vec::new();
            let mut decode_ms = Vec::new();
            let mut encode_ms = Vec::new();

            for _ in 0..reps {
                let start = Instant::now();
                let out =
                    huffman::decode(&exponent_plane, element_count as usize).expect("huffman");
                huffman_ms.push(start.elapsed().as_secs_f64() * 1e3);
                assert_eq!(out, decoded_exponent, "huffman repeat mismatch");

                let start = Instant::now();
                let out = join_float_tensor_payload(
                    &decoded_exponent,
                    &mantissa_plane,
                    dtype,
                    element_count,
                    &limits,
                )
                .expect("join");
                join_ms.push(start.elapsed().as_secs_f64() * 1e3);
                assert_eq!(out, source, "plane join repeat mismatch");

                let start = Instant::now();
                let out = decode_float_tensor(&parsed, &limits).expect("decode");
                decode_ms.push(start.elapsed().as_secs_f64() * 1e3);
                assert_eq!(out, source, "mode-9 decode repeat mismatch");

                let start = Instant::now();
                let out = encode_float_tensor(&source, dtype, &shape, &limits).expect("encode");
                encode_ms.push(start.elapsed().as_secs_f64() * 1e3);
                assert_eq!(out, frame, "mode-9 encode repeat mismatch");
            }

            let mib = bytes as f64 / (1024.0 * 1024.0);
            let physical = frame.len();
            for (component, samples) in [
                ("huffman_decode", &huffman_ms),
                ("plane_join", &join_ms),
                ("mode9_decode", &decode_ms),
                ("mode9_encode", &encode_ms),
            ] {
                let ms = median(samples.clone());
                println!(
                    "{}",
                    json!({
                        "schema": "weft.decode-floor-bench.v1",
                        "event": "row",
                        "component": component,
                        "dtype": dtype.as_str(),
                        "size": size_name,
                        "decoded_bytes": bytes,
                        "physical_bytes": physical,
                        "median_ms": ms,
                        "mib_per_s": mib / (ms / 1e3),
                        "samples_ms": samples,
                    })
                );
            }
            // Isolated exponent-plane unpack rate, in decoded exponent MiB/s.
            let exponent_mib = element_count as f64 / (1024.0 * 1024.0);
            let huffman_median = median(huffman_ms.clone());
            println!(
                "{}",
                json!({
                    "schema": "weft.decode-floor-bench.v1",
                    "event": "row",
                    "component": "huffman_exponent_mib_per_s",
                    "dtype": dtype.as_str(),
                    "size": size_name,
                    "exponent_plane_mib": exponent_mib,
                    "median_ms": huffman_median,
                    "mib_per_s": exponent_mib / (huffman_median / 1e3),
                })
            );
        }
    }
}
